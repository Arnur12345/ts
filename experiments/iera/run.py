"""Run the IERA four-pair go/no-go pilot on cached BioMedCLIP patches."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.residuals.data import load_config, load_dataset
from experiments.residuals.metrics import _average_precision, _auc, _ece, select_temperature, select_threshold

from .episodes import (
    PILOT_PAIRS,
    eligible_directed_pairs,
    generate_pair_episodes,
    patient_counts,
    split_indices,
    stratum_pools,
    validate_pair_episodes,
)
from .labels import restore_raw_target_status
from .model import IERA, METHODS
from .patch_cache import load_patch_cache


def _ids(data, names: tuple[str, str]) -> tuple[int, int]:
    missing = [name for name in names if name not in data.class_names]
    if missing:
        raise ValueError(f"cache lacks labels {missing}")
    return data.class_names.index(names[0]), data.class_names.index(names[1])


def _gather(patches: torch.Tensor, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return patches[indices].to(device=device, dtype=torch.float32, non_blocking=True)


def _episode_batch(patches, episode, start, end, shot, device):
    return (
        _gather(patches, episode["positive"][start:end, :, :shot], device),
        _gather(patches, episode["negative"][start:end, :, :shot], device),
        _gather(patches, episode["query"][start:end], device),
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train(model, method, patches, data, config, args, device, run_seed) -> list[float]:
    if args.train_steps == 0 or method == "positive_prototype":
        return []
    train_indices = split_indices(data, "train", args.split_seed)
    evaluation_targets = {_ids(data, pair)[0] for pair in PILOT_PAIRS}
    base_ids = [data.class_names.index(name) for name in config["class_partitions"]["base"] if data.class_names.index(name) not in evaluation_targets]
    pairs = eligible_directed_pairs(
        data, train_indices, base_ids, args.min_stratum_patients,
        confounder_ids=base_ids,
    )
    if not pairs:
        raise ValueError("no eligible base-only meta-training target/confounder pairs")
    random.Random(run_seed).shuffle(pairs)
    pairs = pairs[: min(12, len(pairs))]
    bank = []
    episodes_per_pair = max(2, math.ceil(min(args.train_steps, 120) / len(pairs)))
    for pair_index, (target, confounder) in enumerate(pairs):
        generated = generate_pair_episodes(
            data, train_indices, target, confounder, episodes_per_pair,
            args.train_shot, 1, run_seed + 10_000 + pair_index,
            min_stratum_patients=args.min_stratum_patients,
        )
        for episode_index in range(episodes_per_pair):
            bank.append((generated, episode_index))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(run_seed)
    losses = []
    model.train()
    for step in range(args.train_steps):
        generated, episode_index = bank[int(torch.randint(len(bank), (1,), generator=generator))]
        positive, negative, query = _episode_batch(patches, generated, episode_index, episode_index + 1, args.train_shot, device)
        targets = generated["targets"][episode_index : episode_index + 1].to(device)
        logits = model(positive, negative, query, method)
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0:
            print(f"training {method} seed {run_seed}: {step + 1}/{args.train_steps}, loss={statistics.mean(losses[-25:]):.4f}", flush=True)
    model.eval()
    return losses


def _score(model, patches, episodes, shot, method, batch_size, device):
    logits, panel_zero, panel_one = [], [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(episodes["positive"]), batch_size):
            end = min(start + batch_size, len(episodes["positive"]))
            positive, negative, query = _episode_batch(patches, episodes, start, end, shot, device)
            logits.append(model(positive, negative, query, method).cpu())
            zero, one = model.swapped_logits(positive, negative, query, method)
            panel_zero.append(zero.cpu())
            panel_one.append(one.cpu())
    return torch.cat(logits).flatten(), torch.cat(panel_zero).flatten(), torch.cat(panel_one).flatten()


def _metrics(logits, panel_zero, panel_one, targets, nuisance, temperature, threshold):
    probability = torch.sigmoid(logits / temperature).clamp(1e-7, 1 - 1e-7)
    prediction = probability.ge(threshold)
    target = targets.bool()
    tp, fp = (target & prediction).sum(), (~target & prediction).sum()
    fn = (target & ~prediction).sum()
    raw_shift = (panel_one - panel_zero).abs()
    shift_scale = torch.cat((panel_zero, panel_one)).std().clamp_min(1e-6)
    panel_zero_prediction = panel_zero.ge(0)
    panel_one_prediction = panel_one.ge(0)
    panel_zero_error = panel_zero_prediction.ne(target)
    panel_one_error = panel_one_prediction.ne(target)
    result = {
        "auroc": _auc(target, probability),
        "auprc": _average_precision(target, probability),
        "f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
        "brier": float((probability - targets).square().mean()),
        "ece": _ece(probability, target),
        "false_positive_c0d1": float(probability[(~target) & nuisance.eq(1)].mean()),
        "sms_raw_logit": float(raw_shift.mean()),
        "sms_normalized_logit": float(raw_shift.mean() / shift_scale),
        "support_swap_flip_rate": float(panel_zero_prediction.ne(panel_one_prediction).float().mean()),
        "support_swap_error_gap": float((panel_zero_error.float().mean() - panel_one_error.float().mean()).abs()),
        "worst_support_panel_error": float(torch.maximum(panel_zero_error.float().mean(), panel_one_error.float().mean())),
    }
    nuisance_auc, nuisance_auprc = [], []
    for value in (0, 1):
        mask = nuisance.eq(value)
        nuisance_auc.append(_auc(target[mask], probability[mask]))
        nuisance_auprc.append(_average_precision(target[mask], probability[mask]))
        result[f"d{value}_auroc"] = nuisance_auc[-1]
        result[f"d{value}_auprc"] = nuisance_auprc[-1]
    result["worst_nuisance_auroc"] = min(nuisance_auc)
    result["worst_nuisance_auprc"] = min(nuisance_auprc)
    return result


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    keys = ("pair", "target", "confounder", "method", "shot", "metric")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row["value"]))
    return [
        {
            **dict(zip(keys, key)),
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for key, values in groups.items()
    ]


def _decision(summary: list[dict]) -> dict:
    lookup = {(row["pair"], row["method"], int(row["shot"]), row["metric"]): row["mean"] for row in summary}
    pairs = sorted({row["pair"] for row in summary})
    available_shots = sorted({int(row["shot"]) for row in summary})
    shot = 3 if 3 in available_shots else available_shots[0]
    evidence = []
    for pair in pairs:
        baseline_sms = lookup.get((pair, "positive_prototype", shot, "sms_normalized_logit"), float("nan"))
        iera_sms = lookup.get((pair, "iera", shot, "sms_normalized_logit"), float("nan"))
        reduction = 1 - iera_sms / baseline_sms if baseline_sms > 0 else float("nan")
        worst_gain = lookup.get((pair, "iera", shot, "worst_nuisance_auroc"), float("nan")) - lookup.get((pair, "positive_prototype", shot, "worst_nuisance_auroc"), float("nan"))
        ordinary_loss = lookup.get((pair, "positive_prototype", shot, "auroc"), float("nan")) - lookup.get((pair, "iera", shot, "auroc"), float("nan"))
        evidence.append({"pair": pair, "baseline_sms": baseline_sms, "iera_sms": iera_sms, "sms_reduction": reduction, "worst_auroc_gain": worst_gain, "ordinary_auroc_loss": ordinary_loss})
    passing = [row for row in evidence if row["sms_reduction"] >= 0.30 and row["worst_auroc_gain"] >= 0.02 and row["ordinary_auroc_loss"] < 0.01]
    return {
        "status": "continue" if len(passing) >= 3 else "stop_or_revise",
        "rule": "at least three pairs: uncalibrated normalized-logit SMS reduction >=30%, worst-nuisance AUROC gain >=0.02, ordinary AUROC loss <0.01",
        "passing_pairs": len(passing),
        "evidence": evidence,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--patch-cache", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True, help="Original MIMIC CheXpert CSV; blanks remain unknown")
    parser.add_argument("--config", type=Path, default=Path("configs/mimic_cxr_protocol_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/iera/pilot"))
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--queries-per-stratum", type=int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--train-shot", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--min-stratum-patients", type=int, default=50)
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    started = time.perf_counter()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    data = load_dataset(args.embeddings, args.manifest)
    restore_raw_target_status(data, args.raw_labels)
    config = load_config(args.config)
    patches, patch_metadata = load_patch_cache(args.patch_cache, data.manifest_sha256)
    validation_indices = split_indices(data, "validate", args.split_seed)
    test_indices = split_indices(data, "test", args.split_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eligibility = []
    eligible_pairs = []
    for pair_index, names in enumerate(PILOT_PAIRS):
        target_id, confounder_id = _ids(data, names)
        partition_counts = {
            "validation": patient_counts(data, stratum_pools(data, validation_indices, target_id, confounder_id)),
            "test": patient_counts(data, stratum_pools(data, test_indices, target_id, confounder_id)),
        }
        eligible = all(
            min(counts.values()) >= args.min_stratum_patients
            for counts in partition_counts.values()
        )
        eligibility.append(
            {
                "pair": f"{names[0]}__{names[1]}",
                "target": names[0],
                "confounder": names[1],
                "minimum_required_per_stratum": args.min_stratum_patients,
                "counts": {
                    partition: {f"c{key[0]}d{key[1]}": value for key, value in counts.items()}
                    for partition, counts in partition_counts.items()
                },
                "eligible": eligible,
            }
        )
        if eligible:
            eligible_pairs.append((pair_index, names, target_id, confounder_id))
    (args.output_dir / "eligibility.json").write_text(json.dumps(eligibility, indent=2) + "\n", encoding="utf-8")
    if not eligible_pairs:
        raise ValueError(
            "none of the four evaluation pairs satisfies --min-stratum-patients "
            "in both validation and test; inspect eligibility.json"
        )

    max_shot = max(args.shots)
    episode_sets = {}
    for pair_index, names, target_id, confounder_id in eligible_pairs:
        for seed in args.seeds:
            validation = generate_pair_episodes(
                data, validation_indices, target_id, confounder_id, args.episodes,
                max_shot, args.queries_per_stratum,
                args.seed + pair_index * 10_000 + seed,
                min_stratum_patients=args.min_stratum_patients,
            )
            test = generate_pair_episodes(
                data, test_indices, target_id, confounder_id, args.episodes,
                max_shot, args.queries_per_stratum,
                args.seed + 100_000 + pair_index * 10_000 + seed,
                min_stratum_patients=args.min_stratum_patients,
            )
            validate_pair_episodes(validation, data)
            validate_pair_episodes(test, data)
            episode_sets[(pair_index, seed)] = (validation, test)
            torch.save(
                {"validation": validation, "test": test},
                args.output_dir / f"episodes_{pair_index:02d}_seed_{seed:03d}.pt",
            )

    rows = []
    training_runs = []
    for seed in args.seeds:
        run_seed = args.seed + seed
        for method in args.methods:
            _set_seed(run_seed)
            model = IERA(patches.shape[-1], args.projection_dim).to(device)
            losses = _train(model, method, patches, data, config, args, device, run_seed)
            if method != "positive_prototype":
                torch.save(
                    {
                        "method": method,
                        "seed": seed,
                        "training_seed": run_seed,
                        "state_dict": model.state_dict(),
                        "parameters": model.parameters_dict(),
                    },
                    args.output_dir / f"model_{method}_seed_{seed:03d}.pt",
                )
            training_runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "training_seed": run_seed,
                    "final_loss": losses[-1] if losses else None,
                    "learned_parameters": model.parameters_dict() if method != "positive_prototype" else None,
                }
            )
            for pair_index, names, _target_id, _confounder_id in eligible_pairs:
                pair_name = f"{names[0]}__{names[1]}"
                validation, test = episode_sets[(pair_index, seed)]
                for shot in args.shots:
                    val_logits, _, _ = _score(model, patches, validation, shot, method, args.episode_batch_size, device)
                    val_targets = validation["targets"].flatten()
                    temperature = select_temperature(val_logits[:, None], val_targets[:, None], "multi_label")
                    threshold = select_threshold(val_logits[:, None], val_targets[:, None], temperature)
                    logits, panel_zero, panel_one = _score(model, patches, test, shot, method, args.episode_batch_size, device)
                    metrics = _metrics(logits, panel_zero, panel_one, test["targets"].flatten(), test["nuisance"].flatten(), temperature, threshold)
                    for metric, value in metrics.items():
                        rows.append({"pair": pair_name, "target": names[0], "confounder": names[1], "method": method, "shot": shot, "seed": seed, "temperature": temperature, "threshold": threshold, "metric": metric, "value": value})
                    print(f"finished {pair_name}, {method}, {shot}-shot", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    summary = _summaries(rows)
    _write(args.output_dir / "per_seed_metrics.csv", rows)
    _write(args.output_dir / "summary_metrics.csv", summary)
    decision = _decision(summary)
    (args.output_dir / "decision.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "method": "Invariant Evidence-Ratio Attention",
                "patch_cache": patch_metadata,
                "raw_labels": str(args.raw_labels),
                "blank_and_uncertain_policy": "unknown/excluded",
                "shots": args.shots,
                "episodes": args.episodes,
                "seeds": args.seeds,
                "seed_semantics": "independent initialization, meta-training, validation, and test episode run",
                "patient_split": {"seed": args.split_seed, "fractions": [0.70, 0.15, 0.15]},
                "train_steps": args.train_steps,
                "meta_training_labels": "target and confounder both restricted to non-evaluation base classes",
                "training_runs": training_runs,
                "eligible_pairs": [item["pair"] for item in eligibility if item["eligible"]],
                "elapsed_seconds": time.perf_counter() - started,
                "readout_note": "Evidence weights only positive support patches; queries use support-independent prototype-to-patch local matching.",
                "sms_policy": "uncalibrated raw and within-method normalized logit shift; calibration is not used for SMS",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"results written to {args.output_dir}; decision={decision['status']}")


if __name__ == "__main__":
    main()
