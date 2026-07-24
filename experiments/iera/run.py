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

from .episodes import PILOT_PAIRS, eligible_directed_pairs, generate_pair_episodes, split_indices, validate_pair_episodes
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


def _train(model, patches, data, config, args, device) -> list[float]:
    if args.train_steps == 0:
        return []
    train_indices = split_indices(data, "train", args.split_seed)
    evaluation_targets = {_ids(data, pair)[0] for pair in PILOT_PAIRS}
    base_ids = [data.class_names.index(name) for name in config["class_partitions"]["base"] if data.class_names.index(name) not in evaluation_targets]
    pairs = eligible_directed_pairs(data, train_indices, base_ids, args.min_stratum_patients)
    if not pairs:
        raise ValueError("no eligible meta-training target/confounder pairs")
    random.Random(args.seed).shuffle(pairs)
    pairs = pairs[: min(12, len(pairs))]
    bank = []
    episodes_per_pair = max(2, math.ceil(min(args.train_steps, 120) / len(pairs)))
    for pair_index, (target, confounder) in enumerate(pairs):
        generated = generate_pair_episodes(
            data, train_indices, target, confounder, episodes_per_pair,
            args.train_shot, 1, args.seed + 10_000 + pair_index,
        )
        for episode_index in range(episodes_per_pair):
            bank.append((generated, episode_index))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    losses = []
    model.train()
    for step in range(args.train_steps):
        generated, episode_index = bank[int(torch.randint(len(bank), (1,), generator=generator))]
        positive, negative, query = _episode_batch(patches, generated, episode_index, episode_index + 1, args.train_shot, device)
        targets = generated["targets"][episode_index : episode_index + 1].to(device)
        logits = model(positive, negative, query, "iera")
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 25 == 0:
            print(f"meta-training {step + 1}/{args.train_steps}: loss={statistics.mean(losses[-25:]):.4f}", flush=True)
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
    result = {
        "auroc": _auc(target, probability),
        "auprc": _average_precision(target, probability),
        "f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
        "brier": float((probability - targets).square().mean()),
        "ece": _ece(probability, target),
        "false_positive_c0d1": float(probability[(~target) & nuisance.eq(1)].mean()),
        "sms": float((torch.sigmoid(panel_one / temperature) - torch.sigmoid(panel_zero / temperature)).abs().mean()),
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
    shot = 3
    evidence = []
    for pair in pairs:
        baseline_sms = lookup.get((pair, "positive_prototype", shot, "sms"), float("nan"))
        iera_sms = lookup.get((pair, "iera", shot, "sms"), float("nan"))
        reduction = 1 - iera_sms / baseline_sms if baseline_sms > 0 else float("nan")
        worst_gain = lookup.get((pair, "iera", shot, "worst_nuisance_auroc"), float("nan")) - lookup.get((pair, "positive_prototype", shot, "worst_nuisance_auroc"), float("nan"))
        ordinary_loss = lookup.get((pair, "positive_prototype", shot, "auroc"), float("nan")) - lookup.get((pair, "iera", shot, "auroc"), float("nan"))
        evidence.append({"pair": pair, "baseline_sms": baseline_sms, "iera_sms": iera_sms, "sms_reduction": reduction, "worst_auroc_gain": worst_gain, "ordinary_auroc_loss": ordinary_loss})
    passing = [row for row in evidence if row["sms_reduction"] >= 0.30 and row["worst_auroc_gain"] >= 0.02 and row["ordinary_auroc_loss"] < 0.01]
    return {
        "status": "continue" if len(passing) >= 3 else "stop_or_revise",
        "rule": "at least three pairs: SMS reduction >=30%, worst-nuisance AUROC gain >=0.02, ordinary AUROC loss <0.01",
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
    model = IERA(patches.shape[-1], args.projection_dim).to(device)
    losses = _train(model, patches, data, config, args, device)

    validation_indices = split_indices(data, "validate", args.split_seed)
    test_indices = split_indices(data, "test", args.split_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "parameters": model.parameters_dict()}, args.output_dir / "iera_model.pt")
    rows = []
    max_shot = max(args.shots)
    for pair_index, names in enumerate(PILOT_PAIRS):
        target_id, confounder_id = _ids(data, names)
        pair_name = f"{names[0]}__{names[1]}"
        validation = generate_pair_episodes(
            data, validation_indices, target_id, confounder_id, args.episodes,
            max_shot, args.queries_per_stratum, args.seed + pair_index * 10_000,
        )
        test_by_seed = {
            seed: generate_pair_episodes(
                data, test_indices, target_id, confounder_id, args.episodes,
                max_shot, args.queries_per_stratum,
                args.seed + 100_000 + pair_index * 10_000 + seed,
            )
            for seed in args.seeds
        }
        validate_pair_episodes(validation, data)
        for test in test_by_seed.values():
            validate_pair_episodes(test, data)
        torch.save(
            {"validation": validation, "test_by_seed": test_by_seed},
            args.output_dir / f"episodes_{pair_index:02d}.pt",
        )
        for shot in args.shots:
            for method in args.methods:
                val_logits, _, _ = _score(model, patches, validation, shot, method, args.episode_batch_size, device)
                val_targets = validation["targets"].flatten()
                temperature = select_temperature(val_logits[:, None], val_targets[:, None], "multi_label")
                threshold = select_threshold(val_logits[:, None], val_targets[:, None], temperature)
                for seed in args.seeds:
                    test = test_by_seed[seed]
                    logits, panel_zero, panel_one = _score(model, patches, test, shot, method, args.episode_batch_size, device)
                    metrics = _metrics(logits, panel_zero, panel_one, test["targets"].flatten(), test["nuisance"].flatten(), temperature, threshold)
                    for metric, value in metrics.items():
                        rows.append({"pair": pair_name, "target": names[0], "confounder": names[1], "method": method, "shot": shot, "seed": seed, "temperature": temperature, "threshold": threshold, "metric": metric, "value": value})
                print(f"finished {pair_name}, {method}, {shot}-shot", flush=True)
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
                "patient_split": {"seed": args.split_seed, "fractions": [0.70, 0.15, 0.15]},
                "train_steps": args.train_steps,
                "final_train_loss": losses[-1] if losses else None,
                "learned_parameters": model.parameters_dict(),
                "elapsed_seconds": time.perf_counter() - started,
                "readout_note": "The proposal omits p_c/readout equations; implementation uses evidence-weighted positive and query tokens with cosine scoring.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"results written to {args.output_dir}; decision={decision['status']}")


if __name__ == "__main__":
    main()
