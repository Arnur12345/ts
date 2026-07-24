"""Run the IERA four-pair go/no-go pilot on cached BioMedCLIP patches."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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


def _meta_split(data, indices, split_seed):
    train, validation = [], []
    for index in indices.tolist():
        subject = data.subject_ids[index]
        bucket = int.from_bytes(
            hashlib.sha256(f"iera-meta|{split_seed}|{subject}".encode()).digest()[:8], "big"
        ) % 10_000
        (validation if bucket >= 8500 else train).append(index)
    return torch.tensor(train, dtype=torch.long), torch.tensor(validation, dtype=torch.long)


def _normalized_consistency(panel_zero: torch.Tensor, panel_one: torch.Tensor) -> torch.Tensor:
    variance = torch.cat((panel_zero, panel_one), dim=-1).var(unbiased=False)
    return (panel_zero - panel_one).square().mean() / (variance + 1e-6)


def _objective(
    model, method, positive, negative, query, targets, args,
    uniform_reference_model=None,
) -> dict[str, torch.Tensor]:
    logits = model(positive, negative, query, method)
    classification = F.binary_cross_entropy_with_logits(logits, targets)
    zero = classification.new_zeros(())
    invariance = zero
    budget_excess = zero
    uniform_reference = zero
    if method == "anchored_iera":
        panel_zero, panel_one = model.swapped_logits(positive, negative, query, method)
        invariance = _normalized_consistency(panel_zero, panel_one)
        if uniform_reference_model is None:
            raise ValueError("anchored_iera requires a fixed learned-uniform reference model")
        with torch.no_grad():
            uniform_zero, uniform_one = uniform_reference_model.swapped_logits(
                positive, negative, query, "learned_uniform"
            )
            uniform_reference = _normalized_consistency(uniform_zero, uniform_one)
        budget_excess = F.relu(
            invariance - args.invariance_budget * uniform_reference.detach()
        )
    total = classification + args.invariance_weight * (invariance + budget_excess)
    return {
        "total": total,
        "classification": classification,
        "invariance": invariance,
        "budget_excess": budget_excess,
        "uniform_reference": uniform_reference,
    }


def _validation_objective(
    model, method, patches, bank, shot, device, args,
    uniform_reference_model=None,
) -> dict[str, float]:
    totals = defaultdict(float)
    total_queries = 0
    model.eval()
    with torch.inference_mode():
        for generated in bank:
            positive, negative, query = _episode_batch(
                patches, generated, 0, len(generated["positive"]), shot, device
            )
            targets = generated["targets"].to(device)
            components = _objective(
                model, method, positive, negative, query, targets, args,
                uniform_reference_model,
            )
            queries = targets.numel()
            for name, value in components.items():
                totals[name] += float(value) * queries
            total_queries += queries
    return {name: value / total_queries for name, value in totals.items()}


def _train(
    model, method, patches, data, config, args, device, run_seed,
    uniform_reference_model=None,
) -> dict:
    if method == "frozen_protonet":
        return {
            "method": method, "steps_run": 0, "best_step": 0,
            "best_validation_loss": None, "stopped_early": False, "curve": [],
        }
    all_train_indices = split_indices(data, "train", args.split_seed)
    train_indices, early_stop_indices = _meta_split(data, all_train_indices, args.split_seed)
    evaluation_targets = {_ids(data, pair)[0] for pair in PILOT_PAIRS}
    base_ids = [data.class_names.index(name) for name in config["class_partitions"]["base"] if data.class_names.index(name) not in evaluation_targets]
    train_pairs = eligible_directed_pairs(
        data, train_indices, base_ids, args.min_stratum_patients,
        confounder_ids=base_ids,
    )
    early_stop_minimum = max(args.train_shot + 1, 10)
    early_stop_pairs = eligible_directed_pairs(
        data, early_stop_indices, base_ids, early_stop_minimum,
        confounder_ids=base_ids,
    )
    pairs = sorted(set(train_pairs) & set(early_stop_pairs))
    if not pairs:
        raise ValueError("no base-only target/confounder pairs support disjoint meta-train/early-stop episodes")
    random.Random(run_seed).shuffle(pairs)
    pairs = pairs[: min(12, len(pairs))]
    bank = []
    validation_bank = []
    episodes_per_pair = max(2, math.ceil(min(args.max_train_steps, 120) / len(pairs)))
    for pair_index, (target, confounder) in enumerate(pairs):
        generated = generate_pair_episodes(
            data, train_indices, target, confounder, episodes_per_pair,
            args.train_shot, 1, run_seed + 10_000 + pair_index,
            min_stratum_patients=args.min_stratum_patients,
        )
        for episode_index in range(episodes_per_pair):
            bank.append((generated, episode_index))
        validation_bank.append(
            generate_pair_episodes(
                data, early_stop_indices, target, confounder,
                args.early_stopping_episodes_per_pair, args.train_shot, 1,
                run_seed + 50_000 + pair_index,
                min_stratum_patients=early_stop_minimum,
            )
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(run_seed)
    component_history = defaultdict(list)
    initial_validation = _validation_objective(
        model, method, patches, validation_bank, args.train_shot, device, args,
        uniform_reference_model,
    )
    best_validation = initial_validation["total"]
    best_step = 0
    best_state = copy.deepcopy(model.state_dict())
    curve = [
        {
            "step": 0,
            "train_loss": None,
            "train_classification": None,
            "train_invariance": None,
            "train_budget_excess": None,
            **{f"validation_{name}": value for name, value in initial_validation.items()},
        }
    ]
    checks_without_improvement = 0
    stopped_early = False
    model.train()
    for step in range(args.max_train_steps):
        generated, episode_index = bank[int(torch.randint(len(bank), (1,), generator=generator))]
        positive, negative, query = _episode_batch(patches, generated, episode_index, episode_index + 1, args.train_shot, device)
        targets = generated["targets"][episode_index : episode_index + 1].to(device)
        components = _objective(
            model, method, positive, negative, query, targets, args,
            uniform_reference_model,
        )
        loss = components["total"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        for name, value in components.items():
            component_history[name].append(float(value.detach()))
        completed = step + 1
        if completed % args.validation_interval == 0 or completed == args.max_train_steps:
            validation = _validation_objective(
                model, method, patches, validation_bank, args.train_shot, device, args,
                uniform_reference_model,
            )
            recent = slice(-args.validation_interval, None)
            curve.append(
                {
                    "step": completed,
                    "train_loss": statistics.mean(component_history["total"][recent]),
                    "train_classification": statistics.mean(component_history["classification"][recent]),
                    "train_invariance": statistics.mean(component_history["invariance"][recent]),
                    "train_budget_excess": statistics.mean(component_history["budget_excess"][recent]),
                    **{f"validation_{name}": value for name, value in validation.items()},
                }
            )
            print(
                f"training {method} seed {run_seed}: {completed}/{args.max_train_steps}, "
                f"train={curve[-1]['train_loss']:.4f}, val={validation['total']:.4f}",
                flush=True,
            )
            if validation["total"] < best_validation - args.early_stopping_min_delta:
                best_validation = validation["total"]
                best_step = completed
                best_state = copy.deepcopy(model.state_dict())
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
                if checks_without_improvement >= args.early_stopping_patience:
                    stopped_early = True
                    break
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    return {
        "method": method,
        "steps_run": len(component_history["total"]),
        "best_step": best_step,
        "best_validation_loss": best_validation,
        "stopped_early": stopped_early,
        "curve": curve,
        "meta_training_pairs": [
            {"target": data.class_names[target], "confounder": data.class_names[confounder]}
            for target, confounder in pairs
        ],
    }


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
    panel_zero_probability = torch.sigmoid(panel_zero / temperature)
    panel_one_probability = torch.sigmoid(panel_one / temperature)
    panel_zero_prediction = panel_zero_probability.ge(threshold)
    panel_one_prediction = panel_one_probability.ge(threshold)
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
    methods = ("frozen_protonet", "learned_uniform", "iera", "anchored_iera")
    evidence = []
    for pair in pairs:
        method_values = {
            method: {
                "sms": lookup.get((pair, method, shot, "sms_normalized_logit"), float("nan")),
                "worst_nuisance_auroc": lookup.get((pair, method, shot, "worst_nuisance_auroc"), float("nan")),
                "auroc": lookup.get((pair, method, shot, "auroc"), float("nan")),
            }
            for method in methods
        }
        baseline, anchored = method_values["learned_uniform"], method_values["anchored_iera"]
        evidence.append(
            {
                "pair": pair,
                "methods": method_values,
                "anchored_reduces_sms": anchored["sms"] < baseline["sms"],
                "anchored_retains_auroc": anchored["auroc"] >= baseline["auroc"],
                "anchored_retains_worst_nuisance_auroc": (
                    anchored["worst_nuisance_auroc"] >= baseline["worst_nuisance_auroc"]
                ),
            }
        )
    anchored_consistent = bool(evidence) and all(
        row["anchored_reduces_sms"]
        and row["anchored_retains_auroc"]
        and row["anchored_retains_worst_nuisance_auroc"]
        for row in evidence
    )
    pneumothorax = next(
        (row for row in evidence if row["pair"].startswith("Pneumothorax__")), None
    )
    if anchored_consistent:
        status = "continue_anchored_iera"
        recommendation = "Anchored IERA beats learned-uniform SMS while retaining AUROC on both pairs."
    elif pneumothorax is not None and not pneumothorax["anchored_reduces_sms"]:
        status = "abandon_due_identifiability"
        recommendation = (
            "Directly constrained Anchored IERA cannot reduce Pneumothorax SMS; "
            "treat pathology-versus-device evidence as observationally non-identifiable."
        )
    else:
        status = "stop_or_revise_anchor"
        recommendation = "The anchor reduces sensitivity but does not retain both AUROC criteria."
    return {
        "status": status,
        "rule": "Anchored IERA must beat learned-uniform SMS and retain AUROC and worst-nuisance AUROC on both eligible pairs.",
        "required_pairs": len(pairs),
        "anchored_iera_consistent": anchored_consistent,
        "recommendation": recommendation,
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
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--alpha-max", type=float, default=0.25)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--invariance-budget", type=float, default=0.7)
    parser.add_argument(
        "--max-train-steps", "--train-steps", dest="max_train_steps",
        type=int, default=1000,
        help="Maximum optimization steps; --train-steps is a deprecated alias",
    )
    parser.add_argument("--train-shot", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-episodes-per-pair", type=int, default=2)
    parser.add_argument("--min-stratum-patients", type=int, default=50)
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.max_train_steps <= 0 or args.validation_interval <= 0:
        parser.error("max-train-steps and validation-interval must be positive")
    if args.early_stopping_patience <= 0 or args.early_stopping_episodes_per_pair <= 0:
        parser.error("early-stopping patience and episodes-per-pair must be positive")
    if not 0 < args.alpha_max <= 1 or args.invariance_weight < 0:
        parser.error("alpha-max must be in (0,1] and invariance-weight must be non-negative")
    if not 0 < args.invariance_budget <= 1:
        parser.error("invariance-budget must be in (0,1]")
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
    training_curve_rows = []
    for seed in args.seeds:
        run_seed = args.seed + seed
        uniform_reference_model = None
        for method in METHODS:
            _set_seed(run_seed)
            model = IERA(
                patches.shape[-1], args.projection_dim, alpha_max=args.alpha_max
            ).to(device)
            training_info = _train(
                model, method, patches, data, config, args, device, run_seed,
                uniform_reference_model,
            )
            if method == "learned_uniform":
                uniform_reference_model = copy.deepcopy(model).eval().requires_grad_(False)
            if method != "frozen_protonet":
                torch.save(
                    {
                        "method": method,
                        "seed": seed,
                        "training_seed": run_seed,
                        "state_dict": model.state_dict(),
                        "parameters": model.parameters_dict(),
                        "training": training_info,
                    },
                    args.output_dir / f"model_{method}_seed_{seed:03d}.pt",
                )
            training_runs.append({
                **{key: value for key, value in training_info.items() if key != "curve"},
                "seed": seed,
                "training_seed": run_seed,
                "learned_parameters": model.parameters_dict() if method != "frozen_protonet" else None,
            })
            for point in training_info["curve"]:
                training_curve_rows.append(
                    {"method": method, "seed": seed, "training_seed": run_seed, **point}
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
        del uniform_reference_model
    summary = _summaries(rows)
    _write(args.output_dir / "per_seed_metrics.csv", rows)
    _write(args.output_dir / "summary_metrics.csv", summary)
    if training_curve_rows:
        _write(args.output_dir / "training_curves.csv", training_curve_rows)
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
                "methods": list(METHODS),
                "episodes": args.episodes,
                "seeds": args.seeds,
                "seed_semantics": "independent initialization, meta-training, validation, and test episode run",
                "patient_split": {"seed": args.split_seed, "fractions": [0.70, 0.15, 0.15]},
                "max_train_steps": args.max_train_steps,
                "anchored_iera": {
                    "alpha_max": args.alpha_max,
                    "invariance_weight": args.invariance_weight,
                    "uniform_sensitivity_budget": args.invariance_budget,
                    "objective": "classification + lambda * (normalized consistency + budget excess)",
                },
                "early_stopping": {
                    "validation_interval": args.validation_interval,
                    "patience": args.early_stopping_patience,
                    "min_delta": args.early_stopping_min_delta,
                    "episodes_per_pair": args.early_stopping_episodes_per_pair,
                    "selection_data": "patient-disjoint base-class episodes only",
                },
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
