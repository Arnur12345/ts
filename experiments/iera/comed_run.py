"""Run the gated CoMeD variants on fixed global features and episodes."""

from __future__ import annotations

import argparse
import copy
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

from experiments.residuals.data import ResidualDataset
from experiments.residuals.metrics import _auc, _average_precision

from .comed import CoMeD, distillation_loss, grouped_rex_loss, swap_loss
from .episodes import generate_pair_episodes, split_indices
from .robust_metrics import normalized_sms, ranking_disagreement


VARIANTS = (
    "text_only",
    "comed_no_nuisance",
    "comed_nuisance",
    "comed_learned_metric",
    "full_comed",
)


def _load_cache(root: Path):
    import numpy as np

    metadata = json.loads(
        (root / "comed_cache.json").read_text(encoding="utf-8")
    )
    if metadata.get("complete") is not True:
        raise ValueError("CoMeD cache is incomplete")
    index = torch.load(
        root / metadata["dataset_index"],
        map_location="cpu",
        weights_only=False,
    )
    arrays = {
        "rad": np.load(metadata["rad_features"], mmap_mode="r"),
        "labels": np.load(root / metadata["labels"], mmap_mode="r"),
        "known": np.load(root / metadata["known"], mmap_mode="r"),
        "nuisance": np.load(root / metadata["nuisance"], mmap_mode="r"),
        "prior": np.load(root / metadata["semantic_priors"], mmap_mode="r"),
        "teacher": np.load(root / metadata["teacher_scores"], mmap_mode="r"),
    }
    count = len(index["subject_ids"])
    if any(len(value) != count for value in arrays.values()):
        raise ValueError("CoMeD cache arrays have inconsistent lengths")
    data = ResidualDataset(
        images=torch.empty(count, 0),
        labels=torch.as_tensor(arrays["labels"].astype("bool")),
        known=torch.as_tensor(arrays["known"].astype("bool")),
        metadata=torch.empty(count, 0),
        class_names=list(index["class_names"]),
        subject_ids=list(index["subject_ids"]),
        dicom_ids=[
            str(row.get("dicom_id", position))
            for position, row in enumerate(index["rows"])
        ],
        rows=list(index["rows"]),
        manifest_sha256=index["manifest_sha256"],
    )
    return arrays, metadata, data


def _balanced_support(episodes: dict, episode: int, shot: int) -> torch.Tensor:
    selected = []
    for name, offset in (("positive", 0), ("negative", 1)):
        counters = [0, 0]
        for position in range(shot):
            environment = (position + offset) % 2
            selected.append(
                int(episodes[name][episode, environment, counters[environment]])
            )
            counters[environment] += 1
    return torch.tensor(selected, dtype=torch.long)


def _panel_support(
    episodes: dict, episode: int, shot: int, environment: int
) -> torch.Tensor:
    return torch.cat(
        (
            episodes["positive"][episode, environment, :shot],
            episodes["negative"][episode, environment, :shot],
        )
    ).long()


def _tensor(array, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(
        array[indices.cpu().numpy()], dtype=torch.float32, device=device
    )


def _protonet(
    rad,
    episodes,
    episode,
    shot,
    environment,
    query,
    device,
) -> torch.Tensor:
    positive = episodes["positive"][episode, environment, :shot]
    negative = episodes["negative"][episode, environment, :shot]
    positive_prototype = F.normalize(_tensor(rad, positive, device).mean(0), dim=-1)
    negative_prototype = F.normalize(_tensor(rad, negative, device).mean(0), dim=-1)
    query_z = F.normalize(_tensor(rad, query, device), dim=-1)
    return query_z @ positive_prototype - query_z @ negative_prototype


def _score_support(
    model,
    arrays,
    class_id,
    support,
    query,
    device,
):
    if model is None:
        return _tensor(arrays["prior"][:, class_id], query, device)
    support_target = _tensor(
        arrays["labels"][:, class_id], support, device
    )
    return model(
        _tensor(arrays["rad"], support, device),
        support_target.mul(2).sub(1),
        _tensor(arrays["nuisance"], support, device),
        _tensor(arrays["prior"][:, class_id], support, device),
        _tensor(arrays["rad"], query, device),
        _tensor(arrays["prior"][:, class_id], query, device),
    )


def _episode_outputs(
    model,
    arrays,
    episodes,
    episode,
    shot,
    device,
):
    class_id = int(episodes["target_id"])
    query = episodes["query"][episode].long()
    ordinary = _balanced_support(episodes, episode, shot)
    panel_zero = _panel_support(episodes, episode, shot, 0)
    panel_one = _panel_support(episodes, episode, shot, 1)
    logits = _score_support(
        model, arrays, class_id, ordinary, query, device
    )
    zero = _score_support(
        model, arrays, class_id, panel_zero, query, device
    )
    one = _score_support(
        model, arrays, class_id, panel_one, query, device
    )
    reference_zero = _protonet(
        arrays["rad"], episodes, episode, shot, 0, query, device
    )
    reference_one = _protonet(
        arrays["rad"], episodes, episode, shot, 1, query, device
    )
    return logits, zero, one, reference_zero, reference_one


def _evaluate(model, arrays, episodes, shot, device) -> dict:
    destinations = ([], [], [], [], [])
    with torch.inference_mode():
        for episode in range(len(episodes["positive"])):
            values = _episode_outputs(
                model, arrays, episodes, episode, shot, device
            )
            for destination, value in zip(destinations, values):
                destination.append(value.cpu())
    logits, zero, one, reference_zero, reference_one = (
        torch.cat(values) for values in destinations
    )
    target = episodes["targets"].flatten().bool()
    nuisance = episodes["nuisance"].flatten().long()
    result = {
        "auroc": _auc(target, logits),
        "auprc": _average_precision(target, logits),
        "sms_fixed_reference": float(
            normalized_sms(zero, one, reference_zero, reference_one)
        ),
        "ranking_disagreement": ranking_disagreement(zero, one),
    }
    nuisance_aurocs = []
    for value in (0, 1):
        mask = nuisance.eq(value)
        observed = _auc(target[mask], logits[mask])
        nuisance_aurocs.append(observed)
        result[f"device_{value}_auroc"] = observed
    result["worst_device_auroc"] = min(nuisance_aurocs)
    return result


def _mean_metrics(metrics: list[dict]) -> dict:
    return {
        key: statistics.mean(float(item[key]) for item in metrics)
        for key in metrics[0]
    }


def _build_base_episodes(data, metadata, args):
    path = args.output_dir / "base_episodes.pt"
    signature = {
        "manifest_sha256": data.manifest_sha256,
        "split_seed": args.split_seed,
        "shot": args.shot,
        "train_episodes_per_class": args.train_episodes_per_class,
        "validation_episodes_per_class": args.validation_episodes_per_class,
        "queries_per_stratum": args.queries_per_stratum,
        "seed": args.seed,
        "base_target_ids": metadata["base_target_ids"],
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("existing base episodes use another protocol")
        return saved
    train_indices = split_indices(data, "train", args.split_seed)
    validation_indices = split_indices(data, "validate", args.split_seed)
    device_id = data.class_names.index("Support Devices")
    train_bank, validation_bank, skipped = [], [], []
    for position, target_id in enumerate(metadata["base_target_ids"]):
        try:
            train_bank.append(
                generate_pair_episodes(
                    data,
                    train_indices,
                    target_id,
                    device_id,
                    args.train_episodes_per_class,
                    args.shot,
                    args.queries_per_stratum,
                    args.seed + 10_000 + position,
                )
            )
            validation_bank.append(
                generate_pair_episodes(
                    data,
                    validation_indices,
                    target_id,
                    device_id,
                    args.validation_episodes_per_class,
                    args.shot,
                    args.queries_per_stratum,
                    args.seed + 20_000 + position,
                )
            )
        except ValueError as error:
            skipped.append(
                {
                    "target": data.class_names[target_id],
                    "reason": str(error),
                }
            )
    if not train_bank:
        raise ValueError("no base pathology supports paired CoMeD episodes")
    saved = {
        "signature": signature,
        "train": train_bank,
        "validate": validation_bank,
        "skipped": skipped,
    }
    torch.save(saved, path)
    return saved


def _base_metrics(model, arrays, bank, args, device):
    return _mean_metrics(
        [
            _evaluate(model, arrays, episodes, args.shot, device)
            for episodes in bank
        ]
    )


def _feasible(metrics, reference, args):
    return (
        metrics["sms_fixed_reference"] <= args.sms_budget
        and metrics["worst_device_auroc"]
        >= reference["worst_device_auroc"] - args.worst_device_tolerance
        and metrics["ranking_disagreement"]
        <= (
            reference["ranking_disagreement"]
            - args.min_ranking_improvement
        )
    )


def _checkpoint_key(metrics, reference, args):
    if _feasible(metrics, reference, args):
        return 1.0, metrics["auroc"]
    violation = (
        max(0.0, metrics["sms_fixed_reference"] - args.sms_budget)
        + max(
            0.0,
            reference["worst_device_auroc"]
            - args.worst_device_tolerance
            - metrics["worst_device_auroc"],
        )
        + max(
            0.0,
            metrics["ranking_disagreement"]
            - reference["ranking_disagreement"]
            + args.min_ranking_improvement,
        )
    )
    return 0.0, -violation


def _train(
    model,
    variant,
    arrays,
    bank,
    validation_bank,
    reference,
    args,
    device,
):
    learn_metric = variant in {"comed_learned_metric", "full_comed"}
    learn_nuisance = variant in {
        "comed_nuisance",
        "comed_learned_metric",
        "full_comed",
    }
    parameters = model.configure_trainable(
        learn_metric,
        learn_nuisance,
        learn_noise=learn_metric,
    )
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    initial = _base_metrics(
        model, arrays, validation_bank, args, device
    )
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = initial
    best_key = _checkpoint_key(initial, reference, args)
    best_step = 0
    curve = [{"step": 0, **initial}]
    choices = [
        (pair, episode)
        for pair, episodes in enumerate(bank)
        for episode in range(len(episodes["positive"]))
    ]
    generator = random.Random(args.seed)
    model.train()
    for step in range(1, args.train_steps + 1):
        pair, episode = generator.choice(choices)
        episodes = bank[pair]
        class_id = int(episodes["target_id"])
        query = episodes["query"][episode].long()
        support_zero = _panel_support(
            episodes, episode, args.shot, 0
        )
        support_one = _panel_support(
            episodes, episode, args.shot, 1
        )
        scores = (
            _score_support(
                model, arrays, class_id, support_zero, query, device
            ),
            _score_support(
                model, arrays, class_id, support_one, query, device
            ),
        )
        target = episodes["targets"][episode].to(device)
        nuisance = episodes["nuisance"][episode].to(device)
        classification = grouped_rex_loss(
            scores, target, nuisance, args.beta_rex
        )
        teacher = _tensor(
            arrays["teacher"][:, class_id], query, device
        )
        teacher_loss = distillation_loss(scores, teacher)
        stability = swap_loss(*scores)
        if variant == "full_comed":
            loss = (
                classification
                + args.lambda_teacher * teacher_loss
                + args.lambda_swap * stability
            )
        else:
            loss = classification
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        optimizer.step()
        if step % args.validation_interval == 0 or step == args.train_steps:
            metrics = _base_metrics(
                model, arrays, validation_bank, args, device
            )
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "classification": float(classification.detach()),
                    "teacher": float(teacher_loss.detach()),
                    "swap": float(stability.detach()),
                    **metrics,
                }
            )
            key = _checkpoint_key(metrics, reference, args)
            if key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics
                best_step = step
            print(
                f"{variant}: step={step}, "
                f"AUROC={metrics['auroc']:.4f}, "
                f"SMS={metrics['sms_fixed_reference']:.3f}, "
                f"worst={metrics['worst_device_auroc']:.4f}",
                flush=True,
            )
    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_step": best_step,
        "best_base_validation": best_metrics,
        "feasible": _feasible(best_metrics, reference, args),
        "curve": curve,
    }


def _load_target_episodes(path, data, args):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved["signature"]["manifest_sha256"] != data.manifest_sha256:
        raise ValueError("evaluation episodes and CoMeD cache differ")
    pair_ids = [
        pair_id for pair_id, names in saved["pairs"].items()
        if names == ("Pneumothorax", "Support Devices")
        or list(names) == ["Pneumothorax", "Support Devices"]
    ]
    if len(pair_ids) != 1:
        raise ValueError("expected one Pneumothorax-Support Devices pair")
    pair_id = pair_ids[0]
    result = {}
    for seed in args.seeds:
        for partition in ("validate", "test"):
            episodes = saved["episodes"][(pair_id, seed, partition)]
            result[(seed, partition)] = {
                key: (
                    value[: args.episodes_per_seed]
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == saved["signature"]["episodes"]
                    else value
                )
                for key, value in episodes.items()
            }
    return result


def _target_metrics(model, arrays, target_episodes, args, device, partition):
    return [
        _evaluate(
            model,
            arrays,
            target_episodes[(seed, partition)],
            args.shot,
            device,
        )
        for seed in args.seeds
    ]


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows):
    groups = defaultdict(list)
    keys = ("partition", "method", "shot", "metric")
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row["value"]))
    output = []
    for key, values in groups.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half = 1.96 * std / math.sqrt(len(values))
        output.append(
            {
                **dict(zip(keys, key)),
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--train-episodes-per-class", type=int, default=40)
    parser.add_argument("--validation-episodes-per-class", type=int, default=20)
    parser.add_argument("--queries-per-stratum", type=int, default=2)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--lambda-teacher", type=float, default=0.3)
    parser.add_argument("--lambda-swap", type=float, default=0.3)
    parser.add_argument("--beta-rex", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--sms-budget", type=float, default=0.30)
    parser.add_argument("--worst-device-tolerance", type=float, default=0.0)
    parser.add_argument(
        "--min-ranking-improvement", type=float, default=1e-4
    )
    parser.add_argument("--anchor-auroc", type=float, default=0.539)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if tuple(args.variants[:2]) != VARIANTS[:2]:
        parser.error("variants must begin with text_only comed_no_nuisance")
    if any(variant not in VARIANTS for variant in args.variants):
        parser.error(f"variants must be selected from {VARIANTS}")
    expected_order = [
        variant for variant in VARIANTS if variant in set(args.variants)
    ]
    if list(args.variants) != expected_order:
        parser.error(f"variants must preserve this order: {VARIANTS}")
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    arrays, metadata, data = _load_cache(args.cache)
    target_episodes = _load_target_episodes(
        args.episodes, data, args
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = _build_base_episodes(data, metadata, args)
    models = {
        "text_only": None,
        "comed_no_nuisance": CoMeD(
            dim=arrays["rad"].shape[1],
            rank=args.rank,
            metric_mode="identity",
            use_nuisance=False,
        ).to(device).eval().requires_grad_(False),
    }
    validation_metrics = {}
    for variant in VARIANTS[:2]:
        values = _target_metrics(
            models[variant],
            arrays,
            target_episodes,
            args,
            device,
            "validate",
        )
        validation_metrics[variant] = _mean_metrics(values)
    early_pass = (
        validation_metrics["comed_no_nuisance"]["auroc"]
        > validation_metrics["text_only"]["auroc"]
        and validation_metrics["comed_no_nuisance"]["auroc"]
        > args.anchor_auroc
    )
    training = {}
    if early_pass:
        base_reference = _base_metrics(
            models["comed_no_nuisance"],
            arrays,
            base["validate"],
            args,
            device,
        )
        for variant in args.variants[2:]:
            if variant == "comed_nuisance":
                model = CoMeD(
                    arrays["rad"].shape[1],
                    args.rank,
                    metric_mode="identity",
                    use_nuisance=True,
                )
            else:
                model = CoMeD(
                    arrays["rad"].shape[1],
                    args.rank,
                    metric_mode="learned",
                    use_nuisance=True,
                )
                if (
                    variant == "full_comed"
                    and "comed_learned_metric" in models
                ):
                    model.load_state_dict(
                        models["comed_learned_metric"].state_dict()
                    )
            model = model.to(device)
            model_path = args.output_dir / f"model_{variant}.pt"
            if model_path.exists():
                saved = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                model.load_state_dict(saved["state_dict"])
                training[variant] = saved["training"]
                print(f"reusing {variant}", flush=True)
            else:
                training[variant] = _train(
                    model,
                    variant,
                    arrays,
                    base["train"],
                    base["validate"],
                    base_reference,
                    args,
                    device,
                )
                torch.save(
                    {
                        "variant": variant,
                        "state_dict": model.state_dict(),
                        "training": training[variant],
                    },
                    model_path,
                )
            model.eval()
            models[variant] = model
            validation_metrics[variant] = _mean_metrics(
                _target_metrics(
                    model,
                    arrays,
                    target_episodes,
                    args,
                    device,
                    "validate",
                )
            )
    available = list(models)
    reference = validation_metrics["comed_no_nuisance"]
    feasible = [
        variant for variant in available
        if variant not in {"text_only", "comed_no_nuisance"}
        and _feasible(validation_metrics[variant], reference, args)
    ]
    if feasible:
        selected = max(
            feasible,
            key=lambda variant: validation_metrics[variant]["auroc"],
        )
    else:
        selected = max(
            available,
            key=lambda variant: validation_metrics[variant]["auroc"],
        )
    rows = []
    if early_pass:
        for variant in available:
            for partition in ("validate", "test"):
                values = _target_metrics(
                    models[variant],
                    arrays,
                    target_episodes,
                    args,
                    device,
                    partition,
                )
                for seed, metrics in zip(args.seeds, values):
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "partition": partition,
                                "method": variant,
                                "seed": seed,
                                "shot": args.shot,
                                "metric": metric,
                                "value": value,
                            }
                        )
    else:
        for variant in VARIANTS[:2]:
            for seed, metrics in zip(
                args.seeds,
                _target_metrics(
                    models[variant],
                    arrays,
                    target_episodes,
                    args,
                    device,
                    "validate",
                ),
            ):
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "partition": "validate",
                            "method": variant,
                            "seed": seed,
                            "shot": args.shot,
                            "metric": metric,
                            "value": value,
                        }
                    )
    _write(args.output_dir / "per_seed_metrics.csv", rows)
    summaries = _summaries(rows)
    _write(args.output_dir / "summary_metrics.csv", summaries)
    selection = {
        "selection_partition": "validate",
        "early_gate": {
            "passed": early_pass,
            "criterion": (
                "comed_no_nuisance AUROC must exceed text_only and 0.539"
            ),
            "text_only_auroc": validation_metrics["text_only"]["auroc"],
            "comed_no_nuisance_auroc": validation_metrics[
                "comed_no_nuisance"
            ]["auroc"],
            "anchor_auroc_threshold": args.anchor_auroc,
        },
        "constraints": {
            "sms_budget": args.sms_budget,
            "worst_device_tolerance": args.worst_device_tolerance,
            "ranking_reference": "comed_no_nuisance",
            "minimum_ranking_improvement": args.min_ranking_improvement,
        },
        "validation_metrics": validation_metrics,
        "feasible_learned_variants": feasible,
        "selected": selected,
    }
    if not early_pass:
        status = "stop_support_correction_insufficient"
    elif not feasible:
        status = "stop_no_learned_variant_satisfies_constraints"
    else:
        status = "comed_candidate_selected"
    decision = {
        "status": status,
        "selected_variant": selected,
        "test_accessed": early_pass,
        "selected_test_metrics": (
            {
                row["metric"]: row["mean"]
                for row in summaries
                if row["partition"] == "test"
                and row["method"] == selected
            }
            if early_pass
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training.json").write_text(
        json.dumps(training, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"CoMeD results written to {args.output_dir}; decision={status}"
    )


if __name__ == "__main__":
    main()
