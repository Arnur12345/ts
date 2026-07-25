"""Factorial diagnostic for a genuine episodic Pneumothorax detector.

This deliberately excludes IERA and SMS regularization. It compares the old
positive-only score with a proper positive-minus-negative binary ProtoNet over
multiple frozen patch-token caches and retained spatial resolutions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.residuals.data import load_dataset
from experiments.residuals.metrics import _average_precision, _auc

from .episodes import generate_pair_episodes, patient_counts, split_indices, stratum_pools
from .labels import restore_raw_target_status
from .patch_cache import load_patch_cache


HEADS = ("positive_only", "binary_protonet")


def _parse_cache(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("cache must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("cache must be NAME=PATH")
    return name, Path(raw_path)


def _pool(tokens: torch.Tensor, source_grid: int, retained_grid: int) -> torch.Tensor:
    if retained_grid > source_grid:
        raise ValueError(
            f"cannot retain {retained_grid}x{retained_grid} from "
            f"{source_grid}x{source_grid} cached tokens"
        )
    if retained_grid == source_grid:
        return F.normalize(tokens.float(), dim=-1)
    leading = tokens.shape[:-2]
    width = tokens.shape[-1]
    spatial = tokens.float().reshape(-1, source_grid, source_grid, width)
    spatial = spatial.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(spatial, (retained_grid, retained_grid))
    return F.normalize(
        pooled.permute(0, 2, 3, 1).reshape(
            *leading, retained_grid * retained_grid, width
        ),
        dim=-1,
    )


def _prototype(tokens: torch.Tensor) -> torch.Tensor:
    return F.normalize(tokens.float().mean(dim=(1, 2, 3)), dim=-1)


def _detector_logits(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    head: str,
) -> torch.Tensor:
    if head not in HEADS:
        raise ValueError(f"unknown detector head {head!r}")
    positive_prototype = _prototype(positive)
    query_representation = F.normalize(query.float().mean(dim=2), dim=-1)
    logits = torch.einsum(
        "bqd,bd->bq", query_representation, positive_prototype
    )
    if head == "binary_protonet":
        negative_prototype = _prototype(negative)
        logits = logits - torch.einsum(
            "bqd,bd->bq", query_representation, negative_prototype
        )
    return logits


def _score_factorial(
    patches: torch.Tensor,
    metadata: dict,
    episodes: dict[str, torch.Tensor],
    retained_grid: int,
    shots: list[int] | tuple[int, ...],
    batch_size: int,
    device: torch.device,
) -> dict[tuple[int, str], torch.Tensor]:
    source_grid = int(metadata["pool_grid"])
    scored = defaultdict(list)
    with torch.inference_mode():
        for start in range(0, len(episodes["positive"]), batch_size):
            end = min(start + batch_size, len(episodes["positive"]))
            positive = patches[episodes["positive"][start:end]].to(
                device, non_blocking=True
            )
            negative = patches[episodes["negative"][start:end]].to(
                device, non_blocking=True
            )
            query = patches[episodes["query"][start:end]].to(
                device, non_blocking=True
            )
            positive = _pool(positive, source_grid, retained_grid)
            negative = _pool(negative, source_grid, retained_grid)
            query = _pool(query, source_grid, retained_grid)
            for shot in shots:
                for head in HEADS:
                    scored[(shot, head)].append(
                        _detector_logits(
                            positive[:, :, :shot],
                            negative[:, :, :shot],
                            query,
                            head,
                        ).cpu()
                    )
    return {
        key: torch.cat(values).flatten()
        for key, values in scored.items()
    }


def _metrics(
    logits: torch.Tensor, targets: torch.Tensor, nuisance: torch.Tensor
) -> dict[str, float]:
    target = targets.bool()
    result = {
        "auroc": _auc(target, logits),
        "auprc": _average_precision(target, logits),
    }
    nuisance_aurocs = []
    for value in (0, 1):
        mask = nuisance.eq(value)
        nuisance_aurocs.append(_auc(target[mask], logits[mask]))
        result[f"d{value}_auroc"] = nuisance_aurocs[-1]
    result["worst_nuisance_auroc"] = min(nuisance_aurocs)
    return result


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict]) -> list[dict]:
    keys = (
        "partition", "cache", "model", "source_grid", "retained_grid",
        "head", "shot", "metric",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row["value"]))
    summaries = []
    for key, values in groups.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half_width = 1.96 * std / math.sqrt(len(values))
        summaries.append(
            {
                **dict(zip(keys, key)),
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )
    return summaries


def _paired_deltas(rows: list[dict]) -> list[dict]:
    values = {
        (
            row["partition"], row["cache"], row["model"],
            int(row["source_grid"]), int(row["retained_grid"]),
            int(row["shot"]), int(row["seed"]), row["metric"], row["head"],
        ): float(row["value"])
        for row in rows
    }
    collapsed = defaultdict(list)
    for binary_key in values:
        if binary_key[-1] != "binary_protonet":
            continue
        prefix = binary_key[:-1]
        positive_key = (*prefix, "positive_only")
        if positive_key in values and binary_key in values:
            summary_key = (*prefix[:6], prefix[7])
            collapsed[summary_key].append(
                values[binary_key] - values[positive_key]
            )
    result = []
    summary_names = (
        "partition", "cache", "model", "source_grid", "retained_grid",
        "shot", "metric",
    )
    for key, deltas in collapsed.items():
        mean = statistics.mean(deltas)
        std = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        half_width = 1.96 * std / math.sqrt(len(deltas))
        result.append(
            {
                **dict(zip(summary_names, key)),
                "n_seeds": len(deltas),
                "binary_minus_positive_mean": mean,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )
    return result


def _decision(summary: list[dict], primary_shot: int) -> dict:
    candidates = [
        row for row in summary
        if row["partition"] == "validation"
        and row["head"] == "binary_protonet"
        and int(row["shot"]) == primary_shot
        and row["metric"] == "auroc"
    ]
    if not candidates:
        raise ValueError("no validation binary-ProtoNet AUROC candidates")
    selected = max(candidates, key=lambda row: float(row["mean"]))
    test = next(
        row for row in summary
        if row["partition"] == "test"
        and row["cache"] == selected["cache"]
        and int(row["retained_grid"]) == int(selected["retained_grid"])
        and row["head"] == "binary_protonet"
        and int(row["shot"]) == primary_shot
        and row["metric"] == "auroc"
    )
    credible = float(test["ci95_low"]) > 0.5
    return {
        "status": (
            "credible_pneumothorax_detector"
            if credible
            else "pneumothorax_detector_not_established"
        ),
        "rule": "Select on validation AUROC; require test AUROC 95% CI above 0.5.",
        "primary_shot": primary_shot,
        "selected_configuration": {
            "cache": selected["cache"],
            "model": selected["model"],
            "retained_grid": int(selected["retained_grid"]),
            "validation_auroc": float(selected["mean"]),
            "test_auroc": float(test["mean"]),
            "test_auroc_ci95": [
                float(test["ci95_low"]), float(test["ci95_high"])
            ],
        },
        "proceed_to_falsification_baselines": credible,
        "iera_or_sms_used": False,
    }


def _score_cache_worker(
    cache_name: str,
    cache_path: Path,
    manifest_hash: str,
    episode_sets: dict,
    grids: list[int],
    shots: list[int],
    batch_size: int,
    device_name: str,
    checkpoint_path: Path,
) -> None:
    """Score one large mmap in an isolated process, then exit to unmap it."""
    device = torch.device(device_name)
    patches, metadata = load_patch_cache(
        cache_path, manifest_hash, expected_model=None
    )
    for grid in grids:
        if grid > int(metadata["pool_grid"]):
            raise ValueError(
                f"cache {cache_name!r} cannot supply retained grid {grid}; "
                f"source grid is {metadata['pool_grid']}"
            )
    rows = []
    partitions = sorted({partition for partition, _seed in episode_sets})
    seeds = sorted({seed for _partition, seed in episode_sets})
    for grid in grids:
        for partition in partitions:
            for seed in seeds:
                generated = episode_sets[(partition, seed)]
                targets = generated["targets"].flatten()
                nuisance = generated["nuisance"].flatten()
                logits_by_factor = _score_factorial(
                    patches, metadata, generated, grid, shots,
                    batch_size, device,
                )
                for shot in shots:
                    for head in HEADS:
                        logits = logits_by_factor[(shot, head)]
                        for metric, value in _metrics(
                            logits, targets, nuisance
                        ).items():
                            rows.append(
                                {
                                    "partition": partition,
                                    "cache": cache_name,
                                    "model": metadata["model"],
                                    "source_grid": metadata["pool_grid"],
                                    "retained_grid": grid,
                                    "head": head,
                                    "shot": shot,
                                    "seed": seed,
                                    "metric": metric,
                                    "value": value,
                                }
                            )
                print(
                    f"finished {cache_name}, {grid}x{grid}, "
                    f"{partition}, seed {seed}",
                    flush=True,
                )
    temporary_path = checkpoint_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {"metadata": {"name": cache_name, **metadata}, "rows": rows}
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(checkpoint_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument(
        "--cache", action="append", type=_parse_cache, required=True,
        help="Repeat NAME=PATH for each frozen backbone patch cache",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/iera/detector_factorial_v1"),
    )
    parser.add_argument("--target", default="Pneumothorax")
    parser.add_argument("--confounder", default="Support Devices")
    parser.add_argument("--grids", type=int, nargs="+", default=(4, 14))
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--primary-shot", type=int, default=3)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=tuple(range(10))
    )
    parser.add_argument("--queries-per-stratum", type=int, default=1)
    parser.add_argument("--min-stratum-patients", type=int, default=50)
    parser.add_argument("--episode-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.primary_shot not in args.shots:
        parser.error("primary-shot must be included in --shots")
    if min(args.grids) <= 0 or min(args.shots) <= 0:
        parser.error("grids and shots must be positive")
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    data = load_dataset(args.embeddings, args.manifest)
    restore_raw_target_status(data, args.raw_labels)
    missing = [
        name for name in (args.target, args.confounder)
        if name not in data.class_names
    ]
    if missing:
        raise ValueError(f"cache lacks labels {missing}")
    target_id = data.class_names.index(args.target)
    confounder_id = data.class_names.index(args.confounder)
    partitions = {
        name: split_indices(data, name, args.split_seed)
        for name in ("validate", "test")
    }
    counts = {
        name: patient_counts(
            data,
            stratum_pools(data, indices, target_id, confounder_id),
        )
        for name, indices in partitions.items()
    }
    if any(
        min(partition_counts.values()) < args.min_stratum_patients
        for partition_counts in counts.values()
    ):
        raise ValueError(
            "Pneumothorax/Support Devices lacks the requested patients per stratum"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_sets = {}
    max_shot = max(args.shots)
    for partition_index, (partition, indices) in enumerate(partitions.items()):
        for seed in args.seeds:
            generated = generate_pair_episodes(
                data, indices, target_id, confounder_id, args.episodes,
                max_shot, args.queries_per_stratum,
                args.seed + partition_index * 100_000 + seed,
                min_stratum_patients=args.min_stratum_patients,
            )
            episode_sets[(partition, seed)] = generated
            torch.save(
                generated,
                args.output_dir / f"episodes_{partition}_seed_{seed:03d}.pt",
            )
    rows = []
    cache_metadata = []
    # torch.from_file keeps a very large private mmap alive longer than the
    # tensor's Python reference. Score each backbone in a spawned process so
    # process exit guarantees unmapping before the next ~50 GB cache is opened.
    context = mp.get_context("spawn")
    for cache_name, cache_path in args.cache:
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in cache_name
        )
        checkpoint_path = args.output_dir / f"cache_rows_{safe_name}.json"
        if not checkpoint_path.exists():
            process = context.Process(
                target=_score_cache_worker,
                args=(
                    cache_name,
                    cache_path,
                    data.manifest_sha256,
                    episode_sets,
                    list(args.grids),
                    list(args.shots),
                    args.episode_batch_size,
                    str(device),
                    checkpoint_path,
                ),
            )
            process.start()
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"cache worker {cache_name!r} failed with exit code "
                    f"{process.exitcode}"
                )
        else:
            print(f"reusing completed cache rows for {cache_name}", flush=True)
        completed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        cache_metadata.append(completed["metadata"])
        rows.extend(completed["rows"])
    summary = _summaries(rows)
    paired = _paired_deltas(rows)
    decision = _decision(summary, args.primary_shot)
    _write(args.output_dir / "per_seed_metrics.csv", rows)
    _write(args.output_dir / "summary_metrics.csv", summary)
    _write(args.output_dir / "paired_head_deltas.csv", paired)
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": "Pneumothorax detector factorial",
                "target": args.target,
                "confounder": args.confounder,
                "heads": list(HEADS),
                "grids": args.grids,
                "shots": args.shots,
                "seeds": args.seeds,
                "episodes": args.episodes,
                "partition_counts": {
                    partition: {
                        f"c{key[0]}d{key[1]}": value
                        for key, value in partition_counts.items()
                    }
                    for partition, partition_counts in counts.items()
                },
                "caches": cache_metadata,
                "classifier": (
                    "cos(q,p+) versus cos(q,p+)-cos(q,p-); no learned "
                    "projection, IERA, SMS loss, or calibration"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"results written to {args.output_dir}; decision={decision['status']}"
    )


if __name__ == "__main__":
    main()
