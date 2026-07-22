"""Run the PAIR-FSL kill experiment in single- and multi-label regimes."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from .data import ResidualDataset, load_config, load_dataset
from .episodes import batch, generate_episodes, validate_episodes
from .metrics import evaluate_multilabel, evaluate_single, select_temperature, select_threshold
from .model import METHODS, pair_fsl_logits


def _class_ids(data: ResidualDataset, config: dict, partition: str) -> list[int]:
    missing = [name for name in config["class_partitions"][partition] if name not in data.class_names]
    if missing:
        raise ValueError(f"embedding cache lacks configured classes: {missing}")
    return [data.class_names.index(name) for name in config["class_partitions"][partition]]


def _episodes(
    path: Path,
    data: ResidualDataset,
    config: dict,
    regimes: list[str],
    seeds: list[int],
    episode_count: int,
    max_shot: int,
    max_controls: int,
    queries: int,
) -> dict:
    settings = {
        "version": 1,
        "manifest_sha256": data.manifest_sha256,
        "regimes": regimes,
        "seeds": seeds,
        "episode_count": episode_count,
        "max_shot": max_shot,
        "max_controls": max_controls,
        "queries_per_class": queries,
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if any(saved.get(key) != value for key, value in settings.items()):
            raise ValueError("saved residual episodes do not match this run; choose a new output directory")
        return saved
    saved: dict = {**settings, "blocks": {}}
    for regime_index, regime in enumerate(regimes):
        saved["blocks"][regime] = {}
        for partition_index, partition in enumerate(("validation_novel", "test_novel")):
            indices = data.partition_indices(partition, config)
            class_ids = _class_ids(data, config, partition)
            runs = []
            for seed in seeds:
                run = generate_episodes(
                    data,
                    indices,
                    class_ids,
                    regime,
                    episode_count,
                    max_shot,
                    max_controls,
                    queries,
                    seed=20260722 + regime_index * 100_000 + partition_index * 10_000 + seed,
                )
                validate_episodes(run, data)
                runs.append(run)
            saved["blocks"][regime][partition] = runs
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    return saved


def _predict(data, runs, method, shot, controls, device, match_temperature, center):
    outputs = []
    for run in runs:
        values = batch(data, run, shot, controls, device)
        logits = pair_fsl_logits(
            values["positive"],
            values["negative"],
            values["query"],
            method,
            values["positive_metadata"],
            values["negative_metadata"],
            values["query_metadata"],
            match_temperature,
            center,
        )
        outputs.append((logits.detach().cpu(), values["targets"].detach().cpu(), values["query_indices"]))
    return outputs


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


class _PredictionWriter:
    """Stream the large per-query table instead of retaining it in memory."""

    fields = (
        "regime", "method", "shot", "negative_shot", "support_protocol", "seed",
        "temperature", "threshold", "query_position", "sample_index", "dicom_id",
        "class", "target", "logit", "probability",
    )

    def __init__(self, path: Path) -> None:
        self.handle = gzip.open(path, "wt", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fields, lineterminator="\n")
        self.writer.writeheader()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)

    def close(self) -> None:
        self.handle.close()


def _summary(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row["value"]))
    result = []
    for group, values in groups.items():
        result.append(
            {
                **dict(zip(keys, group)),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mimic_cxr_protocol_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/residuals"))
    parser.add_argument("--regime", choices=("both", "single_label", "multi_label"), default="both")
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5))
    parser.add_argument("--abundant-controls", type=int, default=20)
    parser.add_argument("--queries-per-class", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--match-temperature", type=float, default=0.1)
    parser.add_argument("--center", choices=("mean", "geometric_median"), default="geometric_median")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.shots) <= 0 or args.abundant_controls <= 0:
        parser.error("shots and abundant controls must be positive")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    )
    data = load_dataset(args.embeddings, args.manifest)
    config = load_config(args.config)
    regimes = ["single_label", "multi_label"] if args.regime == "both" else [args.regime]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = _episodes(
        args.output_dir / "residual_episodes.pt",
        data,
        config,
        regimes,
        list(args.seeds),
        args.episodes,
        max(args.shots),
        max(max(args.shots), args.abundant_controls),
        args.queries_per_class,
    )

    overall_rows, class_rows = [], []
    prediction_writer = _PredictionWriter(args.output_dir / "per_query_predictions.csv.gz")
    for regime in regimes:
        validation_runs = saved["blocks"][regime]["validation_novel"]
        test_runs = saved["blocks"][regime]["test_novel"]
        class_names = [data.class_names[index] for index in test_runs[0]["class_ids"]]
        for shot in args.shots:
            for controls in sorted({shot, args.abundant_controls}):
                protocol = "balanced" if controls == shot else "abundant_controls"
                for method in args.methods:
                    print(f"running {regime}: {method}, {shot}+/{controls}-", flush=True)
                    validation = _predict(data, validation_runs, method, shot, controls, device, args.match_temperature, args.center)
                    validation_logits = torch.cat([value[0].reshape(-1, value[0].shape[-1]) for value in validation])
                    validation_targets = torch.cat([value[1].reshape(-1, *value[1].shape[2:]) if value[1].ndim == 3 else value[1].flatten() for value in validation])
                    temperature = select_temperature(validation_logits, validation_targets, regime)
                    threshold = select_threshold(validation_logits, validation_targets, temperature) if regime == "multi_label" else float("nan")
                    test = _predict(data, test_runs, method, shot, controls, device, args.match_temperature, args.center)
                    for seed, (raw_logits, raw_targets, query_indices) in zip(args.seeds, test):
                        logits = raw_logits.reshape(-1, raw_logits.shape[-1]) / temperature
                        targets = raw_targets.reshape(-1, raw_targets.shape[-1]) if regime == "multi_label" else raw_targets.flatten()
                        if regime == "single_label":
                            overall, per_class, probability = evaluate_single(logits, targets, class_names)
                        else:
                            overall, per_class, probability = evaluate_multilabel(logits, targets, class_names, threshold)
                        shared = {
                            "regime": regime,
                            "method": method,
                            "shot": shot,
                            "negative_shot": controls,
                            "support_protocol": protocol,
                            "seed": seed,
                            "temperature": temperature,
                            "threshold": threshold,
                        }
                        for metric, value in overall.items():
                            overall_rows.append({**shared, "metric": metric, "value": value})
                        for class_id, row in enumerate(per_class):
                            for metric in ("auroc", "auprc", "f1", "accuracy", "nll", "calibration_error"):
                                class_rows.append({**shared, "class": row["class"], "metric": metric, "value": row[metric]})
                        flat_indices = query_indices.flatten().tolist()
                        for query_position, sample_index in enumerate(flat_indices):
                            for class_id, class_name in enumerate(class_names):
                                target = int(targets[query_position] == class_id) if regime == "single_label" else int(targets[query_position, class_id])
                                prediction_writer.write(
                                    {
                                        **shared,
                                        "query_position": query_position,
                                        "sample_index": sample_index,
                                        "dicom_id": data.dicom_ids[sample_index],
                                        "class": class_name,
                                        "target": target,
                                        "logit": float(logits[query_position, class_id]),
                                        "probability": float(probability[query_position, class_id]),
                                    }
                                )

    prediction_writer.close()
    common = ("regime", "method", "shot", "negative_shot", "support_protocol", "temperature", "threshold", "metric")
    _write_csv(args.output_dir / "per_seed_metrics.csv", overall_rows)
    _write_csv(args.output_dir / "overall_metrics.csv", _summary(overall_rows, common))
    _write_csv(args.output_dir / "per_class_per_seed.csv", class_rows)
    _write_csv(args.output_dir / "per_class_metrics.csv", _summary(class_rows, common[:-1] + ("class", "metric")))
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "proposal": "PAIR-FSL: pathology residuals from real matched controls",
                "embeddings": str(args.embeddings),
                "manifest": str(args.manifest),
                "regimes": regimes,
                "methods": list(args.methods),
                "shots": list(args.shots),
                "abundant_controls": args.abundant_controls,
                "queries_per_class": args.queries_per_class,
                "episodes": args.episodes,
                "seeds": list(args.seeds),
                "match_temperature": args.match_temperature,
                "robust_center": args.center,
                "selection_firewall": "temperature and multi-label threshold selected on validation_novel only",
                "query_reports_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
