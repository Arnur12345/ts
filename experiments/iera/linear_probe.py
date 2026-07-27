"""Deterministic frozen Rad-DINO global linear-probe diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch

from experiments.residuals.data import load_dataset
from experiments.residuals.metrics import _auc, _average_precision

from .episodes import split_indices
from .labels import restore_raw_target_status
from .patch_cache import RAD_DINO_MODEL


METHODS = ("class_balanced", "four_group_balanced")


def four_group_weights(target, device):
    """Give each observed target-by-device group equal total weight."""
    import numpy as np

    target = np.asarray(target, dtype=np.int64)
    device = np.asarray(device, dtype=np.int64)
    groups = 2 * target + device
    counts = np.bincount(groups, minlength=4)
    if (counts == 0).any():
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"four-group training split lacks groups {missing}")
    return len(groups) / (4.0 * counts[groups]), counts


def _global_features(cache_dir: Path, output_dir: Path, manifest_hash: str):
    import numpy as np

    metadata_path = cache_dir / "patch_cache.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("complete") is not True:
        raise ValueError("Rad-DINO patch cache is incomplete")
    if metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Rad-DINO cache and manifest differ")
    if metadata.get("model") != RAD_DINO_MODEL:
        raise ValueError(
            f"expected {RAD_DINO_MODEL!r}, found {metadata.get('model')!r}"
        )
    if metadata.get("global_indices") is not None:
        raise ValueError(
            "linear probe requires the dense all-study Rad-DINO cache"
        )
    shape = tuple(int(value) for value in metadata["shape"])
    token_path = cache_dir / metadata["tokens"]
    global_path = output_dir / "rad_dino_global.float32.npy"
    global_metadata_path = output_dir / "rad_dino_global.json"
    signature = {
        "source_cache": str(cache_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "source_shape": list(shape),
        "aggregation": "l2(mean(l2_normalized_patch_tokens))",
    }
    if global_path.exists() and global_metadata_path.exists():
        saved = json.loads(global_metadata_path.read_text(encoding="utf-8"))
        if saved == signature:
            print(f"reusing pooled global embeddings from {global_path}")
            return np.load(global_path, mmap_mode="r"), metadata
        raise ValueError(
            "existing pooled global embeddings have another signature; "
            "choose a new output directory"
        )
    tokens = np.memmap(token_path, dtype=np.float16, mode="r", shape=shape)
    pooled = np.lib.format.open_memmap(
        global_path,
        mode="w+",
        dtype=np.float32,
        shape=(shape[0], shape[2]),
    )
    batch_size = 256
    for start in range(0, shape[0], batch_size):
        end = min(start + batch_size, shape[0])
        current = np.asarray(tokens[start:end], dtype=np.float32).mean(axis=1)
        current /= np.maximum(
            np.linalg.norm(current, axis=1, keepdims=True), 1e-12
        )
        pooled[start:end] = current
        if end % (batch_size * 20) == 0 or end == shape[0]:
            pooled.flush()
            print(f"pooled global embeddings {end:,}/{shape[0]:,}", flush=True)
    pooled.flush()
    global_metadata_path.write_text(
        json.dumps(signature, indent=2) + "\n", encoding="utf-8"
    )
    return np.load(global_path, mmap_mode="r"), metadata


def _metric_rows(method: str, scores, target, device, device_known) -> list[dict]:
    target_tensor = torch.as_tensor(target, dtype=torch.bool)
    score_tensor = torch.as_tensor(scores, dtype=torch.float32)
    rows = [
        {
            "method": method,
            "partition": "test",
            "metric": "auroc",
            "value": _auc(target_tensor, score_tensor),
        },
        {
            "method": method,
            "partition": "test",
            "metric": "auprc",
            "value": _average_precision(target_tensor, score_tensor),
        },
    ]
    nuisance_values = []
    for value in (0, 1):
        mask = device_known & (device == value)
        observed = _auc(target_tensor[mask], score_tensor[mask])
        nuisance_values.append(observed)
        rows.append(
            {
                "method": method,
                "partition": "test",
                "metric": f"device_{value}_auroc",
                "value": observed,
            }
        )
    rows.append(
        {
            "method": method,
            "partition": "test",
            "metric": "worst_device_auroc",
            "value": min(nuisance_values),
        }
    )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _interpret(rows: list[dict]) -> dict:
    points = {}
    for method in METHODS:
        points[method] = {
            row["metric"]: float(row["value"])
            for row in rows
            if row["method"] == method
        }
    best = max(
        METHODS,
        key=lambda method: (
            points[method]["auroc"],
            points[method]["worst_device_auroc"],
        ),
    )
    auroc = points[best]["auroc"]
    worst = points[best]["worst_device_auroc"]
    if auroc >= 0.70 and worst >= 0.65:
        status = "episodic_prototype_mechanism_is_the_likely_bottleneck"
    elif auroc >= 0.60:
        status = "frozen_signal_exists_but_is_weak"
    else:
        status = "frozen_global_rad_dino_is_likely_the_main_limitation"
    return {
        "status": status,
        "best_probe_for_interpretation": best,
        "test_auroc": auroc,
        "test_worst_device_auroc": worst,
        "thresholds": {
            "strong": "AUROC >= 0.70 and worst-device AUROC >= 0.65",
            "weak": "0.60 <= AUROC < 0.70",
            "limited": "AUROC < 0.60",
        },
        "all_test_metrics": points,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument("--rad-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cs", type=float, nargs="+", default=(0.01, 0.1, 1.0, 10.0)
    )
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise SystemExit("Install the probe dependency: pip install scikit-learn") from error
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.embeddings, args.manifest)
    restore_raw_target_status(data, args.raw_labels)
    target_id = data.class_names.index("Pneumothorax")
    device_id = data.class_names.index("Support Devices")
    features, cache_metadata = _global_features(
        args.rad_cache, args.output_dir, data.manifest_sha256
    )
    if len(features) != len(data.rows):
        raise ValueError("dense Rad-DINO cache and manifest lengths differ")
    target = data.labels[:, target_id].numpy().astype(np.int64)
    target_known = data.known[:, target_id].numpy()
    device = data.labels[:, device_id].numpy().astype(np.int64)
    device_known = data.known[:, device_id].numpy()
    partitions = {}
    for partition in ("train", "validate", "test"):
        values = split_indices(data, partition, args.split_seed).numpy()
        partitions[partition] = values[target_known[values]]
    candidate_rows, selected, test_rows = [], {}, []
    for method in METHODS:
        train = partitions["train"]
        if method == "four_group_balanced":
            train = train[device_known[train]]
            sample_weight, train_group_counts = four_group_weights(
                target[train], device[train]
            )
            class_weight = None
        else:
            sample_weight = None
            train_group_counts = None
            class_weight = "balanced"
        validation = partitions["validate"]
        x_train = np.asarray(features[train], dtype=np.float32)
        x_validation = np.asarray(features[validation], dtype=np.float32)
        candidates = []
        for c_value in args.cs:
            model = LogisticRegression(
                C=c_value,
                class_weight=class_weight,
                solver="lbfgs",
                max_iter=args.max_iter,
                random_state=args.seed,
            )
            model.fit(
                x_train,
                target[train],
                sample_weight=sample_weight,
            )
            validation_score = model.predict_proba(x_validation)[:, 1]
            validation_auroc = _auc(
                torch.as_tensor(target[validation], dtype=torch.bool),
                torch.as_tensor(validation_score, dtype=torch.float32),
            )
            row = {
                "method": method,
                "C": c_value,
                "validation_auroc": validation_auroc,
            }
            candidate_rows.append(row)
            candidates.append(row)
            print(
                f"{method}, C={c_value:g}, "
                f"validation AUROC={validation_auroc:.4f}",
                flush=True,
            )
        winner = max(candidates, key=lambda row: row["validation_auroc"])
        selected[method] = {
            **winner,
            "train_examples": int(len(train)),
            "train_group_counts": (
                None
                if train_group_counts is None
                else train_group_counts.tolist()
            ),
        }
        train_validation = np.concatenate(
            (partitions["train"], partitions["validate"])
        )
        if method == "four_group_balanced":
            train_validation = train_validation[
                device_known[train_validation]
            ]
            refit_weight, _ = four_group_weights(
                target[train_validation], device[train_validation]
            )
        else:
            refit_weight = None
        final_model = LogisticRegression(
            C=winner["C"],
            class_weight=class_weight,
            solver="lbfgs",
            max_iter=args.max_iter,
            random_state=args.seed,
        )
        final_model.fit(
            np.asarray(features[train_validation], dtype=np.float32),
            target[train_validation],
            sample_weight=refit_weight,
        )
        test = partitions["test"]
        scores = final_model.predict_proba(
            np.asarray(features[test], dtype=np.float32)
        )[:, 1]
        test_rows.extend(
            _metric_rows(
                method,
                scores,
                target[test],
                device[test],
                device_known[test],
            )
        )
    decision = _interpret(test_rows)
    _write_csv(args.output_dir / "candidate_metrics.csv", candidate_rows)
    _write_csv(args.output_dir / "test_metrics.csv", test_rows)
    (args.output_dir / "selection.json").write_text(
        json.dumps(
            {
                "selection_partition": "validate",
                "criterion": "highest AUROC",
                "selected": selected,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "stage": "frozen_global_linear_probe",
                "backbone": cache_metadata["model"],
                "global_embedding": (
                    "L2-normalized mean of cached normalized patch tokens"
                ),
                "episodic_sampling": False,
                "split": "existing deterministic patient-level 70/15/15",
                "split_seed": args.split_seed,
                "C_grid": args.cs,
                "refit_after_selection": "train_plus_validation",
                "test_examples": int(len(partitions["test"])),
                "test_device_known": int(
                    device_known[partitions["test"]].sum()
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"linear-probe results written to {args.output_dir}; "
        f"decision={decision['status']}"
    )


if __name__ == "__main__":
    main()
