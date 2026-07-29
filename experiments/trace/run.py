"""Run TRACE's covariance and temporal-identifiability kill tests on MIMIC-CXR."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import statistics
import time
from collections import Counter
from pathlib import Path

import torch

from experiments.iera.episodes import split_indices
from experiments.iera.labels import restore_raw_target_status
from experiments.iera.patch_cache import (
    RAD_DINO_MODEL,
    load_patch_cache,
)
from experiments.residuals.data import load_dataset
from experiments.residuals.metrics import _auc, _average_precision

from .core import (
    TransitionPair,
    apply_shrinkage_precision,
    canonical_pathology_atom,
    consecutive_transitions,
    covariance_eigendecomposition,
    localization_statistics,
    select_transition_pairs,
    temporal_partition,
    transition_counts,
    transition_feature_batch,
)
from .evaluation import (
    classification_metrics,
    score_episode_bank,
    summarize_metric_rows,
)


def _open_csv(path: Path):
    return (
        gzip.open(path, "rt", encoding="utf-8", newline="")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8", newline="")
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _sha256_json(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _device(name: str) -> torch.device:
    return torch.device(
        "cuda"
        if name == "auto" and torch.cuda.is_available()
        else name if name != "auto" else "cpu"
    )


def _global_features(path: Path, metadata_path: Path, data):
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(
            f"missing pooled RAD-DINO features: {path}. Recreate them with "
            "experiments.iera.linear_probe before running TRACE."
        )
    features = np.load(path, mmap_mode="r")
    if features.ndim != 2 or len(features) != len(data.rows):
        raise ValueError("RAD-DINO global features and manifest differ")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("manifest_sha256") != data.manifest_sha256:
        raise ValueError("RAD-DINO global metadata and manifest differ")
    expected = "l2(mean(l2_normalized_patch_tokens))"
    if metadata.get("aggregation") != expected:
        raise ValueError(
            f"TRACE requires aggregation {expected!r}; found "
            f"{metadata.get('aggregation')!r}"
        )
    return features, metadata


def _load_target_episodes(
    path: Path,
    manifest_hash: str,
    target: str,
    confounder: str,
    seeds: list[int],
) -> tuple[dict, dict]:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("signature", {}).get("manifest_sha256") != manifest_hash:
        raise ValueError("saved episodes and manifest differ")
    available = set(saved.get("signature", {}).get("seeds", []))
    if not set(seeds).issubset(available):
        raise ValueError(
            f"requested seeds {seeds} are not all in saved episode seeds "
            f"{sorted(available)}"
        )
    pair_id = None
    for candidate, names in saved["pairs"].items():
        if tuple(names) == (target, confounder):
            pair_id = candidate
            break
    if pair_id is None:
        raise ValueError(
            f"saved episodes contain no {target}/{confounder} pair"
        )
    banks = {
        (seed, partition): saved["episodes"][(pair_id, seed, partition)]
        for seed in seeds
        for partition in ("validate", "test")
    }
    return banks, {
        "path": str(path),
        "pair_id": pair_id,
        "target": target,
        "confounder": confounder,
        "seeds": seeds,
        "signature": saved["signature"],
    }


def _covariance_state(
    features,
    data,
    args,
    output_dir: Path,
    device: torch.device,
) -> dict:
    path = output_dir / "covariance_state.pt"
    signature = {
        "manifest_sha256": data.manifest_sha256,
        "split_seed": args.split_seed,
        "estimator": "unlabeled_train_mean_and_full_sample_covariance",
        "width": int(features.shape[1]),
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError(
                "existing covariance state has a different protocol; "
                "choose another output directory"
            )
        print(f"reusing unlabeled covariance from {path}", flush=True)
        return saved
    train = split_indices(data, "train", args.split_seed)
    mean, eigenvalues, eigenvectors = covariance_eigendecomposition(
        features,
        train,
        device,
        args.covariance_batch_size,
    )
    state = {
        "signature": signature,
        "mean": mean,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "train_examples": len(train),
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "condition_without_ridge": float(
            eigenvalues.max() / eigenvalues.clamp_min(1e-12).min()
        ),
    }
    torch.save(state, path)
    print(f"saved unlabeled covariance to {path}", flush=True)
    return state


def _method_rows(
    method: str,
    scores: dict,
    episodes: dict,
    seed: int,
    shot: int,
    partition: str,
    candidate: float | str,
) -> list[dict]:
    metrics = classification_metrics(
        scores, episodes["targets"], episodes["nuisance"]
    )
    return [
        {
            "partition": partition,
            "method": method,
            "candidate": candidate,
            "seed": seed,
            "shot": shot,
            "metric": metric,
            "value": value,
        }
        for metric, value in metrics.items()
    ]


def _mean_metric(
    rows: list[dict],
    method: str,
    candidate,
    shot: int,
    metric: str,
    partition: str,
) -> float:
    values = [
        float(row["value"])
        for row in rows
        if row["method"] == method
        and str(row["candidate"]) == str(candidate)
        and int(row["shot"]) == shot
        and row["metric"] == metric
        and row["partition"] == partition
    ]
    if not values:
        raise ValueError(
            f"missing {partition} {method} {candidate} {shot} {metric}"
        )
    return statistics.mean(values)


def run_covariance(
    features,
    episodes: dict,
    covariance: dict,
    args,
    output_root: Path,
    device: torch.device,
) -> dict:
    output_dir = output_root / "covariance"
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_rows = []
    for seed in args.seeds:
        bank = episodes[(seed, "validate")]
        for shot in args.shots:
            baseline = score_episode_bank(
                features, bank, shot, device
            )
            validation_rows.extend(
                _method_rows(
                    "binary_protonet",
                    baseline,
                    bank,
                    seed,
                    shot,
                    "validate",
                    "none",
                )
            )
            for ridge in args.ridges:
                scores = score_episode_bank(
                    features,
                    bank,
                    shot,
                    device,
                    covariance["eigenvalues"],
                    covariance["eigenvectors"],
                    ridge,
                )
                validation_rows.extend(
                    _method_rows(
                        "shrinkage_lda",
                        scores,
                        bank,
                        seed,
                        shot,
                        "validate",
                        ridge,
                    )
                )
        print(f"validated covariance seed={seed}", flush=True)
    winner = max(
        args.ridges,
        key=lambda ridge: (
            _mean_metric(
                validation_rows,
                "shrinkage_lda",
                ridge,
                args.covariance_primary_shot,
                "auroc",
                "validate",
            ),
            -_mean_metric(
                validation_rows,
                "shrinkage_lda",
                ridge,
                args.covariance_primary_shot,
                "sms_fixed_reference",
                "validate",
            ),
        ),
    )
    test_rows = []
    for seed in args.seeds:
        bank = episodes[(seed, "test")]
        for shot in args.shots:
            for method, ridge in (
                ("binary_protonet", None),
                ("shrinkage_lda", winner),
            ):
                scores = score_episode_bank(
                    features,
                    bank,
                    shot,
                    device,
                    (
                        covariance["eigenvalues"]
                        if ridge is not None
                        else None
                    ),
                    (
                        covariance["eigenvectors"]
                        if ridge is not None
                        else None
                    ),
                    ridge,
                )
                test_rows.extend(
                    _method_rows(
                        method,
                        scores,
                        bank,
                        seed,
                        shot,
                        "test",
                        "none" if ridge is None else ridge,
                    )
                )
        print(f"tested covariance seed={seed}", flush=True)
    all_rows = validation_rows + test_rows
    _write_csv(output_dir / "per_seed_metrics.csv", all_rows)
    _write_csv(
        output_dir / "summary_metrics.csv",
        summarize_metric_rows(
            all_rows,
            ("partition", "method", "candidate", "shot", "metric"),
        ),
    )
    selected_test = _mean_metric(
        test_rows,
        "shrinkage_lda",
        winner,
        args.covariance_primary_shot,
        "auroc",
        "test",
    )
    empirical_baseline = _mean_metric(
        test_rows,
        "binary_protonet",
        "none",
        args.covariance_primary_shot,
        "auroc",
        "test",
    )
    decision = {
        "status": (
            "anisotropic_estimation_is_a_major_bottleneck"
            if selected_test >= args.existing_10shot_auroc + 0.03
            else "covariance_does_not_close_the_few_shot_gap"
        ),
        "selection_partition": "validate",
        "selection_criterion": (
            "highest mean validation AUROC at the primary shot; "
            "lower fixed-reference SMS breaks ties"
        ),
        "primary_shot": args.covariance_primary_shot,
        "selected_ridge": winner,
        "test_auroc": selected_test,
        "same_episode_binary_protonet_test_auroc": empirical_baseline,
        "published_existing_10shot_auroc": args.existing_10shot_auroc,
        "substantial_gain_threshold": args.existing_10shot_auroc + 0.03,
        "covariance_examples": covariance["train_examples"],
        "labels_used_for_covariance": False,
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"covariance kill test written to {output_dir}; "
        f"decision={decision['status']}",
        flush=True,
    )
    return decision


def _merge_metadata(rows: list[dict], path: Path | None) -> None:
    if path is None:
        return
    with _open_csv(path) as handle:
        source = list(csv.DictReader(handle))
    if not source:
        raise ValueError("metadata CSV is empty")
    by_dicom = {
        str(row.get("dicom_id", "")).strip(): row
        for row in source
        if str(row.get("dicom_id", "")).strip()
    }
    by_study = {
        str(row.get("study_id", "")).strip(): row
        for row in source
        if str(row.get("study_id", "")).strip()
    }
    matched = 0
    for row in rows:
        extra = by_dicom.get(str(row.get("dicom_id", "")).strip())
        if extra is None:
            extra = by_study.get(str(row.get("study_id", "")).strip())
        if extra is None:
            continue
        matched += 1
        for key, value in extra.items():
            if not str(row.get(key, "")).strip():
                row[key] = value
    if not matched:
        raise ValueError("metadata CSV did not match any manifest row")


def _pair_signature(pairs: list[TransitionPair]) -> str:
    return _sha256_json(
        [
            (
                pair.before,
                pair.after,
                pair.target_before,
                pair.target_after,
                pair.device_before,
                pair.device_after,
            )
            for pair in pairs
        ]
    )


def _extract_transition_split(
    name: str,
    pairs: list[TransitionPair],
    patches,
    cache_metadata: dict,
    data,
    args,
    output_dir: Path,
    device: torch.device,
) -> dict:
    path = output_dir / f"transition_{name}.pt"
    signature = {
        "manifest_sha256": data.manifest_sha256,
        "patch_cache_model": cache_metadata["model"],
        "patch_cache_grid": cache_metadata["pool_grid"],
        "pairs_sha256": _pair_signature(pairs),
        "max_registration_shift": args.max_registration_shift,
        "feature": "mean_absolute_registered_token_residual_plus_energy_map",
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("signature") != signature:
            raise ValueError(
                f"existing {name} transition features differ; "
                "choose another output directory"
            )
        print(f"reusing {name} transition features from {path}", flush=True)
        return saved
    values: dict[str, list[torch.Tensor]] = {
        "features": [],
        "signed_mean": [],
        "energy_map": [],
        "valid": [],
        "shifts": [],
        "registration_score": [],
    }
    for start in range(0, len(pairs), args.transition_batch_size):
        current = pairs[start : start + args.transition_batch_size]
        indices = torch.tensor(
            [[pair.before, pair.after] for pair in current],
            dtype=torch.long,
        )
        tokens = patches[indices].to(device, non_blocking=True)
        extracted = transition_feature_batch(
            tokens[:, 0],
            tokens[:, 1],
            int(cache_metadata["pool_grid"]),
            args.max_registration_shift,
        )
        for key in values:
            values[key].append(extracted[key].detach().cpu())
        end = min(start + len(current), len(pairs))
        if end % 256 < args.transition_batch_size or end == len(pairs):
            print(
                f"extracted {name} transitions {end:,}/{len(pairs):,}",
                flush=True,
            )
    payload = {
        "signature": signature,
        **{key: torch.cat(value) for key, value in values.items()},
        "before": torch.tensor([pair.before for pair in pairs]),
        "after": torch.tensor([pair.after for pair in pairs]),
        "subject_id": [pair.subject_id for pair in pairs],
        "target_before": torch.tensor(
            [pair.target_before for pair in pairs], dtype=torch.long
        ),
        "target_after": torch.tensor(
            [pair.target_after for pair in pairs], dtype=torch.long
        ),
        "device_before": torch.tensor(
            [pair.device_before for pair in pairs], dtype=torch.long
        ),
        "device_after": torch.tensor(
            [pair.device_after for pair in pairs], dtype=torch.long
        ),
        "target_changed": torch.tensor(
            [pair.target_changed for pair in pairs], dtype=torch.bool
        ),
        "device_changed": torch.tensor(
            [pair.device_changed for pair in pairs], dtype=torch.bool
        ),
        "view_changed": torch.tensor(
            [pair.view_changed for pair in pairs], dtype=torch.bool
        ),
        "group": [pair.stratum for pair in pairs],
    }
    torch.save(payload, path)
    return payload


def _binary_auc(target, score) -> float:
    return _auc(
        torch.as_tensor(target, dtype=torch.bool),
        torch.as_tensor(score, dtype=torch.float32),
    )


def _fit_transition_classifier(
    fit: dict,
    validation: dict,
    test: dict,
    args,
    output_dir: Path,
) -> tuple[dict, object]:
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError as error:
        raise SystemExit(
            "Install the established diagnostic dependency: "
            "pip install scikit-learn"
        ) from error
    x_fit = fit["features"].numpy().astype(np.float32)
    x_validation = validation["features"].numpy().astype(np.float32)
    x_test = test["features"].numpy().astype(np.float32)
    mean = x_fit.mean(axis=0)
    scale = x_fit.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x_fit = (x_fit - mean) / scale
    x_validation = (x_validation - mean) / scale
    x_test = (x_test - mean) / scale
    y_fit = fit["target_changed"].numpy().astype(np.int64)
    device_change_fit = fit["device_changed"].numpy().astype(np.int64)
    y_validation = validation["target_changed"].numpy().astype(np.int64)
    y_test = test["target_changed"].numpy().astype(np.int64)
    fit_groups = 2 * y_fit + device_change_fit
    group_counts = np.bincount(fit_groups, minlength=4)
    expected_groups = np.asarray([0, 1, 2])
    if (group_counts[expected_groups] == 0).any():
        raise ValueError(
            "internal temporal fit split lacks required stable, device-only, "
            "or disease-only groups "
            f"{expected_groups[group_counts[expected_groups] == 0].tolist()}; inspect "
            "transition_counts.json"
        )
    if group_counts[3] != 0:
        raise ValueError("simultaneous disease/device changes must be excluded")
    sample_weight = len(fit_groups) / (3.0 * group_counts[fit_groups])
    rows = []
    candidates = []
    for c_value in args.transition_cs:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=args.transition_max_iter,
            random_state=args.seed,
        )
        model.fit(x_fit, y_fit, sample_weight=sample_weight)
        score = model.decision_function(x_validation)
        stable_device = ~validation["device_changed"].numpy()
        primary = _binary_auc(
            y_validation[stable_device], score[stable_device]
        )
        disease_stable = ~validation["target_changed"].numpy()
        device_target = validation["device_changed"].numpy()[disease_stable]
        device_score = score[disease_stable]
        device_auc = (
            _binary_auc(device_target, device_score)
            if len(np.unique(device_target)) == 2
            else float("nan")
        )
        rows.extend(
            [
                {
                    "partition": "validate",
                    "C": c_value,
                    "metric": "disease_change_auroc_device_stable",
                    "value": primary,
                },
                {
                    "partition": "validate",
                    "C": c_value,
                    "metric": "device_only_activation_auroc",
                    "value": device_auc,
                },
            ]
        )
        candidates.append(
            (
                primary,
                -abs(device_auc - 0.5)
                if math_is_finite(device_auc)
                else -1.0,
                c_value,
                model,
            )
        )
    _, _, selected_c, model = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    test_score = model.decision_function(x_test)
    stable_device = ~test["device_changed"].numpy()
    primary = _binary_auc(y_test[stable_device], test_score[stable_device])
    overall = _binary_auc(y_test, test_score)
    disease_stable = ~test["target_changed"].numpy()
    device_target = test["device_changed"].numpy()[disease_stable]
    device_score = test_score[disease_stable]
    device_auc = (
        _binary_auc(device_target, device_score)
        if len(np.unique(device_target)) == 2
        else float("nan")
    )
    rows.extend(
        [
            {
                "partition": "test",
                "C": selected_c,
                "metric": "disease_change_auroc_device_stable",
                "value": primary,
            },
            {
                "partition": "test",
                "C": selected_c,
                "metric": "disease_change_auroc_all",
                "value": overall,
            },
            {
                "partition": "test",
                "C": selected_c,
                "metric": "device_only_activation_auroc",
                "value": device_auc,
            },
        ]
    )
    _write_csv(output_dir / "transition_classifier_metrics.csv", rows)
    np.savez(
        output_dir / "transition_classifier.npz",
        mean=mean,
        scale=scale,
        coefficient=model.coef_,
        intercept=model.intercept_,
        selected_c=np.asarray([selected_c]),
    )
    decision = {
        "selection_partition": "internal_train_patient_validation",
        "selected_C": selected_c,
        "fit_disease_by_device_change_counts": group_counts.tolist(),
        "fit_group_weighting": (
            "equal total mass for both-stable, device-only, and disease-only "
            "strata; simultaneous changes excluded"
        ),
        "test_disease_change_auroc_device_stable": primary,
        "test_disease_change_auroc_all": overall,
        "test_device_only_activation_auroc": device_auc,
        "device_only_deviation_from_chance": abs(device_auc - 0.5),
    }
    return decision, test_score


def math_is_finite(value: float) -> bool:
    return value == value and abs(value) != float("inf")


def _save_heatmap(path: Path, values: torch.Tensor, grid: int) -> None:
    try:
        from PIL import Image
    except ImportError:
        return
    current = values.float().reshape(grid, grid)
    current = current - current.min()
    current = current / current.max().clamp_min(1e-8)
    red = (255 * current).byte().numpy()
    green = (80 * current.sqrt()).byte().numpy()
    blue = (30 * (1 - current)).byte().numpy()
    image = Image.fromarray(
        __import__("numpy").stack((red, green, blue), axis=-1),
        mode="RGB",
    )
    image.resize((grid * 32, grid * 32), Image.Resampling.NEAREST).save(
        path
    )


def _save_localization(
    test: dict,
    test_score,
    data,
    args,
    output_dir: Path,
) -> dict:
    import numpy as np

    grid = args.retained_grid
    statistics_by_group = localization_statistics(
        test["energy_map"], test["group"], grid
    )
    means = {}
    for group in sorted(set(test["group"])):
        selected = torch.tensor(
            [value == group for value in test["group"]], dtype=torch.bool
        )
        means[group] = test["energy_map"][selected].float().mean(dim=0)
        _save_heatmap(
            output_dir / f"heatmap_{group}.png", means[group], grid
        )
    torch.save(means, output_dir / "mean_transition_heatmaps.pt")
    visual_rows = []
    disease_only = [
        index
        for index, group in enumerate(test["group"])
        if group == "disease_change_device_stable"
    ]
    disease_only.sort(key=lambda index: float(test_score[index]), reverse=True)
    for rank, index in enumerate(
        disease_only[: args.visual_examples], start=1
    ):
        after = int(test["after"][index])
        row = data.rows[after]
        visual_rows.append(
            {
                "rank": rank,
                "before_index": int(test["before"][index]),
                "after_index": after,
                "subject_id": data.subject_ids[after],
                "study_id": row.get("study_id", ""),
                "dicom_id": row.get("dicom_id", ""),
                "relative_path": row.get("relative_path", ""),
                "classifier_score": float(test_score[index]),
                "heatmap_tensor_row": index,
            }
        )
    if visual_rows:
        _write_csv(output_dir / "visual_review_examples.csv", visual_rows)
    (output_dir / "localization.json").write_text(
        json.dumps(
            {
                "review_status": args.pleural_review,
                "warning": (
                    "The pleural band is a coarse review proxy, not a "
                    "radiologist annotation or pneumothorax ROI."
                ),
                "statistics": statistics_by_group,
                "saved_mean_heatmaps": sorted(means),
                "visual_examples": len(visual_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez(
        output_dir / "transition_heatmaps.npz",
        **{name: value.numpy() for name, value in means.items()},
    )
    return statistics_by_group


def _temporal_atom_fewshot(
    fit: dict,
    features,
    episodes: dict,
    covariance: dict,
    selected_ridge: float,
    args,
    output_dir: Path,
    device: torch.device,
) -> dict:
    atom = canonical_pathology_atom(
        fit["signed_mean"],
        fit["target_before"],
        fit["target_after"],
        fit["device_changed"],
    )
    atom = apply_shrinkage_precision(
        atom[None],
        covariance["eigenvalues"],
        covariance["eigenvectors"],
        selected_ridge,
    )[0]
    torch.save(
        {
            "atom": atom,
            "construction": (
                "onset-oriented mean registered residual from "
                "disease-change/device-stable training-patient pairs, then "
                "the validation-selected unlabeled covariance precision"
            ),
            "diagnostic_only": True,
        },
        output_dir / "pneumothorax_temporal_atom.pt",
    )
    validation_rows = []
    for weight in args.atom_weights:
        for seed in args.seeds:
            bank = episodes[(seed, "validate")]
            scores = score_episode_bank(
                features,
                bank,
                args.temporal_primary_shot,
                device,
                covariance["eigenvalues"],
                covariance["eigenvectors"],
                selected_ridge,
                atom,
                weight,
            )
            validation_rows.extend(
                _method_rows(
                    "temporal_atom_lda",
                    scores,
                    bank,
                    seed,
                    args.temporal_primary_shot,
                    "validate",
                    weight,
                )
            )
    feasible = [
        weight
        for weight in args.atom_weights
        if _mean_metric(
            validation_rows,
            "temporal_atom_lda",
            weight,
            args.temporal_primary_shot,
            "sms_fixed_reference",
            "validate",
        )
        <= args.sms_budget
    ]
    candidates = feasible if feasible else args.atom_weights
    selected_weight = max(
        candidates,
        key=lambda weight: _mean_metric(
            validation_rows,
            "temporal_atom_lda",
            weight,
            args.temporal_primary_shot,
            "auroc",
            "validate",
        ),
    )
    test_rows = []
    for seed in args.seeds:
        bank = episodes[(seed, "test")]
        for shot in args.shots:
            scores = score_episode_bank(
                features,
                bank,
                shot,
                device,
                covariance["eigenvalues"],
                covariance["eigenvectors"],
                selected_ridge,
                atom,
                selected_weight,
            )
            test_rows.extend(
                _method_rows(
                    "temporal_atom_lda",
                    scores,
                    bank,
                    seed,
                    shot,
                    "test",
                    selected_weight,
                )
            )
    all_rows = validation_rows + test_rows
    _write_csv(output_dir / "temporal_fewshot_per_seed.csv", all_rows)
    _write_csv(
        output_dir / "temporal_fewshot_summary.csv",
        summarize_metric_rows(
            all_rows,
            ("partition", "method", "candidate", "shot", "metric"),
        ),
    )
    test_auroc = _mean_metric(
        test_rows,
        "temporal_atom_lda",
        selected_weight,
        args.temporal_primary_shot,
        "auroc",
        "test",
    )
    test_sms = _mean_metric(
        test_rows,
        "temporal_atom_lda",
        selected_weight,
        args.temporal_primary_shot,
        "sms_fixed_reference",
        "test",
    )
    return {
        "selected_atom_weight": selected_weight,
        "validation_sms_feasible": bool(feasible),
        "test_3shot_auroc": test_auroc,
        "test_3shot_sms_fixed_reference": test_sms,
        "existing_3shot_auroc": args.existing_3shot_auroc,
        "required_3shot_auroc": args.existing_3shot_auroc + 0.05,
        "diagnostic_only_target_labels_used_for_atom": True,
    }


def run_temporal(
    features,
    episodes: dict,
    covariance: dict,
    covariance_decision: dict | None,
    data,
    args,
    output_root: Path,
    device: torch.device,
) -> dict:
    output_dir = output_root / "temporal"
    output_dir.mkdir(parents=True, exist_ok=True)
    _merge_metadata(data.rows, args.metadata_csv)
    target_id = data.class_names.index(args.target)
    device_id = data.class_names.index(args.confounder)
    main_train = split_indices(data, "train", args.split_seed)
    all_pairs = consecutive_transitions(
        data.rows,
        data.subject_ids,
        data.labels,
        data.known,
        target_id,
        device_id,
        main_train.tolist(),
    )
    count_payload = {
        "all_main_training_patients": transition_counts(all_pairs),
        "internal_partitions": {},
    }
    partitions = {}
    for partition_index, partition in enumerate(("fit", "validate", "test")):
        available = [
            pair
            for pair in all_pairs
            if temporal_partition(pair.subject_id, args.split_seed) == partition
        ]
        count_payload["internal_partitions"][partition] = transition_counts(
            available
        )
        partitions[partition] = select_transition_pairs(
            available,
            args.max_pairs_per_stratum,
            args.seed + partition_index * 10_000,
        )
        count_payload["internal_partitions"][partition][
            "selected_for_feature_pilot"
        ] = len(partitions[partition])
    (output_dir / "transition_counts.json").write_text(
        json.dumps(count_payload, indent=2) + "\n", encoding="utf-8"
    )
    patches, cache_metadata = load_patch_cache(
        args.rad_cache,
        data.manifest_sha256,
        expected_model=RAD_DINO_MODEL,
        expected_pool_grid=args.retained_grid,
        access_mode="stream",
    )
    try:
        payload = {
            partition: _extract_transition_split(
                partition,
                partitions[partition],
                patches,
                cache_metadata,
                data,
                args,
                output_dir,
                device,
            )
            for partition in ("fit", "validate", "test")
        }
    finally:
        if hasattr(patches, "close"):
            patches.close()
    classifier, test_score = _fit_transition_classifier(
        payload["fit"],
        payload["validate"],
        payload["test"],
        args,
        output_dir,
    )
    localization = _save_localization(
        payload["test"], test_score, data, args, output_dir
    )
    selected_ridge = (
        covariance_decision["selected_ridge"]
        if covariance_decision is not None
        else args.ridges[0]
    )
    fewshot = _temporal_atom_fewshot(
        payload["fit"],
        features,
        episodes,
        covariance,
        selected_ridge,
        args,
        output_dir,
        device,
    )
    gates = {
        "disease_change_auroc_at_least_0_75": (
            classifier["test_disease_change_auroc_device_stable"] >= 0.75
        ),
        "device_only_activation_near_chance": (
            classifier["device_only_deviation_from_chance"]
            <= args.device_chance_tolerance
        ),
        "pleural_localization_review_passed": args.pleural_review == "pass",
        "three_shot_gain_at_least_0_05": (
            fewshot["test_3shot_auroc"]
            >= fewshot["required_3shot_auroc"]
        ),
        "three_shot_sms_at_most_0_6": (
            fewshot["test_3shot_sms_fixed_reference"] <= args.sms_budget
        ),
    }
    numeric = all(
        value
        for key, value in gates.items()
        if key != "pleural_localization_review_passed"
    )
    if numeric and args.pleural_review == "pending":
        status = "manual_pleural_localization_review_required"
    elif all(gates.values()):
        status = "continue_to_target_masked_trace_pretraining"
    else:
        status = "stop_trace_temporal_identifiability_not_supported"
    decision = {
        "status": status,
        "scope": "diagnostic kill test only",
        "main_dataset": "MIMIC-CXR-JPG",
        "target": args.target,
        "confounder": args.confounder,
        "registered_grid": args.retained_grid,
        "classifier": classifier,
        "fewshot": fewshot,
        "localization": localization,
        "gates": gates,
        "strict_few_shot_warning": (
            "This pilot deliberately uses Pneumothorax transition labels to "
            "test identifiability. It cannot support a strict novel-class "
            "claim. A passing pilot must be followed by target-name-masked, "
            "base-disease temporal pretraining using training patients only."
        ),
    }
    (output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"temporal kill test written to {output_dir}; "
        f"decision={status}",
        flush=True,
    )
    return decision


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.target == args.confounder:
        parser.error("target and confounder must differ")
    if any(value <= 0 for value in args.ridges):
        parser.error("ridges must be positive")
    if args.covariance_primary_shot not in args.shots:
        parser.error("covariance-primary-shot must be present in shots")
    if args.temporal_primary_shot not in args.shots:
        parser.error("temporal-primary-shot must be present in shots")
    if any(value < 0 for value in args.atom_weights):
        parser.error("atom-weights must be nonnegative")
    if not 0 < args.sms_budget:
        parser.error("sms-budget must be positive")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument("--rad-global", type=Path, required=True)
    parser.add_argument("--rad-global-metadata", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--rad-cache", type=Path)
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="Optional MIMIC metadata CSV with StudyDate/StudyTime",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/trace/kill_tests_v1"),
    )
    parser.add_argument(
        "--stage", choices=("covariance", "temporal", "both"), default="both"
    )
    parser.add_argument("--target", default="Pneumothorax")
    parser.add_argument("--confounder", default="Support Devices")
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=tuple(range(10))
    )
    parser.add_argument(
        "--ridges",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1),
    )
    parser.add_argument("--atom-weights", type=float, nargs="+", default=(0, 0.1, 0.3, 1, 3))
    parser.add_argument("--covariance-primary-shot", type=int, default=10)
    parser.add_argument("--temporal-primary-shot", type=int, default=3)
    parser.add_argument("--existing-10shot-auroc", type=float, default=0.6085)
    parser.add_argument("--existing-3shot-auroc", type=float, default=0.5686)
    parser.add_argument("--sms-budget", type=float, default=0.6)
    parser.add_argument("--retained-grid", type=int, default=14)
    parser.add_argument("--max-registration-shift", type=int, default=2)
    parser.add_argument("--max-pairs-per-stratum", type=int, default=2000)
    parser.add_argument("--transition-batch-size", type=int, default=16)
    parser.add_argument(
        "--transition-cs",
        type=float,
        nargs="+",
        default=(0.01, 0.1, 1.0, 10.0),
    )
    parser.add_argument("--transition-max-iter", type=int, default=1000)
    parser.add_argument("--covariance-batch-size", type=int, default=4096)
    parser.add_argument("--device-chance-tolerance", type=float, default=0.1)
    parser.add_argument("--visual-examples", type=int, default=12)
    parser.add_argument(
        "--pleural-review",
        choices=("pending", "pass", "fail"),
        default="pending",
    )
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    _validate_args(parser, args)
    if args.stage in {"temporal", "both"} and args.rad_cache is None:
        parser.error("temporal stage requires --rad-cache")
    started = time.perf_counter()
    device = _device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.embeddings, args.manifest)
    restore_raw_target_status(data, args.raw_labels)
    if args.target not in data.class_names or args.confounder not in data.class_names:
        raise ValueError("target or confounder is absent from cached labels")
    features, global_metadata = _global_features(
        args.rad_global, args.rad_global_metadata, data
    )
    episodes, episode_metadata = _load_target_episodes(
        args.episodes,
        data.manifest_sha256,
        args.target,
        args.confounder,
        list(args.seeds),
    )
    covariance = _covariance_state(
        features, data, args, args.output_dir, device
    )
    covariance_decision = None
    temporal_decision = None
    if args.stage in {"covariance", "both"}:
        covariance_decision = run_covariance(
            features,
            episodes,
            covariance,
            args,
            args.output_dir,
            device,
        )
    elif (args.output_dir / "covariance" / "decision.json").exists():
        covariance_decision = json.loads(
            (args.output_dir / "covariance" / "decision.json").read_text(
                encoding="utf-8"
            )
        )
    if args.stage in {"temporal", "both"}:
        if covariance_decision is None:
            raise ValueError(
                "temporal atom scoring requires a validation-selected "
                "covariance ridge; run --stage covariance first or use "
                "--stage both"
            )
        temporal_decision = run_temporal(
            features,
            episodes,
            covariance,
            covariance_decision,
            data,
            args,
            args.output_dir,
            device,
        )
    experiment = {
        "stage": args.stage,
        "dataset": "MIMIC-CXR-JPG",
        "target": args.target,
        "confounder": args.confounder,
        "split": "deterministic patient-level 70/15/15",
        "split_seed": args.split_seed,
        "episodes": episode_metadata,
        "global_features": global_metadata,
        "covariance": covariance_decision,
        "temporal": temporal_decision,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "experiment.json").write_text(
        json.dumps(experiment, indent=2) + "\n", encoding="utf-8"
    )
    final_status = (
        temporal_decision["status"]
        if temporal_decision is not None
        else covariance_decision["status"]
    )
    print(
        f"TRACE kill tests written to {args.output_dir}; "
        f"decision={final_status}"
    )


if __name__ == "__main__":
    main()
