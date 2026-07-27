"""Prepare frozen Rad-DINO/VLM features, priors, nuisance data and teachers."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.residuals.data import load_config, load_dataset
from experiments.residuals.metrics import _auc

from .episodes import split_indices
from .labels import restore_raw_target_status


CS = (0.01, 0.1, 1.0, 10.0)


def prompt_ensembles(disease: str) -> tuple[list[str], list[str]]:
    disease_lower = disease.lower()
    if disease == "Pneumothorax":
        return (
            [
                "a chest radiograph showing pneumothorax",
                "visible pleural air consistent with pneumothorax",
                "radiographic evidence of pneumothorax",
            ],
            [
                "a chest radiograph without pneumothorax",
                "no evidence of pneumothorax",
                "normal pleural space",
            ],
        )
    return (
        [
            f"a chest radiograph showing {disease_lower}",
            f"visible findings consistent with {disease_lower}",
            f"radiographic evidence of {disease_lower}",
        ],
        [
            f"a chest radiograph without {disease_lower}",
            f"no evidence of {disease_lower}",
            f"no radiographic findings of {disease_lower}",
        ],
    )


def _scores(model, features, indices):
    return model.decision_function(features[indices])


def _fit_logistic_grid(
    features,
    labels,
    train,
    validation,
    cs,
    seed,
    device=None,
    device_known=None,
    reject_bad_device_negative: bool = False,
):
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    candidates = []
    for c_value in cs:
        model = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=seed,
        )
        model.fit(
            np.asarray(features[train], dtype=np.float32), labels[train]
        )
        score = _scores(model, features, validation)
        overall = _auc(
            torch.as_tensor(labels[validation], dtype=torch.bool),
            torch.as_tensor(score, dtype=torch.float32),
        )
        device_zero = float("nan")
        device_one = float("nan")
        worst_device = float("nan")
        catastrophic = False
        if device is not None and device_known is not None:
            mask = device_known[validation] & (device[validation] == 0)
            device_zero = _auc(
                torch.as_tensor(labels[validation][mask], dtype=torch.bool),
                torch.as_tensor(score[mask], dtype=torch.float32),
            )
            mask_one = device_known[validation] & (device[validation] == 1)
            device_one = _auc(
                torch.as_tensor(
                    labels[validation][mask_one], dtype=torch.bool
                ),
                torch.as_tensor(score[mask_one], dtype=torch.float32),
            )
            worst_device = min(device_zero, device_one)
            catastrophic = (
                not math.isfinite(device_zero)
                or device_zero < 0.55
                or overall - device_zero > 0.20
            )
        candidates.append(
            {
                "C": float(c_value),
                "validation_auroc": overall,
                "device_negative_auroc": device_zero,
                "device_positive_auroc": device_one,
                "worst_device_auroc": worst_device,
                "catastrophic_device_negative": catastrophic,
                "model": model,
            }
        )
    feasible = [
        item for item in candidates
        if not (
            reject_bad_device_negative
            and item["catastrophic_device_negative"]
        )
    ]
    pool = feasible or candidates
    selected = max(
        pool,
        key=lambda item: (
            item["validation_auroc"],
            item["device_negative_auroc"]
            if math.isfinite(item["device_negative_auroc"])
            else -math.inf,
        ),
    )
    return selected, candidates, bool(feasible)


def _encode_text(class_names, model_name, device):
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError as error:
        raise SystemExit(
            "Install embedding dependencies with: pip install -e '.[embedding]'"
        ) from error
    model, _ = create_model_from_pretrained(model_name)
    tokenizer = get_tokenizer(model_name)
    model.to(device).eval().requires_grad_(False)
    positives, negatives, prompt_records = [], [], {}
    with torch.inference_mode():
        for disease in class_names:
            positive_prompts, negative_prompts = prompt_ensembles(disease)
            positive = F.normalize(
                model.encode_text(tokenizer(positive_prompts).to(device)).float(),
                dim=-1,
            )
            negative = F.normalize(
                model.encode_text(tokenizer(negative_prompts).to(device)).float(),
                dim=-1,
            )
            positives.append(F.normalize(positive.mean(0), dim=-1).cpu())
            negatives.append(F.normalize(negative.mean(0), dim=-1).cpu())
            prompt_records[disease] = {
                "positive": positive_prompts,
                "negative": negative_prompts,
            }
    return torch.stack(positives), torch.stack(negatives), prompt_records


def _calibrate_prior(margins, labels, known, validation, base_ids):
    values, targets, weights = [], [], []
    for class_id in base_ids:
        selected = validation[known[validation, class_id]]
        current_target = labels[selected, class_id].float()
        positive = int(current_target.sum())
        negative = len(current_target) - positive
        if positive == 0 or negative == 0:
            continue
        current_weight = torch.where(
            current_target.bool(),
            torch.full_like(current_target, 0.5 / positive),
            torch.full_like(current_target, 0.5 / negative),
        )
        values.append(margins[selected, class_id])
        targets.append(current_target)
        weights.append(current_weight)
    if not values:
        raise ValueError("no base-validation labels can calibrate the VLM prior")
    values = torch.cat(values)
    targets = torch.cat(targets)
    weights = torch.cat(weights)
    log_a = torch.tensor(0.0, requires_grad=True)
    bias = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam((log_a, bias), lr=0.05)
    for _ in range(500):
        scale = F.softplus(log_a)
        losses = F.binary_cross_entropy_with_logits(
            scale * values + bias, targets, reduction="none"
        )
        loss = (losses * weights).sum() / len(base_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return float(F.softplus(log_a).detach()), float(bias.detach()), float(loss)


def _raw_nuisance(rows, device_probability, train_indices):
    import numpy as np

    values = [np.asarray(device_probability, dtype=np.float32)]
    names = ["device_probability"]
    missing = [np.zeros(len(rows), dtype=np.float32)]
    views = []
    for row in rows:
        view = str(
            row.get("view", row.get("ViewPosition", ""))
        ).strip().upper()
        views.append(1.0 if view == "AP" else 0.0 if view == "PA" else np.nan)
    values.append(np.asarray(views, dtype=np.float32))
    names.append("is_ap_view")
    portable_keys = (
        "is_portable",
        "portable",
        "procedure_description",
        "PerformedProcedureStepDescription",
    )
    portable = []
    for row in rows:
        observed = ""
        for key in portable_keys:
            if str(row.get(key, "")).strip():
                observed = str(row[key]).strip().lower()
                break
        if not observed:
            portable.append(np.nan)
        elif observed in {"1", "1.0", "true", "yes"}:
            portable.append(1.0)
        elif observed in {"0", "0.0", "false", "no"}:
            portable.append(0.0)
        else:
            portable.append(float("portable" in observed or "mobile" in observed))
    portable = np.asarray(portable, dtype=np.float32)
    if np.isfinite(portable).any():
        values.append(portable)
        names.append("is_portable")
    site_keys = ("site", "site_id", "institution", "hospital_id")
    site_key = next(
        (
            key for key in site_keys
            if any(str(row.get(key, "")).strip() for row in rows)
        ),
        None,
    )
    if site_key is not None:
        train_sites = [
            str(rows[index].get(site_key, "")).strip()
            for index in train_indices
            if str(rows[index].get(site_key, "")).strip()
        ]
        categories = sorted(set(train_sites))[:16]
        for category in categories:
            values.append(
                np.asarray(
                    [
                        (
                            float(str(row.get(site_key, "")).strip() == category)
                            if str(row.get(site_key, "")).strip()
                            else np.nan
                        )
                        for row in rows
                    ],
                    dtype=np.float32,
                )
            )
            names.append(f"site={category}")
    standardized, output_names = [], []
    statistics = {}
    train_indices = np.asarray(train_indices)
    for name, value in zip(names, values):
        present = np.isfinite(value)
        train_present = present[train_indices]
        mean = (
            float(value[train_indices][train_present].mean())
            if train_present.any()
            else 0.0
        )
        std = (
            float(value[train_indices][train_present].std())
            if train_present.any()
            else 1.0
        )
        std = max(std, 1e-6)
        filled = np.where(present, value, mean)
        standardized.append((filled - mean) / std)
        output_names.append(name)
        statistics[name] = {"mean": mean, "std": std}
        if not present.all():
            standardized.append((~present).astype(np.float32))
            output_names.append(f"{name}_missing")
    return (
        np.stack(standardized, axis=1).astype(np.float32),
        output_names,
        statistics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument("--rad-global", type=Path, required=True)
    parser.add_argument("--rad-global-metadata", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mimic_cxr_protocol_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cs", type=float, nargs="+", default=CS)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        import numpy as np
    except ImportError as error:
        raise SystemExit("Install numpy and scikit-learn") from error
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(args.embeddings, args.manifest)
    restore_raw_target_status(data, args.raw_labels)
    config = load_config(args.config)
    rad_metadata = json.loads(
        args.rad_global_metadata.read_text(encoding="utf-8")
    )
    if rad_metadata.get("manifest_sha256") != data.manifest_sha256:
        raise ValueError("Rad-DINO global cache and manifest differ")
    rad = np.load(args.rad_global, mmap_mode="r")
    if len(rad) != len(data.rows) or rad.shape[1] != 768:
        raise ValueError("expected dense [N,768] Rad-DINO global features")
    vlm = F.normalize(data.images.float(), dim=-1)
    labels = data.labels
    known = data.known
    partitions = {
        name: split_indices(data, name, args.split_seed).numpy()
        for name in ("train", "validate", "test")
    }
    target_id = data.class_names.index("Pneumothorax")
    device_id = data.class_names.index("Support Devices")
    device_labels = labels[:, device_id].numpy().astype(np.int64)
    device_known = known[:, device_id].numpy()
    device_train = partitions["train"][
        device_known[partitions["train"]]
    ]
    device_validation = partitions["validate"][
        device_known[partitions["validate"]]
    ]
    selected_device, device_candidates, _ = _fit_logistic_grid(
        rad,
        device_labels,
        device_train,
        device_validation,
        args.cs,
        args.seed,
    )
    device_probability = selected_device["model"].predict_proba(rad)[:, 1]
    nuisance, nuisance_names, nuisance_statistics = _raw_nuisance(
        data.rows, device_probability, partitions["train"]
    )
    positive_text, negative_text, prompts = _encode_text(
        data.class_names,
        str(torch.load(
            args.embeddings, map_location="cpu", weights_only=False
        ).get(
            "model",
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        )),
        torch.device(
            "cuda"
            if args.device == "auto" and torch.cuda.is_available()
            else args.device if args.device != "auto" else "cpu"
        ),
    )
    margins = vlm @ positive_text.T - vlm @ negative_text.T
    base_ids = [
        data.class_names.index(name)
        for name in config["class_partitions"]["base"]
        if name in data.class_names and name != "Pneumothorax"
    ]
    calibration_a, calibration_b, calibration_loss = _calibrate_prior(
        margins,
        labels,
        known,
        torch.as_tensor(partitions["validate"]),
        base_ids,
    )
    priors = calibration_a * margins + calibration_b
    teacher_scores = np.full(
        (len(data.rows), len(data.class_names)),
        np.nan,
        dtype=np.float32,
    )
    teacher_records = {}
    for class_id in base_ids:
        train = partitions["train"][
            known[partitions["train"], class_id].numpy()
        ]
        validation = partitions["validate"][
            known[partitions["validate"], class_id].numpy()
        ]
        selected, candidates, feasible = _fit_logistic_grid(
            rad,
            labels[:, class_id].numpy().astype(np.int64),
            train,
            validation,
            args.cs,
            args.seed + class_id,
            device=device_labels,
            device_known=device_known,
            reject_bad_device_negative=True,
        )
        teacher_scores[:, class_id] = selected["model"].decision_function(rad)
        teacher_records[data.class_names[class_id]] = {
            "selected_C": selected["C"],
            "validation_auroc": selected["validation_auroc"],
            "device_negative_auroc": selected["device_negative_auroc"],
            "device_positive_auroc": selected["device_positive_auroc"],
            "worst_device_auroc": selected["worst_device_auroc"],
            "noncatastrophic_candidate_exists": feasible,
            "candidates": [
                {key: value for key, value in candidate.items() if key != "model"}
                for candidate in candidates
            ],
            "weight": selected["model"].coef_[0].tolist(),
            "bias": float(selected["model"].intercept_[0]),
        }
        print(
            f"teacher {data.class_names[class_id]}: "
            f"AUROC={selected['validation_auroc']:.4f}, "
            f"device-={selected['device_negative_auroc']:.4f}",
            flush=True,
        )
    np.save(args.output_dir / "vlm_features.float16.npy", vlm.numpy().astype(np.float16))
    np.save(args.output_dir / "labels.uint8.npy", labels.numpy().astype(np.uint8))
    np.save(args.output_dir / "known.uint8.npy", known.numpy().astype(np.uint8))
    np.save(args.output_dir / "nuisance.float32.npy", nuisance)
    np.save(args.output_dir / "semantic_priors.float32.npy", priors.numpy())
    np.save(args.output_dir / "teacher_scores.float32.npy", teacher_scores)
    torch.save(
        {
            "class_names": data.class_names,
            "subject_ids": data.subject_ids,
            "rows": data.rows,
            "manifest_sha256": data.manifest_sha256,
        },
        args.output_dir / "dataset_index.pt",
    )
    metadata = {
        "version": 1,
        "manifest_sha256": data.manifest_sha256,
        "rad_features": str(args.rad_global.expanduser().resolve()),
        "rad_feature_shape": list(rad.shape),
        "rad_aggregation": rad_metadata["aggregation"],
        "vlm_features": "vlm_features.float16.npy",
        "labels": "labels.uint8.npy",
        "known": "known.uint8.npy",
        "nuisance": "nuisance.float32.npy",
        "nuisance_names": nuisance_names,
        "nuisance_statistics": nuisance_statistics,
        "semantic_priors": "semantic_priors.float32.npy",
        "semantic_calibration": {
            "a": calibration_a,
            "b": calibration_b,
            "base_validation_loss": calibration_loss,
            "base_classes": [data.class_names[index] for index in base_ids],
        },
        "prompts": prompts,
        "teachers": "teachers.json",
        "teacher_scores": "teacher_scores.float32.npy",
        "base_target_ids": base_ids,
        "held_out_target": "Pneumothorax",
        "device_detector": {
            "selected_C": selected_device["C"],
            "validation_auroc": selected_device["validation_auroc"],
            "candidates": [
                {key: value for key, value in item.items() if key != "model"}
                for item in device_candidates
            ],
        },
        "split_seed": args.split_seed,
        "dataset_index": "dataset_index.pt",
        "elapsed_seconds": time.perf_counter() - started,
        "complete": True,
    }
    (args.output_dir / "teachers.json").write_text(
        json.dumps(teacher_records, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comed_cache.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"CoMeD cache written to {args.output_dir}")


if __name__ == "__main__":
    main()
