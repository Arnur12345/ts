"""Load one embedding cache without discarding native multi-label targets."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _number(value: str) -> float | None:
    value = str(value).strip()
    if value.lower() in {"", "nan", "na", "none"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


@dataclass(slots=True)
class ResidualDataset:
    images: torch.Tensor
    labels: torch.Tensor
    known: torch.Tensor
    metadata: torch.Tensor
    class_names: list[str]
    subject_ids: list[str]
    dicom_ids: list[str]
    rows: list[dict[str, str]]
    manifest_sha256: str

    def partition_indices(self, partition: str, config: dict[str, Any]) -> torch.Tensor:
        """Return the official patient split assigned to a class partition."""
        official = config["official_split_for_partition"][partition]
        selected = []
        for index, row in enumerate(self.rows):
            protocol_partition = row.get("protocol_partition", "").strip()
            if protocol_partition:
                keep = protocol_partition == partition
            else:
                keep = row.get("official_split", "").strip().lower() == official
            if keep:
                selected.append(index)
        if not selected:
            raise ValueError(f"manifest has no rows for {partition!r}")
        return torch.tensor(selected, dtype=torch.long)


def _metadata(rows: list[dict[str, str]]) -> torch.Tensor:
    """Encode view, sex, and age when present; missing fields remain neutral."""
    ages = []
    for row in rows:
        ages.append(_number(row.get("age", row.get("anchor_age", ""))))
    observed = [value for value in ages if value is not None]
    age_mean = sum(observed) / len(observed) if observed else 0.0
    age_scale = max(10.0, (sum((value - age_mean) ** 2 for value in observed) / max(1, len(observed))) ** 0.5)

    features = []
    for row, age in zip(rows, ages):
        view = row.get("view", row.get("ViewPosition", "")).strip().upper()
        sex = row.get("sex", row.get("gender", "")).strip().upper()
        features.append(
            [
                float(view == "AP"),
                float(view == "PA"),
                float(view not in {"AP", "PA"}),
                float(sex in {"M", "MALE"}),
                float(sex in {"F", "FEMALE"}),
                float(sex not in {"M", "MALE", "F", "FEMALE"}),
                0.0 if age is None else (age - age_mean) / age_scale,
                float(age is not None),
            ]
        )
    return torch.tensor(features, dtype=torch.float32)


def load_dataset(embedding_path: Path, manifest_path: Path) -> ResidualDataset:
    cache = torch.load(embedding_path, map_location="cpu", weights_only=False)
    with _open_csv(manifest_path) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest is empty")
    images = cache["image_embeddings"].float()
    if len(rows) != len(images):
        raise ValueError("embedding and manifest lengths differ")
    dicoms = [row["dicom_id"] for row in rows]
    if cache.get("dicom_ids") is not None and list(cache["dicom_ids"]) != dicoms:
        raise ValueError("embedding and manifest row order differs")
    source_hash = _hash(manifest_path)
    if cache.get("manifest_sha256") not in (None, source_hash):
        raise ValueError("embedding file was created from a different manifest")

    class_names = list(cache.get("class_names", []))
    if not class_names:
        raise ValueError("embedding cache must contain class_names")
    if "label_matrix" in cache:
        raw = torch.as_tensor(cache["label_matrix"])
    elif all(name in rows[0] for name in class_names):
        raw = torch.tensor(
            [[-1 if _number(row[name]) == -1 else int(_number(row[name]) or 0) for name in class_names] for row in rows]
        )
    elif "labels" in cache and torch.as_tensor(cache["labels"]).ndim == 1:
        raw = F.one_hot(torch.as_tensor(cache["labels"]).long(), len(class_names))
    else:
        raise ValueError("cache needs label_matrix, manifest label columns, or 1-D labels")
    if raw.shape != (len(rows), len(class_names)):
        raise ValueError("label matrix shape does not match samples and class_names")
    if not torch.isin(raw, torch.tensor([-1, 0, 1])).all():
        raise ValueError("labels must use -1 (uncertain), 0, or 1")

    subjects = [str(row.get("subject_id", value)) for row, value in zip(rows, cache.get("subject_ids", range(len(rows))))]
    return ResidualDataset(
        images=F.normalize(images, dim=-1),
        labels=raw.eq(1),
        known=raw.ne(-1),
        metadata=_metadata(rows),
        class_names=class_names,
        subject_ids=subjects,
        dicom_ids=dicoms,
        rows=rows,
        manifest_sha256=source_hash,
    )


def load_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
