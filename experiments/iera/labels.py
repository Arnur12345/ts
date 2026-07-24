"""Restore explicit target status from the raw MIMIC-CXR CheXpert table."""

from __future__ import annotations

import csv
import gzip
from pathlib import Path

import torch

from experiments.residuals.data import ResidualDataset


def _open(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(newline="", encoding="utf-8")


def restore_raw_target_status(data: ResidualDataset, raw_labels: Path) -> None:
    """Mutate labels/known so blank and uncertain are unknown, never negative."""
    with _open(raw_labels) as handle:
        reader = csv.DictReader(handle)
        required = {"study_id", *data.class_names}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"raw label table is missing columns: {sorted(missing)}")
        by_study = {str(int(float(row["study_id"]))): row for row in reader}
    labels = torch.zeros_like(data.labels)
    known = torch.zeros_like(data.known)
    missing_studies = []
    for index, row in enumerate(data.rows):
        study_id = str(int(float(row["study_id"])))
        source = by_study.get(study_id)
        if source is None:
            missing_studies.append(study_id)
            continue
        for class_id, name in enumerate(data.class_names):
            raw = str(source[name]).strip().lower()
            if raw in {"", "nan", "na", "none", "-1", "-1.0"}:
                continue
            value = float(raw)
            if value not in {0.0, 1.0}:
                raise ValueError(f"unexpected raw label {raw!r} for {name}")
            known[index, class_id] = True
            labels[index, class_id] = value == 1.0
    if missing_studies:
        raise ValueError(f"raw label table lacks {len(missing_studies)} cached studies")
    data.labels = labels
    data.known = known
