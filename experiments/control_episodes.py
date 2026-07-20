"""Frozen, patient-disjoint episodes for the control-only experiment."""

from __future__ import annotations

import csv
import gzip
import hashlib
from collections import defaultdict
from pathlib import Path

import torch


def rotating_splits(class_names: list[str], fold_count: int = 5) -> list[dict]:
    """Rotate 3 test and 3 validation classes; the other 8 are base classes."""
    if len(class_names) != 14 or not 1 <= fold_count <= 14:
        raise ValueError("rotating controls require 14 classes and 1-14 folds")
    folds = []
    for fold in range(fold_count):
        test = [(3 * fold + i) % 14 for i in range(3)]
        validation = [(3 * fold + 3 + i) % 14 for i in range(3)]
        base = [i for i in range(14) if i not in test + validation]
        folds.append({"fold": fold, "base_class_ids": base, "validation_class_ids": validation, "test_class_ids": test})
    return folds


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pools(data: dict) -> dict[int, dict[str, list[int]]]:
    pools: dict[int, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, (label, subject) in enumerate(zip(data["labels"].tolist(), data["subject_ids"].tolist())):
        pools[int(label)][str(subject)].append(index)
    return pools


def _runs(data: dict, class_ids: list[int], episode_count: int, seeds: tuple[int, ...], offset: int):
    pools = _pools(data)
    for class_id in class_ids:
        if len(pools[class_id]) < 6:
            name = data["class_names"][class_id]
            raise ValueError(f"{name} needs six distinct patients; found {len(pools[class_id])}")

    runs = []
    for seed in seeds:
        generator = torch.Generator().manual_seed(20260720 + offset + seed)
        support = torch.empty(episode_count, 3, 5, dtype=torch.long)
        query = torch.empty(episode_count, 3, 1, dtype=torch.long)
        for episode in range(episode_count):
            used_patients: set[str] = set()
            for local_class, class_id in enumerate(class_ids):
                available = sorted(set(pools[class_id]) - used_patients)
                chosen = [available[i] for i in torch.randperm(len(available), generator=generator)[:6].tolist()]
                indices = []
                for patient in chosen:
                    candidates = pools[class_id][patient]
                    pick = torch.randint(len(candidates), (1,), generator=generator).item()
                    indices.append(candidates[pick])
                support[episode, local_class] = torch.tensor(indices[:5])
                query[episode, local_class] = indices[5]
                used_patients.update(chosen)
        runs.append({"support": support, "query": query})
    return runs


def load_or_create(
    path: Path,
    data: dict,
    manifest_path: Path,
    episode_count: int = 500,
    seeds: tuple[int, ...] = tuple(range(10)),
    fold_count: int = 5,
) -> dict:
    source_hash = _sha256(manifest_path)
    if data.get("manifest_sha256") not in (None, source_hash):
        raise ValueError("embedding file was created from a different manifest")
    settings = {"version": 2, "manifest_sha256": source_hash, "episode_count": episode_count, "seeds": list(seeds), "fold_count": fold_count}
    if path.exists():
        saved = torch.load(path, map_location="cpu")
        if any(saved.get(key) != value for key, value in settings.items()):
            raise ValueError("saved control episodes do not match this run")
        validate(saved, data)
        return saved

    saved = {**settings, "folds": []}
    names = data["class_names"]
    for split in rotating_splits(names, fold_count):
        fold = {**split}
        for partition, class_key, partition_offset in (
            ("validation_novel", "validation_class_ids", 10_000),
            ("test_novel", "test_class_ids", 20_000),
        ):
            class_ids = split[class_key]
            fold[partition] = {
                "class_ids": class_ids,
                "class_names": [names[i] for i in class_ids],
                "runs": _runs(data, class_ids, episode_count, seeds, split["fold"] * 100_000 + partition_offset),
            }
        saved["folds"].append(fold)

    validate(saved, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    return saved


def validate(saved: dict, data: dict) -> None:
    labels = data["labels"].long()
    subjects = [str(value) for value in data["subject_ids"].tolist()]
    if len(subjects) != len(set(subjects)):
        raise ValueError("the embedding subset must contain one row per patient")
    for fold in saved["folds"]:
        assigned = fold["base_class_ids"] + fold["validation_class_ids"] + fold["test_class_ids"]
        if len(assigned) != 14 or len(set(assigned)) != 14:
            raise ValueError("base/validation/test classes must be disjoint")
        for partition in ("validation_novel", "test_novel"):
            block = fold[partition]
            for run in block["runs"]:
                support, query = run["support"], run["query"]
                if support.shape != (saved["episode_count"], 3, 5) or query.shape != (saved["episode_count"], 3, 1):
                    raise ValueError("incorrect episode tensor shape")
                for local_class, class_id in enumerate(block["class_ids"]):
                    indices = torch.cat([support[:, local_class], query[:, local_class]], 1)
                    if not labels[indices].eq(class_id).all():
                        raise ValueError("episode contains an index from the wrong class")
                for episode in range(saved["episode_count"]):
                    indices = torch.cat([support[episode].flatten(), query[episode].flatten()]).tolist()
                    if len({subjects[i] for i in indices}) != 18:
                        raise ValueError("support/query patients overlap inside an episode")


def write_episode_ids(path: Path, saved: dict, data: dict) -> None:
    fields = [
        "fold", "partition", "seed", "episode_id", "class_position", "class_id", "class_name",
        "support_indices", "support_subject_ids", "support_dicom_ids",
        "query_index", "query_subject_id", "query_dicom_id",
    ]
    subjects = [str(value) for value in data["subject_ids"].tolist()]
    dicoms = data["dicom_ids"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for fold in saved["folds"]:
            for partition in ("validation_novel", "test_novel"):
                block = fold[partition]
                for seed, run in zip(saved["seeds"], block["runs"]):
                    for episode in range(saved["episode_count"]):
                        episode_id = f"f{fold['fold']:02d}-{partition}-s{seed:02d}-e{episode:04d}"
                        for position, (class_id, class_name) in enumerate(zip(block["class_ids"], block["class_names"])):
                            support = run["support"][episode, position].tolist()
                            query = run["query"][episode, position, 0].item()
                            writer.writerow({
                                "fold": fold["fold"], "partition": partition, "seed": seed, "episode_id": episode_id,
                                "class_position": position, "class_id": class_id, "class_name": class_name,
                                "support_indices": "|".join(map(str, support)),
                                "support_subject_ids": "|".join(subjects[i] for i in support),
                                "support_dicom_ids": "|".join(dicoms[i] for i in support),
                                "query_index": query, "query_subject_id": subjects[query], "query_dicom_id": dicoms[query],
                            })


def batch(data: dict, run: dict, shot: int, device: torch.device):
    support_index = run["support"][:, :, :shot].reshape(len(run["support"]), -1)
    query_index = run["query"].reshape(len(run["query"]), -1)
    labels = torch.arange(3, device=device).repeat_interleave(shot)
    query_labels = torch.arange(3).repeat(len(query_index))
    images = data["image_embeddings"].float()
    reports = data["report_embeddings"].float()
    return images[support_index].to(device), reports[support_index].to(device), labels, images[query_index].to(device), query_labels, query_index
