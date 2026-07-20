from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import torch


def load_dataset(embedding_path: Path, manifest_path: Path) -> dict:
    data = torch.load(embedding_path, map_location="cpu")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(data["labels"]):
        raise ValueError("embedding and manifest lengths differ")
    if [row["dicom_id"] for row in rows] != data["dicom_ids"]:
        raise ValueError("embedding and manifest row order differs")
    subjects = [str(value) for value in data.get("subject_ids", [])]
    if subjects and len(set(subjects)) != len(subjects):
        raise ValueError("subset must contain at most one study per patient")
    data["rows"] = rows
    return data


def _manifest_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_or_create(
    path: Path,
    data: dict,
    manifest_path: Path,
    episode_count: int = 500,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
) -> dict:
    source_hash = _manifest_hash(manifest_path)
    if path.exists():
        saved = torch.load(path, map_location="cpu")
        if saved["manifest_sha256"] != source_hash or saved["episode_count"] != episode_count:
            raise ValueError("saved episodes do not match this subset")
        return saved

    labels = data["labels"].long()
    class_names = data["class_names"]
    rows = data["rows"]
    saved = {"manifest_sha256": source_hash, "episode_count": episode_count, "seeds": list(seeds)}
    split_offsets = {"validation_novel": 10_000, "test_novel": 20_000}

    for partition, offset in split_offsets.items():
        class_ids = sorted(
            {int(labels[i]) for i, row in enumerate(rows) if row["protocol_partition"] == partition}
        )
        if len(class_ids) != 3:
            raise ValueError(f"{partition} must contain exactly three classes")
        pools = [
            torch.tensor(
                [i for i, row in enumerate(rows) if int(labels[i]) == class_id and row["protocol_partition"] == partition]
            )
            for class_id in class_ids
        ]
        if min(map(len, pools)) < 6:
            raise ValueError(f"{partition} needs at least six samples per class")

        partition_runs = []
        for seed in seeds:
            generator = torch.Generator().manual_seed(offset + seed)
            support = torch.empty(episode_count, 3, 5, dtype=torch.long)
            query = torch.empty(episode_count, 3, 1, dtype=torch.long)
            for episode in range(episode_count):
                for class_index, pool in enumerate(pools):
                    chosen = pool[torch.randperm(len(pool), generator=generator)[:6]]
                    support[episode, class_index] = chosen[:5]
                    query[episode, class_index] = chosen[5:]
            partition_runs.append({"support": support, "query": query})
        saved[partition] = {
            "class_ids": class_ids,
            "class_names": [class_names[index] for index in class_ids],
            "runs": partition_runs,
        }

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(saved, path)
    return saved


def batch(data: dict, episode_run: dict, shot: int, device: torch.device):
    support_index = episode_run["support"][:, :, :shot].reshape(len(episode_run["support"]), -1)
    query_index = episode_run["query"].reshape(len(episode_run["query"]), -1)
    labels = torch.arange(3, device=device).repeat_interleave(shot)
    query_labels = torch.arange(3).repeat(len(query_index))
    images = data["image_embeddings"].float()
    reports = data["report_embeddings"].float()
    return (
        images[support_index].to(device),
        reports[support_index].to(device),
        labels,
        images[query_index].to(device),
        query_labels,
    )
