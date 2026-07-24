"""Patient-disjoint target-confounder episodes for the IERA pilot."""

from __future__ import annotations

import hashlib

import torch

from experiments.residuals.data import ResidualDataset
from experiments.residuals.episodes import _draw


PILOT_PAIRS = (
    ("Pneumothorax", "Support Devices"),
    ("Edema", "Cardiomegaly"),
    ("Pleural Effusion", "Atelectasis"),
    ("Pneumonia", "Consolidation"),
)


def split_indices(data: ResidualDataset, partition: str, seed: int = 2026) -> torch.Tensor:
    """Deterministic 70/15/15 patient split over the full MIMIC cohort."""
    if partition not in {"train", "validate", "test"}:
        raise ValueError("partition must be train, validate, or test")
    bounds = {"train": (0, 7000), "validate": (7000, 8500), "test": (8500, 10000)}
    low, high = bounds[partition]
    values = []
    for index, subject in enumerate(data.subject_ids):
        bucket = int.from_bytes(hashlib.sha256(f"iera|{seed}|{subject}".encode()).digest()[:8], "big") % 10_000
        if low <= bucket < high:
            values.append(index)
    if not values:
        raise ValueError(f"manifest contains no rows in deterministic {partition!r} split")
    return torch.tensor(values, dtype=torch.long)


def stratum_pools(data: ResidualDataset, indices: torch.Tensor, target_id: int, confounder_id: int) -> dict[tuple[int, int], torch.Tensor]:
    known = data.known[indices, target_id] & data.known[indices, confounder_id]
    selected = indices[known]
    return {
        (target, confounder): selected[
            data.labels[selected, target_id].eq(bool(target))
            & data.labels[selected, confounder_id].eq(bool(confounder))
        ]
        for target in (0, 1)
        for confounder in (0, 1)
    }


def patient_counts(data: ResidualDataset, pools: dict[tuple[int, int], torch.Tensor]) -> dict[tuple[int, int], int]:
    return {key: len({data.subject_ids[index] for index in pool.tolist()}) for key, pool in pools.items()}


def eligible_directed_pairs(
    data: ResidualDataset,
    indices: torch.Tensor,
    target_ids: list[int],
    min_patients: int,
) -> list[tuple[int, int]]:
    pairs = []
    for target in target_ids:
        for confounder in range(len(data.class_names)):
            if target == confounder:
                continue
            counts = patient_counts(data, stratum_pools(data, indices, target, confounder))
            if min(counts.values()) >= min_patients:
                pairs.append((target, confounder))
    return pairs


def generate_pair_episodes(
    data: ResidualDataset,
    indices: torch.Tensor,
    target_id: int,
    confounder_id: int,
    episode_count: int,
    max_shot: int,
    queries_per_stratum: int,
    seed: int,
    max_attempts: int = 200,
) -> dict:
    pools = stratum_pools(data, indices, target_id, confounder_id)
    counts = patient_counts(data, pools)
    required = max_shot + queries_per_stratum
    if min(counts.values()) < required:
        raise ValueError(
            f"{data.class_names[target_id]} / {data.class_names[confounder_id]} needs {required} "
            f"patients per stratum; found {counts}"
        )
    order = sorted(pools, key=lambda key: counts[key])
    generator = torch.Generator().manual_seed(seed)
    positive_runs, negative_runs, query_runs, target_runs, nuisance_runs = [], [], [], [], []
    for _ in range(episode_count):
        for _attempt in range(max_attempts):
            used: set[str] = set()
            selected: dict[tuple[int, int], list[int]] = {}
            try:
                for key in order:
                    selected[key] = _draw(pools[key], required, used, data.subject_ids, generator)
                positive_runs.append(torch.tensor([selected[(1, env)][:max_shot] for env in (0, 1)]))
                negative_runs.append(torch.tensor([selected[(0, env)][:max_shot] for env in (0, 1)]))
                queries, targets, nuisance = [], [], []
                for target in (0, 1):
                    for env in (0, 1):
                        values = selected[(target, env)][max_shot:]
                        queries.extend(values)
                        targets.extend([target] * len(values))
                        nuisance.extend([env] * len(values))
                query_runs.append(torch.tensor(queries))
                target_runs.append(torch.tensor(targets, dtype=torch.float32))
                nuisance_runs.append(torch.tensor(nuisance, dtype=torch.long))
                break
            except RuntimeError:
                continue
        else:
            raise ValueError("could not create a patient-disjoint four-stratum episode")
    return {
        "target_id": target_id,
        "confounder_id": confounder_id,
        "positive": torch.stack(positive_runs),
        "negative": torch.stack(negative_runs),
        "query": torch.stack(query_runs),
        "targets": torch.stack(target_runs),
        "nuisance": torch.stack(nuisance_runs),
        "patient_counts": {f"c{key[0]}d{key[1]}": value for key, value in counts.items()},
    }


def validate_pair_episodes(episodes: dict, data: ResidualDataset) -> None:
    target_id, confounder_id = episodes["target_id"], episodes["confounder_id"]
    for episode in range(len(episodes["positive"])):
        indices = torch.cat(
            (episodes["positive"][episode].flatten(), episodes["negative"][episode].flatten(), episodes["query"][episode])
        ).tolist()
        subjects = [data.subject_ids[index] for index in indices]
        if len(subjects) != len(set(subjects)):
            raise ValueError("patient overlap inside IERA episode")
        for env in (0, 1):
            positive = episodes["positive"][episode, env]
            negative = episodes["negative"][episode, env]
            if not data.labels[positive, target_id].all() or data.labels[negative, target_id].any():
                raise ValueError("support target status is incorrect")
            if not data.labels[positive, confounder_id].eq(bool(env)).all():
                raise ValueError("positive support environment is incorrect")
            if not data.labels[negative, confounder_id].eq(bool(env)).all():
                raise ValueError("negative support environment is incorrect")
