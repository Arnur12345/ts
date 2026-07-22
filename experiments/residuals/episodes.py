"""Deterministic, patient-disjoint episodes for both label regimes."""

from __future__ import annotations

from collections import defaultdict

import torch

from .data import ResidualDataset


def _draw(
    pool: torch.Tensor,
    count: int,
    used_subjects: set[str],
    subjects: list[str],
    generator: torch.Generator,
) -> list[int]:
    by_subject: dict[str, list[int]] = defaultdict(list)
    for index in pool.tolist():
        if subjects[index] not in used_subjects:
            by_subject[subjects[index]].append(index)
    available = sorted(by_subject)
    if len(available) < count:
        raise RuntimeError(f"need {count} unused patients, found {len(available)}")
    order = torch.randperm(len(available), generator=generator)[:count].tolist()
    chosen = []
    for position in order:
        subject = available[position]
        candidates = by_subject[subject]
        offset = int(torch.randint(len(candidates), (1,), generator=generator))
        chosen.append(candidates[offset])
        used_subjects.add(subject)
    return chosen


def _pools(data: ResidualDataset, indices: torch.Tensor, class_ids: list[int], single_label: bool):
    labels = data.labels[indices]
    known = data.known[indices]
    if single_label:
        eligible = known.all(1) & labels.sum(1).eq(1)
    else:
        eligible = torch.ones(len(indices), dtype=torch.bool)
    result = []
    for class_id in class_ids:
        certain = known[:, class_id] & eligible
        result.append(
            (
                indices[certain & labels[:, class_id]],
                indices[certain & ~labels[:, class_id]],
            )
        )
    return result


def generate_episodes(
    data: ResidualDataset,
    partition_indices: torch.Tensor,
    class_ids: list[int],
    regime: str,
    episode_count: int,
    positive_shots: int,
    negative_shots: int,
    queries_per_class: int,
    seed: int,
    max_attempts: int = 200,
) -> dict[str, torch.Tensor | str | int | list[int]]:
    """Generate maximum-shot episodes; callers take nested prefixes."""
    if regime not in {"single_label", "multi_label"}:
        raise ValueError("regime must be single_label or multi_label")
    if min(episode_count, positive_shots, negative_shots, queries_per_class) <= 0:
        raise ValueError("episode sizes must be positive")
    pools = _pools(data, partition_indices, class_ids, regime == "single_label")
    for class_id, (positive, negative) in zip(class_ids, pools):
        if not len(positive) or not len(negative):
            raise ValueError(f"class {data.class_names[class_id]!r} lacks positive or negative samples")

    generator = torch.Generator().manual_seed(seed)
    positive_runs, negative_runs, query_runs, target_runs = [], [], [], []
    for _ in range(episode_count):
        last_error: Exception | None = None
        for _attempt in range(max_attempts):
            used: set[str] = set()
            positive_support, negative_support = [], []
            queries: list[int] = []
            try:
                # Scarce positives are allocated before the usually abundant controls.
                for positive, _ in pools:
                    positive_support.append(_draw(positive, positive_shots, used, data.subject_ids, generator))
                for _, negative in pools:
                    negative_support.append(_draw(negative, negative_shots, used, data.subject_ids, generator))
                if regime == "single_label":
                    for positive, _ in pools:
                        queries.extend(_draw(positive, queries_per_class, used, data.subject_ids, generator))
                    targets = torch.arange(len(class_ids)).repeat_interleave(queries_per_class)
                else:
                    for positive, negative in pools:
                        queries.extend(_draw(positive, queries_per_class, used, data.subject_ids, generator))
                        queries.extend(_draw(negative, queries_per_class, used, data.subject_ids, generator))
                    query_known = data.known[queries][:, class_ids]
                    query_labels = data.labels[queries][:, class_ids].float()
                    targets = torch.where(query_known, query_labels, -torch.ones_like(query_labels))
                positive_runs.append(torch.tensor(positive_support, dtype=torch.long))
                negative_runs.append(torch.tensor(negative_support, dtype=torch.long))
                query_runs.append(torch.tensor(queries, dtype=torch.long))
                target_runs.append(targets)
                break
            except RuntimeError as error:
                last_error = error
        else:
            raise ValueError(f"could not create a patient-disjoint {regime} episode: {last_error}")

    return {
        "regime": regime,
        "seed": seed,
        "class_ids": class_ids,
        "positive": torch.stack(positive_runs),
        "negative": torch.stack(negative_runs),
        "query": torch.stack(query_runs),
        "targets": torch.stack(target_runs),
    }


def validate_episodes(episodes: dict, data: ResidualDataset) -> None:
    class_ids = episodes["class_ids"]
    positive = episodes["positive"]
    negative = episodes["negative"]
    query = episodes["query"]
    for episode in range(len(positive)):
        all_indices = torch.cat((positive[episode].flatten(), negative[episode].flatten(), query[episode])).tolist()
        subjects = [data.subject_ids[index] for index in all_indices]
        if len(subjects) != len(set(subjects)):
            raise ValueError("support/control/query patient overlap")
        for position, class_id in enumerate(class_ids):
            if not data.labels[positive[episode, position], class_id].all():
                raise ValueError("positive support has a negative target")
            if data.labels[negative[episode, position], class_id].any():
                raise ValueError("negative support has a positive target")


def batch(
    data: ResidualDataset,
    episodes: dict,
    positive_shot: int,
    negative_shot: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    positive = episodes["positive"][:, :, :positive_shot]
    negative = episodes["negative"][:, :, :negative_shot]
    query = episodes["query"]
    return {
        "positive": data.images[positive].to(device),
        "negative": data.images[negative].to(device),
        "query": data.images[query].to(device),
        "positive_metadata": data.metadata[positive].to(device),
        "negative_metadata": data.metadata[negative].to(device),
        "query_metadata": data.metadata[query].to(device),
        "targets": episodes["targets"].to(device),
        "query_indices": query,
    }
