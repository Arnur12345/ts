"""Identical candidate panels with random and nuisance-balanced selections."""

from __future__ import annotations

import torch


def environment_choices(
    episode_count: int,
    support_count: int,
    nuisance_one_probability: float,
    seed: int,
) -> torch.Tensor:
    if not 0 <= nuisance_one_probability <= 1:
        raise ValueError("nuisance probability must be in [0,1]")
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(
        episode_count, support_count, generator=generator
    ).lt(nuisance_one_probability).long()


def balanced_choices(
    episode_count: int, support_count: int, seed: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(2, (episode_count, 1), generator=generator)
    offsets = torch.arange(support_count)[None]
    return (starts + offsets).remainder(2).long()


def select_supports(
    panels: torch.Tensor,
    choices: torch.Tensor,
    support_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select and pad per-environment supports.

    panels are [B,2,C,P,D], choices are [B,T]. The result stays [B,2,T,P,D]
    with a [B,2,T] validity mask, allowing naturally imbalanced episodes
    without silently reweighting environments.
    """
    if panels.ndim != 5 or panels.shape[1] != 2:
        raise ValueError("panels must be [B,2,C,P,D]")
    if choices.shape[0] != panels.shape[0] or support_count > choices.shape[1]:
        raise ValueError("choices are incompatible with panels")
    selected = panels.new_zeros(
        panels.shape[0], 2, support_count, panels.shape[3], panels.shape[4]
    )
    mask = torch.zeros(
        panels.shape[0], 2, support_count,
        dtype=torch.bool, device=panels.device,
    )
    counts = torch.zeros(
        panels.shape[0], 2, dtype=torch.long, device=panels.device
    )
    for position in range(support_count):
        environment = choices[:, position].to(panels.device)
        for batch in range(panels.shape[0]):
            env = int(environment[batch])
            source = int(counts[batch, env])
            if source >= panels.shape[2]:
                raise ValueError(
                    "candidate panel is too small for sampled nuisance allocation"
                )
            selected[batch, env, source] = panels[batch, env, source]
            mask[batch, env, source] = True
            counts[batch, env] += 1
    return selected, mask


def nuisance_probability(patient_counts: dict, target: int) -> float:
    zero = float(patient_counts[f"c{target}d0"])
    one = float(patient_counts[f"c{target}d1"])
    if zero + one == 0:
        raise ValueError("empty target stratum")
    return one / (zero + one)
