"""Fixed-episode scoring and metrics for TRACE kill tests."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict

import torch
import torch.nn.functional as F

from experiments.iera.robust_metrics import normalized_sms
from experiments.residuals.metrics import _auc, _average_precision

from .core import apply_shrinkage_precision, select_support_indices


def _gather(features, indices: torch.Tensor, device: torch.device) -> torch.Tensor:
    import numpy as np

    values = np.array(
        features[indices.detach().cpu().numpy()],
        dtype=np.float32,
        copy=True,
    )
    return torch.from_numpy(values).to(device)


def _prototype_direction(
    positive: torch.Tensor, negative: torch.Tensor
) -> torch.Tensor:
    positive_prototype = F.normalize(positive.mean(dim=-2), dim=-1)
    negative_prototype = F.normalize(negative.mean(dim=-2), dim=-1)
    return positive_prototype - negative_prototype


def _lda_direction(
    positive: torch.Tensor,
    negative: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    difference = positive.mean(dim=-2) - negative.mean(dim=-2)
    return apply_shrinkage_precision(
        difference, eigenvalues, eigenvectors, ridge
    )


def score_episode_bank(
    features,
    episodes: dict,
    shot: int,
    device: torch.device,
    eigenvalues: torch.Tensor | None = None,
    eigenvectors: torch.Tensor | None = None,
    ridge: float | None = None,
    atom: torch.Tensor | None = None,
    atom_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Score exactly the saved random supports and support-swap panels."""

    if shot <= 0:
        raise ValueError("shot must be positive")
    use_lda = eigenvalues is not None or eigenvectors is not None or ridge is not None
    if use_lda and (
        eigenvalues is None or eigenvectors is None or ridge is None
    ):
        raise ValueError("LDA requires eigenvalues, eigenvectors, and ridge")
    # K means K supports per target class in the paper. Candidate panels
    # remain stratified by device, but the sampled mixture totals exactly K.
    support_count = shot
    positive_index = select_support_indices(
        episodes["positive"],
        episodes["random_positive_env"],
        support_count,
    )
    negative_index = select_support_indices(
        episodes["negative"],
        episodes["random_negative_env"],
        support_count,
    )
    positive = _gather(features, positive_index, device)
    negative = _gather(features, negative_index, device)
    query = _gather(features, episodes["query"], device)
    panel_positive = _gather(
        features, episodes["positive"][:, :, :shot], device
    )
    panel_negative = _gather(
        features, episodes["negative"][:, :, :shot], device
    )
    reference_direction = _prototype_direction(
        panel_positive, panel_negative
    )
    reference_zero = torch.einsum(
        "eqd,ed->eq", query, reference_direction[:, 0]
    )
    reference_one = torch.einsum(
        "eqd,ed->eq", query, reference_direction[:, 1]
    )
    if use_lda:
        eigenvalues = eigenvalues.to(device)
        eigenvectors = eigenvectors.to(device)
        direction = _lda_direction(
            positive, negative, eigenvalues, eigenvectors, float(ridge)
        )
        panel_direction = _lda_direction(
            panel_positive,
            panel_negative,
            eigenvalues,
            eigenvectors,
            float(ridge),
        )
    else:
        direction = _prototype_direction(positive, negative)
        panel_direction = reference_direction
    if atom is not None and atom_weight:
        fixed = F.normalize(atom.to(device).float(), dim=-1)
        direction = F.normalize(
            direction + atom_weight * fixed, dim=-1
        )
        panel_direction = F.normalize(
            panel_direction + atom_weight * fixed[None, None], dim=-1
        )
    logits = torch.einsum("eqd,ed->eq", query, direction)
    panel_zero = torch.einsum(
        "eqd,ed->eq", query, panel_direction[:, 0]
    )
    panel_one = torch.einsum(
        "eqd,ed->eq", query, panel_direction[:, 1]
    )
    return {
        "logits": logits.detach().cpu().flatten(),
        "panel_zero": panel_zero.detach().cpu().flatten(),
        "panel_one": panel_one.detach().cpu().flatten(),
        "reference_zero": reference_zero.detach().cpu().flatten(),
        "reference_one": reference_one.detach().cpu().flatten(),
    }


def classification_metrics(
    scores: dict[str, torch.Tensor],
    targets: torch.Tensor,
    nuisance: torch.Tensor,
) -> dict[str, float]:
    target = targets.flatten().bool().cpu()
    device = nuisance.flatten().long().cpu()
    logits = scores["logits"].flatten().float().cpu()
    result = {
        "auroc": _auc(target, logits),
        "auprc": _average_precision(target, logits),
        "sms_fixed_reference": float(
            normalized_sms(
                scores["panel_zero"],
                scores["panel_one"],
                scores["reference_zero"],
                scores["reference_one"],
            )
        ),
    }
    nuisance_aurocs = []
    for value in (0, 1):
        selected = device.eq(value)
        observed = _auc(target[selected], logits[selected])
        nuisance_aurocs.append(observed)
        result[f"device_{value}_auroc"] = observed
    result["worst_device_auroc"] = min(nuisance_aurocs)
    return result


def summarize_metric_rows(
    rows: list[dict],
    group_keys: tuple[str, ...],
) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_keys)].append(
            float(row["value"])
        )
    result = []
    for key, values in sorted(groups.items()):
        mean = statistics.mean(values)
        standard_deviation = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
        half_width = 1.96 * standard_deviation / math.sqrt(len(values))
        result.append(
            {
                **dict(zip(group_keys, key)),
                "n": len(values),
                "mean": mean,
                "std": standard_deviation,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }
        )
    return result
