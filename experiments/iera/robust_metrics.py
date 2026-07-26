"""AUROC-paired support-swap metrics with a fixed reference scale."""

from __future__ import annotations

import torch

from experiments.residuals.metrics import (
    _average_precision,
    _auc,
    _ece,
)


def normalized_sms(
    panel_zero: torch.Tensor,
    panel_one: torch.Tensor,
    reference_zero: torch.Tensor,
    reference_one: torch.Tensor,
) -> torch.Tensor:
    scale = torch.cat((reference_zero, reference_one)).std().clamp_min(1e-6)
    return (panel_one - panel_zero).abs().mean() / scale


def ranking_disagreement(
    panel_zero: torch.Tensor, panel_one: torch.Tensor
) -> float:
    """Fraction of query pairs whose ordering changes under support swap."""
    zero_difference = panel_zero[:, None] - panel_zero[None, :]
    one_difference = panel_one[:, None] - panel_one[None, :]
    upper = torch.triu(
        torch.ones_like(zero_difference, dtype=torch.bool), diagonal=1
    )
    non_tied = upper & zero_difference.ne(0) & one_difference.ne(0)
    if not non_tied.any():
        return 0.0
    return float(
        zero_difference[non_tied]
        .sign()
        .ne(one_difference[non_tied].sign())
        .float()
        .mean()
    )


def evaluate(
    logits: torch.Tensor,
    panel_zero: torch.Tensor,
    panel_one: torch.Tensor,
    reference_zero: torch.Tensor,
    reference_one: torch.Tensor,
    targets: torch.Tensor,
    nuisance: torch.Tensor,
    temperature: float,
    threshold: float,
) -> dict[str, float]:
    target = targets.bool()
    probability = torch.sigmoid(logits / temperature).clamp(1e-7, 1 - 1e-7)
    prediction = probability.ge(threshold)
    tp = (target & prediction).sum()
    fp = (~target & prediction).sum()
    fn = (target & ~prediction).sum()
    panel_zero_prediction = torch.sigmoid(
        panel_zero / temperature
    ).ge(threshold)
    panel_one_prediction = torch.sigmoid(
        panel_one / temperature
    ).ge(threshold)
    result = {
        "auroc": _auc(target, probability),
        "auprc": _average_precision(target, probability),
        "f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
        "brier": float((probability - targets.float()).square().mean()),
        "ece": _ece(probability, target),
        "sms_raw_logit": float((panel_one - panel_zero).abs().mean()),
        "sms_fixed_reference": float(
            normalized_sms(
                panel_zero, panel_one, reference_zero, reference_one
            )
        ),
        "ranking_disagreement": ranking_disagreement(panel_zero, panel_one),
        "support_swap_flip_rate": float(
            panel_zero_prediction.ne(panel_one_prediction).float().mean()
        ),
    }
    nuisance_aurocs = []
    for value in (0, 1):
        mask = nuisance.eq(value)
        nuisance_aurocs.append(_auc(target[mask], probability[mask]))
        result[f"d{value}_auroc"] = nuisance_aurocs[-1]
    result["worst_nuisance_auroc"] = min(nuisance_aurocs)
    return result
