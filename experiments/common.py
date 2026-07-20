from __future__ import annotations

import torch
import torch.nn.functional as F


def prototypes(features: torch.Tensor, labels: torch.Tensor, ways: int) -> torch.Tensor:
    """Mean normalized prototype per class; features are [episodes, samples, d]."""
    one_hot = F.one_hot(labels, ways).to(features.dtype)
    means = torch.einsum("bnc,bnd->bcd", one_hot.expand(len(features), -1, -1), features)
    counts = one_hot.sum(0).clamp_min(1)[None, :, None]
    return F.normalize(means / counts, dim=-1)


def text_prototypes(
    support_reports: torch.Tensor,
    labels: torch.Tensor,
    ways: int,
    shuffled: bool,
) -> torch.Tensor:
    text = prototypes(support_reports, labels, ways)
    return text.roll(1, dims=1) if shuffled else text


def cosine_logits(query: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bqd,bcd->bqc", F.normalize(query, dim=-1), F.normalize(class_weights, dim=-1))


def one_hot(labels: torch.Tensor, ways: int, batch: int) -> torch.Tensor:
    return F.one_hot(labels, ways).float().expand(batch, -1, -1)
