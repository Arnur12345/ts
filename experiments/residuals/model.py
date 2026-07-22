"""Embedding-only PAIR-FSL and its preregistered subtraction controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F


METHODS = (
    "positive_prototype",
    "global_negative_centroid",
    "random_residual",
    "metadata_matched_residual",
    "full_embedding_matched_residual",
    "anatomy_matched_residual",
    "shuffled_anatomy_match",
)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.einsum("eqd,ecd->eqc", F.normalize(left, dim=-1), F.normalize(right, dim=-1))


def _robust_center(values: torch.Tensor, method: str) -> torch.Tensor:
    if method == "mean" or values.shape[2] <= 2:
        return values.mean(2)
    if method != "geometric_median":
        raise ValueError("center must be mean or geometric_median")
    center = values.mean(2)
    for _ in range(8):
        distance = (values - center[:, :, None]).norm(dim=-1).clamp_min(1e-5)
        weight = distance.reciprocal()
        center = (values * weight[..., None]).sum(2) / weight.sum(2, keepdim=True)
    return center


def _project_out(values: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = F.normalize(direction, dim=-1)
    return values - (values * direction.unsqueeze(-2)).sum(-1, keepdim=True) * direction.unsqueeze(-2)


def _weights(anchor: torch.Tensor, candidates: torch.Tensor, mode: str, temperature: float) -> torch.Tensor:
    # anchor [E,C,A,D], candidates [E,C,M,D]
    if mode == "metadata":
        score = -(anchor.unsqueeze(3) - candidates.unsqueeze(2)).square().sum(-1)
    else:
        score = torch.einsum(
            "ecad,ecmd->ecam", F.normalize(anchor, dim=-1), F.normalize(candidates, dim=-1)
        )
    return (score / temperature).softmax(-1)


def _matched(anchor: torch.Tensor, candidates: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.einsum("ecam,ecmd->ecad", weights, candidates)


def pair_fsl_logits(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    method: str,
    positive_metadata: torch.Tensor | None = None,
    negative_metadata: torch.Tensor | None = None,
    query_metadata: torch.Tensor | None = None,
    match_temperature: float = 0.1,
    center: str = "geometric_median",
) -> torch.Tensor:
    """Score every query against every episode class.

    Shapes are positive [E,C,K,D], negative [E,C,M,D], query [E,Q,D].
    Matching never sees query labels or report embeddings.
    """
    if method not in METHODS:
        raise ValueError(f"unknown residual method {method!r}")
    if match_temperature <= 0:
        raise ValueError("match_temperature must be positive")
    episodes, classes, _, width = positive.shape
    if negative.shape[:2] != (episodes, classes) or negative.shape[-1] != width:
        raise ValueError("positive and negative support shapes are incompatible")
    if query.shape[0] != episodes or query.shape[-1] != width:
        raise ValueError("query shape is incompatible with supports")
    if method == "positive_prototype":
        return _cosine(query, _robust_center(positive, center))

    query_by_class = query[:, None].expand(-1, classes, -1, -1)
    if method == "global_negative_centroid":
        control = negative.mean(2)
        disease = _robust_center(positive - control[:, :, None], center)
        residual_query = query_by_class - control[:, :, None]
    elif method == "random_residual":
        positive_control = negative[:, :, torch.arange(positive.shape[2], device=negative.device) % negative.shape[2]]
        query_control = negative[:, :, torch.arange(query.shape[1], device=negative.device) % negative.shape[2]]
        disease = _robust_center(positive - positive_control, center)
        residual_query = query_by_class - query_control
    else:
        if method == "metadata_matched_residual":
            if positive_metadata is None or negative_metadata is None or query_metadata is None:
                raise ValueError("metadata matching requires support and query metadata")
            positive_match = positive_metadata
            negative_match = negative_metadata
            query_match = query_metadata[:, None].expand(-1, classes, -1, -1)
            mode = "metadata"
        else:
            preliminary = positive.mean(2) - negative.mean(2)
            positive_match = positive
            negative_match = negative
            query_match = query_by_class
            mode = "embedding"
            if method in {"anatomy_matched_residual", "shuffled_anatomy_match"}:
                positive_match = _project_out(positive_match, preliminary)
                negative_match = _project_out(negative_match, preliminary)
                query_match = _project_out(query_match, preliminary)
            if method == "shuffled_anatomy_match":
                flat = negative_match.reshape(episodes, classes * negative.shape[2], width)
                flat = flat.roll(1, dims=1)
                negative_match = flat.reshape_as(negative_match)

        positive_weight = _weights(positive_match, negative_match, mode, match_temperature)
        query_weight = _weights(query_match, negative_match, mode, match_temperature)
        positive_control = _matched(positive_match, negative, positive_weight)
        query_control = _matched(query_match, negative, query_weight)
        disease = _robust_center(positive - positive_control, center)
        residual_query = query_by_class - query_control

    return torch.einsum(
        "ecqd,ecd->eqc", F.normalize(residual_query, dim=-1), F.normalize(disease, dim=-1)
    )
