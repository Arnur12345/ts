"""Frozen native-token matching primitives for Stable Region Witnesses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def relational_descriptor(tokens: torch.Tensor) -> torch.Tensor:
    """Apply fixed A0=[I,I]/sqrt(2) to token and 3x3 local residual."""
    side = math.isqrt(tokens.shape[-2])
    if side * side != tokens.shape[-2]:
        raise ValueError("relational descriptors require a square token grid")
    normalized = F.normalize(tokens.float(), dim=-1)
    leading = normalized.shape[:-2]
    width = normalized.shape[-1]
    spatial = normalized.reshape(-1, side, side, width).permute(0, 3, 1, 2)
    neighbourhood = F.avg_pool2d(
        spatial,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    )
    residual = spatial - neighbourhood
    projected = (spatial + residual) / math.sqrt(2)
    return F.normalize(
        projected.permute(0, 2, 3, 1).reshape(
            *leading, side * side, width
        ),
        dim=-1,
    )


def compact_support_images(
    tokens: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Remove padded environment slots, preserving one row per support image."""
    if tokens.ndim != 5 or mask.shape != tokens.shape[:3]:
        raise ValueError("support tokens/mask have incompatible shapes")
    counts = mask.bool().flatten(1).sum(dim=1)
    if not counts.eq(counts[0]).all():
        raise ValueError("support-image count must be constant in a batch")
    flattened = tokens.flatten(1, 2)
    valid = mask.bool().flatten(1)
    return torch.stack(
        [current[current_mask] for current, current_mask in zip(flattened, valid)]
    )


def witness_confidence(
    same_class: torch.Tensor,
    opposite_class: torch.Tensor,
    token_chunk_size: int = 128,
) -> torch.Tensor:
    """Cross-image same-class reproducibility minus opposite-class similarity."""
    if same_class.ndim != 4 or opposite_class.ndim != 4:
        raise ValueError("witness supports must be [B,images,patches,width]")
    if same_class.shape[0] != opposite_class.shape[0]:
        raise ValueError("witness support batches differ")
    if same_class.shape[1] < 2:
        raise ValueError("witness certification needs two same-class images")
    if token_chunk_size <= 0:
        raise ValueError("token_chunk_size must be positive")
    same_class = F.normalize(same_class.float(), dim=-1)
    opposite_class = F.normalize(opposite_class.float(), dim=-1)
    image_confidences = []
    for image_index in range(same_class.shape[1]):
        token_confidences = []
        image = same_class[:, image_index]
        other_indices = [
            index
            for index in range(same_class.shape[1])
            if index != image_index
        ]
        other = same_class[:, other_indices]
        for start in range(0, image.shape[1], token_chunk_size):
            tokens = image[:, start : start + token_chunk_size]
            same_similarity = torch.einsum(
                "bcd,bipd->bcip", tokens, other
            ).amax(dim=-1)
            ordered = same_similarity.sort(dim=-1).values
            middle = ordered.shape[-1] // 2
            same_reproducibility = (
                ordered[..., middle]
                if ordered.shape[-1] % 2
                else (ordered[..., middle - 1] + ordered[..., middle]) / 2
            )
            opposite_similarity = torch.einsum(
                "bcd,bipd->bcip", tokens, opposite_class
            ).amax(dim=(-1, -2))
            token_confidences.append(
                same_reproducibility - opposite_similarity
            )
        image_confidences.append(torch.cat(token_confidences, dim=1))
    return torch.stack(image_confidences, dim=1)


def _witness_bank(
    support: torch.Tensor,
    confidence: torch.Tensor,
    maximum_fraction: float,
) -> torch.Tensor:
    flattened = support.flatten(1, 2)
    flattened_confidence = confidence.flatten(1)
    count = max(1, math.ceil(maximum_fraction * flattened.shape[1]))
    indices = flattened_confidence.topk(count, dim=1).indices
    return torch.gather(
        flattened,
        1,
        indices[..., None].expand(-1, -1, flattened.shape[-1]),
    )


def _maximum_similarity(
    query: torch.Tensor,
    bank: torch.Tensor,
    counts: dict[float, int],
    query_chunk_size: int,
) -> dict[float, torch.Tensor]:
    destinations = {fraction: [] for fraction in counts}
    for start in range(0, query.shape[1], query_chunk_size):
        chunk = query[:, start : start + query_chunk_size]
        similarity = torch.einsum("bqtd,bkd->bqtk", chunk, bank)
        for fraction, count in counts.items():
            destinations[fraction].append(
                similarity[..., :count].amax(dim=-1)
            )
    return {
        fraction: torch.cat(chunks, dim=1)
        for fraction, chunks in destinations.items()
    }


def aggregate_top_evidence(
    evidence_map: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    if not 0 < fraction <= 1:
        raise ValueError("witness fraction must be in (0,1]")
    count = max(1, math.ceil(fraction * evidence_map.shape[-1]))
    return evidence_map.topk(count, dim=-1).values.mean(dim=-1)


def certified_witness_scores(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    fractions: list[float] | tuple[float, ...],
    *,
    token_chunk_size: int = 128,
    query_chunk_size: int = 1,
) -> dict[float, tuple[torch.Tensor, torch.Tensor]]:
    """Return top-evidence logits and fields for every witness fraction."""
    fractions = tuple(sorted(set(float(value) for value in fractions)))
    if not fractions or min(fractions) <= 0 or max(fractions) > 1:
        raise ValueError("witness fractions must be in (0,1]")
    positive = F.normalize(positive.float(), dim=-1)
    negative = F.normalize(negative.float(), dim=-1)
    query = F.normalize(query.float(), dim=-1)
    positive_confidence = witness_confidence(
        positive, negative, token_chunk_size
    )
    negative_confidence = witness_confidence(
        negative, positive, token_chunk_size
    )
    maximum_fraction = max(fractions)
    positive_bank = _witness_bank(
        positive, positive_confidence, maximum_fraction
    )
    negative_bank = _witness_bank(
        negative, negative_confidence, maximum_fraction
    )
    total_tokens = positive.shape[1] * positive.shape[2]
    counts = {
        fraction: max(1, math.ceil(fraction * total_tokens))
        for fraction in fractions
    }
    positive_match = _maximum_similarity(
        query, positive_bank, counts, query_chunk_size
    )
    negative_match = _maximum_similarity(
        query, negative_bank, counts, query_chunk_size
    )
    result = {}
    for fraction in fractions:
        field = positive_match[fraction] - negative_match[fraction]
        result[fraction] = (
            aggregate_top_evidence(field, fraction),
            field,
        )
    return result


def dn4_hard_knn_score(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    neighbours: int = 3,
    query_chunk_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """DN4 image-to-class score with hard top-k local descriptor matching."""
    if neighbours <= 0:
        raise ValueError("DN4 neighbours must be positive")
    positive = F.normalize(positive.float().flatten(1, 2), dim=-1)
    negative = F.normalize(negative.float().flatten(1, 2), dim=-1)
    query = F.normalize(query.float(), dim=-1)
    if neighbours > min(positive.shape[1], negative.shape[1]):
        raise ValueError("DN4 neighbours exceed the support descriptor bank")
    fields = []
    for start in range(0, query.shape[1], query_chunk_size):
        chunk = query[:, start : start + query_chunk_size]
        positive_similarity = torch.einsum(
            "bqtd,bkd->bqtk", chunk, positive
        ).topk(neighbours, dim=-1).values.mean(dim=-1)
        negative_similarity = torch.einsum(
            "bqtd,bkd->bqtk", chunk, negative
        ).topk(neighbours, dim=-1).values.mean(dim=-1)
        fields.append(positive_similarity - negative_similarity)
    evidence_map = torch.cat(fields, dim=1)
    return evidence_map.mean(dim=-1), evidence_map


def border_maximum(evidence_map: torch.Tensor) -> torch.Tensor:
    """One Boolean per query indicating whether its evidence maximum is border."""
    side = math.isqrt(evidence_map.shape[-1])
    if side * side != evidence_map.shape[-1]:
        raise ValueError("border metric requires a square evidence map")
    maximum = evidence_map.argmax(dim=-1)
    row = maximum.div(side, rounding_mode="floor")
    column = maximum.remainder(side)
    return row.eq(0) | row.eq(side - 1) | column.eq(0) | column.eq(side - 1)
