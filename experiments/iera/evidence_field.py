"""Prototype-free, image-balanced support-to-query evidence fields."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn.functional as F


SupportAdapter = Callable[[torch.Tensor], torch.Tensor] | torch.nn.Module


def _validate(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    tau_support: float,
    tau_query: float,
    query_chunk_size: int,
) -> None:
    if positive.ndim != 5 or negative.ndim != 5:
        raise ValueError("support tensors must be [B,environments,shots,patches,width]")
    if query.ndim != 4:
        raise ValueError("query must be [B,queries,patches,width]")
    if positive.shape[0] != query.shape[0] or negative.shape[0] != query.shape[0]:
        raise ValueError("support and query batch sizes differ")
    if positive.shape[-1] != query.shape[-1] or negative.shape[-1] != query.shape[-1]:
        raise ValueError("support and query widths differ")
    if positive_mask.shape != positive.shape[:3]:
        raise ValueError("positive mask shape does not match positive supports")
    if negative_mask.shape != negative.shape[:3]:
        raise ValueError("negative mask shape does not match negative supports")
    if not positive_mask.bool().flatten(1).any(dim=1).all():
        raise ValueError("every batch item needs a valid positive support image")
    if not negative_mask.bool().flatten(1).any(dim=1).all():
        raise ValueError("every batch item needs a valid negative support image")
    if tau_support <= 0 or tau_query <= 0:
        raise ValueError("evidence-field temperatures must be positive")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")


def adapt_support_tokens(
    tokens: torch.Tensor,
    adapter: SupportAdapter | None,
) -> torch.Tensor:
    """Normalize and optionally adapt support tokens; query tokens never enter."""
    tokens = F.normalize(tokens.float(), dim=-1)
    if adapter is None:
        return tokens
    if hasattr(adapter, "support_down") and hasattr(adapter, "support_up"):
        residual = adapter.support_up(F.gelu(adapter.support_down(tokens)))
        adapted = tokens + residual
    else:
        adapted = adapter(tokens)
    if adapted.shape != tokens.shape:
        raise ValueError("support adapter changed the token shape")
    return F.normalize(adapted.float(), dim=-1)


def image_match(
    query: torch.Tensor,
    support: torch.Tensor,
    tau: float,
    query_chunk_size: int = 1,
) -> torch.Tensor:
    """Log-mean-exp query-patch match to each support image independently.

    The inputs must already be normalized. The result is
    ``[B,queries,query_patches,environments,shots]``.
    """
    if tau <= 0:
        raise ValueError("support temperature must be positive")
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if query.ndim != 4 or support.ndim != 5:
        raise ValueError("query/support ranks must be four/five")
    if query.shape[0] != support.shape[0] or query.shape[-1] != support.shape[-1]:
        raise ValueError("query and support are incompatible")
    support_patch_count = support.shape[-2]
    normalizer = math.log(support_patch_count)
    matches = []
    for start in range(0, query.shape[1], query_chunk_size):
        chunk = query[:, start : start + query_chunk_size]
        similarity = torch.einsum(
            "bqtd,bikpd->bqtikp", chunk, support
        )
        matches.append(
            tau
            * (
                torch.logsumexp(similarity / tau, dim=-1)
                - normalizer
            )
        )
    return torch.cat(matches, dim=1)


def _density_grid(
    query: torch.Tensor,
    support: torch.Tensor,
    mask: torch.Tensor,
    temperatures: tuple[float, ...],
    pooling_modes: tuple[str, ...],
    query_chunk_size: int,
) -> dict[tuple[str, float], torch.Tensor]:
    """Evaluate a temperature/pooling grid while computing similarity once."""
    destinations = {
        (mode, temperature): []
        for mode in pooling_modes
        for temperature in temperatures
    }
    valid = mask.bool()[:, None, None, :, :, None]
    valid_images = mask.bool().flatten(1).sum(dim=1)
    support_patches = support.shape[-2]
    for start in range(0, query.shape[1], query_chunk_size):
        chunk = query[:, start : start + query_chunk_size]
        similarity = torch.einsum(
            "bqtd,bikpd->bqtikp", chunk, support
        )
        for temperature in temperatures:
            scaled = similarity / temperature
            if "image_balanced" in pooling_modes:
                matches = temperature * (
                    torch.logsumexp(scaled, dim=-1)
                    - math.log(support_patches)
                )
                weights = mask.bool()[:, None, None].to(matches.dtype)
                density = (matches * weights).sum(
                    dim=(-1, -2)
                ) / valid_images.to(matches.dtype)[:, None, None]
                destinations[("image_balanced", temperature)].append(density)
            if "dense" in pooling_modes:
                flattened = scaled.masked_fill(~valid, -torch.inf).flatten(-3)
                density = temperature * (
                    torch.logsumexp(flattened, dim=-1)
                    - (valid_images * support_patches)
                    .to(similarity.dtype)
                    .log()[:, None, None]
                )
                destinations[("dense", temperature)].append(density)
    return {
        key: torch.cat(chunks, dim=1)
        for key, chunks in destinations.items()
    }


def evidence_field_grid(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    tau_supports: tuple[float, ...] | list[float],
    tau_queries: tuple[float, ...] | list[float],
    *,
    adapter: SupportAdapter | None = None,
    pooling_modes: tuple[str, ...] = ("image_balanced",),
    query_chunk_size: int = 1,
) -> dict[tuple[str, float, float], tuple[torch.Tensor, torch.Tensor]]:
    """Score a fixed-temperature grid with shared patch similarities."""
    tau_supports = tuple(float(value) for value in tau_supports)
    tau_queries = tuple(float(value) for value in tau_queries)
    if not tau_supports or not tau_queries:
        raise ValueError("temperature grids cannot be empty")
    if not pooling_modes or any(
        mode not in {"image_balanced", "dense"} for mode in pooling_modes
    ):
        raise ValueError("pooling_modes must contain image_balanced and/or dense")
    _validate(
        positive,
        negative,
        query,
        positive_mask,
        negative_mask,
        min(tau_supports),
        min(tau_queries),
        query_chunk_size,
    )
    positive = adapt_support_tokens(positive, adapter)
    negative = adapt_support_tokens(negative, adapter)
    query = F.normalize(query.float(), dim=-1)
    positive_density = _density_grid(
        query,
        positive,
        positive_mask,
        tau_supports,
        pooling_modes,
        query_chunk_size,
    )
    negative_density = _density_grid(
        query,
        negative,
        negative_mask,
        tau_supports,
        pooling_modes,
        query_chunk_size,
    )
    result = {}
    for mode in pooling_modes:
        for tau_support in tau_supports:
            evidence_map = (
                positive_density[(mode, tau_support)]
                - negative_density[(mode, tau_support)]
            )
            for tau_query in tau_queries:
                logits = tau_query * (
                    torch.logsumexp(evidence_map / tau_query, dim=-1)
                    - math.log(evidence_map.shape[-1])
                )
                result[(mode, tau_support, tau_query)] = (
                    logits,
                    evidence_map,
                )
    return result


def evidence_field_score(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    tau_support: float,
    tau_query: float,
    *,
    adapter: SupportAdapter | None = None,
    image_balanced: bool = True,
    query_chunk_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return prototype-free binary logits and per-query-patch evidence.

    Support tokens may pass through the existing residual adapter. Query tokens
    are only L2-normalized and are therefore frozen by construction.
    """
    mode = "image_balanced" if image_balanced else "dense"
    return evidence_field_grid(
        positive,
        negative,
        query,
        positive_mask,
        negative_mask,
        (tau_support,),
        (tau_query,),
        adapter=adapter,
        pooling_modes=(mode,),
        query_chunk_size=query_chunk_size,
    )[(mode, float(tau_support), float(tau_query))]
