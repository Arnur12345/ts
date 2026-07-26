"""Frozen global/local binary evidence heads for controlled diagnostics."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def support_adapter(tokens: torch.Tensor, model) -> torch.Tensor:
    """Apply an already-trained support adapter without changing the query."""
    tokens = F.normalize(tokens.float(), dim=-1)
    residual = model.support_up(F.gelu(model.support_down(tokens)))
    return F.normalize(tokens + residual, dim=-1)


def global_prototype(
    tokens: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if mask is None:
        return F.normalize(tokens.float().mean(dim=(1, 2, 3)), dim=-1)
    weights = mask[..., None, None].to(tokens.dtype)
    total = (tokens * weights).sum(dim=(1, 2, 3))
    denominator = (
        mask.sum(dim=(1, 2)).clamp_min(1).to(tokens.dtype)
        * tokens.shape[3]
    )
    return F.normalize(total / denominator[:, None], dim=-1)


def selected_local_prototypes(
    positive: torch.Tensor,
    negative: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    temperature: float,
    uniform_weights: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select positive patches toward d and negative patches toward -d."""
    if temperature <= 0:
        raise ValueError("patch-selection temperature must be positive")
    positive_global = global_prototype(positive, positive_mask)
    negative_global = global_prototype(negative, negative_mask)
    direction = F.normalize(positive_global - negative_global, dim=-1)

    def select(
        tokens: torch.Tensor,
        mask: torch.Tensor,
        signed_direction: torch.Tensor,
    ) -> torch.Tensor:
        batch, environments, shots, patches, width = tokens.shape
        flattened = tokens.reshape(batch, environments * shots * patches, width)
        valid = mask[..., None].expand(
            -1, -1, -1, patches
        ).reshape(batch, -1)
        if uniform_weights:
            logits = torch.zeros_like(valid, dtype=tokens.dtype)
        else:
            logits = torch.einsum(
                "bnd,bd->bn", flattened, signed_direction
            ) / temperature
        logits = logits.masked_fill(~valid, -torch.inf)
        weights = logits.softmax(dim=-1)
        return F.normalize(
            torch.einsum("bn,bnd->bd", weights, flattened), dim=-1
        )

    return (
        select(positive, positive_mask, direction),
        select(negative, negative_mask, -direction),
    )


def global_binary_score(
    query: torch.Tensor,
    positive_prototype: torch.Tensor,
    negative_prototype: torch.Tensor,
) -> torch.Tensor:
    query_mean = F.normalize(query.float().mean(dim=2), dim=-1)
    return (
        torch.einsum("bqd,bd->bq", query_mean, positive_prototype)
        - torch.einsum("bqd,bd->bq", query_mean, negative_prototype)
    )


def local_binary_score(
    query: torch.Tensor,
    positive_prototype: torch.Tensor,
    negative_prototype: torch.Tensor,
    query_temperature: float = 0.1,
) -> torch.Tensor:
    if query_temperature <= 0:
        raise ValueError("query temperature must be positive")

    def score(prototype: torch.Tensor) -> torch.Tensor:
        similarity = torch.einsum(
            "bqpd,bd->bqp", query.float(), prototype
        )
        return query_temperature * (
            torch.logsumexp(similarity / query_temperature, dim=-1)
            - math.log(similarity.shape[-1])
        )

    return score(positive_prototype) - score(negative_prototype)


def dual_scores(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    patch_temperature: float,
    query_temperature: float = 0.1,
    uniform_local_weights: bool = False,
) -> dict[str, torch.Tensor]:
    """Return global, current-local, and selected-local binary logits."""
    positive_global = global_prototype(positive, positive_mask)
    negative_global = global_prototype(negative, negative_mask)
    positive_local, negative_local = selected_local_prototypes(
        positive,
        negative,
        positive_mask,
        negative_mask,
        patch_temperature,
        uniform_weights=uniform_local_weights,
    )
    return {
        "global": global_binary_score(
            query, positive_global, negative_global
        ),
        "current_local": local_binary_score(
            query,
            positive_global,
            negative_global,
            query_temperature,
        ),
        "selected_local": local_binary_score(
            query,
            positive_local,
            negative_local,
            query_temperature,
        ),
    }


def fused_score(
    global_score: torch.Tensor,
    local_score: torch.Tensor,
    local_weight: float,
) -> torch.Tensor:
    if not 0 <= local_weight <= 1:
        raise ValueError("local fusion weight must be in [0,1]")
    return (1 - local_weight) * global_score + local_weight * local_score
