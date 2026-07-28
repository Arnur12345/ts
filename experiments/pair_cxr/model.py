"""Small frozen-feature evidence router for PAIR-CXR."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PAIRRouter(nn.Module):
    def __init__(self, width: int = 768, bottleneck: int = 64) -> None:
        super().__init__()
        if width <= 0 or bottleneck <= 0:
            raise ValueError("width and bottleneck must be positive")
        self.width = int(width)
        self.key_down = nn.Linear(width, bottleneck, bias=False)
        self.key_up = nn.Linear(bottleneck, width, bias=False)
        self.query_down = nn.Linear(width, bottleneck, bias=False)
        self.query_up = nn.Linear(bottleneck, width, bias=False)
        nn.init.xavier_uniform_(self.key_down.weight)
        nn.init.zeros_(self.key_up.weight)
        nn.init.xavier_uniform_(self.query_down.weight)
        nn.init.zeros_(self.query_up.weight)
        self.raw_temperature = nn.Parameter(torch.tensor(-2.3))
        self.raw_local_weight = nn.Parameter(torch.tensor(-1.1))
        self.raw_scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _positive(value: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
        return F.softplus(value) + floor

    def project_query(self, text_query: torch.Tensor) -> torch.Tensor:
        text_query = F.normalize(text_query.float(), dim=-1)
        residual = self.query_up(F.gelu(self.query_down(text_query)))
        return F.normalize(text_query + residual, dim=-1)

    def encode_tokens(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError("tokens must be [images,patches,width]")
        tokens = F.normalize(tokens.float(), dim=-1)
        keys = F.normalize(
            tokens + self.key_up(F.gelu(self.key_down(tokens))), dim=-1
        )
        global_feature = F.normalize(tokens.mean(dim=1), dim=-1)
        return keys, global_feature

    def route_encoded(
        self,
        encoded: tuple[torch.Tensor, torch.Tensor],
        query: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        keys, global_feature = encoded
        query = F.normalize(query.float(), dim=-1)
        similarity = torch.einsum("npd,d->np", keys, query)
        temperature = self._positive(self.raw_temperature)
        weights = torch.softmax(similarity / temperature, dim=-1)
        local = (weights * similarity).sum(dim=-1)
        global_score = global_feature @ query
        local_weight = torch.sigmoid(self.raw_local_weight)
        evidence = (
            (1 - local_weight) * global_score + local_weight * local
        )
        return evidence, weights

    def route(
        self, tokens: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.route_encoded(self.encode_tokens(tokens), query)

    def score_encoded(
        self,
        encoded: tuple[torch.Tensor, torch.Tensor],
        query: torch.Tensor,
        raw_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        evidence, weights = self.route_encoded(encoded, query)
        scale = self.raw_scale if raw_scale is None else raw_scale
        current_bias = self.bias if bias is None else bias
        return self._positive(scale) * evidence + current_bias, weights

    def score_with_query(
        self,
        tokens: torch.Tensor,
        query: torch.Tensor,
        raw_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.score_encoded(
            self.encode_tokens(tokens), query, raw_scale, bias
        )

    def forward(
        self, tokens: torch.Tensor, text_query: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.score_with_query(
            tokens, self.project_query(text_query)
        )


def intervention_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    devices: torch.Tensor,
    beta_rex: float,
    lambda_invariance: float,
    lambda_responsiveness: float,
    minimum_margin: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    group_losses, margins = [], []
    for target in (0, 1):
        for device in (0, 1):
            mask = targets.eq(target) & devices.eq(device)
            if mask.any():
                group_losses.append(
                    F.binary_cross_entropy_with_logits(
                        logits[mask], targets[mask].float()
                    )
                )
    classification = torch.stack(group_losses).mean()
    rex = (
        torch.stack(group_losses).var(unbiased=False)
        if len(group_losses) > 1
        else logits.new_tensor(0.0)
    )
    for device in (0, 1):
        positive = targets.eq(1) & devices.eq(device)
        negative = targets.eq(0) & devices.eq(device)
        if positive.any() and negative.any():
            margins.append(logits[positive].mean() - logits[negative].mean())
    if margins:
        margin_tensor = torch.stack(margins)
        responsiveness = F.relu(
            float(minimum_margin) - margin_tensor
        ).mean()
        invariance = (
            margin_tensor.var(unbiased=False)
            if len(margin_tensor) > 1
            else logits.new_tensor(0.0)
        )
    else:
        responsiveness = logits.new_tensor(0.0)
        invariance = logits.new_tensor(0.0)
    loss = (
        classification
        + float(beta_rex) * rex
        + float(lambda_invariance) * invariance
        + float(lambda_responsiveness) * responsiveness
    )
    return loss, {
        "classification": classification,
        "rex": rex,
        "invariance": invariance,
        "responsiveness": responsiveness,
    }


def normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    if weights.shape[-1] <= 1:
        return weights.new_tensor(0.0)
    entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(-1)
    return entropy.mean() / math.log(weights.shape[-1])
