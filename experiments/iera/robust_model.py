"""Repaired localized binary detector and controlled robustness ablations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


LEARNED_METHODS = ("rex", "adapter_only", "anchor_only", "full_iera")


class RobustBinaryModel(nn.Module):
    """Frozen-query binary detector with optional support-side mechanisms."""

    def __init__(
        self,
        width: int,
        adapter_dim: int = 16,
        alpha_max: float = 0.25,
        local_temperature: float = 0.1,
        proposal_grid: int | None = None,
    ) -> None:
        super().__init__()
        if width <= 0 or adapter_dim <= 0:
            raise ValueError("width and adapter_dim must be positive")
        if not 0 < alpha_max <= 1 or local_temperature <= 0:
            raise ValueError("alpha_max and local_temperature are invalid")
        if proposal_grid is not None and proposal_grid <= 0:
            raise ValueError("proposal_grid must be positive")
        self.width = int(width)
        self.adapter_dim = int(adapter_dim)
        self.alpha_max = float(alpha_max)
        self.local_temperature = float(local_temperature)
        self.proposal_grid = proposal_grid
        self.support_down = nn.Linear(width, adapter_dim, bias=False)
        self.support_up = nn.Linear(adapter_dim, width, bias=False)
        nn.init.xavier_uniform_(self.support_down.weight)
        nn.init.zeros_(self.support_up.weight)
        self.raw_evidence_temperature = nn.Parameter(torch.tensor(-2.3))
        self.raw_attention_temperature = nn.Parameter(torch.tensor(-2.3))
        self.raw_soft_min = nn.Parameter(torch.tensor(-1.5))
        self.raw_anchor_bias = nn.Parameter(torch.tensor(0.0))
        self.raw_anchor_slope = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _positive(value: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
        return F.softplus(value) + floor

    @staticmethod
    def _prototype(
        tokens: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is None:
            return F.normalize(tokens.mean(dim=(1, 2, 3)), dim=-1)
        weights = mask[..., None, None].to(tokens.dtype)
        total = (tokens * weights).sum(dim=(1, 2, 3))
        denominator = (
            mask.sum(dim=(1, 2)).clamp_min(1).to(tokens.dtype)
            * tokens.shape[3]
        )
        return F.normalize(total / denominator[:, None], dim=-1)

    def _adapt(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = self.support_up(F.gelu(self.support_down(tokens)))
        return F.normalize(tokens + residual, dim=-1)

    def _lme(
        self,
        tokens: torch.Tensor,
        bank: torch.Tensor,
        bank_mask: torch.Tensor,
        self_offset: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        similarity = torch.einsum("bnpd,bad->bnpa", tokens, bank)
        similarity = similarity / self._positive(self.raw_evidence_temperature)
        effective_mask = bank_mask.clone()
        if self_offset is not None:
            patch_count = tokens.shape[2]
            effective_mask[:, self_offset : self_offset + patch_count] = False
        similarity = similarity.masked_fill(
            ~effective_mask[:, None, None], -torch.inf
        )
        bank_size = effective_mask.sum(dim=-1)
        available = bank_size.gt(0)
        value = torch.logsumexp(similarity, dim=-1) - bank_size.clamp_min(
            1
        ).log()[:, None, None]
        return torch.where(available[:, None, None], value, 0.0), available

    def _robust_evidence(
        self,
        tokens: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
        self_environment: int,
        self_shot: int,
    ) -> torch.Tensor:
        ratios = []
        available_groups = []
        for environment in range(positive.shape[1]):
            offset = (
                self_shot * tokens.shape[2]
                if environment == self_environment
                else None
            )
            positive_bank_mask = positive_mask[:, environment, :, None].expand(
                -1, -1, positive.shape[3]
            ).flatten(1)
            negative_bank_mask = negative_mask[:, environment, :, None].expand(
                -1, -1, negative.shape[3]
            ).flatten(1)
            positive_score, positive_available = self._lme(
                tokens,
                positive[:, environment].flatten(1, 2),
                positive_bank_mask,
                offset,
            )
            negative_score, negative_available = self._lme(
                tokens,
                negative[:, environment].flatten(1, 2),
                negative_bank_mask,
            )
            ratios.append(positive_score - negative_score)
            available_groups.append(positive_available & negative_available)
        stacked = torch.stack(ratios, dim=-1)
        available = torch.stack(available_groups, dim=-1)
        beta = self._positive(self.raw_soft_min)
        terms = (-stacked / beta).masked_fill(
            ~available[:, None, None], -torch.inf
        )
        count = available.sum(dim=-1)
        terms = torch.where(
            count[:, None, None, None].gt(0),
            terms,
            torch.zeros_like(terms),
        )
        result = -beta * (
            torch.logsumexp(terms, dim=-1)
            - count.clamp_min(1).log()[:, None, None]
        )
        return torch.where(count[:, None, None].gt(0), result, 0.0)

    def _evidence_prototype(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_groups, evidence_groups, token_mask_groups = [], [], []
        for environment in range(positive.shape[1]):
            for shot in range(positive.shape[2]):
                if not positive_mask[:, environment, shot].any():
                    continue
                tokens = positive[:, environment, shot : shot + 1]
                token_groups.append(tokens[:, 0])
                token_mask_groups.append(
                    positive_mask[:, environment, shot : shot + 1].expand(
                        -1, positive.shape[3]
                    )
                )
                evidence_groups.append(
                    self._robust_evidence(
                        tokens,
                        positive,
                        negative,
                        positive_mask,
                        negative_mask,
                        environment,
                        shot,
                    )[:, 0]
                )
        tokens = torch.cat(token_groups, dim=1)
        evidence = torch.cat(evidence_groups, dim=1)
        token_mask = torch.cat(token_mask_groups, dim=1)
        attention_logits = (
            evidence / self._positive(self.raw_attention_temperature)
        ).masked_fill(~token_mask, -torch.inf)
        attention = attention_logits.softmax(dim=-1)
        return F.normalize(
            torch.einsum("bn,bnd->bd", attention, tokens), dim=-1
        )

    def _anchored_positive(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        positive_mask: torch.Tensor,
        negative_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uniform = self._prototype(positive, positive_mask)
        evidence_positive = self._proposal_tokens(positive)
        evidence_negative = self._proposal_tokens(negative)
        evidence = self._evidence_prototype(
            evidence_positive,
            evidence_negative,
            positive_mask,
            negative_mask,
        )
        disagreement = 1 - (uniform * evidence).sum(dim=-1, keepdim=True)
        alpha = self.alpha_max * torch.sigmoid(
            self.raw_anchor_bias + self.raw_anchor_slope * disagreement
        )
        anchored = F.normalize(
            uniform + alpha * (evidence - uniform), dim=-1
        )
        return anchored, alpha.squeeze(-1)

    def _proposal_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.proposal_grid is None:
            return tokens
        side = math.isqrt(tokens.shape[3])
        if side * side != tokens.shape[3]:
            raise ValueError("support patch count is not a square grid")
        if self.proposal_grid > side:
            raise ValueError("proposal grid exceeds the support token grid")
        if self.proposal_grid == side:
            return tokens
        leading = tokens.shape[:3]
        width = tokens.shape[-1]
        spatial = tokens.flatten(0, 2).transpose(1, 2).reshape(
            -1, width, side, side
        )
        pooled = F.adaptive_avg_pool2d(
            spatial, (self.proposal_grid, self.proposal_grid)
        )
        return F.normalize(
            pooled.flatten(2).transpose(1, 2).reshape(
                *leading, self.proposal_grid**2, width
            ),
            dim=-1,
        )

    def _local_score(
        self, query: torch.Tensor, prototype: torch.Tensor
    ) -> torch.Tensor:
        similarity = torch.einsum("bqpd,bd->bqp", query, prototype)
        tau = self.local_temperature
        return tau * (
            torch.logsumexp(similarity / tau, dim=-1)
            - math.log(similarity.shape[-1])
        )

    def forward(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        query: torch.Tensor,
        method: str,
        positive_mask: torch.Tensor | None = None,
        negative_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if method not in ("uniform", *LEARNED_METHODS):
            raise ValueError(f"unknown robust method {method!r}")
        positive = F.normalize(positive.float(), dim=-1)
        negative = F.normalize(negative.float(), dim=-1)
        query = F.normalize(query.float(), dim=-1)
        if positive_mask is None:
            positive_mask = torch.ones(
                positive.shape[:3], dtype=torch.bool, device=positive.device
            )
        if negative_mask is None:
            negative_mask = torch.ones(
                negative.shape[:3], dtype=torch.bool, device=negative.device
            )
        if method in {"rex", "adapter_only", "full_iera"}:
            positive = self._adapt(positive)
            negative = self._adapt(negative)
        if method in {"anchor_only", "full_iera"}:
            positive_prototype, _ = self._anchored_positive(
                positive, negative, positive_mask, negative_mask
            )
        else:
            positive_prototype = self._prototype(positive, positive_mask)
        negative_prototype = self._prototype(negative, negative_mask)
        return (
            self._local_score(query, positive_prototype)
            - self._local_score(query, negative_prototype)
        )

    def swapped_logits(
        self,
        positive: torch.Tensor,
        negative: torch.Tensor,
        query: torch.Tensor,
        method: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        panels = []
        for environment in (0, 1):
            panels.append(
                self(
                    positive[:, environment : environment + 1],
                    negative[:, environment : environment + 1],
                    query,
                    method,
                )
            )
        return panels[0], panels[1]

    def configure_trainable(self, method: str) -> list[nn.Parameter]:
        if method not in LEARNED_METHODS:
            raise ValueError(f"{method!r} is not learned")
        adapter = method in {"rex", "adapter_only", "full_iera"}
        evidence_anchor = method in {"anchor_only", "full_iera"}
        for name, parameter in self.named_parameters():
            if name.startswith("support_"):
                parameter.requires_grad_(adapter)
            else:
                parameter.requires_grad_(evidence_anchor)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def localized_binary_logits(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Parameter-free repaired detector shared by all cheap baselines."""
    model = RobustBinaryModel(
        positive.shape[-1], local_temperature=temperature
    ).to(positive.device)
    return model(positive, negative, query, "uniform")


def project_direction(
    tokens: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    direction = F.normalize(direction.float(), dim=-1)
    while direction.ndim < tokens.ndim:
        direction = direction.unsqueeze(1)
    return F.normalize(
        tokens.float()
        - torch.einsum("...d,...d->...", tokens.float(), direction)[..., None]
        * direction,
        dim=-1,
    )
