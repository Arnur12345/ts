"""Invariant evidence-ratio attention over positive and negative patch banks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = ("frozen_protonet", "learned_uniform", "iera", "anchored_iera")


class IERA(nn.Module):
    def __init__(
        self,
        input_dim: int,
        projection_dim: int = 128,
        alpha_max: float = 0.25,
        support_adapter_dim: int = 16,
    ) -> None:
        super().__init__()
        if not 0 < alpha_max <= 1:
            raise ValueError("alpha_max must be in (0, 1]")
        if support_adapter_dim <= 0:
            raise ValueError("support_adapter_dim must be positive")
        self.alpha_max = float(alpha_max)
        self.support_adapter_dim = int(support_adapter_dim)
        self.projection = nn.Linear(input_dim, projection_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.support_adapter_down = nn.Linear(
            projection_dim, support_adapter_dim, bias=False
        )
        self.support_adapter_up = nn.Linear(
            support_adapter_dim, projection_dim, bias=False
        )
        nn.init.xavier_uniform_(self.support_adapter_down.weight)
        nn.init.zeros_(self.support_adapter_up.weight)
        self.raw_tau = nn.Parameter(torch.tensor(-2.3))
        self.raw_tau_attention = nn.Parameter(torch.tensor(-2.3))
        self.raw_tau_query = nn.Parameter(torch.tensor(-2.3))
        self.raw_beta = nn.Parameter(torch.tensor(-1.5))
        self.raw_gamma = nn.Parameter(torch.tensor(2.0))
        self.raw_anchor_bias = nn.Parameter(torch.tensor(0.0))
        self.raw_anchor_slope = nn.Parameter(torch.tensor(0.0))

    @staticmethod
    def _positive(value: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
        return F.softplus(value) + floor

    def parameters_dict(self) -> dict[str, float | int]:
        return {
            "tau": float(self._positive(self.raw_tau).detach()),
            "tau_attention": float(self._positive(self.raw_tau_attention).detach()),
            "tau_query": float(self._positive(self.raw_tau_query).detach()),
            "beta": float(self._positive(self.raw_beta).detach()),
            "gamma": float(self._positive(self.raw_gamma).detach()),
            "alpha_max": self.alpha_max,
            "support_adapter_dim": self.support_adapter_dim,
            "support_adapter_norm": float(
                self.support_adapter_up.weight.detach().norm()
            ),
            "anchor_bias": float(self.raw_anchor_bias.detach()),
            "anchor_slope": float(self.raw_anchor_slope.detach()),
        }

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(tokens.float()), dim=-1)

    def _adapt_support(self, projected: torch.Tensor) -> torch.Tensor:
        residual = self.support_adapter_up(
            F.gelu(self.support_adapter_down(projected))
        )
        return F.normalize(projected + residual, dim=-1)

    def _lme(self, tokens: torch.Tensor, bank: torch.Tensor, self_image_offset: int | None = None) -> torch.Tensor:
        # tokens [B,N,P,D], bank [B,A,D] -> [B,N,P]
        similarity = torch.einsum("bnpd,bad->bnpa", tokens, bank) / self._positive(self.raw_tau)
        bank_size = bank.shape[1]
        if self_image_offset is not None:
            patch_count = tokens.shape[2]
            similarity[:, :, :, self_image_offset : self_image_offset + patch_count] = -torch.inf
            bank_size -= patch_count
        if bank_size <= 0:
            raise ValueError("evidence bank is empty after excluding the source radiograph")
        return torch.logsumexp(similarity, dim=-1) - math.log(bank_size)

    def _robust_evidence(
        self,
        tokens: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        method: str,
        self_environment: int | None = None,
        self_shot: int | None = None,
    ) -> torch.Tensor:
        environments = positive.shape[1]
        ratios = []
        for environment in range(environments):
            positive_bank = positive[:, environment].flatten(1, 2)
            self_offset: int | None = None
            if environment == self_environment and self_shot is not None:
                # With one shot there is no independent positive evidence in this
                # environment. Use the other environment instead of leaking the
                # source radiograph into its own evidence estimate.
                if positive.shape[2] == 1:
                    continue
                self_offset = self_shot * tokens.shape[2]
            ratio = self._lme(tokens, positive_bank, self_offset)
            if method != "iera_no_negatives":
                negative_bank = negative[:, environment].flatten(1, 2)
                ratio = ratio - self._lme(tokens, negative_bank)
            ratios.append(ratio)
        if not ratios:
            return torch.zeros(tokens.shape[:-1], dtype=tokens.dtype, device=tokens.device)
        evidence = torch.stack(ratios, dim=-1)
        if evidence.shape[-1] == 1:
            return evidence.mean(-1)
        beta = self._positive(self.raw_beta)
        return -beta * (torch.logsumexp(-evidence / beta, dim=-1) - math.log(evidence.shape[-1]))

    @staticmethod
    def _uniform_prototype(positive: torch.Tensor) -> torch.Tensor:
        return F.normalize(positive.mean(dim=(1, 2, 3)), dim=-1)

    def _evidence_prototype(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        batch, environments, shots, patches, width = positive.shape
        token_groups, evidence_groups = [], []
        for environment in range(environments):
            for shot in range(shots):
                tokens = positive[:, environment, shot : shot + 1]
                token_groups.append(tokens[:, 0])
                evidence_groups.append(
                    self._robust_evidence(
                        tokens, positive, negative, "iera",
                        self_environment=environment, self_shot=shot,
                    )[:, 0]
                )
        tokens = torch.cat(token_groups, dim=1)
        evidence = torch.cat(evidence_groups, dim=1)
        attention = (evidence / self._positive(self.raw_tau_attention)).softmax(-1)
        return F.normalize(torch.einsum("bn,bnd->bd", attention, tokens), dim=-1)

    def _anchored_prototype(
        self, positive: torch.Tensor, negative: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        uniform = self._uniform_prototype(positive)
        evidence = self._evidence_prototype(positive, negative)
        disagreement = 1 - (uniform * evidence).sum(-1, keepdim=True)
        alpha = self.alpha_max * torch.sigmoid(
            self.raw_anchor_bias + self.raw_anchor_slope * disagreement
        )
        prototype = F.normalize(uniform + alpha * (evidence - uniform), dim=-1)
        return prototype, alpha.squeeze(-1)

    def anchor_weight(self, positive_tokens: torch.Tensor, negative_tokens: torch.Tensor) -> torch.Tensor:
        positive = self._adapt_support(self._project(positive_tokens))
        negative = self._adapt_support(self._project(negative_tokens))
        return self._anchored_prototype(positive, negative)[1]

    def _local_logits(self, query: torch.Tensor, prototype: torch.Tensor) -> torch.Tensor:
        patch_similarity = torch.einsum("bnpd,bd->bnp", query, prototype)
        tau_query = self._positive(self.raw_tau_query)
        local_score = tau_query * (
            torch.logsumexp(patch_similarity / tau_query, dim=-1)
            - math.log(patch_similarity.shape[-1])
        )
        return self._positive(self.raw_gamma) * local_score

    def forward(self, positive_tokens: torch.Tensor, negative_tokens: torch.Tensor, query_tokens: torch.Tensor, method: str = "iera") -> torch.Tensor:
        if method not in METHODS:
            raise ValueError(f"unknown IERA method {method!r}")
        if method == "frozen_protonet":
            # A fair frozen-space ProtoNet baseline: it does not inherit the
            # projection or scale learned with an IERA objective.
            positive = F.normalize(positive_tokens.float(), dim=-1)
            query = F.normalize(query_tokens.float(), dim=-1)
            prototype = F.normalize(positive.mean(dim=(1, 2, 3)), dim=-1)
            query_representation = F.normalize(query.mean(2), dim=-1)
            return torch.einsum("bnd,bd->bn", query_representation, prototype)

        positive = self._project(positive_tokens)
        negative = self._project(negative_tokens)
        query = self._project(query_tokens)
        if method == "learned_uniform":
            prototype = self._uniform_prototype(positive)
        elif method == "iera":
            prototype = self._evidence_prototype(positive, negative)
        else:
            # Only Anchored IERA may adapt support representations. Query tokens
            # always remain in the frozen learned-uniform projection space.
            positive = self._adapt_support(positive)
            negative = self._adapt_support(negative)
            prototype, _ = self._anchored_prototype(positive, negative)
        return self._local_logits(query, prototype)

    def swapped_logits(self, positive: torch.Tensor, negative: torch.Tensor, query: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Score identical queries using d=0-only versus d=1-only support panels."""
        panels = []
        for environment in (0, 1):
            panels.append(
                self(
                    positive[:, environment : environment + 1].expand(-1, 2, -1, -1, -1),
                    negative[:, environment : environment + 1].expand(-1, 2, -1, -1, -1),
                    query,
                    method,
                )
            )
        return panels[0], panels[1]
