"""Invariant evidence-ratio attention over positive and negative patch banks."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


METHODS = ("positive_prototype", "iera", "iera_e1", "iera_no_negatives", "iera_mean_env")


class IERA(nn.Module):
    def __init__(self, input_dim: int, projection_dim: int = 128) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, projection_dim, bias=False)
        nn.init.orthogonal_(self.projection.weight)
        self.raw_tau = nn.Parameter(torch.tensor(-2.3))
        self.raw_tau_attention = nn.Parameter(torch.tensor(-2.3))
        self.raw_tau_query = nn.Parameter(torch.tensor(-2.3))
        self.raw_beta = nn.Parameter(torch.tensor(-1.5))
        self.raw_gamma = nn.Parameter(torch.tensor(2.0))

    @staticmethod
    def _positive(value: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
        return F.softplus(value) + floor

    def parameters_dict(self) -> dict[str, float]:
        return {
            "tau": float(self._positive(self.raw_tau).detach()),
            "tau_attention": float(self._positive(self.raw_tau_attention).detach()),
            "tau_query": float(self._positive(self.raw_tau_query).detach()),
            "beta": float(self._positive(self.raw_beta).detach()),
            "gamma": float(self._positive(self.raw_gamma).detach()),
        }

    def _project(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(tokens.float()), dim=-1)

    def _lme(self, tokens: torch.Tensor, bank: torch.Tensor, self_patch_offset: int | None = None) -> torch.Tensor:
        # tokens [B,N,P,D], bank [B,A,D] -> [B,N,P]
        similarity = torch.einsum("bnpd,bad->bnpa", tokens, bank) / self._positive(self.raw_tau)
        if self_patch_offset is not None:
            patch_count = tokens.shape[2]
            diagonal = torch.arange(patch_count, device=tokens.device)
            similarity[:, 0, diagonal, self_patch_offset + diagonal] = -torch.inf
        return torch.logsumexp(similarity, dim=-1) - math.log(bank.shape[1])

    def _robust_evidence(
        self,
        tokens: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
        method: str,
        self_environment: int | None = None,
        self_shot: int | None = None,
    ) -> torch.Tensor:
        environments = 1 if method == "iera_e1" else positive.shape[1]
        ratios = []
        for environment in range(environments):
            positive_bank = positive[:, environment].flatten(1, 2)
            self_offset = None
            if environment == self_environment and self_shot is not None:
                self_offset = self_shot * tokens.shape[2]
            ratio = self._lme(tokens, positive_bank, self_offset)
            if method != "iera_no_negatives":
                negative_bank = negative[:, environment].flatten(1, 2)
                ratio = ratio - self._lme(tokens, negative_bank)
            ratios.append(ratio)
        evidence = torch.stack(ratios, dim=-1)
        if method == "iera_mean_env" or evidence.shape[-1] == 1:
            return evidence.mean(-1)
        beta = self._positive(self.raw_beta)
        return -beta * (torch.logsumexp(-evidence / beta, dim=-1) - math.log(evidence.shape[-1]))

    def _prototype(self, positive: torch.Tensor, negative: torch.Tensor, method: str) -> torch.Tensor:
        batch, environments, shots, patches, width = positive.shape
        if method == "iera_e1":
            environments = 1
        token_groups, evidence_groups = [], []
        for environment in range(environments):
            for shot in range(shots):
                tokens = positive[:, environment, shot : shot + 1]
                token_groups.append(tokens[:, 0])
                evidence_groups.append(
                    self._robust_evidence(
                        tokens, positive, negative, method,
                        self_environment=environment, self_shot=shot,
                    )[:, 0]
                )
        tokens = torch.cat(token_groups, dim=1)
        evidence = torch.cat(evidence_groups, dim=1)
        attention = (evidence / self._positive(self.raw_tau_attention)).softmax(-1)
        return F.normalize(torch.einsum("bn,bnd->bd", attention, tokens), dim=-1)

    def forward(self, positive_tokens: torch.Tensor, negative_tokens: torch.Tensor, query_tokens: torch.Tensor, method: str = "iera") -> torch.Tensor:
        if method not in METHODS:
            raise ValueError(f"unknown IERA method {method!r}")
        positive = self._project(positive_tokens)
        negative = self._project(negative_tokens)
        query = self._project(query_tokens)
        if method == "positive_prototype":
            prototype = F.normalize(positive.mean(dim=(1, 2, 3)), dim=-1)
            query_representation = F.normalize(query.mean(2), dim=-1)
        else:
            prototype = self._prototype(positive, negative, method)
            evidence = self._robust_evidence(query, positive, negative, method)
            attention = (evidence / self._positive(self.raw_tau_query)).softmax(-1)
            query_representation = F.normalize(torch.einsum("bnp,bnpd->bnd", attention, query), dim=-1)
        gamma = self._positive(self.raw_gamma)
        return gamma * torch.einsum("bnd,bd->bn", query_representation, prototype)

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
