"""Counterfactual Metric Distillation (CoMeD) probabilistic head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores - scores.mean()
    return scores / scores.norm().clamp_min(1e-8)


class CoMeD(nn.Module):
    """A low-rank PSD metric and support-only nuisance covariance."""

    def __init__(
        self,
        dim: int = 768,
        rank: int = 16,
        metric_mode: str = "learned",
        use_nuisance: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0 or rank <= 0:
            raise ValueError("dim and rank must be positive")
        if metric_mode not in {"identity", "learned"}:
            raise ValueError("metric_mode must be identity or learned")
        self.dim = int(dim)
        self.rank = int(rank)
        self.metric_mode = metric_mode
        self.use_nuisance = bool(use_nuisance)
        self.log_diag = nn.Parameter(torch.zeros(dim))
        self.low_rank = nn.Parameter(torch.randn(dim, rank) * 0.01)
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.log_noise = nn.Parameter(torch.tensor(-2.0))

    def transform(self, features: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features.float(), dim=-1)
        if self.metric_mode == "identity":
            return features
        diagonal = F.softplus(self.log_diag) + 1e-4
        diagonal_part = features * diagonal.sqrt()
        low_rank_part = features @ self.low_rank
        return torch.cat((diagonal_part, low_rank_part), dim=-1)

    def configure_trainable(
        self,
        learn_metric: bool,
        learn_nuisance: bool,
        learn_noise: bool = True,
    ) -> list[nn.Parameter]:
        self.log_diag.requires_grad_(learn_metric)
        self.low_rank.requires_grad_(learn_metric)
        self.log_tau.requires_grad_(learn_nuisance)
        self.log_noise.requires_grad_(learn_noise)
        return [
            parameter for parameter in self.parameters()
            if parameter.requires_grad
        ]

    def forward(
        self,
        support_z: torch.Tensor,
        support_y_pm: torch.Tensor,
        support_nuisance: torch.Tensor,
        support_prior: torch.Tensor,
        query_z: torch.Tensor,
        query_prior: torch.Tensor,
    ) -> torch.Tensor:
        if support_z.ndim != 2 or query_z.ndim != 2:
            raise ValueError("support_z and query_z must be matrices")
        if support_z.shape[1] != self.dim or query_z.shape[1] != self.dim:
            raise ValueError("feature width differs from CoMeD dimension")
        if support_y_pm.shape != (len(support_z),):
            raise ValueError("support labels have wrong shape")
        if support_prior.shape != support_y_pm.shape:
            raise ValueError("support prior has wrong shape")
        if query_prior.shape != (len(query_z),):
            raise ValueError("query prior has wrong shape")
        support_phi = self.transform(support_z)
        query_phi = self.transform(query_z)
        disease_kernel = support_phi @ support_phi.T
        if self.use_nuisance:
            if (
                support_nuisance.ndim != 2
                or support_nuisance.shape[0] != len(support_z)
            ):
                raise ValueError("support nuisance matrix has wrong shape")
            tau = F.softplus(self.log_tau)
            nuisance_kernel = (
                tau * support_nuisance.float() @ support_nuisance.float().T
            )
        else:
            nuisance_kernel = torch.zeros_like(disease_kernel)
        noise = F.softplus(self.log_noise) + 1e-4
        eye = torch.eye(
            len(support_z),
            device=support_z.device,
            dtype=disease_kernel.dtype,
        )
        kernel = disease_kernel + nuisance_kernel + noise * eye
        residual = support_y_pm.float() - support_prior.float()
        chol = torch.linalg.cholesky(kernel + 1e-5 * eye)
        alpha = torch.cholesky_solve(residual[:, None], chol)
        # Deliberately exclude nuisance from the query-to-support kernel:
        # inference is the counterfactual do(nuisance=0) prediction.
        cross_kernel = query_phi @ support_phi.T
        correction = cross_kernel @ alpha
        return query_prior.float() + correction.squeeze(-1)


def grouped_rex_loss(
    scores: tuple[torch.Tensor, torch.Tensor],
    target: torch.Tensor,
    nuisance: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Mean plus REx variance over the four disease-by-device groups."""
    losses = []
    for disease in (0, 1):
        for device in (0, 1):
            mask = target.eq(disease) & nuisance.eq(device)
            if mask.any():
                panel_losses = [
                    F.binary_cross_entropy_with_logits(
                        current[mask], target[mask].float()
                    )
                    for current in scores
                ]
                losses.append(torch.stack(panel_losses).mean())
    if len(losses) != 4:
        raise ValueError("paired query set must contain all four groups")
    values = torch.stack(losses)
    return values.mean() + float(beta) * values.var(unbiased=False)


def distillation_loss(
    scores: tuple[torch.Tensor, torch.Tensor],
    teacher: torch.Tensor,
) -> torch.Tensor:
    teacher = normalize_scores(teacher.float())
    return sum(
        1 - F.cosine_similarity(
            normalize_scores(current)[None], teacher[None]
        ).squeeze(0)
        for current in scores
    )


def swap_loss(
    scores_zero: torch.Tensor, scores_one: torch.Tensor
) -> torch.Tensor:
    return 1 - F.cosine_similarity(
        normalize_scores(scores_zero)[None],
        normalize_scores(scores_one)[None],
    ).squeeze(0)
