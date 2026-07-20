"""TCLA final-layer ablation: prototype basis plus residual logit correction."""

import torch

from .common import cosine_logits, one_hot, prototypes, text_prototypes


def predict(
    support_images,
    support_reports,
    labels,
    query_images,
    shuffled=False,
    beta=5.0,
    ridge=1.0,
):
    batch = len(support_images)
    visual = prototypes(support_images, labels, 3)
    text = text_prototypes(support_reports, labels, 3, shuffled)
    support_basis = torch.exp(beta * (cosine_logits(support_images, visual) - 1))
    query_basis = torch.exp(beta * (cosine_logits(query_images, visual) - 1))
    support_prior = cosine_logits(support_images, text)
    residual = one_hot(labels, 3, batch).to(support_images.device) - support_prior
    eye = torch.eye(3, device=support_images.device)[None]
    mapping = torch.linalg.solve(
        support_basis.transpose(1, 2) @ support_basis + ridge * eye,
        support_basis.transpose(1, 2) @ residual,
    )
    fitted = support_basis @ mapping
    eta = (residual * fitted).sum((1, 2)) / fitted.square().sum((1, 2)).clamp_min(1e-8)
    return cosine_logits(query_images, text) + eta[:, None, None] * (query_basis @ mapping)
