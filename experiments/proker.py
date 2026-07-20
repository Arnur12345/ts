"""ProKeR equation 14: proximal RBF kernel ridge correction of text logits."""

import torch

from .common import cosine_logits, one_hot, text_prototypes


def predict(
    support_images,
    support_reports,
    labels,
    query_images,
    shuffled=False,
    beta=5.0,
    ridge=1.0,
):
    batch, samples, _ = support_images.shape
    text = text_prototypes(support_reports, labels, 3, shuffled)
    targets = one_hot(labels, 3, batch).to(support_images.device)
    support_prior = cosine_logits(support_images, text)
    gram = torch.exp(-beta * (1 - torch.bmm(support_images, support_images.transpose(1, 2))))
    system = torch.eye(samples, device=gram.device)[None] + gram / ridge
    gamma = torch.linalg.solve(system, targets - support_prior)
    query_kernel = torch.exp(-beta * (1 - torch.bmm(query_images, support_images.transpose(1, 2))))
    return cosine_logits(query_images, text) + torch.bmm(query_kernel, gamma)
