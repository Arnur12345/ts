"""Tip-Adapter-F: fine-tune support cache keys, then add cache and text logits."""

import torch
import torch.nn.functional as F

from .common import cosine_logits, one_hot, text_prototypes


def predict(
    support_images,
    support_reports,
    labels,
    query_images,
    shuffled=False,
    alpha=1.0,
    beta=5.0,
    epochs=20,
):
    batch = len(support_images)
    text = text_prototypes(support_reports, labels, 3, shuffled)
    values = one_hot(labels, 3, batch).to(support_images.device)
    keys = support_images.detach().clone().requires_grad_()
    target = labels.expand(batch, -1).reshape(-1)
    optimizer = torch.optim.AdamW([keys], lr=0.01, weight_decay=1e-4)

    for _ in range(epochs):
        affinity = torch.bmm(support_images, F.normalize(keys, dim=-1).transpose(1, 2))
        cache = torch.bmm(torch.exp(-beta * (1 - affinity)), values)
        logits = cosine_logits(support_images, text) + alpha * cache
        loss = F.cross_entropy((10 * logits).reshape(-1, 3), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    affinity = torch.bmm(query_images, F.normalize(keys.detach(), dim=-1).transpose(1, 2))
    cache = torch.bmm(torch.exp(-beta * (1 - affinity)), values)
    return cosine_logits(query_images, text) + alpha * cache
