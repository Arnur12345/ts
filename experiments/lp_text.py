"""Medical LP+text: a visual linear probe plus class-wise text logits."""

import torch
import torch.nn.functional as F

from .common import cosine_logits, prototypes, text_prototypes


def predict(support_images, support_reports, labels, query_images, shuffled=False, epochs=100):
    batch = len(support_images)
    text = text_prototypes(support_reports, labels, 3, shuffled).detach()
    weights = prototypes(support_images, labels, 3).detach().clone().requires_grad_()
    bias = torch.zeros(batch, 3, device=weights.device, requires_grad=True)
    alpha = torch.ones(batch, 3, device=weights.device, requires_grad=True)
    target = labels.expand(batch, -1).reshape(-1)
    optimizer = torch.optim.Adam([weights, bias, alpha], lr=0.03)

    for _ in range(epochs):
        visual = cosine_logits(support_images, weights) + bias[:, None, :]
        text_logits = cosine_logits(support_images, text)
        loss = F.cross_entropy((10 * (visual + alpha[:, None, :] * text_logits)).reshape(-1, 3), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    visual = cosine_logits(query_images, weights.detach()) + bias.detach()[:, None, :]
    text_logits = cosine_logits(query_images, text)
    return (visual + alpha.detach()[:, None, :] * text_logits).detach()
