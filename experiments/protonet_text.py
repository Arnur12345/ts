"""Average visual and real-report class prototypes."""

import torch.nn.functional as F

from .common import cosine_logits, prototypes, text_prototypes


def predict(support_images, support_reports, labels, query_images, shuffled=False, text_weight=0.5):
    visual = prototypes(support_images, labels, 3)
    text = text_prototypes(support_reports, labels, 3, shuffled)
    mixed = F.normalize((1 - text_weight) * visual + text_weight * text, dim=-1)
    return cosine_logits(query_images, mixed)
