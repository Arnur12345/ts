"""Standard visual Prototypical Network on frozen image embeddings."""

from .common import cosine_logits, prototypes


def predict(support_images, support_reports, labels, query_images, shuffled=False):
    del support_reports, shuffled
    return cosine_logits(query_images, prototypes(support_images, labels, 3))
