"""Zero visual-support classifier using real support-report prototypes."""

from .common import cosine_logits, text_prototypes


def predict(support_images, support_reports, labels, query_images, shuffled=False):
    del support_images
    text = text_prototypes(support_reports, labels, 3, shuffled)
    return cosine_logits(query_images, text)
