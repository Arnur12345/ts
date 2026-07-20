"""Only the diagnostic controls requested after the initial run."""

from .common import cosine_logits, prototypes
from .protonet_text import predict as protonet_text
from .text_only_zero_shot import predict as text_only
from .visual_protonet import predict as visual_protonet


def shuffled_text(*args, **kwargs):
    return protonet_text(*args, **kwargs, shuffled=True)


def permuted_support_labels(support_images, support_reports, labels, query_images, shuffled=False):
    """Visual ProtoNet after cyclically assigning every support to a wrong class."""
    del support_reports, shuffled
    wrong_labels = (labels + 1) % 3
    return cosine_logits(query_images, prototypes(support_images, wrong_labels, 3))


def duplicated_support(support_images, support_reports, labels, query_images, shuffled=False):
    """Repeat the first support instead of using independent 3/5-shot supports."""
    del support_reports, shuffled
    episodes, samples, width = support_images.shape
    shot = samples // 3
    repeated = support_images.reshape(episodes, 3, shot, width)[:, :, :1].expand(-1, -1, shot, -1)
    return cosine_logits(query_images, prototypes(repeated.reshape(episodes, samples, width), labels, 3))


METHODS = {
    "text_only": (text_only, "text_only"),
    "visual_protonet": (visual_protonet, "visual_protonet"),
    "protonet_text": (protonet_text, "protonet_text"),
    "protonet_text_shuffled_text": (shuffled_text, "protonet_text"),
    "visual_protonet_permuted_support_labels": (permuted_support_labels, "visual_protonet"),
    "visual_protonet_duplicated_support": (duplicated_support, "visual_protonet"),
}
