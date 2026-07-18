"""A small, self-contained implementation of Prototypical Networks.

An episode consists of a labeled support set and an unlabeled query set.  The
model embeds every example, averages the support embeddings for each class,
and classifies queries by their distance to those class prototypes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConvBlock(nn.Sequential):
    """The convolutional block used by the original ProtoNet image encoder."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )


class ConvEncoder(nn.Module):
    """A basic four-block CNN that maps images to embedding vectors."""

    def __init__(self, in_channels: int = 3, hidden_size: int = 64) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
            ConvBlock(hidden_size, hidden_size),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images: Tensor) -> Tensor:
        return self.pool(self.features(images)).flatten(start_dim=1)


def squared_euclidean_distance(x: Tensor, y: Tensor) -> Tensor:
    """Return all pairwise squared Euclidean distances between two batches."""
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("x and y must be matrices of shape [samples, features]")
    if x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have the same feature dimension")
    return (x[:, None, :] - y[None, :, :]).square().sum(dim=-1)


class PrototypicalNetwork(nn.Module):
    """Prototypical Network with an arbitrary embedding module.

    The columns of the returned logits correspond to the sorted unique labels
    in ``support_labels``.  Set ``return_classes=True`` when that mapping is
    needed explicitly.
    """

    def __init__(self, encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else ConvEncoder()

    def _encode(self, inputs: Tensor) -> Tensor:
        embeddings = self.encoder(inputs)
        if embeddings.ndim < 2:
            raise ValueError("the encoder must keep a batch dimension")
        return embeddings.flatten(start_dim=1)

    @staticmethod
    def compute_prototypes(
        support_embeddings: Tensor, support_labels: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Average support embeddings by class and return prototypes/classes."""
        if support_embeddings.ndim != 2:
            raise ValueError("support_embeddings must have shape [samples, features]")

        labels = support_labels.reshape(-1).to(support_embeddings.device)
        if support_embeddings.shape[0] != labels.numel():
            raise ValueError("each support embedding must have one label")
        if labels.numel() == 0:
            raise ValueError("the support set cannot be empty")

        classes, inverse = torch.unique(labels, sorted=True, return_inverse=True)
        prototypes = support_embeddings.new_zeros(
            (classes.numel(), support_embeddings.shape[1])
        )
        prototypes.index_add_(0, inverse, support_embeddings)
        counts = torch.bincount(inverse, minlength=classes.numel()).to(
            dtype=support_embeddings.dtype
        )
        prototypes = prototypes / counts[:, None]
        return prototypes, classes

    def forward(
        self,
        support_inputs: Tensor,
        support_labels: Tensor,
        query_inputs: Tensor,
        *,
        return_classes: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return query logits for one few-shot episode."""
        support_embeddings = self._encode(support_inputs)
        query_embeddings = self._encode(query_inputs)
        prototypes, classes = self.compute_prototypes(
            support_embeddings, support_labels
        )
        logits = -squared_euclidean_distance(query_embeddings, prototypes)
        return (logits, classes) if return_classes else logits

    @torch.no_grad()
    def predict(
        self, support_inputs: Tensor, support_labels: Tensor, query_inputs: Tensor
    ) -> Tensor:
        """Predict original class labels for every query example."""
        logits, classes = self(
            support_inputs, support_labels, query_inputs, return_classes=True
        )
        return classes[logits.argmax(dim=1)]


def prototypical_loss(
    model: PrototypicalNetwork,
    support_inputs: Tensor,
    support_labels: Tensor,
    query_inputs: Tensor,
    query_labels: Tensor,
) -> Tensor:
    """Compute cross-entropy loss for a single training episode."""
    logits, classes = model(
        support_inputs, support_labels, query_inputs, return_classes=True
    )
    query_labels = query_labels.reshape(-1).to(classes.device)
    if query_labels.numel() != logits.shape[0]:
        raise ValueError("each query input must have one label")

    matches = query_labels[:, None].eq(classes[None, :])
    if not matches.any(dim=1).all():
        raise ValueError("every query label must occur in the support set")
    targets = matches.to(torch.long).argmax(dim=1)
    return F.cross_entropy(logits, targets)


# A shorter alias commonly used in few-shot learning code.
ProtoNet = PrototypicalNetwork

