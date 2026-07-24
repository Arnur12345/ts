from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from experiments.iera.episodes import generate_pair_episodes, validate_pair_episodes
from experiments.iera.model import IERA, METHODS
from experiments.iera.patch_cache import extract_patch_tokens
from experiments.residuals.data import ResidualDataset


def _data() -> ResidualDataset:
    labels = []
    for target in (0, 1):
        for nuisance in (0, 1):
            labels.extend([[target, nuisance, (target + nuisance) % 2]] * 20)
    values = torch.tensor(labels, dtype=torch.bool)
    count = len(values)
    rows = [{"subject_id": str(1000 + i), "dicom_id": f"d{i}", "official_split": "test"} for i in range(count)]
    return ResidualDataset(
        images=torch.randn(count, 8), labels=values, known=torch.ones_like(values),
        metadata=torch.zeros(count, 8), class_names=["target", "nuisance", "other"],
        subject_ids=[row["subject_id"] for row in rows], dicom_ids=[row["dicom_id"] for row in rows],
        rows=rows, manifest_sha256="synthetic",
    )


class _Trunk(nn.Module):
    num_prefix_tokens = 1

    def forward_features(self, images):
        batch = len(images)
        return torch.arange(batch * 197 * 8, dtype=torch.float32).reshape(batch, 197, 8)


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = _Trunk()
        self.head = nn.Identity()


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()


class IERATest(unittest.TestCase):
    def test_patch_extraction_removes_prefix_and_pools(self) -> None:
        tokens = extract_patch_tokens(_Model(), torch.randn(2, 3, 224, 224), pool_grid=7)
        self.assertEqual(tuple(tokens.shape), (2, 49, 8))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_four_stratum_episodes_are_patient_disjoint(self) -> None:
        data = _data()
        episodes = generate_pair_episodes(data, torch.arange(len(data.labels)), 0, 1, 3, 5, 2, seed=9)
        validate_pair_episodes(episodes, data)
        self.assertEqual(tuple(episodes["positive"].shape), (3, 2, 5))
        self.assertEqual(tuple(episodes["query"].shape), (3, 8))

    def test_all_ablation_scores_are_finite_and_trainable(self) -> None:
        generator = torch.Generator().manual_seed(4)
        positive = torch.randn(2, 2, 3, 9, 12, generator=generator)
        negative = torch.randn(2, 2, 3, 9, 12, generator=generator)
        query = torch.randn(2, 8, 9, 12, generator=generator)
        model = IERA(12, 6)
        for method in METHODS:
            logits = model(positive, negative, query, method)
            self.assertEqual(tuple(logits.shape), (2, 8))
            self.assertTrue(torch.isfinite(logits).all())
        model(positive, negative, query, "iera").sum().backward()
        self.assertIsNotNone(model.projection.weight.grad)


if __name__ == "__main__":
    unittest.main()
