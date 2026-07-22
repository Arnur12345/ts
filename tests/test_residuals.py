from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from experiments.residuals.data import ResidualDataset
from experiments.residuals.episodes import batch, generate_episodes, validate_episodes
from experiments.residuals.model import METHODS, pair_fsl_logits


def _dataset() -> ResidualDataset:
    labels = []
    for class_id in range(3):
        for _ in range(20):
            row = [False, False, False]
            row[class_id] = True
            labels.append(row)
    for first, second in ((0, 1), (1, 2), (0, 2)):
        for _ in range(10):
            row = [False, False, False]
            row[first] = row[second] = True
            labels.append(row)
    label_tensor = torch.tensor(labels)
    count = len(labels)
    images = F.normalize(torch.randn(count, 12, generator=torch.Generator().manual_seed(7)), dim=-1)
    rows = [
        {
            "dicom_id": f"d{index}",
            "subject_id": str(10_000 + index),
            "official_split": "validate",
            "view": "PA" if index % 2 else "AP",
        }
        for index in range(count)
    ]
    return ResidualDataset(
        images=images,
        labels=label_tensor,
        known=torch.ones_like(label_tensor),
        metadata=torch.randn(count, 8, generator=torch.Generator().manual_seed(8)),
        class_names=["A", "B", "C"],
        subject_ids=[row["subject_id"] for row in rows],
        dicom_ids=[row["dicom_id"] for row in rows],
        rows=rows,
        manifest_sha256="synthetic",
    )


class ResidualExperimentTest(unittest.TestCase):
    def test_both_regimes_are_patient_disjoint_and_have_expected_targets(self) -> None:
        data = _dataset()
        indices = torch.arange(len(data.images))
        for regime in ("single_label", "multi_label"):
            episodes = generate_episodes(data, indices, [0, 1, 2], regime, 2, 2, 2, 1, seed=11)
            validate_episodes(episodes, data)
            self.assertEqual(tuple(episodes["positive"].shape), (2, 3, 2))
            self.assertEqual(tuple(episodes["negative"].shape), (2, 3, 2))
            if regime == "single_label":
                self.assertEqual(tuple(episodes["targets"].shape), (2, 3))
            else:
                self.assertEqual(tuple(episodes["targets"].shape), (2, 6, 3))

    def test_every_residual_arm_scores_both_episode_types(self) -> None:
        data = _dataset()
        indices = torch.arange(len(data.images))
        for regime in ("single_label", "multi_label"):
            episodes = generate_episodes(data, indices, [0, 1, 2], regime, 2, 3, 4, 1, seed=19)
            values = batch(data, episodes, 3, 4, torch.device("cpu"))
            for method in METHODS:
                logits = pair_fsl_logits(
                    values["positive"], values["negative"], values["query"], method,
                    values["positive_metadata"], values["negative_metadata"], values["query_metadata"],
                )
                self.assertEqual(logits.shape[:2], values["query"].shape[:2])
                self.assertEqual(logits.shape[2], 3)
                self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
