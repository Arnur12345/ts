from __future__ import annotations

import unittest
import json
import math
import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from experiments.iera.episodes import generate_pair_episodes, validate_pair_episodes
from experiments.iera.model import IERA, METHODS
from experiments.iera.patch_cache import MODEL, extract_patch_tokens, load_patch_cache
from experiments.iera.run import _decision, _meta_split, _metrics
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
        with self.assertRaisesRegex(ValueError, "needs 21 patients"):
            generate_pair_episodes(
                data, torch.arange(len(data.labels)), 0, 1, 1, 5, 2,
                seed=9, min_stratum_patients=21,
            )

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

    def test_self_exclusion_masks_the_complete_source_image(self) -> None:
        model = IERA(3, 3)
        tokens = torch.nn.functional.normalize(torch.tensor([[[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]]), dim=-1)
        bank = torch.nn.functional.normalize(
            torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0]]]),
            dim=-1,
        )
        observed = model._lme(tokens, bank, self_image_offset=0)
        similarity = torch.einsum("bnpd,bad->bnpa", tokens, bank[:, 2:]) / model._positive(model.raw_tau)
        expected = torch.logsumexp(similarity, -1) - math.log(2)
        torch.testing.assert_close(observed, expected)

    def test_frozen_prototype_does_not_use_iera_projection(self) -> None:
        generator = torch.Generator().manual_seed(12)
        positive = torch.randn(1, 2, 2, 4, 6, generator=generator)
        negative = torch.randn(1, 2, 2, 4, 6, generator=generator)
        query = torch.randn(1, 3, 4, 6, generator=generator)
        model = IERA(6, 3)
        before = model(positive, negative, query, "positive_prototype")
        with torch.no_grad():
            model.projection.weight.zero_()
            model.raw_gamma.fill_(20)
        after = model(positive, negative, query, "positive_prototype")
        torch.testing.assert_close(before, after)

    def test_patch_cache_requires_complete_consistent_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (2, 4, 3)
            token_path = root / "tokens.bin"
            torch.zeros(math.prod(shape), dtype=torch.float16).numpy().tofile(token_path)
            metadata = {
                "tokens": token_path.name, "shape": list(shape), "dtype": "float16",
                "pool_grid": 2, "manifest_sha256": "manifest", "model": MODEL,
                "completed": 2, "complete": False,
            }
            (root / "patch_cache.json").write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                load_patch_cache(root, "manifest")
            metadata["complete"] = True
            (root / "patch_cache.json").write_text(json.dumps(metadata), encoding="utf-8")
            tokens, _ = load_patch_cache(root, "manifest", expected_pool_grid=2)
            self.assertEqual(tuple(tokens.shape), shape)

    def test_sms_is_independent_of_calibration_temperature(self) -> None:
        logits = torch.tensor([-1.0, -0.5, 0.5, 1.0])
        panel_zero = torch.tensor([-1.0, -0.2, 0.3, 0.8])
        panel_one = torch.tensor([-0.5, 0.2, 0.7, 1.1])
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
        nuisance = torch.tensor([0, 1, 0, 1])
        cold = _metrics(logits, panel_zero, panel_one, targets, nuisance, 0.1, 0.5)
        warm = _metrics(logits, panel_zero, panel_one, targets, nuisance, 10.0, 0.5)
        self.assertEqual(cold["sms_raw_logit"], warm["sms_raw_logit"])
        self.assertEqual(cold["sms_normalized_logit"], warm["sms_normalized_logit"])

    def test_meta_early_stop_split_is_patient_disjoint(self) -> None:
        data = _data()
        # Add a second study per patient and prove both studies stay together.
        original = len(data.subject_ids)
        data.subject_ids.extend(data.subject_ids.copy())
        train, validation = _meta_split(data, torch.arange(original * 2), split_seed=17)
        train_subjects = {data.subject_ids[index] for index in train.tolist()}
        validation_subjects = {data.subject_ids[index] for index in validation.tolist()}
        self.assertFalse(train_subjects & validation_subjects)

    def test_decision_requires_consistency_across_both_pairs(self) -> None:
        rows = []
        values = {
            "positive_prototype": (1.0, 0.60, 0.70),
            "iera": (0.8, 0.65, 0.71),
            "iera_no_negatives": (0.9, 0.63, 0.70),
            "iera_mean_env": (0.95, 0.61, 0.70),
        }
        for pair in ("pair_a", "pair_b"):
            for method, (sms, worst, auroc) in values.items():
                for metric, mean in (
                    ("sms_normalized_logit", sms),
                    ("worst_nuisance_auroc", worst),
                    ("auroc", auroc),
                ):
                    rows.append({"pair": pair, "method": method, "shot": 3, "metric": metric, "mean": mean})
        decision = _decision(rows)
        self.assertEqual(decision["required_pairs"], 2)
        self.assertEqual(decision["status"], "continue_full_iera")


if __name__ == "__main__":
    unittest.main()
