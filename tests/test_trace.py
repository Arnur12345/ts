from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from experiments.trace.core import (
    apply_shrinkage_precision,
    canonical_pathology_atom,
    consecutive_transitions,
    consecutive_transitions_from_canonical_timeline,
    covariance_eigendecomposition,
    localization_statistics,
    register_translation,
    select_transition_pairs,
    select_support_indices,
    study_timestamp,
    transition_counts,
    transition_feature_batch,
)
from experiments.trace.evaluation import (
    classification_metrics,
    score_episode_bank,
)
from experiments.trace import evaluation as trace_evaluation


class TraceTest(unittest.TestCase):
    def test_chronology_uses_dates_and_never_study_id_order(self) -> None:
        rows = [
            {
                "subject_id": "1",
                "study_id": "999",
                "StudyDate": "20200101",
                "StudyTime": "080000",
                "ViewPosition": "AP",
            },
            {
                "subject_id": "1",
                "study_id": "100",
                "StudyDate": "20200102",
                "StudyTime": "080000",
                "ViewPosition": "AP",
            },
            {
                "subject_id": "1",
                "study_id": "500",
                "StudyDate": "20200103",
                "StudyTime": "080000",
                "ViewPosition": "PA",
            },
        ]
        labels = torch.tensor([[0, 0], [1, 0], [0, 1]], dtype=torch.bool)
        known = torch.ones_like(labels)
        pairs = consecutive_transitions(
            rows, ["1", "1", "1"], labels, known, 0, 1, range(3)
        )
        self.assertEqual(
            [(pair.before, pair.after) for pair in pairs],
            [(0, 1), (1, 2)],
        )
        self.assertEqual(
            transition_counts(pairs)["disease_change_device_stable"], 1
        )
        self.assertTrue(pairs[1].view_changed)
        self.assertEqual(study_timestamp(rows[0]), (20200101, 80000.0))

    def test_unknown_intermediate_study_is_not_skipped(self) -> None:
        rows = [
            {
                "StudyDate": f"2020010{index + 1}",
                "StudyTime": "0",
                "study_id": str(index),
            }
            for index in range(3)
        ]
        labels = torch.tensor([[0, 0], [0, 0], [1, 0]], dtype=torch.bool)
        known = torch.ones_like(labels)
        known[1, 0] = False
        with self.assertRaisesRegex(ValueError, "no consecutive pair"):
            consecutive_transitions(
                rows, ["1", "1", "1"], labels, known, 0, 1, range(3)
            )

    def test_filtered_endpoint_cannot_jump_over_canonical_intermediate(self) -> None:
        canonical = [
            {
                "subject_id": "1",
                "study_id": str(index + 10),
                "dicom_id": f"d{index}",
                "StudyDate": f"2020010{index + 1}",
                "StudyTime": "0",
                "view": "AP",
            }
            for index in range(3)
        ]
        cached = [canonical[0], canonical[2]]
        labels = torch.tensor([[0], [1]], dtype=torch.bool)
        known = torch.ones_like(labels)
        transitions = {
            ("10", "11"): "stable_absent",
            ("11", "12"): "stable_absent",
            # This annotation must not make 10->12 consecutive.
            ("10", "12"): "stable_absent",
        }
        with self.assertRaisesRegex(ValueError, "raw-consecutive"):
            consecutive_transitions_from_canonical_timeline(
                canonical,
                cached,
                ["1", "1"],
                labels,
                known,
                target_id=0,
                allowed_subjects={"1"},
                intervention_transitions=transitions,
            )

    def test_simultaneous_disease_device_changes_are_excluded(self) -> None:
        rows = [
            {
                "StudyDate": f"2020010{index + 1}",
                "StudyTime": "0",
                "study_id": str(index),
            }
            for index in range(5)
        ]
        labels = torch.tensor(
            [[0, 0], [1, 0], [1, 1], [1, 1], [0, 0]],
            dtype=torch.bool,
        )
        known = torch.ones_like(labels)
        pairs = consecutive_transitions(
            rows, ["1"] * 5, labels, known, 0, 1, range(5)
        )
        selected = select_transition_pairs(pairs, 10, seed=3)
        self.assertNotIn("both_change", [pair.stratum for pair in selected])

    def test_integer_registration_recovers_known_shift(self) -> None:
        torch.manual_seed(7)
        grid, width = 6, 12
        before = F.normalize(torch.randn(grid, grid, width), dim=-1)
        after = torch.roll(before, shifts=(1, -1), dims=(0, 1))
        aligned, valid, shift, score = register_translation(
            before.flatten(0, 1), after.flatten(0, 1), grid, max_shift=2
        )
        self.assertEqual(tuple(shift.tolist()), (1, -1))
        self.assertGreater(float(score), 0.99)
        original = before.flatten(0, 1)
        torch.testing.assert_close(
            aligned[valid], original[valid], atol=1e-5, rtol=1e-5
        )
        extracted = transition_feature_batch(
            original, after.flatten(0, 1), grid, max_shift=2
        )
        self.assertEqual(tuple(extracted["features"].shape), (width + grid**2,))

    def test_canonical_atom_orients_onset_and_resolution_together(self) -> None:
        residual = torch.tensor(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]
        )
        atom = canonical_pathology_atom(
            residual,
            torch.tensor([0, 1, 0]),
            torch.tensor([1, 0, 1]),
            torch.tensor([0, 0, 1]),
        )
        torch.testing.assert_close(atom, torch.tensor([1.0, 0.0]))

    def test_unlabeled_covariance_precision_downweights_high_variance_axis(self) -> None:
        features = np.asarray(
            [
                [-10.0, -1.0],
                [-5.0, 1.0],
                [5.0, -1.0],
                [10.0, 1.0],
            ],
            dtype=np.float32,
        )
        _, eigenvalues, eigenvectors = covariance_eigendecomposition(
            features, torch.arange(4), torch.device("cpu"), batch_size=2
        )
        direction = apply_shrinkage_precision(
            torch.tensor([[1.0, 1.0]]),
            eigenvalues,
            eigenvectors,
            ridge=0.01,
        )[0]
        self.assertGreater(abs(float(direction[1])), abs(float(direction[0])))

    def test_saved_episode_scoring_uses_random_supports_and_fixed_sms(self) -> None:
        features = np.asarray(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.8, 0.2],
                [0.7, 0.3],
                [-1.0, 0.0],
                [-0.9, -0.1],
                [-0.8, -0.2],
                [-0.7, -0.3],
                [1.0, 0.0],
                [-1.0, 0.0],
                [0.8, 0.2],
                [-0.8, -0.2],
            ],
            dtype=np.float32,
        )
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        episodes = {
            "positive": torch.tensor([[[0, 1], [2, 3]]]),
            "negative": torch.tensor([[[4, 5], [6, 7]]]),
            "random_positive_env": torch.tensor([[0, 1]]),
            "random_negative_env": torch.tensor([[0, 1]]),
            "query": torch.tensor([[8, 9, 10, 11]]),
            "targets": torch.tensor([[1.0, 0.0, 1.0, 0.0]]),
            "nuisance": torch.tensor([[0, 0, 1, 1]]),
        }
        with patch.object(
            trace_evaluation,
            "select_support_indices",
            wraps=select_support_indices,
        ) as selector:
            scores = score_episode_bank(
                features, episodes, shot=1, device=torch.device("cpu")
            )
        self.assertEqual([call.args[2] for call in selector.call_args_list], [1, 1])
        metrics = classification_metrics(
            scores, episodes["targets"], episodes["nuisance"]
        )
        self.assertEqual(tuple(scores["logits"].shape), (4,))
        self.assertGreater(metrics["auroc"], 0.99)
        self.assertGreaterEqual(metrics["sms_fixed_reference"], 0.0)

    def test_localization_reports_border_and_pleural_proxy_separately(self) -> None:
        grid = 7
        values = torch.zeros(2, grid * grid)
        values[0, 0] = 2
        values[1, grid + 1] = 2
        stats = localization_statistics(
            values, ["disease", "disease"], grid
        )["disease"]
        self.assertEqual(stats["border_max_fraction"], 0.5)
        self.assertEqual(stats["pleural_proxy_max_fraction"], 0.5)

    def test_support_index_selection_tracks_each_environment_cursor(self) -> None:
        panels = torch.tensor([[[10, 11, 12], [20, 21, 22]]])
        choices = torch.tensor([[1, 0, 1, 1]])
        selected = select_support_indices(panels, choices, 4)
        torch.testing.assert_close(
            selected, torch.tensor([[20, 10, 21, 22]])
        )


if __name__ == "__main__":
    unittest.main()
