from __future__ import annotations

import unittest
import hashlib
import json
import math
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from experiments.iera.detector_diagnostic import (
    _decision as _detector_decision,
    _detector_logits,
    _pool,
)
from experiments.iera.dual_head import (
    dual_scores,
    fused_score,
    selected_local_prototypes,
    support_adapter,
)
from experiments.iera.evidence_field import (
    evidence_field_grid,
    evidence_field_score,
    image_match,
)
from experiments.iera.evidence_field_diagnostic import (
    _compact as _compact_evidence_supports,
    _decision as _evidence_field_decision,
)
from experiments.iera.episodes import generate_pair_episodes, validate_pair_episodes
from experiments.iera.model import IERA, METHODS
from experiments.iera.patch_cache import (
    MODEL,
    episode_cache_indices,
    extract_patch_tokens,
    extract_rad_dino_patch_tokens,
    load_patch_cache,
)
from experiments.iera.run import (
    _checkpoint_key,
    _configure_optimizer,
    _decision,
    _meta_split,
    _metrics,
    _normalized_consistency,
    _objective,
    _update_lagrange,
)
from experiments.iera.robust_metrics import (
    normalized_sms as fixed_reference_sms,
    ranking_disagreement,
)
from experiments.iera.robust_model import RobustBinaryModel, project_direction
from experiments.iera.robust_support import (
    balanced_choices,
    environment_choices,
    select_supports,
)
from experiments.iera.representation_adaptation import (
    RadDinoTail,
    configure_tail,
)
from experiments.iera.linear_probe import four_group_weights
from experiments.iera.stable_witness import (
    border_maximum,
    certified_witness_scores,
    dn4_hard_knn_score,
    relational_descriptor,
    witness_confidence,
)
from experiments.iera.stable_witness_diagnostic import (
    _decision as _witness_decision,
)
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


class _RadModel(nn.Module):
    def forward(self, pixel_values):
        batch = len(pixel_values)
        # One CLS token plus a native 4x4 patch grid.
        hidden = torch.arange(
            batch * 17 * 8, dtype=torch.float32
        ).reshape(batch, 17, 8)
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeAttention(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)

    def forward(self, values):
        return self.query(values) + self.key(values) + self.value(values)


class _FakeBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attention = _FakeAttention(width)

    def forward(self, hidden):
        return (hidden + self.attention(hidden),)


class IERATest(unittest.TestCase):
    def test_four_group_probe_weights_equalize_total_group_mass(self) -> None:
        import numpy as np

        target = np.asarray([0, 0, 0, 1, 1, 1, 1])
        device = np.asarray([0, 0, 1, 0, 1, 1, 1])
        weights, counts = four_group_weights(target, device)
        groups = 2 * target + device
        totals = np.asarray(
            [weights[groups == group].sum() for group in range(4)]
        )
        np.testing.assert_allclose(totals, np.repeat(len(groups) / 4, 4))
        np.testing.assert_array_equal(counts, np.asarray([2, 1, 1, 3]))

    def test_last_block_and_lora_configuration_are_strictly_bounded(self) -> None:
        layers = nn.ModuleList((_FakeBlock(4), _FakeBlock(4)))
        last1 = configure_tail(layers, nn.LayerNorm(4), "last1", 2, 2)
        self.assertFalse(any(p.requires_grad for p in last1.layers[0].parameters()))
        self.assertTrue(any(p.requires_grad for p in last1.layers[1].parameters()))
        lora = configure_tail(layers, nn.LayerNorm(4), "lora2", 2, 2)
        trainable = [
            name for name, parameter in lora.named_parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(trainable)
        self.assertTrue(all(".down." in name or ".up." in name for name in trainable))
        output = lora(torch.randn(2, 5, 4))
        self.assertEqual(tuple(output.shape), (2, 4, 4))
        torch.testing.assert_close(
            output.norm(dim=-1), torch.ones(2, 4), atol=1e-5, rtol=1e-5
        )

    def test_fixed_relational_descriptor_preserves_constant_regions(self) -> None:
        token = torch.tensor([1.0, 2.0, 3.0])
        tokens = token.expand(2, 9, 3).clone()
        observed = relational_descriptor(tokens)
        expected = torch.nn.functional.normalize(tokens, dim=-1)
        torch.testing.assert_close(observed, expected)

    def test_witness_confidence_rewards_cross_image_class_specificity(self) -> None:
        same = torch.tensor(
            [[
                [[1.0, 0.0], [0.0, 1.0]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]]
        )
        opposite = torch.tensor(
            [[
                [[0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]]
        )
        confidence = witness_confidence(same, opposite, token_chunk_size=1)
        torch.testing.assert_close(
            confidence,
            torch.tensor([[[1.0, 0.0], [1.0, 0.0]]]),
        )

    def test_frozen_witness_and_dn4_scores_return_patch_fields(self) -> None:
        positive = torch.tensor(
            [[
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
                [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            ]]
        )
        negative = torch.tensor(
            [[
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
            ]]
        )
        query = positive[:, :1]
        witnesses = certified_witness_scores(
            positive,
            negative,
            query,
            (0.25, 0.5),
            token_chunk_size=2,
        )
        self.assertGreater(float(witnesses[0.25][0][0, 0]), 0.9)
        self.assertEqual(tuple(witnesses[0.5][1].shape), (1, 1, 4))
        dn4, field = dn4_hard_knn_score(
            positive, negative, query, neighbours=3
        )
        self.assertGreater(float(dn4[0, 0]), 0.9)
        self.assertTrue(bool(border_maximum(field)[0, 0]))

    def test_stage1_witness_gate_never_starts_training_automatically(self) -> None:
        rows = []
        for seed in range(5):
            for method, values in {
                "anchor_rho03": {
                    "auroc": 0.541,
                    "auprc": 0.50,
                    "sms_fixed_reference": 0.30,
                    "worst_nuisance_auroc": 0.50,
                    "support_swap_flip_rate": 0.10,
                    "border_max_fraction": 0.20,
                },
                "anchor_plus_relational_witness": {
                    "auroc": 0.56,
                    "auprc": 0.52,
                    "sms_fixed_reference": 0.31,
                    "worst_nuisance_auroc": 0.50,
                    "support_swap_flip_rate": 0.09,
                    "border_max_fraction": 0.15,
                },
            }.items():
                for metric, value in values.items():
                    rows.append(
                        {
                            "partition": "test",
                            "method": method,
                            "seed": seed,
                            "metric": metric,
                            "value": value,
                        }
                    )
        pending = _witness_decision(rows, "pending")
        self.assertEqual(pending["status"], "await_witness_evidence_review")
        self.assertFalse(pending["stage_two_training_started"])
        passed = _witness_decision(rows, "pass")
        self.assertEqual(
            passed["status"], "proceed_to_stage2_descriptor_training"
        )

    def test_image_match_is_log_mean_exp_per_support_image(self) -> None:
        query = torch.nn.functional.normalize(
            torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]), dim=-1
        )
        support = torch.nn.functional.normalize(
            torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]]), dim=-1
        )
        observed = image_match(query, support, tau=0.1)
        similarity = torch.einsum(
            "bqtd,bikpd->bqtikp", query, support
        )
        expected = 0.1 * (
            torch.logsumexp(similarity / 0.1, dim=-1) - math.log(2)
        )
        torch.testing.assert_close(observed, expected)
        self.assertEqual(tuple(observed.shape), (1, 1, 2, 1, 1))

    def test_evidence_field_grid_matches_single_scores_and_chunks(self) -> None:
        generator = torch.Generator().manual_seed(103)
        positive = torch.randn(2, 2, 3, 4, 6, generator=generator)
        negative = torch.randn(2, 2, 3, 4, 6, generator=generator)
        query = torch.randn(2, 5, 4, 6, generator=generator)
        mask = torch.tensor(
            [
                [[True, True, False], [True, False, False]],
                [[True, False, False], [True, True, False]],
            ]
        )
        grid = evidence_field_grid(
            positive,
            negative,
            query,
            mask,
            mask,
            (0.05, 0.1),
            (0.1, 0.2),
            pooling_modes=("image_balanced", "dense"),
            query_chunk_size=2,
        )
        for mode in ("image_balanced", "dense"):
            observed = grid[(mode, 0.1, 0.2)]
            expected = evidence_field_score(
                positive,
                negative,
                query,
                mask,
                mask,
                0.1,
                0.2,
                image_balanced=mode == "image_balanced",
                query_chunk_size=5,
            )
            torch.testing.assert_close(observed[0], expected[0])
            torch.testing.assert_close(observed[1], expected[1])
            self.assertEqual(tuple(observed[1].shape), (2, 5, 4))

    def test_evidence_field_support_adapter_is_differentiable(self) -> None:
        generator = torch.Generator().manual_seed(104)
        positive = torch.randn(1, 1, 2, 4, 6, generator=generator)
        negative = torch.randn(1, 1, 2, 4, 6, generator=generator)
        query = torch.randn(1, 3, 4, 6, generator=generator)
        mask = torch.ones(1, 1, 2, dtype=torch.bool)
        model = RobustBinaryModel(6, adapter_dim=3)
        with torch.no_grad():
            model.support_up.weight.normal_()
        logits, field = evidence_field_score(
            positive,
            negative,
            query,
            mask,
            mask,
            0.1,
            0.1,
            adapter=model,
            query_chunk_size=1,
        )
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertEqual(tuple(field.shape), (1, 3, 4))
        logits.sum().backward()
        self.assertIsNotNone(model.support_up.weight.grad)

    def test_evidence_support_compaction_removes_only_padding(self) -> None:
        tokens = torch.arange(1 * 2 * 3 * 2 * 2.0).reshape(1, 2, 3, 2, 2)
        mask = torch.tensor([[[True, False, True], [False, True, False]]])
        compact, compact_mask = _compact_evidence_supports(tokens, mask)
        self.assertEqual(tuple(compact.shape), (1, 1, 3, 2, 2))
        self.assertTrue(compact_mask.all())
        torch.testing.assert_close(
            compact.flatten(0, 2),
            tokens.flatten(1, 2)[0, mask.flatten()].reshape(3, 2, 2),
        )

    def test_evidence_field_gate_blocks_training_until_visual_review(self) -> None:
        rows = []
        for seed in range(5):
            for method, auroc, sms, worst in (
                ("current_adapter", 0.55, 1.0, 0.50),
                ("evidence_field_frozen_adapter", 0.58, 1.1, 0.50),
            ):
                for metric, value in (
                    ("auroc", auroc),
                    ("sms_fixed_reference", sms),
                    ("worst_nuisance_auroc", worst),
                ):
                    rows.append(
                        {
                            "partition": "test",
                            "pair": "Pneumothorax__Support Devices",
                            "method": method,
                            "shot": 3,
                            "seed": seed,
                            "metric": metric,
                            "value": value,
                        }
                    )
        pending = _evidence_field_decision(rows, 3, 14, 37, "pending")
        self.assertEqual(pending["status"], "await_evidence_map_review")
        passed = _evidence_field_decision(rows, 3, 14, 37, "pass")
        self.assertEqual(
            passed["status"], "proceed_to_counterfactual_field_training"
        )
        self.assertFalse(passed["stage_two_training_started"])

    def test_dual_head_fusion_has_exact_endpoints(self) -> None:
        global_score = torch.tensor([[-1.0, 2.0]])
        local_score = torch.tensor([[3.0, 0.5]])
        torch.testing.assert_close(
            fused_score(global_score, local_score, 0.0), global_score
        )
        torch.testing.assert_close(
            fused_score(global_score, local_score, 1.0), local_score
        )

    def test_selected_local_prototypes_follow_binary_direction(self) -> None:
        positive = torch.tensor(
            [[[[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]]]]
        )
        negative = torch.tensor(
            [[[[[-1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]]]]
        )
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        positive_local, negative_local = selected_local_prototypes(
            positive, negative, mask, mask, temperature=0.05
        )
        self.assertGreater(float(positive_local[0, 0]), 0.99)
        self.assertLess(float(negative_local[0, 0]), -0.99)

    def test_dual_head_uses_frozen_support_adapter_and_binary_scores(self) -> None:
        model = RobustBinaryModel(2, adapter_dim=1)
        positive = torch.tensor([[[[[1.0, 0.0]]]]])
        negative = torch.tensor([[[[[0.0, 1.0]]]]])
        query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        mask = torch.ones(1, 1, 1, dtype=torch.bool)
        adapted_positive = support_adapter(positive, model)
        adapted_negative = support_adapter(negative, model)
        scores = dual_scores(
            adapted_positive,
            adapted_negative,
            query,
            mask,
            mask,
            patch_temperature=0.1,
        )
        torch.testing.assert_close(
            scores["global"], torch.tensor([[1.0, -1.0]])
        )
        self.assertTrue(torch.isfinite(scores["selected_local"]).all())

    def test_robust_methods_share_binary_localized_detector(self) -> None:
        generator = torch.Generator().manual_seed(101)
        positive = torch.randn(2, 2, 3, 9, 8, generator=generator)
        negative = torch.randn(2, 2, 3, 9, 8, generator=generator)
        query = torch.randn(2, 4, 9, 8, generator=generator)
        model = RobustBinaryModel(8, adapter_dim=4)
        for method in (
            "uniform", "rex", "adapter_only", "anchor_only", "full_iera"
        ):
            logits = model(positive, negative, query, method)
            self.assertEqual(tuple(logits.shape), (2, 4))
            self.assertTrue(torch.isfinite(logits).all())
        changed = model(
            positive, -negative, query, "full_iera"
        )
        self.assertFalse(
            torch.allclose(
                model(positive, negative, query, "full_iera"), changed
            )
        )

    def test_iera_proposal_pooling_preserves_full_query_grid(self) -> None:
        generator = torch.Generator().manual_seed(102)
        positive = torch.randn(1, 2, 2, 196, 8, generator=generator)
        negative = torch.randn(1, 2, 2, 196, 8, generator=generator)
        query = torch.randn(1, 4, 196, 8, generator=generator)
        model = RobustBinaryModel(8, adapter_dim=4, proposal_grid=4)
        self.assertEqual(
            tuple(model._proposal_tokens(positive).shape),
            (1, 2, 2, 16, 8),
        )
        logits = model(positive, negative, query, "full_iera")
        self.assertEqual(tuple(logits.shape), (1, 4))
        self.assertTrue(torch.isfinite(logits).all())

    def test_direction_projection_broadcasts_over_episode_tokens(self) -> None:
        tokens = torch.randn(2, 2, 3, 4, 8)
        direction = torch.randn(8)
        projected = project_direction(tokens, direction)
        self.assertEqual(projected.shape, tokens.shape)
        normalized = torch.nn.functional.normalize(direction, dim=0)
        residual = torch.einsum("...d,d->...", projected, normalized)
        torch.testing.assert_close(residual, torch.zeros_like(residual))

    def test_robust_support_mask_ignores_padding(self) -> None:
        panels = torch.arange(2 * 4 * 2.0).reshape(1, 2, 4, 1, 2)
        choices = torch.tensor([[0, 0, 1, 0]])
        selected, mask = select_supports(panels, choices, 4)
        model = RobustBinaryModel(2)
        prototype = model._prototype(selected, mask)
        valid = selected[mask].reshape(-1, 2).mean(0)
        torch.testing.assert_close(
            prototype,
            torch.nn.functional.normalize(valid, dim=0).unsqueeze(0),
        )
        self.assertEqual(mask.sum(dim=2).tolist(), [[3, 1]])

    def test_balanced_and_empirical_support_choices_are_deterministic(self) -> None:
        empirical = environment_choices(1000, 6, 0.8, 7)
        self.assertGreater(float(empirical.float().mean()), 0.75)
        torch.testing.assert_close(
            empirical, environment_choices(1000, 6, 0.8, 7)
        )
        balanced = balanced_choices(5, 6, 9)
        self.assertTrue(balanced.sum(dim=1).eq(3).all())

    def test_fixed_reference_sms_keeps_the_reference_denominator_fixed(self) -> None:
        zero = torch.tensor([-1.0, 0.0, 1.0])
        one = zero + 0.5
        reference_zero = torch.tensor([-2.0, 0.0, 2.0])
        reference_one = reference_zero + 1.0
        base = fixed_reference_sms(
            zero, one, reference_zero, reference_one
        )
        rescaled_method = fixed_reference_sms(
            zero * 0.1, one * 0.1, reference_zero, reference_one
        )
        self.assertAlmostEqual(float(rescaled_method), float(base) * 0.1)
        self.assertEqual(
            ranking_disagreement(zero, torch.flip(zero, dims=(0,))), 1.0
        )

    def test_component_ablation_trainable_parameters_are_disjoint(self) -> None:
        model = RobustBinaryModel(8, adapter_dim=4)
        selected = {
            id(parameter)
            for parameter in model.configure_trainable("adapter_only")
        }
        adapter = {
            name for name, parameter in model.named_parameters()
            if id(parameter) in selected
        }
        selected = {
            id(parameter)
            for parameter in model.configure_trainable("anchor_only")
        }
        anchor = {
            name for name, parameter in model.named_parameters()
            if id(parameter) in selected
        }
        self.assertTrue(all(name.startswith("support_") for name in adapter))
        self.assertTrue(all(not name.startswith("support_") for name in anchor))
        self.assertFalse(adapter & anchor)

    def test_patch_extraction_removes_prefix_and_pools(self) -> None:
        tokens = extract_patch_tokens(_Model(), torch.randn(2, 3, 224, 224), pool_grid=7)
        self.assertEqual(tuple(tokens.shape), (2, 49, 8))
        self.assertTrue(torch.isfinite(tokens).all())

    def test_rad_dino_patch_extraction_removes_cls_and_pools(self) -> None:
        tokens = extract_rad_dino_patch_tokens(
            _RadModel(), torch.randn(2, 3, 8, 8), pool_grid=2
        )
        self.assertEqual(tuple(tokens.shape), (2, 4, 8))
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
        before = model(positive, negative, query, "frozen_protonet")
        with torch.no_grad():
            model.projection.weight.zero_()
            model.raw_gamma.fill_(20)
        after = model(positive, negative, query, "frozen_protonet")
        torch.testing.assert_close(before, after)

    def test_binary_heads_use_negative_supports(self) -> None:
        generator = torch.Generator().manual_seed(13)
        positive = torch.randn(1, 2, 2, 4, 6, generator=generator)
        negative = torch.randn(1, 2, 2, 4, 6, generator=generator)
        changed_negative = -negative
        query = torch.randn(1, 3, 4, 6, generator=generator)
        model = IERA(6, 4)
        for method in ("frozen_protonet", "learned_uniform"):
            before = model(positive, negative, query, method)
            after = model(positive, changed_negative, query, method)
            self.assertFalse(torch.allclose(before, after))

    def test_detector_binary_score_subtracts_negative_prototype(self) -> None:
        positive = torch.tensor([[[[[1.0, 0.0]]]]])
        negative = torch.tensor([[[[[0.0, 1.0]]]]])
        query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
        positive_only = _detector_logits(
            positive, negative, query, "positive_only"
        )
        binary = _detector_logits(
            positive, negative, query, "binary_protonet"
        )
        torch.testing.assert_close(positive_only, torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(binary, torch.tensor([[1.0, -1.0]]))

    def test_detector_can_pool_native_tokens_to_factorial_grid(self) -> None:
        tokens = torch.arange(16.0).reshape(1, 16, 1)
        pooled = _pool(tokens, source_grid=4, retained_grid=2)
        self.assertEqual(tuple(pooled.shape), (1, 4, 1))
        torch.testing.assert_close(pooled.norm(dim=-1), torch.ones(1, 4))

    def test_detector_decision_selects_on_validate_partition(self) -> None:
        rows = []
        for partition, mean, low, high in (
            ("validate", 0.70, 0.65, 0.75),
            ("test", 0.68, 0.60, 0.76),
        ):
            rows.append(
                {
                    "partition": partition,
                    "cache": "biomedclip",
                    "model": "model",
                    "source_grid": 14,
                    "retained_grid": 14,
                    "head": "binary_protonet",
                    "shot": 3,
                    "metric": "auroc",
                    "mean": mean,
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
        decision = _detector_decision(rows, primary_shot=3)
        self.assertEqual(
            decision["status"], "credible_pneumothorax_detector"
        )

    def test_anchor_weight_is_support_dependent_and_bounded(self) -> None:
        generator = torch.Generator().manual_seed(21)
        positive = torch.randn(3, 2, 2, 4, 6, generator=generator)
        negative = torch.randn(3, 2, 2, 4, 6, generator=generator)
        model = IERA(6, 4, alpha_max=0.25)
        with torch.no_grad():
            model.raw_anchor_slope.fill_(2.0)
        alpha = model.anchor_weight(positive, negative)
        self.assertEqual(tuple(alpha.shape), (3,))
        self.assertTrue(alpha.ge(0).all())
        self.assertTrue(alpha.le(0.25).all())

    def test_normalized_consistency_is_zero_for_identical_panels(self) -> None:
        panel = torch.tensor([[0.1, 0.5, -0.2, 1.0]])
        self.assertEqual(float(_normalized_consistency(panel, panel)), 0.0)

    def test_training_sms_exactly_matches_evaluated_normalized_sms(self) -> None:
        panel_zero = torch.tensor([[-1.0, -0.2, 0.3, 0.8]])
        panel_one = torch.tensor([[-0.5, 0.2, 0.7, 1.1]])
        targets = torch.tensor([0.0, 0.0, 1.0, 1.0])
        nuisance = torch.tensor([0, 1, 0, 1])
        evaluated = _metrics(
            torch.zeros(4), panel_zero.flatten(), panel_one.flatten(),
            targets, nuisance, 1.0, 0.5,
        )
        self.assertAlmostEqual(
            float(_normalized_consistency(panel_zero, panel_one)),
            evaluated["sms_normalized_logit"],
            places=7,
        )

    def test_anchored_objective_uses_fixed_uniform_budget(self) -> None:
        generator = torch.Generator().manual_seed(31)
        positive = torch.randn(1, 2, 2, 4, 6, generator=generator)
        negative = torch.randn(1, 2, 2, 4, 6, generator=generator)
        query = torch.randn(1, 4, 4, 6, generator=generator)
        targets = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
        model = IERA(6, 4, alpha_max=0.25)
        reference = IERA(6, 4, alpha_max=0.25).eval().requires_grad_(False)
        args = SimpleNamespace(lagrange_initial=1.0, invariance_budget=0.7)
        components = _objective(
            model, "anchored_iera", positive, negative, query, targets, args,
            uniform_reference_model=reference,
        )
        expected = (
            components["classification"]
            + args.lagrange_initial * components["budget_excess"]
        )
        torch.testing.assert_close(components["total"], expected)
        self.assertGreaterEqual(float(components["budget_excess"]), 0.0)
        components["total"].backward()
        self.assertIsNotNone(model.raw_anchor_bias.grad)
        self.assertTrue(all(parameter.grad is None for parameter in reference.parameters()))

    def test_anchored_optimizer_initially_freezes_uniform_head(self) -> None:
        model = IERA(6, 4)
        args = SimpleNamespace(learning_rate=1e-3)
        optimizer = _configure_optimizer(model, "anchored_iera", args)
        parameters = dict(model.named_parameters())
        self.assertFalse(parameters["projection.weight"].requires_grad)
        self.assertFalse(parameters["raw_tau_query"].requires_grad)
        self.assertFalse(parameters["raw_gamma"].requires_grad)
        self.assertTrue(parameters["raw_tau_attention"].requires_grad)
        self.assertTrue(parameters["raw_anchor_bias"].requires_grad)
        self.assertTrue(parameters["support_adapter_up.weight"].requires_grad)
        self.assertEqual(len(optimizer.param_groups), 1)

    def test_support_adapter_is_anchored_only(self) -> None:
        generator = torch.Generator().manual_seed(44)
        positive = torch.randn(1, 2, 2, 4, 6, generator=generator)
        negative = torch.randn(1, 2, 2, 4, 6, generator=generator)
        query = torch.randn(1, 4, 4, 6, generator=generator)
        model = IERA(6, 4, support_adapter_dim=2)
        uniform_before = model(positive, negative, query, "learned_uniform")
        anchored_before = model(positive, negative, query, "anchored_iera")
        with torch.no_grad():
            model.support_adapter_up.weight.normal_()
        uniform_after = model(positive, negative, query, "learned_uniform")
        anchored_after = model(positive, negative, query, "anchored_iera")
        torch.testing.assert_close(uniform_before, uniform_after)
        self.assertFalse(torch.allclose(anchored_before, anchored_after))

    def test_lagrange_multiplier_adapts_to_constraint(self) -> None:
        increased = _update_lagrange(1.0, 0.5, learning_rate=0.1, maximum=10.0)
        decreased = _update_lagrange(increased, -0.25, learning_rate=0.1, maximum=10.0)
        self.assertGreater(increased, 1.0)
        self.assertLess(decreased, increased)

    def test_anchored_checkpoint_prefers_feasible_worst_auc(self) -> None:
        infeasible = {
            "total": 0.1, "sms_budget_satisfied": 0.0,
            "max_sms_budget_ratio": 1.01, "worst_nuisance_auroc": 0.99,
        }
        feasible_low = {
            "total": 0.8, "sms_budget_satisfied": 1.0,
            "max_sms_budget_ratio": 0.9, "worst_nuisance_auroc": 0.60,
        }
        feasible_high = {
            "total": 1.2, "sms_budget_satisfied": 1.0,
            "max_sms_budget_ratio": 0.8, "worst_nuisance_auroc": 0.70,
        }
        self.assertGreater(
            _checkpoint_key("anchored_iera", feasible_low),
            _checkpoint_key("anchored_iera", infeasible),
        )
        self.assertGreater(
            _checkpoint_key("anchored_iera", feasible_high),
            _checkpoint_key("anchored_iera", feasible_low),
        )

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
            streamed, _ = load_patch_cache(
                root, "manifest", expected_pool_grid=2, access_mode="stream"
            )
            self.assertEqual(tuple(streamed[torch.tensor([1, 0])].shape), shape)
            streamed.close()

    def test_sparse_patch_cache_resolves_global_episode_indices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shape = (2, 4, 3)
            token_path = root / "tokens.bin"
            tokens = torch.arange(math.prod(shape), dtype=torch.float16).reshape(
                shape
            )
            tokens.numpy().tofile(token_path)
            index_path = root / "global_indices.int64.bin"
            indices = torch.tensor([7, 2], dtype=torch.int64)
            indices.numpy().tofile(index_path)
            metadata = {
                "tokens": token_path.name,
                "shape": list(shape),
                "dtype": "float16",
                "pool_grid": 2,
                "manifest_sha256": "manifest",
                "model": MODEL,
                "completed": 2,
                "complete": True,
                "index_mode": "sparse",
                "dataset_size": 10,
                "global_indices": index_path.name,
                "global_indices_sha256": hashlib.sha256(
                    index_path.read_bytes()
                ).hexdigest(),
            }
            (root / "patch_cache.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            streamed, _ = load_patch_cache(
                root, "manifest", expected_pool_grid=2, access_mode="stream"
            )
            torch.testing.assert_close(
                streamed[torch.tensor([2, 7])],
                torch.stack((tokens[1], tokens[0])),
            )
            with self.assertRaisesRegex(IndexError, "absent from sparse cache"):
                streamed[torch.tensor([3])]
            streamed.close()

    def test_episode_cache_selects_only_required_scoring_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            episode_path = Path(temporary) / "episodes.pt"
            episodes = {
                "positive": torch.arange(8).reshape(1, 2, 4),
                "negative": torch.arange(8, 16).reshape(1, 2, 4),
                "query": torch.tensor([[16, 17]]),
                "random_positive_env": torch.tensor([[1, 1]]),
                "random_negative_env": torch.tensor([[0, 0]]),
            }
            torch.save(
                {
                    "signature": {
                        "manifest_sha256": "manifest",
                        "seeds": [0],
                        "episodes": 1,
                    },
                    "pairs": {
                        0: ("Pneumothorax", "Support Devices"),
                        1: ("Edema", "Cardiomegaly"),
                    },
                    "episodes": {
                        (0, 0, "validate"): episodes,
                        (0, 0, "test"): episodes,
                        (1, 0, "validate"): episodes,
                        (1, 0, "test"): episodes,
                    },
                },
                episode_path,
            )
            indices, protocol = episode_cache_indices(
                episode_path,
                "manifest",
                seeds=[0],
                targets=["Pneumothorax"],
                episode_count=1,
                shots=[1],
            )
            self.assertEqual(
                indices,
                [0, 1, 4, 5, 8, 9, 12, 13, 16, 17],
            )
            self.assertEqual(protocol["episode_pair_ids"], [0])
            self.assertEqual(protocol["episode_shots"], [1])

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
            "frozen_protonet": (1.1, 0.58, 0.68),
            "learned_uniform": (1.0, 0.60, 0.70),
            "iera": (0.9, 0.64, 0.71),
            "anchored_iera": (0.7, 0.65, 0.71),
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
        self.assertEqual(decision["status"], "continue_anchored_iera")

    def test_decision_allows_one_point_auroc_loss(self) -> None:
        rows = []
        for pair in ("pair_a", "pair_b"):
            for method, values in {
                "frozen_protonet": (1.1, 0.58, 0.68),
                "learned_uniform": (1.0, 0.70, 0.75),
                "iera": (0.9, 0.69, 0.74),
                "anchored_iera": (0.69, 0.69, 0.74),
            }.items():
                for metric, mean in zip(
                    ("sms_normalized_logit", "worst_nuisance_auroc", "auroc"),
                    values,
                ):
                    rows.append(
                        {
                            "pair": pair, "method": method, "shot": 3,
                            "metric": metric, "mean": mean,
                        }
                    )
        decision = _decision(rows)
        self.assertEqual(decision["status"], "continue_anchored_iera")
        self.assertEqual(decision["auroc_tolerance"], 0.01)
        self.assertEqual(decision["required_sms_ratio"], 0.7)

    def test_infeasible_base_validation_blocks_high_resolution(self) -> None:
        rows = []
        pair = "Pneumothorax__Support Devices"
        for method, values in {
            "frozen_protonet": (1.1, 0.58, 0.68),
            "learned_uniform": (1.0, 0.70, 0.75),
            "iera": (0.9, 0.69, 0.74),
            "anchored_iera": (0.8, 0.70, 0.75),
        }.items():
            for metric, mean in zip(
                ("sms_normalized_logit", "worst_nuisance_auroc", "auroc"),
                values,
            ):
                rows.append(
                    {
                        "pair": pair, "method": method, "shot": 3,
                        "metric": metric, "mean": mean,
                    }
                )
        decision = _decision(
            rows, patch_grid=4, base_validation_feasible=False
        )
        self.assertEqual(
            decision["status"], "constraint_infeasible_on_base_validation"
        )


if __name__ == "__main__":
    unittest.main()
