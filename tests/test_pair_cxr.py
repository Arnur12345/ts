from __future__ import annotations

import unittest

import torch

from experiments.pair_cxr.model import (
    PAIRRouter,
    intervention_loss,
    normalized_entropy,
)


class PAIRCXRTest(unittest.TestCase):
    def test_router_returns_normalized_patch_evidence(self) -> None:
        torch.manual_seed(3)
        model = PAIRRouter(width=8, bottleneck=3)
        logits, weights = model(torch.randn(5, 9, 8), torch.randn(8))
        self.assertEqual(tuple(logits.shape), (5,))
        self.assertEqual(tuple(weights.shape), (5, 9))
        torch.testing.assert_close(
            weights.sum(-1), torch.ones(5), atol=1e-6, rtol=1e-6
        )
        entropy = normalized_entropy(weights)
        self.assertGreaterEqual(float(entropy), 0.0)
        self.assertLessEqual(float(entropy), 1.0)

    def test_intervention_objective_uses_all_four_groups(self) -> None:
        logits = torch.tensor([-2.0, -1.5, 1.5, 2.0], requires_grad=True)
        targets = torch.tensor([0, 0, 1, 1])
        devices = torch.tensor([0, 1, 0, 1])
        loss, components = intervention_loss(
            logits,
            targets,
            devices,
            beta_rex=0.1,
            lambda_invariance=0.3,
            lambda_responsiveness=0.3,
            minimum_margin=0.2,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(
            set(components),
            {"classification", "rex", "invariance", "responsiveness"},
        )
        loss.backward()
        self.assertIsNotNone(logits.grad)


if __name__ == "__main__":
    unittest.main()
