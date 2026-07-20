from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from experiments.control_decision import write_decision


class ControlDecisionTest(unittest.TestCase):
    def test_genuine_support_branch_uses_ten_paired_seeds(self) -> None:
        rows = []
        values = {
            "text_only": 0.55,
            "visual_protonet": 0.70,
            "protonet_text": 0.75,
            "visual_protonet_permuted_support_labels": 0.40,
            "visual_protonet_duplicated_support": 0.60,
        }
        for fold in range(2):
            for seed in range(10):
                for method, value in values.items():
                    rows.append({"method": method, "fold": fold, "seed": seed, "shot": 5, "metric": "auroc", "value": value})
                rows.append({"method": "protonet_text", "fold": fold, "seed": seed, "shot": 5, "metric": "calibration_error", "value": 0.1})
        with tempfile.TemporaryDirectory() as temporary:
            report = write_decision(Path(temporary), rows)
        self.assertEqual(report["decision"], "abandon_text_dominance_claim")
        self.assertEqual(report["five_shot_evidence"]["permuted_labels"]["n"], 10)


if __name__ == "__main__":
    unittest.main()
