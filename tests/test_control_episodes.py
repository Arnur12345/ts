from __future__ import annotations

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # The protocol itself does not require PyTorch.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ControlEpisodeTest(unittest.TestCase):
    def test_nested_patient_disjoint_rotations_and_saved_ids(self) -> None:
        from experiments.control_episodes import load_or_create, write_episode_ids

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "pairs.csv"
            manifest.write_text("dicom_id\nsynthetic\n", encoding="utf-8")
            count = 14 * 7
            data = {
                "labels": torch.arange(14).repeat_interleave(7),
                "class_names": [f"class_{i}" for i in range(14)],
                "subject_ids": torch.arange(10_000, 10_000 + count),
                "dicom_ids": [f"d{i}" for i in range(count)],
            }
            saved = load_or_create(
                root / "episodes.pt", data, manifest,
                episode_count=4, seeds=(0, 1), fold_count=5,
            )

            covered = set()
            for fold in saved["folds"]:
                covered.update(fold["test_class_ids"])
                for partition in ("validation_novel", "test_novel"):
                    for run in fold[partition]["runs"]:
                        for episode in range(4):
                            support = run["support"][episode]
                            query = run["query"][episode]
                            self.assertEqual(len(set(support.flatten().tolist() + query.flatten().tolist())), 18)
                            for class_position in range(3):
                                one = set(support[class_position, :1].tolist())
                                three = set(support[class_position, :3].tolist())
                                five = set(support[class_position, :5].tolist())
                                self.assertTrue(one < three < five)
            self.assertEqual(covered, set(range(14)))

            output = root / "episode_ids.csv.gz"
            write_episode_ids(output, saved, data)
            with gzip.open(output, "rt", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 5 * 2 * 2 * 4 * 3)
            self.assertEqual(len({row["episode_id"] for row in rows}), 5 * 2 * 2 * 4)


if __name__ == "__main__":
    unittest.main()
