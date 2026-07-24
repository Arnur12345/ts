from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import torch

from experiments.iera.labels import restore_raw_target_status
from experiments.residuals.data import ResidualDataset


class IERALabelTest(unittest.TestCase):
    def test_blank_and_uncertain_are_unknown(self) -> None:
        rows = [{"study_id": "10", "subject_id": "1", "dicom_id": "d"}]
        data = ResidualDataset(
            images=torch.zeros(1, 2), labels=torch.zeros(1, 3, dtype=torch.bool),
            known=torch.ones(1, 3, dtype=torch.bool), metadata=torch.zeros(1, 8),
            class_names=["A", "B", "C"], subject_ids=["1"], dicom_ids=["d"],
            rows=rows, manifest_sha256="x",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "labels.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["study_id", "A", "B", "C"])
                writer.writeheader()
                writer.writerow({"study_id": "10", "A": "1.0", "B": "", "C": "-1.0"})
            restore_raw_target_status(data, path)
        self.assertEqual(data.labels.tolist(), [[True, False, False]])
        self.assertEqual(data.known.tolist(), [[True, False, False]])


if __name__ == "__main__":
    unittest.main()
