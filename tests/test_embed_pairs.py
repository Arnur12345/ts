from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from mimic_cxr_protocol.embed_pairs import select_pairs


class PairSelectionTest(unittest.TestCase):
    def test_selection_is_stratified_deterministic_and_patient_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reports = root / "mimic-cxr-reports-2.1.0"
            rows = []
            counter = 0
            for class_name, count in (("A", 2), ("B", 5), ("C", 5)):
                for _ in range(count):
                    subject = str(10000000 + counter)
                    study = str(50000000 + counter)
                    dicom = f"d{counter}"
                    image_path = Path("files") / f"{dicom}.jpg"
                    (root / image_path).parent.mkdir(parents=True, exist_ok=True)
                    (root / image_path).touch()
                    report_path = reports / "files" / f"p{subject[:2]}" / f"p{subject}" / f"s{study}.txt"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(f"original report {counter}", encoding="utf-8")
                    rows.append(
                        {
                            "dicom_id": dicom,
                            "study_id": study,
                            "subject_id": subject,
                            "relative_path": image_path.as_posix(),
                            "class_name": class_name,
                        }
                    )
                    counter += 1

            manifest = root / "protocol_samples.csv"
            with manifest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            first = select_pairs(manifest, root, reports, target=9, seed=2026)
            second = select_pairs(manifest, root, reports, target=9, seed=2026)
            self.assertEqual(first, second)
            self.assertEqual(len({row["subject_id"] for row in first}), 9)
            self.assertEqual(sum(row["class_name"] == "A" for row in first), 2)


if __name__ == "__main__":
    unittest.main()
