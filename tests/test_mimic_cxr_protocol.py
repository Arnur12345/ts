from __future__ import annotations

import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from mimic_cxr_protocol.protocol import (
    build_protocol,
    support_for_shot,
    validate_protocol,
)


def _write_gzip_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_synthetic_dataset(root: Path, config: dict) -> None:
    labels = config["labels"]
    class_to_partition = {
        class_name: partition
        for partition, class_names in config["class_partitions"].items()
        for class_name in class_names
    }
    official = config["official_split_for_partition"]
    metadata_rows: list[dict[str, object]] = []
    split_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    counter = 0

    def add_study(class_name: str | None, split: str, special: str = "") -> None:
        nonlocal counter
        subject_id = 10000000 + counter
        study_id = 50000000 + counter
        pa_dicom = f"dicom-{counter:06d}-pa"
        ap_dicom = f"dicom-{counter:06d}-ap"
        # Including both views verifies deterministic PA preference.
        for dicom_id, view in ((ap_dicom, "AP"), (pa_dicom, "PA")):
            metadata_rows.append(
                {
                    "dicom_id": dicom_id,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "ViewPosition": view,
                }
            )
            split_rows.append(
                {
                    "dicom_id": dicom_id,
                    "subject_id": subject_id,
                    "study_id": study_id,
                    "split": split,
                }
            )
        row: dict[str, object] = {"subject_id": subject_id, "study_id": study_id}
        row.update({label: "" for label in labels})
        if class_name is not None:
            row[class_name] = 1
        if special == "uncertain":
            row["Edema"] = -1
        elif special == "multi":
            row["Edema"] = 1
            row["Pneumonia"] = 1
        label_rows.append(row)
        counter += 1

    # Seven are required by the test config (5 support + 2 query); ten leave margin.
    for class_name in labels:
        split = official[class_to_partition[class_name]]
        for _ in range(10):
            add_study(class_name, split)
    add_study("Atelectasis", "train", "uncertain")
    add_study(None, "train", "multi")
    add_study(None, "train")
    # Eligible for the label audit, but excluded from the model-facing pool:
    # base classes are allowed only from official train patients.
    add_study("Atelectasis", "test")

    _write_gzip_csv(
        root / config["source_files"]["metadata"],
        ["dicom_id", "subject_id", "study_id", "ViewPosition"],
        metadata_rows,
    )
    _write_gzip_csv(
        root / config["source_files"]["official_split"],
        ["dicom_id", "subject_id", "study_id", "split"],
        split_rows,
    )
    _write_gzip_csv(
        root / config["source_files"]["labels"],
        ["subject_id", "study_id", *labels],
        label_rows,
    )


class ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        source_config = (
            Path(__file__).parents[1] / "configs" / "mimic_cxr_protocol_v1.json"
        )
        self.config = json.loads(source_config.read_text(encoding="utf-8"))
        self.config["episodes"]["queries_per_class"] = 2
        self.config["episodes"]["episodes_per_seed"] = 2
        self.config_path = self.root / "test_config.json"
        self.config_path.write_text(json.dumps(self.config), encoding="utf-8")
        self.data_root = self.root / "data"
        self.data_root.mkdir()
        _make_synthetic_dataset(self.data_root, self.config)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _build(self, name: str) -> Path:
        output = self.root / name
        build_protocol(
            data_root=self.data_root,
            output_dir=output,
            config_path=self.config_path,
            check_images=False,
        )
        return output

    def test_builds_valid_nested_patient_disjoint_episodes(self) -> None:
        output = self.root / "protocol"
        summary = build_protocol(
            data_root=self.data_root,
            output_dir=output,
            config_path=self.config_path,
            check_images=False,
        )

        self.assertEqual(summary["eligible_studies"], 141)
        self.assertEqual(summary["protocol_pool_studies"], 140)
        self.assertEqual(
            summary["exclusions"],
            {
                "multiple_positive_labels": 1,
                "no_positive_label": 1,
                "uncertain_label": 1,
            },
        )
        result = validate_protocol(output)
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["episode_files"], 15)
        self.assertEqual(result["episodes"], 30)

        episode_path = output / "episodes" / "validation_novel" / "seed_000.jsonl"
        episode = json.loads(episode_path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(len(support_for_shot(episode, 1)), 3)
        self.assertEqual(len(support_for_shot(episode, 3)), 9)
        self.assertEqual(len(support_for_shot(episode, 5)), 15)
        self.assertEqual(len(episode["query"]), 6)
        all_items = episode["support"] + episode["query"]
        self.assertEqual(len({item["subject_id"] for item in all_items}), len(all_items))
        self.assertTrue(all(item["view"] == "PA" for item in all_items))

    def test_episode_artifacts_are_byte_identical(self) -> None:
        outputs = [self._build("protocol_a"), self._build("protocol_b")]
        files_a = sorted(
            path.relative_to(outputs[0])
            for path in outputs[0].rglob("*")
            if path.is_file()
        )
        files_b = sorted(
            path.relative_to(outputs[1])
            for path in outputs[1].rglob("*")
            if path.is_file()
        )
        self.assertEqual(files_a, files_b)
        for relative_path in files_a:
            self.assertEqual(
                (outputs[0] / relative_path).read_bytes(),
                (outputs[1] / relative_path).read_bytes(),
            )

    def test_validator_detects_modified_episode(self) -> None:
        output = self._build("protocol")
        path = output / "episodes" / "test_novel" / "seed_000.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            validate_protocol(output)


if __name__ == "__main__":
    unittest.main()
