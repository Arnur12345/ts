from __future__ import annotations

import argparse
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path

from subspace_fsl.prepare_data import (
    CHEXPERT_LABELS,
    _resize_one,
    collect_candidates,
    load_single_labels,
)


class PreparationTests(unittest.TestCase):
    def test_resize_letterboxes_without_cropping(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.jpg"
            destination = root / "nested" / "resized.jpg"
            Image.new("L", (40, 20), color=128).save(source)
            ok, error = _resize_one((str(source), str(destination), 32, False))
            self.assertTrue(ok, error)
            with Image.open(destination) as resized:
                self.assertEqual(resized.size, (32, 32))
                self.assertEqual(resized.mode, "L")

    def test_single_label_and_frontal_filter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.csv.gz"
            with gzip.open(labels_path, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["study_id", *CHEXPERT_LABELS])
                writer.writeheader()
                first = {name: "" for name in CHEXPERT_LABELS}
                first.update({"study_id": "10", "Atelectasis": "1"})
                writer.writerow(first)
                multi = {name: "" for name in CHEXPERT_LABELS}
                multi.update({"study_id": "20", "Edema": "1", "Pneumonia": "1"})
                writer.writerow(multi)

            input_path = root / "train.csv"
            with input_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject_id", "AP", "PA"])
                writer.writeheader()
                writer.writerow(
                    {
                        "subject_id": "1",
                        "AP": "['files/p00/p1/s10/a.jpg']",
                        "PA": "[]",
                    }
                )
            validation_path = root / "validation.csv"
            with validation_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["subject_id", "AP", "PA"])
                writer.writeheader()
                writer.writerow(
                    {
                        "subject_id": "1",
                        "AP": "[]",
                        "PA": "['files/p00/p1/s20/b.jpg']",
                    }
                )

            labels, _ = load_single_labels(labels_path, "drop")
            rows, _ = collect_candidates([input_path, validation_path], labels)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "Atelectasis")
            self.assertEqual(rows[0]["view"], "AP")

    def test_official_metadata_reconstructs_image_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            labels_path = root / "labels.csv.gz"
            with gzip.open(labels_path, "wt", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["study_id", *CHEXPERT_LABELS])
                writer.writeheader()
                row = {name: "" for name in CHEXPERT_LABELS}
                row.update({"study_id": "50414267", "Cardiomegaly": "1"})
                writer.writerow(row)

            metadata_path = root / "metadata.csv.gz"
            with gzip.open(metadata_path, "wt", newline="", encoding="utf-8") as handle:
                fields = ["dicom_id", "subject_id", "study_id", "ViewPosition"]
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "dicom_id": "02aa804e-bde0afdd",
                        "subject_id": "10000032",
                        "study_id": "50414267",
                        "ViewPosition": "PA",
                    }
                )
                writer.writerow(
                    {
                        "dicom_id": "lateral-image",
                        "subject_id": "10000032",
                        "study_id": "50414267",
                        "ViewPosition": "LATERAL",
                    }
                )

            labels, _ = load_single_labels(labels_path, "drop")
            rows, stats = collect_candidates(metadata_path, labels)
            self.assertEqual(len(rows), 1)
            self.assertEqual(
                rows[0]["source_path"],
                "files/p10/p10000032/s50414267/02aa804e-bde0afdd.jpg",
            )
            self.assertEqual(stats["dropped_non_frontal"], 1)


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest("PyTorch is not installed") from error

    def test_oracle_is_disjoint_and_end_to_end_runs(self):
        import torch

        from subspace_fsl.evaluate import make_episode_plan, run as geometry_run
        from subspace_fsl.evaluate_text import run as text_run

        indices = {0: torch.arange(30), 1: torch.arange(30, 60), 2: torch.arange(60, 90)}
        oracle, support, query = make_episode_plan(
            torch, indices, [0, 1, 2], oracle_size=8, episodes=5, shots=5, queries=1, seed=7
        )
        self.assertFalse(torch.isin(query, oracle).any().item())
        self.assertFalse(torch.isin(support, oracle).any().item())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = torch.Generator().manual_seed(9)
            dimension = 16
            per_class = 30
            centers = torch.eye(dimension)[:14]
            blocks = []
            labels = []
            for class_id in range(14):
                block = centers[class_id] + 0.08 * torch.randn(
                    per_class, dimension, generator=generator
                )
                blocks.append(torch.nn.functional.normalize(block, dim=-1))
                labels.extend([class_id] * per_class)
            embeddings = root / "embeddings.pt"
            torch.save(
                {
                    "features": torch.cat(blocks).half(),
                    "labels": torch.tensor(labels),
                    "class_names": list(CHEXPERT_LABELS),
                    "normalized": True,
                },
                embeddings,
            )
            output = root / "output"
            geometry_run(
                argparse.Namespace(
                    embeddings=embeddings,
                    output_dir=output,
                    device="cpu",
                    keep_features_cpu=True,
                    split_json=None,
                    split_seed=2026,
                    shots=[1, 3, 5],
                    queries=1,
                    episodes=3,
                    seeds=[0, 1],
                    oracle_size=8,
                    ranks=[1, 2, 4],
                    betas=[0.1, 0.5],
                    base_samples_per_class=20,
                    chunk_size=32,
                )
            )
            self.assertTrue((output / "per_seed_all_settings.csv").exists())
            with (output / "experiment.json").open(encoding="utf-8") as handle:
                experiment = json.load(handle)
            self.assertEqual(experiment["shots"], [1, 3, 5])
            self.assertEqual(experiment["queries_per_class"], 1)

            text_embeddings = root / "text_embeddings.pt"
            torch.save(
                {
                    "features": centers.float(),
                    "class_names": list(CHEXPERT_LABELS),
                    "descriptions": [f"a chest X-ray showing {name}" for name in CHEXPERT_LABELS],
                    "normalized": True,
                },
                text_embeddings,
            )
            text_output = root / "text_output"
            text_run(
                argparse.Namespace(
                    embeddings=embeddings,
                    text_embeddings=text_embeddings,
                    output_dir=text_output,
                    device="cpu",
                    keep_features_cpu=True,
                    split_json=None,
                    split_seed=2026,
                    shots=[1, 3, 5],
                    queries=1,
                    episodes=2,
                    seeds=[0, 1],
                    oracle_size=8,
                    ranks=[1, 2],
                    alphas=[0.0, 0.25],
                    betas=[0.0, 0.5],
                )
            )
            self.assertTrue((text_output / "per_seed_all_settings.csv").exists())
            self.assertTrue((text_output / "semantic_sanity_summary.csv").exists())
            with (text_output / "test_selected_summary.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 18)

    def test_hybrid_distance_retains_prototype_penalty(self):
        import torch

        from subspace_fsl.evaluate import distances

        query = torch.tensor([[[1.0, 0.0]]])
        prototype = torch.tensor([[[0.0, 0.0]]])
        basis = torch.tensor([[1.0], [0.0]])
        hybrid = distances(torch, query, prototype, basis, beta=0.25)
        self.assertAlmostEqual(hybrid.item(), 0.75, places=6)


if __name__ == "__main__":
    unittest.main()
