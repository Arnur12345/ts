from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from lidc_scale.audit import record_audit
from lidc_scale.build import _load_candidate_cache, _prepare_output, resolve_pylidc_root
from lidc_scale.core import Candidate, median, pair_orientations, parse_json_object, physical_grid, select_candidates
from lidc_scale.requests import build_requests, select_frontier_blocks
from lidc_scale.score import score_rows

IMAGING_AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "PIL", "scipy"))


class LIDCScaleCoreTest(unittest.TestCase):
    def test_nested_patient_root_is_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patient_parent = root / "manifest" / "LIDC-IDRI"
            (patient_parent / "LIDC-IDRI-0442").mkdir(parents=True)
            self.assertEqual(
                resolve_pylidc_root(root, "LIDC-IDRI-0442"),
                patient_parent.resolve(),
            )

    def test_matching_config_only_output_is_safe_to_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = {"version": "test"}
            (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
            _prepare_output(output, config)
            (output / "unexpected.png").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                _prepare_output(output, config)

    def test_candidate_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate_pool.jsonl"
            candidate = Candidate("n1", 1, "p1", (2, 3), (5.0, 6.0), 5.5)
            path.write_text(json.dumps({
                "nodule_id": candidate.nodule_id,
                "scan_id": candidate.scan_id,
                "patient_id": candidate.patient_id,
                "annotation_ids": candidate.annotation_ids,
                "reader_diameters_mm": candidate.reader_diameters_mm,
                "diameter_mm": candidate.diameter_mm,
            }) + "\n", encoding="utf-8")
            self.assertEqual(_load_candidate_cache(path), [candidate])

    def test_physical_grid_preserves_requested_spacing_and_center(self) -> None:
        rows, cols = physical_grid((100.25, 200.75), 0.5, 1.0, 5)
        self.assertEqual(rows, [96.25, 98.25, 100.25, 102.25, 104.25])
        self.assertEqual(cols, [196.75, 198.75, 200.75, 202.75, 204.75])

    def test_median_and_json_parser(self) -> None:
        self.assertEqual(median([8, 4, 6, 10]), 7)
        self.assertEqual(parse_json_object('```json\n{"mm": 7.5}\n```'), {"mm": 7.5})

    def test_selection_fulfils_quotas_and_scan_disjointness(self) -> None:
        bins = [
            {"name": "low", "lower": 4, "upper": 7, "quota": 2},
            {"name": "high", "lower": 7, "upper": 10, "quota": 2},
        ]
        candidates = [
            Candidate(f"n{i}", i, f"p{i}", (i,), (d,), d)
            for i, d in enumerate([4.5, 5.5, 7.5, 8.5, 9.0])
        ]
        selected, achieved = select_candidates(candidates, bins, 4, 42, 1)
        self.assertEqual(len(selected), 4)
        self.assertEqual(achieved, {"low": 2, "high": 2})
        self.assertEqual(len({item.scan_id for item in selected}), 4)

    def test_pair_orientations_cover_each_unordered_pair(self) -> None:
        pairs = pair_orientations([0.5, 0.75, 1.0], "n1", 2)
        self.assertEqual(len(pairs), 3)
        self.assertEqual({frozenset(pair) for pair in pairs}, {
            frozenset((0.5, 0.75)), frozenset((0.5, 1.0)), frozenset((0.75, 1.0))
        })


class RequestAndScoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        images = []
        pairs = []
        for nodule in ("n1", "n2", "n3", "n4", "n5", "n6"):
            for spacing in (0.5, 0.75, 1.0):
                images.append({
                    "stimulus_id": f"{nodule}_{spacing}", "nodule_id": nodule,
                    "diameter_mm": 7.0, "threshold_6mm": True,
                    "target_spacing_mm": spacing, "marker_image": "m.png", "scale_bar_image": "b.png",
                })
            for index, (left, right) in enumerate(((0.5, 0.75), (0.5, 1.0), (0.75, 1.0))):
                pairs.append({
                    "stimulus_id": f"{nodule}_p{index}", "nodule_id": nodule,
                    "diameter_mm": 7.0, "left_spacing_mm": left, "right_spacing_mm": right,
                    "marker_image": "pm.png", "scale_bar_image": "pb.png",
                })
        for name, rows in (("images.jsonl", images), ("pairs.jsonl", pairs)):
            (self.root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_factorial_and_frontier_blocks(self) -> None:
        rows = build_requests(self.root)
        self.assertEqual(len(rows), 486)  # 6 nodules * 81 requests each
        self.assertEqual({row["prompt_version"] for row in rows}, {"target-explicit-v2"})
        self.assertTrue(all("not" in row["prompt"] or "must not" in row["prompt"] for row in rows))
        self.assertTrue(all("red" in row["prompt"] for row in rows))
        frontier = select_frontier_blocks(rows, 7, blocks=6)
        self.assertEqual(len(frontier), 54)
        self.assertEqual(len({row["nodule_id"] for row in frontier}), 6)

    def test_score_separates_acquisition_and_paraphrase_variance(self) -> None:
        rows = []
        for spacing, prediction in zip((0.5, 0.75, 1.0), (5.0, 7.0, 9.0)):
            for paraphrase in (1, 2, 3):
                rows.append({
                    "model_id": "model", "question": "q1_absolute", "condition": "A_bare",
                    "nodule_id": "n1", "stimulus_id": f"n1_{spacing}", "paraphrase": paraphrase,
                    "diameter_mm": 7.0, "truth": {"mm": 7.0}, "response_text": json.dumps({"mm": prediction}),
                })
        report = score_rows(rows)
        cell = report["cells"][0]
        self.assertGreater(cell["acquisition"]["mean_group_variance"], 0)
        self.assertEqual(cell["paraphrase"]["mean_group_variance"], 0)
        self.assertEqual(cell["mae_mm"], 4 / 3)

    def test_audit_records_hashed_twenty_image_review(self) -> None:
        from lidc_scale.core import sha256_file

        audit_dir = self.root / "audit"
        audit_dir.mkdir()
        (audit_dir / "contact_sheet_20.png").write_bytes(b"sheet")
        rows = []
        for index in range(20):
            path = self.root / f"image_{index}.png"
            path.write_bytes(f"image-{index}".encode())
            rows.append({
                "nodule_id": f"n{index}",
                "marker_image": path.name,
                "marker_sha256": sha256_file(path),
            })
        (audit_dir / "audit_manifest.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        (self.root / "summary.json").write_text(json.dumps({"status": "built_unreviewed"}), encoding="utf-8")
        review = record_audit(self.root, "tester", "pass", "checked")
        self.assertEqual(review["decision"], "pass")
        summary = json.loads((self.root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "audit_passed")


@unittest.skipUnless(IMAGING_AVAILABLE, "optional LIDC imaging dependencies are not installed")
class RenderTest(unittest.TestCase):
    def test_pylidc_legacy_numpy_aliases_are_restored(self) -> None:
        import numpy as np

        from lidc_scale.build import patch_pylidc_numpy_compatibility

        patch_pylidc_numpy_compatibility(np)
        self.assertIs(np.int, int)
        self.assertIs(np.float, float)
        self.assertTrue("bool" in np.__dict__)

    def test_rendered_footprint_tracks_spacing_and_pair_is_square(self) -> None:
        import numpy as np
        from PIL import Image

        from lidc_scale.render import make_pair, resample_crop_hu, save_render_arms

        source = np.full((512, 512), -1000.0, dtype=np.float32)
        rows, cols = np.ogrid[:512, :512]
        source[(rows - 256) ** 2 + (cols - 256) ** 2 <= 10**2] = 100.0
        widths = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = {
                "output_size_px": 448,
                "outside_hu": -1350,
                "window_center_hu": -600,
                "window_width_hu": 1500,
                "marker": {"radius_px": 32, "width_px": 2, "color_rgb": [255, 80, 80]},
                "scale_bar": {"length_mm": 10, "width_px": 3, "margin_px": 24},
            }
            markers = []
            for spacing in (0.5, 0.75, 1.0):
                crop = resample_crop_hu(source, (256, 256), 0.5, spacing, 448, -1350)
                points = np.argwhere(crop > -500)
                widths.append(int(points[:, 1].max() - points[:, 1].min() + 1))
                marker = root / f"m_{spacing}.png"
                bar = root / f"b_{spacing}.png"
                save_render_arms(source, (256, 256), 0.5, spacing, config, marker, bar)
                with Image.open(marker) as rendered:
                    self.assertEqual(rendered.size, (448, 448))
                    self.assertEqual(rendered.mode, "RGB")
                markers.append(marker)
            self.assertGreater(widths[0], widths[1])
            self.assertGreater(widths[1], widths[2])
            pair = root / "pair.png"
            make_pair(markers[0], markers[2], pair)
            with Image.open(pair) as rendered_pair:
                self.assertEqual(rendered_pair.size, (896, 896))


if __name__ == "__main__":
    unittest.main()
