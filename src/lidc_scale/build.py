from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .core import (
    Candidate,
    canonical_json,
    median,
    pair_orientations,
    select_candidates,
    sha256_file,
    stable_key,
)
from .render import make_contact_sheet, make_pair, save_render_arms


DEFAULT_CONFIG = Path(__file__).parents[2] / "configs" / "lidc_scale_pilot_v1.json"


def _require_dependencies() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import pylidc as pl
        from pylidc.utils import consensus
    except ImportError as error:
        raise SystemExit(
            'Missing LIDC dependencies. Install with: pip install -e ".[lidc-scale]"'
        ) from error
    return np, pl, consensus


def configure_pylidc_root(data_root: Path) -> None:
    """Set pylidc's DICOM root in-process without changing ~/.pylidcrc."""
    scan_module = importlib.import_module("pylidc.Scan")
    scan_module._get_dicom_file_path_from_config_file = lambda: str(data_root.resolve())


def enumerate_candidates(pl: Any, selection: dict[str, Any]) -> list[Candidate]:
    candidates: list[Candidate] = []
    minimum = float(selection["min_diameter_mm"])
    maximum = float(selection["max_diameter_mm"])
    min_readers = int(selection["min_readers"])
    for scan in pl.query(pl.Scan).order_by(pl.Scan.id):
        for cluster_index, annotations in enumerate(scan.cluster_annotations(verbose=False)):
            if len(annotations) < min_readers:
                continue
            reader_diameters = tuple(sorted(float(ann.diameter) for ann in annotations))
            diameter = median(reader_diameters)
            if not minimum <= diameter <= maximum:
                continue
            annotation_ids = tuple(sorted(int(ann.id) for ann in annotations))
            candidates.append(
                Candidate(
                    nodule_id=f"{scan.patient_id}_n{cluster_index:02d}",
                    scan_id=int(scan.id),
                    patient_id=str(scan.patient_id),
                    annotation_ids=annotation_ids,
                    reader_diameters_mm=reader_diameters,
                    diameter_mm=diameter,
                )
            )
    return candidates


def consensus_location(
    np: Any, consensus: Any, annotations: list[Any]
) -> tuple[int, tuple[float, float], int]:
    mask, bbox = consensus(
        annotations, clevel=0.5, pad=None, ret_masks=False, verbose=False
    )
    areas = mask.sum(axis=(0, 1))
    local_slice = int(np.argmax(areas))
    points = np.argwhere(mask[:, :, local_slice])
    if len(points) == 0:
        raise ValueError("empty 50% consensus mask")
    center_local = points.mean(axis=0)
    center = (
        float(bbox[0].start + center_local[0]),
        float(bbox[1].start + center_local[1]),
    )
    slice_index = int(bbox[2].start + local_slice)
    return slice_index, center, int(areas[local_slice])


def spacing_slug(spacing: float) -> str:
    return f"{spacing:.2f}".replace(".", "p")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def _prepare_output(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"refusing to overwrite non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def build_stimuli(
    data_root: Path, output_dir: Path, config_path: Path = DEFAULT_CONFIG
) -> dict[str, Any]:
    np, pl, consensus = _require_dependencies()
    if not data_root.is_dir():
        raise FileNotFoundError(f"LIDC DICOM root does not exist: {data_root}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _prepare_output(output_dir)
    configure_pylidc_root(data_root)

    all_candidates = enumerate_candidates(pl, config["selection"])
    selected, achieved = select_candidates(
        all_candidates,
        config["selection"]["diameter_bins"],
        int(config["selection"]["n_nodules"]),
        int(config["seed"]),
        int(config["selection"]["max_nodules_per_scan"]),
    )
    shutil.copyfile(config_path, output_dir / "config.json")

    nodule_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    image_lookup: dict[tuple[str, float, str], Path] = {}

    for index, candidate in enumerate(selected, start=1):
        scan = pl.query(pl.Scan).filter(pl.Scan.id == candidate.scan_id).one()
        annotation_by_id = {int(ann.id): ann for ann in scan.annotations}
        annotations = [annotation_by_id[value] for value in candidate.annotation_ids]
        slice_index, center, consensus_area_px = consensus_location(np, consensus, annotations)
        volume = scan.to_volume(verbose=False)
        axial_hu = volume[:, :, slice_index]
        nodule_dir = output_dir / "images" / candidate.nodule_id

        nodule_row = {
            **asdict(candidate),
            "annotation_ids": list(candidate.annotation_ids),
            "reader_diameters_mm": list(candidate.reader_diameters_mm),
            "rank": index,
            "source_spacing_mm": float(scan.pixel_spacing),
            "slice_thickness_mm": float(scan.slice_thickness),
            "slice_spacing_mm": float(scan.slice_spacing),
            "slice_index": slice_index,
            "center_row": center[0],
            "center_col": center[1],
            "consensus_area_source_px": consensus_area_px,
        }
        nodule_rows.append(nodule_row)

        for spacing in config["render"]["target_spacings_mm"]:
            spacing = float(spacing)
            slug = spacing_slug(spacing)
            marker_path = nodule_dir / f"spacing_{slug}_marker.png"
            bar_path = nodule_dir / f"spacing_{slug}_scale_bar.png"
            save_render_arms(
                axial_hu,
                center,
                float(scan.pixel_spacing),
                spacing,
                config["render"],
                marker_path,
                bar_path,
            )
            marker_relative = marker_path.relative_to(output_dir)
            bar_relative = bar_path.relative_to(output_dir)
            image_lookup[(candidate.nodule_id, spacing, "marker")] = marker_path
            image_lookup[(candidate.nodule_id, spacing, "scale_bar")] = bar_path
            image_rows.append(
                {
                    "stimulus_id": f"{candidate.nodule_id}_s{slug}",
                    "nodule_id": candidate.nodule_id,
                    "diameter_mm": candidate.diameter_mm,
                    "threshold_6mm": candidate.diameter_mm
                    >= float(config["questions"]["threshold_mm"]),
                    "target_spacing_mm": spacing,
                    "expected_diameter_px": candidate.diameter_mm / spacing,
                    "marker_image": str(marker_relative),
                    "scale_bar_image": str(bar_relative),
                    "marker_sha256": sha256_file(marker_path),
                    "scale_bar_sha256": sha256_file(bar_path),
                }
            )

        spacings = [float(value) for value in config["render"]["target_spacings_mm"]]
        for pair_index, (left, right) in enumerate(
            pair_orientations(spacings, candidate.nodule_id, int(config["seed"])),
            start=1,
        ):
            pair_dir = output_dir / "pairs" / candidate.nodule_id
            marker_pair = pair_dir / f"pair_{pair_index}_marker.png"
            bar_pair = pair_dir / f"pair_{pair_index}_scale_bar.png"
            make_pair(
                image_lookup[(candidate.nodule_id, left, "marker")],
                image_lookup[(candidate.nodule_id, right, "marker")],
                marker_pair,
            )
            make_pair(
                image_lookup[(candidate.nodule_id, left, "scale_bar")],
                image_lookup[(candidate.nodule_id, right, "scale_bar")],
                bar_pair,
            )
            pair_rows.append(
                {
                    "stimulus_id": f"{candidate.nodule_id}_pair{pair_index}",
                    "nodule_id": candidate.nodule_id,
                    "diameter_mm": candidate.diameter_mm,
                    "left_spacing_mm": left,
                    "right_spacing_mm": right,
                    "ground_truth_grew": False,
                    "marker_image": str(marker_pair.relative_to(output_dir)),
                    "scale_bar_image": str(bar_pair.relative_to(output_dir)),
                    "marker_sha256": sha256_file(marker_pair),
                    "scale_bar_sha256": sha256_file(bar_pair),
                }
            )

    _write_jsonl(output_dir / "nodules.jsonl", nodule_rows)
    _write_jsonl(output_dir / "images.jsonl", image_rows)
    _write_jsonl(output_dir / "pairs.jsonl", pair_rows)

    audit_nodules = sorted(
        (row["nodule_id"] for row in nodule_rows),
        key=lambda nodule_id: stable_key(int(config["seed"]), str(nodule_id)),
    )[:20]
    spacings = [float(value) for value in config["render"]["target_spacings_mm"]]
    rows_by_key = {
        (row["nodule_id"], float(row["target_spacing_mm"])): row for row in image_rows
    }
    audit_candidates = [
        rows_by_key[(nodule_id, spacings[index % len(spacings)])]
        for index, nodule_id in enumerate(audit_nodules)
    ]
    audit_items = [
        (
            output_dir / str(row["marker_image"]),
            f"{row['nodule_id'].removeprefix('LIDC-IDRI-')} {row['diameter_mm']:.1f}mm {row['target_spacing_mm']:.2f}mm/px",
        )
        for row in audit_candidates
    ]
    make_contact_sheet(audit_items, output_dir / "audit" / "contact_sheet_20.png")
    _write_jsonl(output_dir / "audit" / "audit_manifest.jsonl", audit_candidates)

    summary = {
        "version": config["version"],
        "status": "built_unreviewed",
        "candidate_nodules": len(all_candidates),
        "selected_nodules": len(nodule_rows),
        "images": len(image_rows),
        "growth_pairs": len(pair_rows),
        "bin_counts": achieved,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pylidc": getattr(pl, "__version__", "unknown"),
        "contact_sheet": "audit/contact_sheet_20.png",
    }
    (output_dir / "summary.json").write_text(canonical_json(summary) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the controlled LIDC scale-perception stimuli"
    )
    parser.add_argument(
        "--data-root", type=Path, required=True, help="LIDC-IDRI DICOM root"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    summary = build_stimuli(args.data_root, args.output_dir, args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
