from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable


CHEXPERT_LABELS = (
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
)

STUDY_RE = re.compile(r"(?:^|/)s(\d+)(?:/|$)")


def _id(value: object) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.removeprefix("s")


def _number(value: object) -> float:
    text = str(value).strip()
    return 0.0 if text in {"", "nan", "None"} else float(text)


def _paths(value: str | None) -> list[str]:
    if not value or value.strip().lower() in {"", "nan", "none", "[]"}:
        return []
    parsed = ast.literal_eval(value)
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a list of image paths, got: {value[:80]!r}")
    return [str(item).replace("\\", "/").lstrip("/") for item in parsed]


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", newline="", encoding="utf-8-sig")
    return path.open(newline="", encoding="utf-8-sig")


def load_single_labels(
    labels_csv: Path, uncertain_policy: str
) -> tuple[dict[str, str], Counter[str]]:
    labels: dict[str, str] = {}
    stats: Counter[str] = Counter()
    with _open_csv(labels_csv) as handle:
        reader = csv.DictReader(handle)
        missing = {"study_id", *CHEXPERT_LABELS}.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{labels_csv} is missing required columns: {sorted(missing)}"
            )
        for row in reader:
            stats["label_rows"] += 1
            values = {name: _number(row[name]) for name in CHEXPERT_LABELS}
            if uncertain_policy == "drop" and any(value == -1 for value in values.values()):
                stats["dropped_uncertain_studies"] += 1
                continue
            positives = [name for name, value in values.items() if value == 1]
            if len(positives) != 1:
                stats["dropped_not_single_label_studies"] += 1
                continue
            labels[_id(row["study_id"])] = positives[0]
            stats["single_label_studies"] += 1
    return labels, stats


def collect_candidates(
    input_csv: Path | list[Path], labels: dict[str, str]
) -> tuple[list[dict[str, str]], Counter[str]]:
    candidates: list[dict[str, str]] = []
    stats: Counter[str] = Counter()
    seen: set[str] = set()
    input_csvs = [input_csv] if isinstance(input_csv, Path) else input_csv
    for csv_path in input_csvs:
        with _open_csv(csv_path) as handle:
            reader = csv.DictReader(handle)
            fields = set(reader.fieldnames or [])
            metadata_fields = {"dicom_id", "subject_id", "study_id", "ViewPosition"}
            if metadata_fields.issubset(fields):
                for row in reader:
                    stats["input_rows"] += 1
                    view = str(row["ViewPosition"]).strip().upper()
                    if view not in {"AP", "PA"}:
                        stats["dropped_non_frontal"] += 1
                        continue
                    subject_id = _id(row["subject_id"])
                    study_id = _id(row["study_id"])
                    dicom_id = str(row["dicom_id"]).strip().removesuffix(".jpg")
                    relative_path = (
                        f"files/p{subject_id[:2]}/p{subject_id}/"
                        f"s{study_id}/{dicom_id}.jpg"
                    )
                    stats["frontal_paths"] += 1
                    label = labels.get(study_id)
                    if label is None:
                        stats["dropped_without_single_label"] += 1
                        continue
                    if relative_path in seen:
                        stats["dropped_duplicate_path"] += 1
                        continue
                    seen.add(relative_path)
                    candidates.append(
                        {
                            "image_id": dicom_id,
                            "subject_id": subject_id,
                            "study_id": study_id,
                            "label": label,
                            "view": view,
                            "source_path": relative_path,
                        }
                    )
                continue
            missing = {"subject_id", "AP", "PA"}.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path} is missing required columns: {sorted(missing)}")
            for row in reader:
                stats["input_rows"] += 1
                for view in ("AP", "PA"):
                    for relative_path in _paths(row.get(view)):
                        stats["frontal_paths"] += 1
                        match = STUDY_RE.search(relative_path)
                        if not match:
                            stats["dropped_bad_path"] += 1
                            continue
                        study_id = _id(match.group(1))
                        label = labels.get(study_id)
                        if label is None:
                            stats["dropped_without_single_label"] += 1
                            continue
                        if relative_path in seen:
                            stats["dropped_duplicate_path"] += 1
                            continue
                        seen.add(relative_path)
                        candidates.append(
                            {
                                "image_id": Path(relative_path).stem,
                                "subject_id": _id(row["subject_id"]),
                                "study_id": study_id,
                                "label": label,
                                "view": view,
                                "source_path": relative_path,
                            }
                        )
    return candidates, stats


def cap_classes(
    rows: list[dict[str, str]], max_per_class: int, seed: int
) -> list[dict[str, str]]:
    if max_per_class <= 0:
        return rows
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)
    rng = random.Random(seed)
    kept: list[dict[str, str]] = []
    for label in sorted(grouped):
        group = grouped[label]
        rng.shuffle(group)
        kept.extend(group[:max_per_class])
    kept.sort(key=lambda row: row["source_path"])
    return kept


def _resize_one(task: tuple[str, str, int, bool]) -> tuple[bool, str]:
    source, destination, size, overwrite = task
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        return True, ""
    try:
        from PIL import Image, ImageOps

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = ImageOps.exif_transpose(image).convert("L")
            image.thumbnail((size, size), Image.Resampling.BICUBIC)
            canvas = Image.new("L", (size, size), color=0)
            offset = ((size - image.width) // 2, (size - image.height) // 2)
            canvas.paste(image, offset)
            temporary = destination_path.with_suffix(destination_path.suffix + ".tmp")
            canvas.save(temporary, format="JPEG", quality=92, optimize=True)
            os.replace(temporary, destination_path)
        return True, ""
    except Exception as error:  # the caller records corrupt/missing images
        return False, f"{source}: {type(error).__name__}: {error}"


def prepare(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir.resolve()
    image_dir = output_dir / f"images_{args.size}"
    output_dir.mkdir(parents=True, exist_ok=True)

    study_labels, label_stats = load_single_labels(
        args.labels_csv, args.uncertain_policy
    )
    candidates, path_stats = collect_candidates(args.input_csv, study_labels)
    before_cap = len(candidates)
    candidates = cap_classes(candidates, args.max_per_class, args.seed)

    tasks: list[tuple[str, str, int, bool]] = []
    for row in candidates:
        source = args.data_root / row["source_path"]
        destination = image_dir / row["source_path"]
        row["resized_path"] = destination.relative_to(output_dir).as_posix()
        tasks.append((str(source), str(destination), args.size, args.overwrite))

    valid: list[dict[str, str]] = []
    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for row, (ok, error) in zip(
            candidates,
            executor.map(_resize_one, tasks, chunksize=32),
            strict=True,
        ):
            if ok:
                valid.append(row)
            else:
                errors.append(error)

    manifest = output_dir / "manifest.csv"
    fieldnames = (
        "image_id",
        "subject_id",
        "study_id",
        "label",
        "view",
        "source_path",
        "resized_path",
    )
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid)

    class_counts = Counter(row["label"] for row in valid)
    summary: dict[str, object] = {
        "input_csv": [str(path.resolve()) for path in args.input_csv],
        "labels_csv": str(args.labels_csv.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(manifest),
        "image_size": args.size,
        "uncertain_policy": args.uncertain_policy,
        "max_per_class": args.max_per_class,
        "candidates_before_cap": before_cap,
        "valid_images": len(valid),
        "failed_images": len(errors),
        "class_counts": dict(sorted(class_counts.items())),
        "label_stats": dict(label_stats),
        "path_stats": dict(path_stats),
        "first_errors": errors[:20],
    }
    with (output_dir / "prepare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a frontal, single-label MIMIC-CXR manifest and 224px cache."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        nargs="+",
        default=[Path("Untitled.csv")],
        help="Official metadata CSV(.gz), or one or more aggregated AP/PA CSVs.",
    )
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Archive root containing paths such as files/p10/...",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument(
        "--uncertain-policy",
        choices=("drop", "negative"),
        default="drop",
        help="Drop studies containing -1 labels (safer) or treat -1 as not positive.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Optional deterministic cap; 0 keeps all data. Episodes remain balanced either way.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = prepare(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
