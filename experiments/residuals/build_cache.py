"""Build a BioMedCLIP image cache that retains MIMIC-CXR's native labels."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path


MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _patients(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    with _open_csv(path) as handle:
        rows = list(csv.DictReader(handle))
    return {str(int(float(row["subject_id"]))): row for row in rows}


def select_multilabel_rows(
    study_manifest: Path,
    data_root: Path,
    labels: list[str],
    patients_path: Path | None = None,
    limit: int | None = None,
    seed: int = 2026,
    check_images: bool = True,
) -> list[dict[str, str]]:
    """Keep certain-label canonical studies, including comorbid and negative controls."""
    patient_rows = _patients(patients_path)
    with _open_csv(study_manifest) as handle:
        reader = csv.DictReader(handle)
        missing = set(labels + ["dicom_id", "subject_id", "official_split", "relative_path"]) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"study manifest is missing columns: {sorted(missing)}")
        selected = []
        for source in reader:
            values = [int(float(source[name] or 0)) for name in labels]
            if -1 in values or source.get("exclusion_reason") == "missing_label_row":
                continue
            if check_images and not (data_root / source["relative_path"]).is_file():
                continue
            row = dict(source)
            patient = patient_rows.get(str(int(float(row["subject_id"]))), {})
            row["gender"] = patient.get("gender", patient.get("sex", ""))
            row["age"] = patient.get("anchor_age", patient.get("age", ""))
            selected.append(row)
    if not selected:
        raise ValueError("no certain-label studies with images were found")
    if limit is not None and len(selected) > limit:
        random.Random(seed).shuffle(selected)
        selected = selected[:limit]
    selected.sort(key=lambda row: (row["official_split"], int(row["subject_id"]), int(row["study_id"])))
    return selected


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.nn.functional as F
        from open_clip import create_model_from_pretrained
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit("Install embedding dependencies with: pip install -e '.[embedding]'") from error

    config = json.loads(args.config.read_text(encoding="utf-8"))
    labels = list(config["labels"])
    data_root = args.data_root.expanduser().resolve()
    rows = select_multilabel_rows(
        args.study_manifest.expanduser().resolve(),
        data_root,
        labels,
        args.patients.expanduser().resolve() if args.patients else None,
        args.limit,
        args.seed,
        not args.skip_image_check,
    )
    _write_manifest(args.output_manifest, rows)
    if args.selection_only:
        print(f"wrote {len(rows):,} rows to {args.output_manifest}")
        return

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu"
    )
    model, preprocess = create_model_from_pretrained(args.model)
    model.to(device).eval().requires_grad_(False)

    class Images(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            with Image.open(data_root / rows[index]["relative_path"]) as image:
                return preprocess(image.convert("RGB")), index

    loader = DataLoader(
        Images(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    features = None
    offset = 0
    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                encoded = F.normalize(model.encode_image(images).float(), dim=-1)
            if features is None:
                features = torch.empty((len(rows), encoded.shape[1]), dtype=torch.float16)
            features[offset : offset + len(images)] = encoded.cpu().half()
            offset += len(images)
            print(f"embedded {offset:,}/{len(rows):,}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_embeddings": features,
            "label_matrix": torch.tensor([[int(float(row[name] or 0)) for name in labels] for row in rows], dtype=torch.int8),
            "class_names": labels,
            "subject_ids": [row["subject_id"] for row in rows],
            "study_ids": [row["study_id"] for row in rows],
            "dicom_ids": [row["dicom_id"] for row in rows],
            "manifest_sha256": _sha256(args.output_manifest),
            "model": args.model,
            "normalized": True,
            "native_multilabel": True,
        },
        args.output,
    )
    print(f"saved {len(rows):,} embeddings to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--study-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/mimic_cxr_protocol_v1.json"))
    parser.add_argument("--patients", type=Path, help="Optional patients.csv with gender and anchor_age")
    parser.add_argument("--output-manifest", type=Path, default=Path("outputs/residuals/multilabel_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/residuals/biomedclip_multilabel.pt"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--skip-image-check", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
