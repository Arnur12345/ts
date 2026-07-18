"""Stratify real MIMIC image-report pairs and embed both with BioMedCLIP."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path


MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(newline="", encoding="utf-8")


def _report_path(report_root: Path, subject: str, study: str) -> Path | None:
    subject = f"{int(subject):08d}"
    relative = Path(f"p{subject[:2]}") / f"p{subject}" / f"s{int(study)}.txt"
    for candidate in (report_root / "files" / relative, report_root / relative):
        if candidate.is_file() and candidate.stat().st_size:
            return candidate
    return None


def select_pairs(
    manifest: Path,
    data_root: Path,
    report_root: Path,
    target: int,
    seed: int,
) -> list[dict[str, str]]:
    """Select a deterministic, class-stratified subset with one study per patient."""
    with _open_csv(manifest) as handle:
        rows = list(csv.DictReader(handle))
    expected_classes = {row["class_name"] for row in rows}

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        image = data_root / row["relative_path"]
        report = _report_path(report_root, row["subject_id"], row["study_id"])
        if not image.is_file() or report is None:
            continue
        item = dict(row)
        item["report_path"] = report.relative_to(data_root).as_posix()
        grouped[row["class_name"]].append(item)

    if not grouped:
        raise ValueError("no image-report pairs were found")
    missing_classes = expected_classes - set(grouped)
    if missing_classes:
        raise ValueError(f"classes without valid image-report pairs: {sorted(missing_classes)}")
    for class_name, items in grouped.items():
        random.Random(f"{seed}:{class_name}").shuffle(items)

    classes = sorted(grouped)
    quotas = {name: target // len(classes) for name in classes}
    for name in classes[: target % len(classes)]:
        quotas[name] += 1
    cursors = Counter()
    counts = Counter()
    used_patients: set[str] = set()
    selected: list[dict[str, str]] = []

    def take(name: str) -> bool:
        items = grouped[name]
        while cursors[name] < len(items):
            item = items[cursors[name]]
            cursors[name] += 1
            if item["subject_id"] in used_patients:
                continue
            used_patients.add(item["subject_id"])
            selected.append(item)
            counts[name] += 1
            return True
        return False

    # Scarce classes choose patients first; common classes cannot consume them.
    for name in sorted(classes, key=lambda value: (len(grouped[value]), value)):
        while counts[name] < quotas[name] and take(name):
            pass

    # If a class cannot meet its quota, redistribute the remainder evenly.
    while len(selected) < target:
        progress = False
        for name in sorted(classes, key=lambda value: (counts[value], value)):
            if len(selected) == target:
                break
            progress |= take(name)
        if not progress:
            raise ValueError(f"only {len(selected):,} unique-patient pairs are available; lower --target")

    random.Random(seed).shuffle(selected)
    return selected


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embed(args: argparse.Namespace) -> None:
    data_root = args.data_root.expanduser().resolve()
    report_root = (args.report_root or data_root / "mimic-cxr-reports-2.1.0").expanduser().resolve()
    manifest = args.manifest or args.protocol_dir / "protocol_samples.csv.gz"
    rows = select_pairs(manifest.resolve(), data_root, report_root, args.target, args.seed)
    _write_manifest(args.subset_manifest, rows)
    print("selected:", json.dumps(Counter(row["class_name"] for row in rows), sort_keys=True))
    if args.selection_only:
        return

    try:
        import torch
        import torch.nn.functional as F
        from open_clip import create_model_from_pretrained, get_tokenizer
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit("Install embedding dependencies with: pip install -e '.[embedding]'") from error

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model, preprocess = create_model_from_pretrained(args.model)
    tokenizer = get_tokenizer(args.model)
    model.to(device).eval().requires_grad_(False)

    class Pairs(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            with Image.open(data_root / row["relative_path"]) as image:
                image = preprocess(image.convert("RGB"))
            report = (data_root / row["report_path"]).read_text(encoding="utf-8", errors="replace")
            return image, report, index

    loader = DataLoader(Pairs(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    image_features = report_features = None
    offset, started = 0, time.perf_counter()
    with torch.inference_mode():
        for images, reports, _ in loader:
            images = images.to(device, non_blocking=True)
            tokens = tokenizer(list(reports), context_length=args.context_length).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                image_batch = F.normalize(model.encode_image(images).float(), dim=-1)
                report_batch = F.normalize(model.encode_text(tokens).float(), dim=-1)
            if image_features is None:
                shape = (len(rows), image_batch.shape[1])
                image_features = torch.empty(shape, dtype=torch.float16)
                report_features = torch.empty(shape, dtype=torch.float16)
            end = offset + len(images)
            image_features[offset:end] = image_batch.cpu().half()
            report_features[offset:end] = report_batch.cpu().half()
            offset = end
            print(f"embedded {offset:,}/{len(rows):,} pairs ({offset / (time.perf_counter() - started):.1f}/s)", flush=True)

    class_names = sorted({row["class_name"] for row in rows})
    class_ids = {name: index for index, name in enumerate(class_names)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_embeddings": image_features,
            "report_embeddings": report_features,
            "labels": torch.tensor([class_ids[row["class_name"]] for row in rows]),
            "class_names": class_names,
            "subject_ids": torch.tensor([int(row["subject_id"]) for row in rows]),
            "study_ids": torch.tensor([int(row["study_id"]) for row in rows]),
            "dicom_ids": [row["dicom_id"] for row in rows],
            "manifest_sha256": _sha256(args.subset_manifest),
            "model": args.model,
            "normalized": True,
        },
        args.output,
    )
    print(f"saved {len(rows):,} paired embeddings to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--target", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--subset-manifest", type=Path, default=Path("outputs/biomedclip_pairs_7000.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/biomedclip_pairs_7000.pt"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selection-only", action="store_true")
    embed(parser.parse_args())


if __name__ == "__main__":
    main()
