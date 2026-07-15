from __future__ import annotations

import argparse
import csv
import hashlib
import time
from pathlib import Path
from typing import Iterable

from .prepare_data import CHEXPERT_LABELS


DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def extract(args: argparse.Namespace) -> None:
    try:
        import torch
        import torch.nn.functional as functional
        from open_clip import create_model_from_pretrained
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit(
            "Embedding dependencies are missing. Install with: pip install -e '.[gpu]'"
        ) from error

    manifest_path = args.manifest.resolve()
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No images found in {manifest_path}")

    known = [name for name in CHEXPERT_LABELS if any(row["label"] == name for row in rows)]
    extras = sorted({row["label"] for row in rows}.difference(known))
    class_names = known + extras
    class_to_id = {name: index for index, name in enumerate(class_names)}

    class ManifestDataset(Dataset):
        def __len__(self) -> int:
            return len(rows)

        def __getitem__(self, index: int):
            row = rows[index]
            image_path = manifest_path.parent / row["resized_path"]
            with Image.open(image_path) as image:
                tensor = preprocess(image.convert("RGB"))
            return tensor, class_to_id[row["label"]], index

    device = _device(torch, args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    print(f"Loading {args.model} on {device} ...", flush=True)
    model, preprocess = create_model_from_pretrained(args.model)
    model.to(device).eval().requires_grad_(False)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    loader = DataLoader(
        ManifestDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.workers > 0,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    features = None
    labels = torch.empty(len(rows), dtype=torch.int64)
    started = time.perf_counter()
    offset = 0
    with torch.inference_mode():
        for batch_number, (images, batch_labels, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    encoded = model.encode_image(images)
            else:
                encoded = model.encode_image(images)
            encoded = functional.normalize(encoded.float(), dim=-1)
            if features is None:
                features = torch.empty(
                    (len(rows), encoded.shape[-1]), dtype=torch.float16
                )
            end = offset + encoded.shape[0]
            features[offset:end].copy_(encoded.cpu().to(torch.float16))
            labels[offset:end].copy_(batch_labels)
            offset = end
            if batch_number % args.log_every == 0 or offset == len(rows):
                rate = offset / max(time.perf_counter() - started, 1e-9)
                print(f"embedded {offset:,}/{len(rows):,} images ({rate:.1f}/s)", flush=True)

    assert features is not None
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "class_names": class_names,
            "subject_ids": torch.tensor(
                [int(row["subject_id"]) for row in rows], dtype=torch.int64
            ),
            "study_ids": torch.tensor(
                [int(row["study_id"]) for row in rows], dtype=torch.int64
            ),
            "manifest_sha256": _sha256(manifest_path),
            "model": args.model,
            "normalized": True,
        },
        output,
    )
    print(f"Saved {features.shape[0]:,} x {features.shape[1]} embeddings to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen BioMedCLIP image embeddings once.")
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/embeddings/biomedclip.pt"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--compile", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    extract(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
