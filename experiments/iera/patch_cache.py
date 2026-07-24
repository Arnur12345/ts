"""Build memory-mapped BioMedCLIP patch tokens aligned to a residual manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path


MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(newline="", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_patch_tokens(model, images, pool_grid: int):
    """Return projected NPD tokens for OpenCLIP's timm-backed BioMedCLIP ViT."""
    import torch
    import torch.nn.functional as F

    visual = model.visual
    if hasattr(visual, "trunk") and hasattr(visual.trunk, "forward_features"):
        tokens = visual.trunk.forward_features(images)
        prefix = int(getattr(visual.trunk, "num_prefix_tokens", 1))
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
            prefix = 0
        if tokens.ndim != 3:
            raise RuntimeError(f"unexpected timm feature shape {tuple(tokens.shape)}")
        tokens = tokens[:, prefix:]
        head = getattr(visual, "head", None)
        if head is not None:
            try:
                projected = head(tokens)
                if projected.ndim == 3:
                    tokens = projected
            except (RuntimeError, TypeError):
                pass
    else:
        raise RuntimeError(
            "BioMedCLIP visual encoder does not expose trunk.forward_features; "
            "use open-clip-torch==2.23.0 as pinned by this repository"
        )

    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise RuntimeError(f"patch count {tokens.shape[1]} is not a square grid")
    spatial = tokens.transpose(1, 2).reshape(len(tokens), tokens.shape[2], side, side)
    if pool_grid != side:
        spatial = F.adaptive_avg_pool2d(spatial, (pool_grid, pool_grid))
    return F.normalize(spatial.flatten(2).transpose(1, 2).float(), dim=-1)


def build(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
        from open_clip import create_model_from_pretrained
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as error:
        raise SystemExit("Install embedding dependencies with: pip install -e '.[embedding]'") from error

    with _open_csv(args.manifest) as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("manifest is empty")
    data_root = args.data_root.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.output_dir / "patch_tokens.float16.bin"
    metadata_path = args.output_dir / "patch_cache.json"
    manifest_hash = _sha256(args.manifest)
    mmap = None
    offset = 0
    if metadata_path.exists() and token_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {"manifest_sha256": manifest_hash, "model": args.model, "pool_grid": args.pool_grid}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("existing patch cache metadata does not match this command; choose a new output directory")
        if saved.get("complete") is True:
            print(f"patch cache is already complete at {metadata_path}")
            return
        mmap = np.memmap(token_path, dtype=np.float16, mode="r+", shape=tuple(saved["shape"]))
        offset = int(saved.get("completed", 0))
        print(f"resuming patch cache at row {offset:,}", flush=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    model, preprocess = create_model_from_pretrained(args.model)
    model.to(device).eval().requires_grad_(False)
    start_offset = offset

    class Images(Dataset):
        def __len__(self):
            return len(rows) - start_offset

        def __getitem__(self, index):
            actual = start_offset + index
            with Image.open(data_root / rows[actual]["relative_path"]) as image:
                return preprocess(image.convert("RGB")), actual

    loader = DataLoader(Images(), batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    def write_progress(complete: bool) -> None:
        metadata_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "tokens": token_path.name,
                    "shape": list(mmap.shape),
                    "dtype": "float16",
                    "pool_grid": args.pool_grid,
                    "manifest_sha256": manifest_hash,
                    "model": args.model,
                    "completed": offset,
                    "complete": complete,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    with torch.inference_mode():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                tokens = extract_patch_tokens(model, images, args.pool_grid)
            if mmap is None:
                shape = (len(rows), tokens.shape[1], tokens.shape[2])
                mmap = np.memmap(token_path, dtype=np.float16, mode="w+", shape=shape)
            elif tuple(mmap.shape[1:]) != tuple(tokens.shape[1:]):
                raise ValueError("resumed patch cache shape differs from encoder output")
            end = offset + len(tokens)
            mmap[offset:end] = tokens.cpu().numpy().astype(np.float16, copy=False)
            offset = end
            if offset % (args.batch_size * 20) == 0 or offset == len(rows):
                mmap.flush()
                write_progress(False)
            print(f"cached patches {offset:,}/{len(rows):,}", flush=True)
    mmap.flush()
    write_progress(True)
    print(f"saved patch cache metadata to {metadata_path}")


def load_patch_cache(cache_dir: Path, manifest_hash: str):
    import torch

    metadata = json.loads((cache_dir / "patch_cache.json").read_text(encoding="utf-8"))
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("patch cache and manifest hashes differ")
    shape = tuple(metadata["shape"])
    tokens = torch.from_file(str(cache_dir / metadata["tokens"]), shared=False, size=math.prod(shape), dtype=torch.float16)
    return tokens.reshape(shape), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/iera/patch_cache"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--pool-grid", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.pool_grid <= 0:
        parser.error("pool-grid must be positive")
    build(args)


if __name__ == "__main__":
    main()
