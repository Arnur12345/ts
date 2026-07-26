"""Build memory-mapped CXR patch tokens aligned to a residual manifest."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path


MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
RAD_DINO_MODEL = "microsoft/rad-dino"


class StreamingPatchCache:
    """Tensor-like row reader that never maps the complete token file."""

    def __init__(
        self,
        path: Path,
        shape: tuple[int, int, int],
        global_indices=None,
    ) -> None:
        import numpy as np

        self.path = path
        self.shape = shape
        self._np = np
        self._row_values = math.prod(shape[1:])
        self._row_bytes = self._row_values * np.dtype(np.float16).itemsize
        self._fd = os.open(path, os.O_RDONLY)
        self._global_to_local = (
            None
            if global_indices is None
            else {
                int(global_index): local_index
                for local_index, global_index in enumerate(global_indices)
            }
        )

    def __getitem__(self, indices):
        import torch

        index_tensor = torch.as_tensor(indices, dtype=torch.long).cpu()
        flat = index_tensor.numpy().reshape(-1)
        if len(flat) == 0:
            return torch.empty(
                (*index_tensor.shape, *self.shape[1:]), dtype=torch.float16
            )
        if self._global_to_local is None:
            if flat.min() < 0 or flat.max() >= self.shape[0]:
                raise IndexError("patch-cache row index is out of bounds")
            local = flat
        else:
            try:
                local = self._np.asarray(
                    [self._global_to_local[int(index)] for index in flat],
                    dtype=self._np.int64,
                )
            except KeyError as error:
                raise IndexError(
                    f"global row {int(error.args[0])} is absent from sparse cache"
                ) from error
        unique, inverse = self._np.unique(local, return_inverse=True)
        rows = self._np.empty(
            (len(unique), *self.shape[1:]), dtype=self._np.float16
        )
        for output_index, row_index in enumerate(unique):
            raw = os.pread(
                self._fd, self._row_bytes, int(row_index) * self._row_bytes
            )
            if len(raw) != self._row_bytes:
                raise OSError(f"short read for patch-cache row {row_index}")
            rows[output_index] = self._np.frombuffer(
                raw, dtype=self._np.float16, count=self._row_values
            ).reshape(self.shape[1:])
        selected = rows[inverse].reshape(
            *index_tensor.shape, *self.shape[1:]
        )
        return torch.from_numpy(selected)

    def close(self) -> None:
        if getattr(self, "_fd", None) is not None:
            os.close(self._fd)
            self._fd = None

    def __del__(self) -> None:
        self.close()


def _open_csv(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open(newline="", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_cache_indices(
    episode_path: Path,
    manifest_hash: str,
    seeds: list[int] | None,
    targets: list[str] | None,
    episode_count: int | None,
    shots: list[int],
) -> tuple[list[int], dict]:
    """Return exact global rows needed by a fixed scoring-only episode subset."""
    import torch

    saved = torch.load(episode_path, map_location="cpu", weights_only=False)
    signature = saved["signature"]
    if signature["manifest_sha256"] != manifest_hash:
        raise ValueError("episodes and manifest hashes differ")
    available_seeds = list(signature["seeds"])
    selected_seeds = available_seeds if seeds is None else list(seeds)
    if not set(selected_seeds).issubset(set(available_seeds)):
        raise ValueError("requested cache seeds are absent from saved episodes")
    if not shots or min(shots) <= 0:
        raise ValueError("episode cache shots must be positive")
    maximum_shot = max(shots)
    selected_pairs = {
        pair_id: names
        for pair_id, names in saved["pairs"].items()
        if targets is None or names[0] in targets
    }
    if not selected_pairs:
        raise ValueError("no saved episode pair matches episode-targets")
    limit = signature["episodes"] if episode_count is None else episode_count
    if limit <= 0 or limit > signature["episodes"]:
        raise ValueError("episode-count exceeds the saved episode bank")
    required = set()
    for pair_id in selected_pairs:
        for seed in selected_seeds:
            for partition in ("validate", "test"):
                episodes = saved["episodes"][(pair_id, seed, partition)]
                required.update(
                    int(index)
                    for index in episodes["query"][:limit].flatten().tolist()
                )
                for class_name in ("positive", "negative"):
                    panels = episodes[class_name][:limit]
                    if 2 * maximum_shot > panels.shape[2]:
                        raise ValueError(
                            "episode cache shot exceeds candidate panel size"
                        )
                    required.update(
                        int(index)
                        for index in panels[:, :, : 2 * maximum_shot]
                        .flatten()
                        .tolist()
                    )
                    choices = episodes[
                        f"random_{class_name}_env"
                    ][:limit, : 2 * maximum_shot]
                    for batch in range(len(panels)):
                        counts = [0, 0]
                        for environment in choices[batch].tolist():
                            source = counts[int(environment)]
                            required.add(
                                int(panels[batch, int(environment), source])
                            )
                            counts[int(environment)] += 1
    protocol = {
        "episode_path": str(episode_path),
        "episode_seeds": selected_seeds,
        "episode_targets": [names[0] for names in selected_pairs.values()],
        "episode_pair_ids": sorted(selected_pairs),
        "episode_count": limit,
        "episode_shots": sorted(set(shots)),
        "partitions": ["validate", "test"],
    }
    return sorted(required), protocol


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


def extract_rad_dino_patch_tokens(model, images, pool_grid: int):
    """Return normalized patch tokens from a Hugging Face RAD-DINO model."""
    import torch.nn.functional as F

    tokens = model(pixel_values=images).last_hidden_state[:, 1:]
    side = math.isqrt(tokens.shape[1])
    if side * side != tokens.shape[1]:
        raise RuntimeError(f"RAD-DINO patch count {tokens.shape[1]} is not square")
    spatial = tokens.transpose(1, 2).reshape(len(tokens), tokens.shape[2], side, side)
    if pool_grid != side:
        spatial = F.adaptive_avg_pool2d(spatial, (pool_grid, pool_grid))
    return F.normalize(spatial.flatten(2).transpose(1, 2).float(), dim=-1)


def build(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        import torch
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
    if args.episodes is None:
        global_indices = list(range(len(rows)))
        episode_protocol = None
    else:
        global_indices, episode_protocol = episode_cache_indices(
            args.episodes,
            manifest_hash,
            args.episode_seeds,
            args.episode_targets,
            args.episode_count,
            args.episode_shots,
        )
        print(
            f"native episode subset contains {len(global_indices):,}/"
            f"{len(rows):,} manifest images",
            flush=True,
        )
    global_index_hash = hashlib.sha256(
        np.asarray(global_indices, dtype="<i8").tobytes()
    ).hexdigest()
    global_index_path = args.output_dir / "global_indices.int64.bin"
    mmap = None
    offset = 0
    if metadata_path.exists() and token_path.exists():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_sha256": manifest_hash,
            "model": args.model,
            "pool_grid": args.pool_grid,
        }
        if episode_protocol is not None:
            expected["global_indices_sha256"] = global_index_hash
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("existing patch cache metadata does not match this command; choose a new output directory")
        if saved.get("complete") is True:
            print(f"patch cache is already complete at {metadata_path}")
            return
        mmap = np.memmap(token_path, dtype=np.float16, mode="r+", shape=tuple(saved["shape"]))
        offset = int(saved.get("completed", 0))
        print(f"resuming patch cache at row {offset:,}", flush=True)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    if args.encoder == "biomedclip":
        from open_clip import create_model_from_pretrained

        model, preprocess = create_model_from_pretrained(args.model)

        def prepare(image):
            return preprocess(image)

        extract = extract_patch_tokens
        grid_size = getattr(
            getattr(model.visual.trunk, "patch_embed", None), "grid_size", 14
        )
        native_grid = int(grid_size[0] if isinstance(grid_size, tuple) else grid_size)
    else:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model)

        def prepare(image):
            return processor(
                images=image, return_tensors="pt"
            ).pixel_values[0]

        extract = extract_rad_dino_patch_tokens
        crop_size = processor.crop_size
        input_size = int(
            crop_size["height"] if isinstance(crop_size, dict) else crop_size.height
        )
        native_grid = input_size // int(model.config.patch_size)
    if args.pool_grid > native_grid:
        raise ValueError("pool-grid exceeds the encoder's native token grid")
    model.to(device).eval().requires_grad_(False)
    start_offset = offset
    if episode_protocol is not None:
        if global_index_path.exists():
            observed_index_hash = hashlib.sha256(
                global_index_path.read_bytes()
            ).hexdigest()
            if observed_index_hash != global_index_hash:
                raise ValueError("existing sparse-cache global indices changed")
        else:
            np.asarray(global_indices, dtype="<i8").tofile(global_index_path)

    class Images(Dataset):
        def __len__(self):
            return len(global_indices) - start_offset

        def __getitem__(self, index):
            actual = global_indices[start_offset + index]
            with Image.open(data_root / rows[actual]["relative_path"]) as image:
                return prepare(image.convert("RGB")), actual

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
                    "native_grid": native_grid,
                    "manifest_sha256": manifest_hash,
                    "model": args.model,
                    "encoder": args.encoder,
                    "index_mode": (
                        "dense" if episode_protocol is None else "sparse"
                    ),
                    "dataset_size": len(rows),
                    "global_indices": (
                        None
                        if episode_protocol is None
                        else global_index_path.name
                    ),
                    "global_indices_sha256": global_index_hash,
                    "episode_protocol": episode_protocol,
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
                tokens = extract(model, images, args.pool_grid)
            if mmap is None:
                shape = (len(global_indices), tokens.shape[1], tokens.shape[2])
                mmap = np.memmap(token_path, dtype=np.float16, mode="w+", shape=shape)
            elif tuple(mmap.shape[1:]) != tuple(tokens.shape[1:]):
                raise ValueError("resumed patch cache shape differs from encoder output")
            end = offset + len(tokens)
            mmap[offset:end] = tokens.cpu().numpy().astype(np.float16, copy=False)
            offset = end
            if (
                offset % (args.batch_size * 20) == 0
                or offset == len(global_indices)
            ):
                mmap.flush()
                write_progress(False)
            print(
                f"cached patches {offset:,}/{len(global_indices):,}",
                flush=True,
            )
    mmap.flush()
    write_progress(True)
    print(f"saved patch cache metadata to {metadata_path}")


def load_patch_cache(
    cache_dir: Path,
    manifest_hash: str,
    expected_model: str | None = MODEL,
    expected_pool_grid: int | None = None,
    access_mode: str = "private",
):
    import torch

    metadata = json.loads((cache_dir / "patch_cache.json").read_text(encoding="utf-8"))
    if metadata["manifest_sha256"] != manifest_hash:
        raise ValueError("patch cache and manifest hashes differ")
    if metadata.get("complete") is not True:
        raise ValueError("patch cache is incomplete; rerun the cache command to resume it")
    if metadata.get("dtype") != "float16":
        raise ValueError(f"unsupported patch cache dtype {metadata.get('dtype')!r}")
    if expected_model is not None and metadata.get("model") != expected_model:
        raise ValueError("patch cache was produced by a different visual encoder")
    pool_grid = int(metadata.get("pool_grid", 0))
    if pool_grid <= 0 or (expected_pool_grid is not None and pool_grid != expected_pool_grid):
        raise ValueError("patch cache pooling configuration does not match this run")
    shape = tuple(metadata["shape"])
    if len(shape) != 3 or shape[1] != pool_grid * pool_grid or int(metadata.get("completed", -1)) != shape[0]:
        raise ValueError("patch cache shape/progress metadata is inconsistent")
    token_path = cache_dir / metadata["tokens"]
    expected_bytes = math.prod(shape) * torch.tensor([], dtype=torch.float16).element_size()
    if not token_path.is_file() or token_path.stat().st_size != expected_bytes:
        raise ValueError("patch cache file size does not match its metadata")
    if metadata.get("index_mode", "dense") == "sparse":
        index_path = cache_dir / metadata["global_indices"]
        import numpy as np

        if (
            not index_path.is_file()
            or index_path.stat().st_size
            != shape[0] * np.dtype(np.int64).itemsize
        ):
            raise ValueError("sparse patch-cache global index file is invalid")
        global_indices = np.fromfile(index_path, dtype=np.int64)
        if hashlib.sha256(index_path.read_bytes()).hexdigest() != metadata.get(
            "global_indices_sha256"
        ):
            raise ValueError("sparse patch-cache global indices changed")
        if access_mode != "stream":
            raise ValueError("sparse patch caches require access_mode='stream'")
        return StreamingPatchCache(
            token_path, shape, global_indices=global_indices
        ), metadata
    if access_mode == "stream":
        return StreamingPatchCache(token_path, shape), metadata
    if access_mode not in {"private", "shared"}:
        raise ValueError("access_mode must be private, shared, or stream")
    tokens = torch.from_file(
        str(token_path),
        shared=access_mode == "shared",
        size=math.prod(shape),
        dtype=torch.float16,
    )
    return tokens.reshape(shape), metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/iera/patch_cache"))
    parser.add_argument(
        "--encoder", choices=("biomedclip", "rad-dino"), default="biomedclip"
    )
    parser.add_argument("--model")
    parser.add_argument("--pool-grid", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--episodes",
        type=Path,
        help="Build a sparse cache containing only rows used by saved episodes",
    )
    parser.add_argument("--episode-seeds", type=int, nargs="+")
    parser.add_argument("--episode-targets", nargs="+")
    parser.add_argument("--episode-count", type=int)
    parser.add_argument(
        "--episode-shots", type=int, nargs="+", default=(1, 3, 5, 10)
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.model is None:
        args.model = MODEL if args.encoder == "biomedclip" else RAD_DINO_MODEL
    if args.pool_grid <= 0:
        parser.error("pool-grid must be positive")
    build(args)


if __name__ == "__main__":
    main()
