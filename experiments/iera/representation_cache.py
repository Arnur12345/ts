"""Cache Rad-DINO activations entering its final two transformer blocks."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from experiments.residuals.data import load_config, load_dataset

from .falsification import _prepare_training
from .labels import restore_raw_target_status
from .patch_cache import RAD_DINO_MODEL, _open_csv


def _indices(episodes: dict) -> set[int]:
    result = set()
    for name in ("positive", "negative", "query"):
        result.update(int(value) for value in episodes[name].flatten().tolist())
    return result


def _layer_output(layer, hidden: torch.Tensor) -> torch.Tensor:
    output = layer(hidden)
    return output[0] if isinstance(output, (tuple, list)) else output


def build(args: argparse.Namespace) -> None:
    try:
        import numpy as np
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise SystemExit(
            "Install embedding dependencies with: pip install -e '.[embedding]'"
        ) from error
    data = load_dataset(args.embeddings, args.manifest)
    restore_raw_target_status(data, args.raw_labels)
    config = load_config(args.config)
    locked = torch.load(
        args.episodes, map_location="cpu", weights_only=False
    )
    if locked["signature"]["manifest_sha256"] != data.manifest_sha256:
        raise ValueError("locked episodes and manifest differ")
    # Reproduce the learned falsification split exactly. That runner excludes
    # only the evaluation pairs that actually survived episode construction,
    # rather than every nominal pilot pair.
    args.evaluation_pair_names = list(locked["pairs"].values())
    pairs = {
        pair_id: names
        for pair_id, names in locked["pairs"].items()
        if names[0] == "Pneumothorax"
        and "Support" in names[1]
        and "Device" in names[1]
    }
    if len(pairs) != 1:
        raise ValueError("expected one locked Pneumothorax-Support Devices pair")
    pair_id = next(iter(pairs))
    base = {}
    required = set()
    for seed in args.seeds:
        train, validation, metadata = _prepare_training(
            data, config, args, args.seed + seed
        )
        base[seed] = {
            "train": train,
            "validate": validation,
            "pairs": metadata,
        }
        for bank in (train, validation):
            for episodes in bank:
                required.update(_indices(episodes))
        for partition in ("validate", "test"):
            episodes = locked["episodes"][(pair_id, seed, partition)]
            subset = {
                name: (
                    value[: args.episodes_per_seed]
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == locked["signature"]["episodes"]
                    else value
                )
                for name, value in episodes.items()
            }
            required.update(_indices(subset))
    global_indices = sorted(required)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bank_path = args.output_dir / "episode_bank.pt"
    bank_signature = {
        "manifest_sha256": data.manifest_sha256,
        "seeds": list(args.seeds),
        "seed": args.seed,
        "split_seed": args.split_seed,
        "train_shot": args.train_shot,
        "max_train_pairs": args.max_train_pairs,
        "max_train_steps": args.max_train_steps,
        "base_validation_episodes": args.base_validation_episodes,
        "queries_per_stratum": args.queries_per_stratum,
        "min_stratum_patients": args.min_stratum_patients,
        "locked_episode_count": args.episodes_per_seed,
        "locked_pair_id": pair_id,
        "evaluation_pair_names": args.evaluation_pair_names,
    }
    if bank_path.exists():
        existing = torch.load(bank_path, map_location="cpu", weights_only=False)
        if existing.get("signature") != bank_signature:
            raise ValueError("existing representation episode bank differs")
    else:
        torch.save(
            {
                "signature": bank_signature,
                "base": base,
                "locked": {
                    "pairs": pairs,
                    "episodes": {
                        key: value
                        for key, value in locked["episodes"].items()
                        if key[0] == pair_id and key[1] in args.seeds
                    },
                },
            },
            bank_path,
        )
    with _open_csv(args.manifest) as handle:
        rows = list(csv.DictReader(handle))
    data_root = args.data_root.expanduser().resolve()
    token_path = args.output_dir / "prefix_tokens.float16.bin"
    index_path = args.output_dir / "global_indices.int64.bin"
    metadata_path = args.output_dir / "representation_cache.json"
    index_bytes = np.asarray(global_indices, dtype="<i8").tobytes()
    index_hash = hashlib.sha256(index_bytes).hexdigest()
    offset, mmap = 0, None
    if metadata_path.exists() and token_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = {
            "manifest_sha256": data.manifest_sha256,
            "model": RAD_DINO_MODEL,
            "global_indices_sha256": index_hash,
            "prefix_blocks": int(args.prefix_blocks),
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise ValueError("existing representation cache differs")
        if metadata.get("complete"):
            print(f"representation cache is complete at {metadata_path}")
            return
        mmap = np.memmap(
            token_path,
            dtype=np.float16,
            mode="r+",
            shape=tuple(metadata["shape"]),
        )
        offset = int(metadata["completed"])
        print(f"resuming representation cache at {offset:,}", flush=True)
    else:
        index_path.write_bytes(index_bytes)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    processor = AutoImageProcessor.from_pretrained(RAD_DINO_MODEL)
    model = AutoModel.from_pretrained(RAD_DINO_MODEL).to(device)
    model.eval().requires_grad_(False)
    layers = model.encoder.layer
    if args.prefix_blocks != len(layers) - 2:
        raise ValueError(
            f"prefix-blocks must equal total blocks minus two ({len(layers)-2})"
        )
    start_offset = offset

    class Images(Dataset):
        def __len__(self):
            return len(global_indices) - start_offset

        def __getitem__(self, index):
            global_index = global_indices[start_offset + index]
            with Image.open(
                data_root / rows[global_index]["relative_path"]
            ) as image:
                pixels = processor(
                    images=image.convert("RGB"), return_tensors="pt"
                ).pixel_values[0]
            return pixels, global_index

    loader = DataLoader(
        Images(),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    def write_metadata(complete: bool) -> None:
        metadata_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "tokens": token_path.name,
                    "shape": list(mmap.shape),
                    "dtype": "float16",
                    "manifest_sha256": data.manifest_sha256,
                    "model": RAD_DINO_MODEL,
                    "prefix_blocks": args.prefix_blocks,
                    "remaining_blocks": 2,
                    "global_indices": index_path.name,
                    "global_indices_sha256": index_hash,
                    "dataset_size": len(rows),
                    "completed": offset,
                    "complete": complete,
                    "episode_bank": bank_path.name,
                    "episode_bank_signature": bank_signature,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    with torch.inference_mode():
        for pixels, _ in loader:
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                hidden = model.embeddings(
                    pixel_values=pixels.to(device, non_blocking=True)
                )
                for layer in layers[: args.prefix_blocks]:
                    hidden = _layer_output(layer, hidden)
            if mmap is None:
                shape = (
                    len(global_indices),
                    hidden.shape[1],
                    hidden.shape[2],
                )
                mmap = np.memmap(
                    token_path, dtype=np.float16, mode="w+", shape=shape
                )
            end = offset + len(hidden)
            mmap[offset:end] = hidden.cpu().numpy().astype(
                np.float16, copy=False
            )
            offset = end
            if (
                offset % (args.batch_size * 20) == 0
                or offset == len(global_indices)
            ):
                mmap.flush()
                write_metadata(False)
            print(
                f"cached prefix activations {offset:,}/{len(global_indices):,}",
                flush=True,
            )
    mmap.flush()
    write_metadata(True)
    print(f"saved representation cache metadata to {metadata_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mimic_cxr_protocol_v1.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--train-shot", type=int, default=3)
    parser.add_argument("--max-train-pairs", type=int, default=8)
    parser.add_argument("--max-train-steps", type=int, default=150)
    parser.add_argument("--base-validation-episodes", type=int, default=20)
    parser.add_argument("--queries-per-stratum", type=int, default=1)
    parser.add_argument("--min-stratum-patients", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--prefix-blocks", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
