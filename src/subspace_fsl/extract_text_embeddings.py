from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from .extract_embeddings import DEFAULT_MODEL
from .prepare_data import CHEXPERT_LABELS


def choose_device(torch, requested: str):
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
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError as error:
        raise SystemExit(
            "Text-embedding dependencies are missing. Install with: "
            ".venv/bin/python -m pip install --no-build-isolation -e '.[gpu]'"
        ) from error

    with args.descriptions.open(encoding="utf-8") as handle:
        description_map = json.load(handle)
    missing = sorted(set(CHEXPERT_LABELS).difference(description_map))
    if missing:
        raise ValueError(f"Description JSON is missing classes: {missing}")

    class_names = list(CHEXPERT_LABELS)
    descriptions = [str(description_map[name]) for name in class_names]
    device = choose_device(torch, args.device)
    print(f"Loading {args.model} on {device} ...", flush=True)
    model, _ = create_model_from_pretrained(args.model)
    tokenizer = get_tokenizer(args.model)
    model.to(device).eval().requires_grad_(False)
    tokens = tokenizer(descriptions, context_length=args.context_length).to(device)
    with torch.inference_mode():
        features = model.encode_text(tokens)
        features = functional.normalize(features.float(), dim=-1).cpu()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "class_names": class_names,
            "descriptions": descriptions,
            "model": args.model,
            "normalized": True,
        },
        output,
    )
    print(f"Saved {len(class_names)} normalized class-text embeddings to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache frozen BioMedCLIP embeddings for class descriptions."
    )
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=Path("configs/mimic_class_descriptions.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/embeddings/biomedclip_text.pt")
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    extract(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
