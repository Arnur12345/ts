"""Build a preregistered BioMedCLIP support-device text direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .patch_cache import MODEL


PRESENT_PROMPTS = (
    "chest radiograph with support devices",
    "chest x-ray with lines and tubes",
    "portable chest radiograph with medical devices",
    "chest radiograph with an endotracheal tube or central venous catheter",
)
ABSENT_PROMPTS = (
    "chest radiograph without support devices",
    "chest x-ray without lines or tubes",
    "chest radiograph with no medical devices",
    "chest radiograph without an endotracheal tube or central venous catheter",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("outputs/iera/biomedclip_device_text_direction.pt"),
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        from open_clip import create_model_from_pretrained, get_tokenizer
    except ImportError as error:
        raise SystemExit(
            "Install embedding dependencies with: pip install -e '.[embedding]'"
        ) from error
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    model, _ = create_model_from_pretrained(args.model)
    tokenizer = get_tokenizer(args.model)
    model.to(device).eval().requires_grad_(False)
    prompts = (*PRESENT_PROMPTS, *ABSENT_PROMPTS)
    with torch.inference_mode():
        features = F.normalize(
            model.encode_text(tokenizer(prompts).to(device)).float(), dim=-1
        )
    split = len(PRESENT_PROMPTS)
    direction = F.normalize(
        features[:split].mean(0) - features[split:].mean(0), dim=-1
    ).cpu()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "direction": direction,
            "model": args.model,
            "present_prompts": PRESENT_PROMPTS,
            "absent_prompts": ABSENT_PROMPTS,
        },
        args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": args.model,
                "width": len(direction),
            }
        )
    )


if __name__ == "__main__":
    main()
