from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .core import canonical_json


MODEL_IDS = {
    "qwen3-vl-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "medgemma-4b": "google/medgemma-4b-it",
}


def _load_requests(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    requests_path: Path,
    output_path: Path,
    model_id: str,
    seed: int = 20260805,
    limit: int | None = None,
) -> None:
    try:
        import torch
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed
    except ImportError as error:
        raise SystemExit(
            'Missing model dependencies. Install with: pip install -e ".[lidc-scale,lidc-models]"'
        ) from error

    set_seed(seed)
    torch.manual_seed(seed)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, torch_dtype="auto", device_map="auto"
    ).eval()
    requests = _load_requests(requests_path)
    if limit is not None:
        requests = requests[:limit]

    completed: set[str] = set()
    if output_path.exists():
        for row in _load_requests(output_path):
            completed.add(str(row["request_id"]))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for index, request in enumerate(requests, start=1):
            if request["request_id"] in completed:
                continue
            image = Image.open(request["image"]).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": request["prompt"]},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=text_prompt, images=image, return_tensors="pt")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            parser = JsonSchemaParser(request["json_schema"])
            prefix_fn = build_transformers_prefix_allowed_tokens_fn(processor.tokenizer, parser)
            started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=48,
                    prefix_allowed_tokens_fn=prefix_fn,
                    use_cache=True,
                )
            prompt_tokens = inputs["input_ids"].shape[1]
            response_text = processor.batch_decode(
                generated[:, prompt_tokens:], skip_special_tokens=True
            )[0].strip()
            result = {
                **request,
                "model_id": model_id,
                "seed": seed,
                "decoding": "greedy_constrained_json",
                "response_text": response_text,
                "latency_seconds": time.perf_counter() - started,
            }
            handle.write(canonical_json(result) + "\n")
            handle.flush()
            print(f"[{index}/{len(requests)}] {request['request_id']} -> {response_text}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Qwen3-VL or MedGemma on frozen LIDC requests")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=sorted(MODEL_IDS), required=True)
    parser.add_argument("--model-id", help="override the pinned Hugging Face model ID")
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    run(args.requests, args.output, args.model_id or MODEL_IDS[args.model], args.seed, args.limit)


if __name__ == "__main__":
    main()
