from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .core import canonical_json


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    requests_path: Path,
    output_path: Path,
    model_id: str,
    seed: int = 20260805,
    max_attempts: int = 5,
) -> None:
    try:
        from google import genai
        from google.genai import types
    except ImportError as error:
        raise SystemExit('Missing Gemini dependency. Install with: pip install -e ".[lidc-frontier]"') from error

    client = genai.Client()
    requests = _read_jsonl(requests_path)
    completed: set[str] = set()
    if output_path.exists():
        completed = {str(row["request_id"]) for row in _read_jsonl(output_path)}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("a", encoding="utf-8") as handle:
        for index, request in enumerate(requests, start=1):
            if request["request_id"] in completed:
                continue
            image_part = types.Part.from_bytes(
                data=Path(request["image"]).read_bytes(), mime_type="image/png"
            )
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                started = time.perf_counter()
                try:
                    response = client.models.generate_content(
                        model=model_id,
                        contents=[image_part, request["prompt"]],
                        config=types.GenerateContentConfig(
                            temperature=0,
                            seed=seed,
                            max_output_tokens=128,
                            response_mime_type="application/json",
                            response_json_schema=request["json_schema"],
                        ),
                    )
                    response_text = response.text.strip()
                    result = {
                        **request,
                        "model_id": model_id,
                        "model_version": getattr(response, "model_version", None),
                        "seed": seed,
                        "decoding": "temperature_0_schema_json",
                        "response_text": response_text,
                        "latency_seconds": time.perf_counter() - started,
                    }
                    handle.write(canonical_json(result) + "\n")
                    handle.flush()
                    print(f"[{index}/{len(requests)}] {request['request_id']} -> {response_text}", flush=True)
                    break
                except Exception as error:  # SDK exception classes vary by backend.
                    last_error = error
                    if attempt == max_attempts:
                        raise RuntimeError(
                            f"Gemini request failed after {max_attempts} attempts: {request['request_id']}"
                        ) from last_error
                    time.sleep(min(2 ** (attempt - 1), 30))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a Gemini frontier model on frozen LIDC requests")
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-id",
        default="gemini-3.5-flash",
        help="Gemini model name; record this exact value in the paper artifact",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-attempts", type=int, default=5)
    args = parser.parse_args(argv)
    run(args.requests, args.output, args.model_id, args.seed, args.max_attempts)


if __name__ == "__main__":
    main()
