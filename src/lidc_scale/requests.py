from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from .core import canonical_json, stable_key


QUESTION_TEXT = {
    "q1_absolute": [
        'The pulmonary nodule to measure is centered inside the thin red circle. What is its largest axial diameter in millimetres? Return only JSON: {"mm": <number>}',
        'Measure the maximum axial width of the pulmonary nodule inside the thin red circle. Report millimetres only as JSON: {"mm": <number>}',
        'Estimate the longest in-plane diameter of the circled pulmonary nodule, in mm. Output only: {"mm": <number>}',
    ],
    "q2_threshold": [
        'The pulmonary nodule is centered inside the thin red circle. Is its largest diameter at least 6 mm? Return only JSON: {"answer": "yes"} or {"answer": "no"}',
        'Does the maximum axial width of the circled pulmonary nodule meet or exceed 6 mm? Output only JSON with answer equal to "yes" or "no".',
        'Classify the marked pulmonary nodule by the 6 mm cutoff: is it 6 mm or larger? Reply only as JSON: {"answer": "yes"|"no"}',
    ],
    "q3_growth": [
        'The left panel is earlier and the right panel is later. Both red circles mark the same pulmonary nodule. Did the nodule grow? Return only JSON: {"answer": "yes"} or {"answer": "no"}',
        'Compare the circled pulmonary nodule from the earlier left image with the later right image. Has its largest diameter increased? Output only JSON with answer equal to "yes" or "no".',
        'These panels show the same marked pulmonary nodule, earlier on the left and later on the right. Is there interval growth? Reply only as JSON: {"answer": "yes"|"no"}',
    ],
}

SCHEMAS = {
    "q1_absolute": {
        "type": "object",
        "properties": {"mm": {"type": "number"}},
        "required": ["mm"],
        "additionalProperties": False,
    },
    "q2_threshold": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
        "additionalProperties": False,
    },
    "q3_growth": {
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
        "additionalProperties": False,
    },
}

CONDITIONS = ("A_bare", "B_spacing_text", "C_spacing_scale_bar")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _spacing_context(question: str, row: dict[str, Any]) -> str:
    if question == "q3_growth":
        return (
            f" The left image pixel spacing is {row['left_spacing_mm']:g} mm per pixel; "
            f"the right image pixel spacing is {row['right_spacing_mm']:g} mm per pixel."
        )
    return f" The image pixel spacing is {row['target_spacing_mm']:g} mm per pixel."


def build_requests(stimulus_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sources = (
        ("q1_absolute", _read_jsonl(stimulus_dir / "images.jsonl")),
        ("q2_threshold", _read_jsonl(stimulus_dir / "images.jsonl")),
        ("q3_growth", _read_jsonl(stimulus_dir / "pairs.jsonl")),
    )
    for question, stimuli in sources:
        for stimulus in stimuli:
            for condition in CONDITIONS:
                image_key = "scale_bar_image" if condition == "C_spacing_scale_bar" else "marker_image"
                for paraphrase_index, base_prompt in enumerate(QUESTION_TEXT[question], start=1):
                    prompt = base_prompt
                    if condition != "A_bare":
                        prompt += _spacing_context(question, stimulus)
                    if condition == "C_spacing_scale_bar":
                        prompt += " A 10 mm scale bar is also rendered in the image."
                    request = {
                        "request_id": f"{stimulus['stimulus_id']}__{question}__{condition}__p{paraphrase_index}",
                        "question": question,
                        "condition": condition,
                        "paraphrase": paraphrase_index,
                        "stimulus_id": stimulus["stimulus_id"],
                        "nodule_id": stimulus["nodule_id"],
                        "image": str(stimulus_dir / str(stimulus[image_key])),
                        "prompt": prompt,
                        "json_schema": SCHEMAS[question],
                        "diameter_mm": stimulus["diameter_mm"],
                    }
                    if question == "q1_absolute":
                        request["truth"] = {"mm": stimulus["diameter_mm"]}
                        request["acquisition"] = stimulus["target_spacing_mm"]
                    elif question == "q2_threshold":
                        request["truth"] = {"answer": "yes" if stimulus["threshold_6mm"] else "no"}
                        request["acquisition"] = stimulus["target_spacing_mm"]
                    else:
                        request["truth"] = {"answer": "no"}
                        request["acquisition"] = [stimulus["left_spacing_mm"], stimulus["right_spacing_mm"]]
                    rows.append(request)
    return rows


def select_frontier_blocks(rows: list[dict[str, Any]], seed: int, blocks: int = 6) -> list[dict[str, Any]]:
    """Select complete 3-acquisition x 3-paraphrase blocks (~9 calls each)."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["nodule_id"]), str(row["question"]), str(row["condition"]))
        groups.setdefault(key, []).append(row)
    complete = [(key, value) for key, value in groups.items() if len(value) == 9]
    complete.sort(key=lambda item: stable_key(seed, ":".join(item[0])))

    selected: list[list[dict[str, Any]]] = []
    question_counts = {name: 0 for name in QUESTION_TEXT}
    condition_counts = {name: 0 for name in CONDITIONS}
    used_nodules: set[str] = set()
    remaining = complete[:]
    while remaining and len(selected) < blocks:
        unused = [item for item in remaining if item[0][0] not in used_nodules]
        pool = unused or remaining
        pool.sort(
            key=lambda item: (
                question_counts[item[0][1]] + condition_counts[item[0][2]],
                question_counts[item[0][1]],
                condition_counts[item[0][2]],
                stable_key(seed, ":".join(item[0])),
            )
        )
        key, group = pool[0]
        remaining.remove((key, group))
        selected.append(group)
        used_nodules.add(key[0])
        question_counts[key[1]] += 1
        condition_counts[key[2]] += 1
    result = [row for group in selected for row in group]
    return sorted(result, key=lambda row: str(row["request_id"]))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Materialize frozen LIDC model requests")
    parser.add_argument("stimulus_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--frontier-output", type=Path)
    parser.add_argument("--frontier-blocks", type=int, default=6, help="nine calls per complete block")
    args = parser.parse_args(argv)
    config = json.loads((args.stimulus_dir / "config.json").read_text(encoding="utf-8"))
    rows = build_requests(args.stimulus_dir)
    output = args.output or args.stimulus_dir / "requests.jsonl"
    frontier_output = args.frontier_output or args.stimulus_dir / "frontier_requests.jsonl"
    frontier = select_frontier_blocks(rows, int(config["seed"]), args.frontier_blocks)
    _write_jsonl(output, rows)
    _write_jsonl(frontier_output, frontier)
    print(json.dumps({"requests": len(rows), "frontier_requests": len(frontier)}, indent=2))


if __name__ == "__main__":
    main()
