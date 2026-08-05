from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .core import canonical_json, parse_json_object


def _prediction(row: dict[str, Any]) -> float:
    parsed = parse_json_object(str(row["response_text"]))
    if row["question"] == "q1_absolute":
        value = float(parsed["mm"])
        if not math.isfinite(value):
            raise ValueError("non-finite diameter")
        return value
    answer = str(parsed["answer"]).strip().lower()
    if answer not in {"yes", "no"}:
        raise ValueError(f"invalid binary answer: {answer}")
    return 1.0 if answer == "yes" else 0.0


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _group_variances(
    rows: list[dict[str, Any]], keys: tuple[str, ...], expected: int = 3
) -> tuple[list[float], list[float], int]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        if "prediction" in row:
            groups[tuple(row[key] for key in keys)].append(float(row["prediction"]))
    variances: list[float] = []
    stds: list[float] = []
    incomplete = 0
    for values in groups.values():
        if len(values) != expected:
            incomplete += 1
            continue
        variances.append(statistics.pvariance(values))
        stds.append(statistics.pstdev(values))
    return variances, stds, incomplete


def score_rows(source_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    valid: list[dict[str, Any]] = []
    invalid = 0
    total = 0
    for source in source_rows:
        total += 1
        row = dict(source)
        try:
            row["prediction"] = _prediction(row)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            invalid += 1
            continue
        valid.append(row)

    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in valid:
        cells[(str(row.get("model_id", "unknown")), row["question"], row["condition"])].append(row)

    reports: list[dict[str, Any]] = []
    for (model_id, question, condition), rows in sorted(cells.items()):
        acquisition_var, acquisition_std, incomplete_acq = _group_variances(
            rows, ("nodule_id", "paraphrase"), expected=3
        )
        paraphrase_var, paraphrase_std, incomplete_para = _group_variances(
            rows, ("nodule_id", "stimulus_id"), expected=3
        )
        mean_acq_var = _mean(acquisition_var)
        mean_para_var = _mean(paraphrase_var)
        variance_ratio: float | None = None
        ratio_is_infinite = False
        if mean_acq_var is not None and mean_para_var is not None:
            if mean_para_var == 0 and mean_acq_var > 0:
                ratio_is_infinite = True
            else:
                variance_ratio = 1.0 if mean_para_var == mean_acq_var == 0 else mean_acq_var / mean_para_var
        report: dict[str, Any] = {
            "model_id": model_id,
            "question": question,
            "condition": condition,
            "n_predictions": len(rows),
            "acquisition": {
                "complete_groups": len(acquisition_var),
                "incomplete_groups": incomplete_acq,
                "mean_group_variance": mean_acq_var,
                "mean_group_std": _mean(acquisition_std),
            },
            "paraphrase": {
                "complete_groups": len(paraphrase_var),
                "incomplete_groups": incomplete_para,
                "mean_group_variance": mean_para_var,
                "mean_group_std": _mean(paraphrase_std),
            },
            "acquisition_to_paraphrase_variance_ratio": variance_ratio,
            "acquisition_to_paraphrase_ratio_is_infinite": ratio_is_infinite,
        }
        if question == "q1_absolute":
            report["mae_mm"] = statistics.fmean(
                abs(float(row["prediction"]) - float(row["diameter_mm"])) for row in rows
            )
            report["bias_mm"] = statistics.fmean(
                float(row["prediction"]) - float(row["diameter_mm"]) for row in rows
            )
            report["mean_per_nodule_acquisition_std_mm"] = _mean(acquisition_std)
        elif question == "q2_threshold":
            report["accuracy"] = statistics.fmean(
                float(row["prediction"] == (1.0 if row["truth"]["answer"] == "yes" else 0.0))
                for row in rows
            )
            acquisition_groups: dict[tuple[str, int], list[float]] = defaultdict(list)
            for row in rows:
                acquisition_groups[(row["nodule_id"], int(row["paraphrase"]))].append(row["prediction"])
            flips = [float(min(values) != max(values)) for values in acquisition_groups.values() if len(values) == 3]
            report["q2_acquisition_flip_rate"] = _mean(flips)
        else:
            report["q3_false_growth_rate"] = statistics.fmean(float(row["prediction"]) for row in rows)
        reports.append(report)

    return {
        "responses": total,
        "valid_responses": len(valid),
        "invalid_responses": invalid,
        "invalid_rate": invalid / total if total else None,
        "cells": reports,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score LIDC scale-perception responses")
    parser.add_argument("responses", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    for path in args.responses:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    report = score_rows(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(report) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
