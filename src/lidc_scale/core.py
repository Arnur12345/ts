from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


@dataclass(frozen=True)
class Candidate:
    nodule_id: str
    scan_id: int
    patient_id: str
    annotation_ids: tuple[int, ...]
    reader_diameters_mm: tuple[float, ...]
    diameter_mm: float


def median(values: Sequence[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("median requires at least one value")
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def bin_for_diameter(diameter_mm: float, bins: Sequence[dict[str, Any]]) -> str | None:
    for item in bins:
        if float(item["lower"]) <= diameter_mm < float(item["upper"]):
            return str(item["name"])
    return None


def select_candidates(
    candidates: Iterable[Candidate],
    bins: Sequence[dict[str, Any]],
    n_nodules: int,
    seed: int,
    max_per_scan: int = 1,
) -> tuple[list[Candidate], dict[str, int]]:
    """Deterministically satisfy bin quotas, then fill any shortfall globally."""
    candidates = sorted(candidates, key=lambda c: stable_key(seed, c.nodule_id))
    by_bin = {str(item["name"]): [] for item in bins}
    for candidate in candidates:
        name = bin_for_diameter(candidate.diameter_mm, bins)
        if name is not None:
            by_bin[name].append(candidate)

    selected: list[Candidate] = []
    selected_ids: set[str] = set()
    scan_counts: dict[int, int] = {}
    achieved = {str(item["name"]): 0 for item in bins}

    def add(candidate: Candidate) -> bool:
        if candidate.nodule_id in selected_ids:
            return False
        if scan_counts.get(candidate.scan_id, 0) >= max_per_scan:
            return False
        selected.append(candidate)
        selected_ids.add(candidate.nodule_id)
        scan_counts[candidate.scan_id] = scan_counts.get(candidate.scan_id, 0) + 1
        name = bin_for_diameter(candidate.diameter_mm, bins)
        if name is not None:
            achieved[name] += 1
        return True

    for item in bins:
        name = str(item["name"])
        target = int(item["quota"])
        for candidate in by_bin[name]:
            if achieved[name] >= target:
                break
            add(candidate)

    # A dataset release or a strict one-scan rule can make a bin short. Fill in a
    # deterministic order rather than silently returning fewer than requested.
    for candidate in candidates:
        if len(selected) >= n_nodules:
            break
        add(candidate)

    if len(selected) != n_nodules:
        raise ValueError(
            f"could select only {len(selected)} of {n_nodules} nodules under "
            f"max_per_scan={max_per_scan}; available={len(candidates)}"
        )
    selected.sort(key=lambda c: (c.diameter_mm, c.nodule_id))
    return selected, achieved


def window_bounds(center_hu: float, width_hu: float) -> tuple[float, float]:
    if width_hu <= 0:
        raise ValueError("window width must be positive")
    return center_hu - width_hu / 2.0, center_hu + width_hu / 2.0


def physical_grid(
    center_row_col: tuple[float, float],
    source_spacing_mm: float,
    target_spacing_mm: float,
    size_px: int,
) -> tuple[list[float], list[float]]:
    """Source coordinates of a centred output sampling grid (test helper)."""
    if source_spacing_mm <= 0 or target_spacing_mm <= 0 or size_px <= 0:
        raise ValueError("spacings and size must be positive")
    scale = target_spacing_mm / source_spacing_mm
    center_out = (size_px - 1) / 2.0
    rows = [center_row_col[0] + (i - center_out) * scale for i in range(size_px)]
    cols = [center_row_col[1] + (j - center_out) * scale for j in range(size_px)]
    return rows, cols


def pair_orientations(spacings: Sequence[float], nodule_id: str, seed: int) -> list[tuple[float, float]]:
    """All unordered pairs, deterministically balanced in left/right direction."""
    pairs: list[tuple[float, float]] = []
    for i, first in enumerate(spacings):
        for second in spacings[i + 1 :]:
            left, right = first, second
            if int(stable_key(seed, f"{nodule_id}:{first}:{second}")[-1], 16) % 2:
                left, right = second, first
            pairs.append((float(left), float(right)))
    return pairs


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model response, tolerating a fenced or prefixed JSON object."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("response JSON is not an object")
    return parsed
