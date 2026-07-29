"""Core covariance, chronology, registration, and transition utilities."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class TransitionPair:
    """One chronologically consecutive, label-known same-patient pair."""

    before: int
    after: int
    subject_id: str
    target_before: int
    target_after: int
    device_before: int
    device_after: int
    view_changed: bool

    @property
    def target_changed(self) -> bool:
        return self.target_before != self.target_after

    @property
    def device_changed(self) -> bool:
        return self.device_before != self.device_after

    @property
    def target_resolved(self) -> bool:
        return self.target_before == 1 and self.target_after == 0

    @property
    def device_remains(self) -> bool:
        return self.device_before == 1 and self.device_after == 1

    @property
    def stratum(self) -> str:
        if self.target_changed and not self.device_changed:
            return "disease_change_device_stable"
        if not self.target_changed and self.device_changed:
            return "disease_stable_device_change"
        if not self.target_changed and not self.device_changed:
            return "both_stable"
        return "both_change"


def subject_bucket(subject_id: str, seed: int, namespace: str) -> int:
    digest = hashlib.sha256(
        f"{namespace}|{seed}|{subject_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % 10_000


def temporal_partition(subject_id: str, seed: int = 2026) -> str:
    """Split main-training patients again without crossing patient boundaries."""

    bucket = subject_bucket(subject_id, seed, "trace-temporal")
    if bucket < 7000:
        return "fit"
    if bucket < 8500:
        return "validate"
    return "test"


def _first(row: Mapping[str, str], names: Sequence[str]) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value and value.lower() not in {"nan", "na", "none"}:
            return value
    return ""


def study_timestamp(row: Mapping[str, str]) -> tuple[int, float] | None:
    """Parse MIMIC StudyDate/StudyTime without treating study_id as time."""

    combined = _first(
        row,
        (
            "study_datetime",
            "StudyDateTime",
            "study_date_time",
        ),
    )
    if combined:
        try:
            parsed = datetime.fromisoformat(combined.replace("Z", "+00:00"))
            return (
                parsed.year * 10_000 + parsed.month * 100 + parsed.day,
                (
                    parsed.hour * 10_000
                    + parsed.minute * 100
                    + parsed.second
                    + parsed.microsecond / 1_000_000
                ),
            )
        except ValueError:
            digits = "".join(character for character in combined if character.isdigit())
            if len(digits) >= 8:
                date = int(digits[:8])
                time = float(digits[8:14] or 0)
                return date, time
    date_value = _first(
        row,
        ("StudyDate", "study_date", "date", "studydate"),
    )
    if not date_value:
        return None
    date_digits = "".join(
        character for character in date_value.split(".", 1)[0]
        if character.isdigit()
    )
    if len(date_digits) < 8:
        return None
    time_value = _first(
        row,
        ("StudyTime", "study_time", "time", "studytime"),
    )
    try:
        time_number = float(time_value) if time_value else 0.0
    except ValueError:
        time_digits = "".join(
            character for character in time_value if character.isdigit()
        )
        time_number = float(time_digits or 0)
    return int(date_digits[:8]), time_number


def _view(row: Mapping[str, str]) -> str:
    return _first(row, ("ViewPosition", "view", "view_position")).upper()


def consecutive_transitions(
    rows: Sequence[Mapping[str, str]],
    subject_ids: Sequence[str],
    labels: torch.Tensor,
    known: torch.Tensor,
    target_id: int,
    device_id: int,
    allowed_indices: Iterable[int],
) -> list[TransitionPair]:
    """Return only genuinely consecutive studies from allowed patients.

    Chronology is formed before unknown-label pairs are removed, preventing an
    unknown intermediate examination from being silently skipped.
    """

    grouped: dict[str, list[tuple[tuple[int, float], int]]] = {}
    missing_timestamp = 0
    for index in allowed_indices:
        timestamp = study_timestamp(rows[index])
        if timestamp is None:
            missing_timestamp += 1
            continue
        grouped.setdefault(subject_ids[index], []).append((timestamp, index))
    if not grouped:
        raise ValueError(
            "no chronological studies found: provide StudyDate/StudyTime in "
            "the manifest or pass --metadata-csv"
        )
    pairs = []
    for subject_id, studies in grouped.items():
        studies.sort(
            key=lambda item: (
                item[0],
                str(rows[item[1]].get("study_id", "")),
                item[1],
            )
        )
        for (_, before), (_, after) in zip(studies, studies[1:]):
            if not (
                bool(known[before, target_id])
                and bool(known[before, device_id])
                and bool(known[after, target_id])
                and bool(known[after, device_id])
            ):
                continue
            pairs.append(
                TransitionPair(
                    before=before,
                    after=after,
                    subject_id=subject_id,
                    target_before=int(labels[before, target_id]),
                    target_after=int(labels[after, target_id]),
                    device_before=int(labels[before, device_id]),
                    device_after=int(labels[after, device_id]),
                    view_changed=_view(rows[before]) != _view(rows[after]),
                )
            )
    if not pairs:
        raise ValueError(
            "chronological studies exist but no consecutive pair has known "
            "target and device labels at both time points"
        )
    return pairs


def transition_counts(pairs: Sequence[TransitionPair]) -> dict[str, int]:
    result = {
        "all_consecutive_known": len(pairs),
        "disease_change_device_stable": 0,
        "disease_stable_device_change": 0,
        "disease_resolves_device_remains": 0,
        "both_stable": 0,
        "both_change": 0,
        "view_changed": 0,
    }
    for pair in pairs:
        result[pair.stratum] += 1
        result["disease_resolves_device_remains"] += int(
            pair.target_resolved and pair.device_remains
        )
        result["view_changed"] += int(pair.view_changed)
    return result


def select_transition_pairs(
    pairs: Sequence[TransitionPair],
    maximum_per_stratum: int,
    seed: int,
) -> list[TransitionPair]:
    """Cap the three interpretable strata and exclude simultaneous changes."""

    if maximum_per_stratum <= 0:
        raise ValueError("maximum_per_stratum must be positive")
    grouped: dict[tuple[bool, bool], list[TransitionPair]] = {}
    for pair in pairs:
        if pair.target_changed and pair.device_changed:
            continue
        grouped.setdefault(
            (pair.target_changed, pair.device_changed), []
        ).append(pair)
    generator = torch.Generator().manual_seed(seed)
    selected = []
    for key in sorted(grouped):
        values = grouped[key]
        order = torch.randperm(len(values), generator=generator)[
            :maximum_per_stratum
        ]
        selected.extend(values[int(index)] for index in order)
    selected.sort(key=lambda pair: (pair.subject_id, pair.before, pair.after))
    return selected


def register_translation(
    before: torch.Tensor,
    after: torch.Tensor,
    grid: int,
    max_shift: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Register token grids by the best global integer cosine translation.

    Returns the aligned follow-up grid, valid overlap mask, `(dy, dx)` shifts,
    and the mean-cosine registration score for every pair.
    """

    squeeze = before.ndim == 2
    if squeeze:
        before = before[None]
        after = after[None]
    if before.shape != after.shape or before.ndim != 3:
        raise ValueError("before and after must have identical [B,P,D] shape")
    if before.shape[1] != grid * grid:
        raise ValueError("patch count does not equal grid squared")
    if max_shift < 0 or max_shift >= grid:
        raise ValueError("max_shift must be in [0, grid)")
    first = F.normalize(before.float(), dim=-1).reshape(
        len(before), grid, grid, before.shape[-1]
    )
    second = F.normalize(after.float(), dim=-1).reshape_as(first)
    shifts = [
        (dy, dx)
        for dy in range(-max_shift, max_shift + 1)
        for dx in range(-max_shift, max_shift + 1)
    ]
    scores = []
    for dy, dx in shifts:
        y0, y1 = max(0, -dy), min(grid, grid - dy)
        x0, x1 = max(0, -dx), min(grid, grid - dx)
        scores.append(
            (
                first[:, y0:y1, x0:x1]
                * second[:, y0 + dy : y1 + dy, x0 + dx : x1 + dx]
            )
            .sum(dim=-1)
            .mean(dim=(1, 2))
        )
    score_matrix = torch.stack(scores, dim=1)
    best_score, best_index = score_matrix.max(dim=1)
    aligned = torch.zeros_like(second)
    valid = torch.zeros(
        len(before), grid, grid, dtype=torch.bool, device=before.device
    )
    best_shifts = torch.empty(
        len(before), 2, dtype=torch.long, device=before.device
    )
    for batch, shift_index in enumerate(best_index.tolist()):
        dy, dx = shifts[shift_index]
        y0, y1 = max(0, -dy), min(grid, grid - dy)
        x0, x1 = max(0, -dx), min(grid, grid - dx)
        aligned[batch, y0:y1, x0:x1] = second[
            batch,
            y0 + dy : y1 + dy,
            x0 + dx : x1 + dx,
        ]
        valid[batch, y0:y1, x0:x1] = True
        best_shifts[batch] = torch.tensor(
            (dy, dx), device=before.device
        )
    aligned = aligned.flatten(1, 2)
    valid = valid.flatten(1)
    if squeeze:
        return aligned[0], valid[0], best_shifts[0], best_score[0]
    return aligned, valid, best_shifts, best_score


def transition_feature_batch(
    before: torch.Tensor,
    after: torch.Tensor,
    grid: int,
    max_shift: int = 2,
) -> dict[str, torch.Tensor]:
    """Create label-free linear features from registered token residuals."""

    first = F.normalize(before.float(), dim=-1)
    aligned, valid, shifts, registration_score = register_translation(
        first, after, grid, max_shift
    )
    squeeze = first.ndim == 2
    if squeeze:
        first = first[None]
        aligned = aligned[None]
        valid = valid[None]
        shifts = shifts[None]
        registration_score = registration_score[None]
    residual = aligned - first
    weights = valid[..., None].to(residual.dtype)
    denominator = valid.sum(dim=1).clamp_min(1).to(residual.dtype)
    signed_mean = (residual * weights).sum(dim=1) / denominator[:, None]
    absolute_mean = (residual.abs() * weights).sum(dim=1) / denominator[:, None]
    energy = residual.square().mean(dim=-1).sqrt()
    energy = energy * valid.to(energy.dtype)
    features = torch.cat((absolute_mean, energy), dim=-1)
    result = {
        "features": features,
        "signed_mean": signed_mean,
        "energy_map": energy,
        "valid": valid,
        "shifts": shifts,
        "registration_score": registration_score,
    }
    if squeeze:
        return {key: value[0] for key, value in result.items()}
    return result


def canonical_pathology_atom(
    signed_residual: torch.Tensor,
    target_before: torch.Tensor,
    target_after: torch.Tensor,
    device_changed: torch.Tensor,
) -> torch.Tensor:
    """Average onset-oriented disease residuals with stable device state."""

    changed = target_before.ne(target_after) & ~device_changed.bool()
    if not changed.any():
        raise ValueError("no disease-change/device-stable residuals")
    orientation = torch.where(
        target_before[changed].eq(0),
        torch.ones_like(target_before[changed], dtype=signed_residual.dtype),
        -torch.ones_like(target_before[changed], dtype=signed_residual.dtype),
    )
    atom = (
        signed_residual[changed] * orientation[:, None]
    ).mean(dim=0)
    if float(atom.norm()) <= 1e-12:
        raise ValueError("canonical temporal atom is numerically zero")
    return F.normalize(atom, dim=0)


def covariance_eigendecomposition(
    features,
    indices: torch.Tensor,
    device: torch.device,
    batch_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate covariance from unlabeled rows and return mean/eigensystem."""

    if len(indices) < 2:
        raise ValueError("covariance estimation needs at least two examples")
    width = int(features.shape[1])
    total = torch.zeros(width, device=device, dtype=torch.float32)
    gram = torch.zeros(width, width, device=device, dtype=torch.float32)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size].cpu().numpy()
        batch = torch.as_tensor(
            features[selected], device=device, dtype=torch.float32
        )
        total += batch.sum(dim=0)
        gram += batch.T @ batch
    count = len(indices)
    mean = total / count
    covariance = (
        gram - count * torch.outer(mean, mean)
    ) / (count - 1)
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    eigenvalues = eigenvalues.clamp_min(0)
    return (
        mean.float().cpu(),
        eigenvalues.float().cpu(),
        eigenvectors.float().cpu(),
    )


def apply_shrinkage_precision(
    directions: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    coordinates = directions.float() @ eigenvectors.float()
    transformed = (
        coordinates / (eigenvalues.float() + ridge)
    ) @ eigenvectors.float().T
    return F.normalize(transformed, dim=-1)


def select_support_indices(
    panels: torch.Tensor,
    environment_choices: torch.Tensor,
    support_count: int,
) -> torch.Tensor:
    """Select global row indices from `[E,2,C]` candidate panels."""

    if panels.ndim != 3 or panels.shape[1] != 2:
        raise ValueError("panels must be [episodes,2,candidates]")
    if support_count <= 0 or support_count > environment_choices.shape[1]:
        raise ValueError("invalid support_count")
    selected = torch.empty(
        panels.shape[0], support_count, dtype=torch.long
    )
    counts = torch.zeros(panels.shape[0], 2, dtype=torch.long)
    for position in range(support_count):
        environment = environment_choices[:, position].long()
        for episode in range(panels.shape[0]):
            current = int(environment[episode])
            source = int(counts[episode, current])
            if source >= panels.shape[2]:
                raise ValueError("candidate panel is too small")
            selected[episode, position] = panels[
                episode, current, source
            ]
            counts[episode, current] += 1
    return selected


def pleural_proxy_mask(grid: int, band: int | None = None) -> torch.Tensor:
    """Coarse peripheral-lung review mask; not a clinical ROI annotation."""

    if grid < 5:
        raise ValueError("pleural proxy requires grid >= 5")
    band = max(1, grid // 7) if band is None else band
    if band <= 0 or 2 * band >= grid:
        raise ValueError("invalid pleural band")
    mask = torch.zeros(grid, grid, dtype=torch.bool)
    # Avoid the literal image frame; emphasize lateral and apical pleura.
    mask[1 : grid - 1, 1 : 1 + band] = True
    mask[1 : grid - 1, grid - 1 - band : grid - 1] = True
    mask[1 : 1 + band, 1 : grid - 1] = True
    return mask.flatten()


def localization_statistics(
    energy_map: torch.Tensor,
    group: Sequence[str],
    grid: int,
) -> dict[str, dict[str, float]]:
    if energy_map.ndim != 2 or energy_map.shape[1] != grid * grid:
        raise ValueError("energy maps must be [N, grid squared]")
    pleural = pleural_proxy_mask(grid).to(energy_map.device)
    border = torch.zeros(grid, grid, dtype=torch.bool, device=energy_map.device)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    border = border.flatten()
    result = {}
    for name in sorted(set(group)):
        selected = torch.tensor(
            [value == name for value in group],
            dtype=torch.bool,
            device=energy_map.device,
        )
        if not selected.any():
            continue
        values = energy_map[selected].float()
        maxima = values.argmax(dim=-1)
        center = ~(pleural | border)
        result[name] = {
            "count": int(selected.sum()),
            "border_max_fraction": float(border[maxima].float().mean()),
            "pleural_proxy_max_fraction": float(
                pleural[maxima].float().mean()
            ),
            "pleural_to_center_energy_ratio": float(
                values[:, pleural].mean()
                / values[:, center].mean().clamp_min(1e-8)
            ),
        }
    return result
