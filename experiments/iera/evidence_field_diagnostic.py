"""Scoring-only evidence-field pilot on frozen Rad-DINO features/adapters."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch

from experiments.residuals.metrics import _auc, select_temperature, select_threshold

from .evidence_field import evidence_field_grid, evidence_field_score
from .falsification import _gather, _selected, _write
from .patch_cache import RAD_DINO_MODEL, load_patch_cache
from .robust_metrics import evaluate, normalized_sms
from .robust_model import RobustBinaryModel


FIELD_FAMILIES = (
    "naive_dense",
    "evidence_field_unadapted",
    "evidence_field_frozen_adapter",
)


def _candidate(family: str, tau_support: float, tau_query: float) -> str:
    return f"{family}|tau_s={tau_support:g}|tau_q={tau_query:g}"


def _candidate_fields(candidate: str) -> dict[str, str]:
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in candidate.split("|")[1:]
    }


def _family(candidate: str) -> str:
    return candidate.split("|", 1)[0]


def _compact(
    tokens: torch.Tensor, mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove environment padding without changing image-level weighting."""
    counts = mask.flatten(1).sum(dim=1)
    if not counts.eq(counts[0]).all():
        raise ValueError("support count must be constant within an episode batch")
    count = int(counts[0])
    flattened = tokens.flatten(1, 2)
    valid = mask.flatten(1)
    selected = torch.stack(
        [values[current] for values, current in zip(flattened, valid)]
    )
    selected = selected.reshape(
        tokens.shape[0], 1, count, tokens.shape[-2], tokens.shape[-1]
    )
    selected_mask = torch.ones(
        tokens.shape[0], 1, count, dtype=torch.bool, device=tokens.device
    )
    return selected, selected_mask


def _field_candidates(
    model: RobustBinaryModel,
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    tau_supports: list[float],
    tau_queries: list[float],
    query_chunk_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    positive, positive_mask = _compact(positive, positive_mask)
    negative, negative_mask = _compact(negative, negative_mask)
    unadapted = evidence_field_grid(
        positive,
        negative,
        query,
        positive_mask,
        negative_mask,
        tau_supports,
        tau_queries,
        pooling_modes=("image_balanced",),
        query_chunk_size=query_chunk_size,
    )
    adapted = evidence_field_grid(
        positive,
        negative,
        query,
        positive_mask,
        negative_mask,
        tau_supports,
        tau_queries,
        adapter=model,
        pooling_modes=("dense", "image_balanced"),
        query_chunk_size=query_chunk_size,
    )
    result = {}
    for tau_support in tau_supports:
        for tau_query in tau_queries:
            result[
                _candidate(
                    "evidence_field_unadapted", tau_support, tau_query
                )
            ] = unadapted[("image_balanced", tau_support, tau_query)]
            result[
                _candidate("naive_dense", tau_support, tau_query)
            ] = adapted[("dense", tau_support, tau_query)]
            result[
                _candidate(
                    "evidence_field_frozen_adapter",
                    tau_support,
                    tau_query,
                )
            ] = adapted[("image_balanced", tau_support, tau_query)]
    return result


def _score(
    model: RobustBinaryModel,
    patches,
    metadata: dict,
    episodes: dict,
    shot: int,
    batch_size: int,
    retained_grid: int,
    device: torch.device,
    tau_supports: list[float],
    tau_queries: list[float],
    query_chunk_size: int,
) -> dict[str, tuple[torch.Tensor, ...]]:
    destinations = defaultdict(lambda: ([], [], [], [], []))
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(episodes["positive"]), batch_size):
            end = min(start + batch_size, len(episodes["positive"]))
            positive_panels = _gather(
                patches,
                episodes["positive"][start:end, :, : 2 * shot],
                metadata,
                retained_grid,
                device,
            )
            negative_panels = _gather(
                patches,
                episodes["negative"][start:end, :, : 2 * shot],
                metadata,
                retained_grid,
                device,
            )
            query = _gather(
                patches,
                episodes["query"][start:end],
                metadata,
                retained_grid,
                device,
            )
            positive, negative, positive_mask, negative_mask = _selected(
                positive_panels,
                negative_panels,
                episodes,
                start,
                end,
                shot,
                False,
            )
            current_logits = model(
                positive,
                negative,
                query,
                "adapter_only",
                positive_mask,
                negative_mask,
            )
            ordinary = _field_candidates(
                model,
                positive,
                negative,
                query,
                positive_mask,
                negative_mask,
                tau_supports,
                tau_queries,
                query_chunk_size,
            )
            panel_fields = []
            for environment in (0, 1):
                panel_positive = positive_panels[
                    :, environment : environment + 1, :shot
                ]
                panel_negative = negative_panels[
                    :, environment : environment + 1, :shot
                ]
                panel_mask = torch.ones(
                    panel_positive.shape[:3],
                    dtype=torch.bool,
                    device=device,
                )
                panel_fields.append(
                    _field_candidates(
                        model,
                        panel_positive,
                        panel_negative,
                        query,
                        panel_mask,
                        panel_mask,
                        tau_supports,
                        tau_queries,
                        query_chunk_size,
                    )
                )
            panel_positive = positive_panels[:, :, :shot]
            panel_negative = negative_panels[:, :, :shot]
            current_zero, current_one = model.swapped_logits(
                panel_positive, panel_negative, query, "adapter_only"
            )
            reference_zero, reference_one = model.swapped_logits(
                panel_positive, panel_negative, query, "uniform"
            )
            candidates = {
                "current_adapter": (
                    current_logits,
                    current_zero,
                    current_one,
                )
            }
            for candidate, (logits, _) in ordinary.items():
                candidates[candidate] = (
                    logits,
                    panel_fields[0][candidate][0],
                    panel_fields[1][candidate][0],
                )
            for candidate, tensors in candidates.items():
                output = destinations[candidate]
                for destination, tensor in zip(
                    output,
                    (*tensors, reference_zero, reference_one),
                ):
                    destination.append(tensor.detach().cpu())
    return {
        candidate: tuple(torch.cat(items).flatten() for items in tensors)
        for candidate, tensors in destinations.items()
    }


def _validation_point(
    scores: dict,
    episodes: dict,
    candidate: str,
    pair_id: int,
    seeds: list[int],
    shot: int,
) -> dict[str, float | str]:
    aurocs, sensitivities = [], []
    for seed in seeds:
        values = scores[(candidate, pair_id, seed, shot, "validate")]
        targets = episodes[(pair_id, seed, "validate")]["targets"].flatten()
        aurocs.append(_auc(targets.bool(), values[0]))
        sensitivities.append(float(normalized_sms(*values[1:])))
    return {
        "candidate": candidate,
        "mean_auroc": statistics.mean(aurocs),
        "mean_sms_fixed_reference": statistics.mean(sensitivities),
    }


def _select(
    scores: dict,
    episodes: dict,
    pair_names: dict,
    seeds: list[int],
    primary_shot: int,
    near_tie: float,
) -> dict[int, dict]:
    candidates = sorted({key[0] for key in scores})
    selection = {}
    for pair_id, names in pair_names.items():
        selected = {
            "pair": f"{names[0]}__{names[1]}",
            "selection_partition": "validate",
            "selection_shot": primary_shot,
            "near_tie_auroc": near_tie,
            "methods": {},
        }
        for family in FIELD_FAMILIES:
            points = [
                _validation_point(
                    scores,
                    episodes,
                    candidate,
                    pair_id,
                    seeds,
                    primary_shot,
                )
                for candidate in candidates
                if _family(candidate) == family
            ]
            best_auroc = max(float(point["mean_auroc"]) for point in points)
            near_best = [
                point
                for point in points
                if float(point["mean_auroc"]) >= best_auroc - near_tie
            ]
            chosen = min(
                near_best,
                key=lambda point: (
                    float(point["mean_sms_fixed_reference"]),
                    -float(point["mean_auroc"]),
                    str(point["candidate"]),
                ),
            )
            fields = _candidate_fields(str(chosen["candidate"]))
            selected["methods"][family] = {
                **chosen,
                "tau_support": float(fields["tau_s"]),
                "tau_query": float(fields["tau_q"]),
                "candidates": sorted(
                    points,
                    key=lambda point: (
                        -float(point["mean_auroc"]),
                        float(point["mean_sms_fixed_reference"]),
                    ),
                ),
            }
        selection[pair_id] = selected
    return selection


def _metric_rows(
    scores: dict,
    episodes: dict,
    pair_names: dict,
    seeds: list[int],
    shots: list[int],
) -> list[dict]:
    rows = []
    candidates = sorted({key[0] for key in scores})
    for pair_id, (target_name, confounder_name) in pair_names.items():
        pair = f"{target_name}__{confounder_name}"
        for candidate in candidates:
            fields = _candidate_fields(candidate)
            for seed in seeds:
                for shot in shots:
                    validation = scores[
                        (candidate, pair_id, seed, shot, "validate")
                    ]
                    validation_targets = episodes[
                        (pair_id, seed, "validate")
                    ]["targets"].flatten()
                    calibration = select_temperature(
                        validation[0][:, None],
                        validation_targets[:, None],
                        "multi_label",
                    )
                    threshold = select_threshold(
                        validation[0][:, None],
                        validation_targets[:, None],
                        calibration,
                    )
                    for partition in ("validate", "test"):
                        values = scores[
                            (candidate, pair_id, seed, shot, partition)
                        ]
                        current = episodes[(pair_id, seed, partition)]
                        metrics = evaluate(
                            *values,
                            current["targets"].flatten(),
                            current["nuisance"].flatten(),
                            calibration,
                            threshold,
                        )
                        for metric, value in metrics.items():
                            rows.append(
                                {
                                    "partition": partition,
                                    "pair": pair,
                                    "target": target_name,
                                    "confounder": confounder_name,
                                    "method": _family(candidate),
                                    "candidate": candidate,
                                    "tau_support": fields.get("tau_s", ""),
                                    "tau_query": fields.get("tau_q", ""),
                                    "shot": shot,
                                    "seed": seed,
                                    "metric": metric,
                                    "value": value,
                                }
                            )
    return rows


def _selected_rows(rows: list[dict], selection: dict) -> list[dict]:
    chosen = {}
    for item in selection.values():
        pair = item["pair"]
        chosen[(pair, "current_adapter")] = "current_adapter"
        for family, details in item["methods"].items():
            chosen[(pair, family)] = details["candidate"]
    return [
        row
        for row in rows
        if chosen.get((row["pair"], row["method"])) == row["candidate"]
    ]


def _summaries(rows: list[dict]) -> list[dict]:
    keys = (
        "partition",
        "pair",
        "target",
        "confounder",
        "method",
        "candidate",
        "tau_support",
        "tau_query",
        "shot",
        "metric",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row["value"]))
    summaries = []
    for key, values in groups.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half = 1.96 * std / math.sqrt(len(values))
        summaries.append(
            {
                **dict(zip(keys, key)),
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            }
        )
    return summaries


def _paired_interval(values: list[float]) -> tuple[float, float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    half = 1.96 * std / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _decision(
    per_seed: list[dict],
    primary_shot: int,
    retained_grid: int,
    native_grid: int,
    visual_review: str,
) -> dict:
    values = defaultdict(dict)
    for row in per_seed:
        if (
            row["partition"] == "test"
            and int(row["shot"]) == primary_shot
            and row["method"]
            in {"current_adapter", "evidence_field_frozen_adapter"}
            and row["metric"]
            in {"auroc", "sms_fixed_reference", "worst_nuisance_auroc"}
        ):
            values[
                (row["pair"], row["method"], int(row["seed"]))
            ][row["metric"]] = float(row["value"])
    comparisons = []
    pairs = sorted(
        {
            key[0]
            for key in values
            if key[0].startswith("Pneumothorax")
            and key[1] == "evidence_field_frozen_adapter"
        }
    )
    for pair in pairs:
        seeds = sorted(
            key[2]
            for key in values
            if key[0] == pair
            and key[1] == "evidence_field_frozen_adapter"
        )
        current = [values[(pair, "current_adapter", seed)] for seed in seeds]
        field = [
            values[(pair, "evidence_field_frozen_adapter", seed)]
            for seed in seeds
        ]
        auroc_delta = [
            candidate["auroc"] - baseline["auroc"]
            for baseline, candidate in zip(current, field)
        ]
        worst_delta = [
            candidate["worst_nuisance_auroc"]
            - baseline["worst_nuisance_auroc"]
            for baseline, candidate in zip(current, field)
        ]
        auroc = _paired_interval(auroc_delta)
        worst = _paired_interval(worst_delta)
        current_sms = statistics.mean(
            item["sms_fixed_reference"] for item in current
        )
        field_sms = statistics.mean(
            item["sms_fixed_reference"] for item in field
        )
        comparisons.append(
            {
                "pair": pair,
                "current_adapter_auroc": statistics.mean(
                    item["auroc"] for item in current
                ),
                "evidence_field_auroc": statistics.mean(
                    item["auroc"] for item in field
                ),
                "paired_auroc_gain": auroc[0],
                "paired_auroc_gain_ci95": [auroc[1], auroc[2]],
                "current_adapter_sms": current_sms,
                "evidence_field_sms": field_sms,
                "sms_ratio_to_current_adapter": field_sms
                / max(current_sms, 1e-12),
                "paired_worst_device_auroc_change": worst[0],
                "paired_worst_device_auroc_change_ci95": [
                    worst[1],
                    worst[2],
                ],
                "quantitative_gate_passed": (
                    auroc[0] >= 0.02
                    and field_sms <= 1.25 * current_sms
                    and worst[0] >= -0.01
                ),
                "credible_057_target_reached": statistics.mean(
                    item["auroc"] for item in field
                )
                >= 0.57,
            }
        )
    quantitative = bool(comparisons) and all(
        item["quantitative_gate_passed"] for item in comparisons
    )
    if quantitative and visual_review == "pass":
        status = "proceed_to_counterfactual_field_training"
    elif quantitative and visual_review == "pending":
        status = "await_evidence_map_review"
    elif quantitative:
        status = "stop_evidence_field_chest_tube_attention"
    elif retained_grid < native_grid:
        status = "try_native_resolution_once"
    else:
        status = "abandon_evidence_field"
    return {
        "status": status,
        "stage_two_training_started": False,
        "primary_shot": primary_shot,
        "retained_grid": retained_grid,
        "native_grid": native_grid,
        "visual_review": visual_review,
        "gate": {
            "minimum_pneumothorax_auroc_gain": 0.02,
            "maximum_sms_ratio_to_current_adapter": 1.25,
            "minimum_worst_device_auroc_change": -0.01,
            "evidence_map_must_not_primarily_highlight_chest_tubes": True,
        },
        "comparisons": comparisons,
    }


def _open_manifest(path: Path) -> list[dict[str, str]]:
    if path.suffix == ".gz":
        handle = gzip.open(path, "rt", encoding="utf-8", newline="")
    else:
        handle = path.open("r", encoding="utf-8", newline="")
    with handle:
        return list(csv.DictReader(handle))


def _render_maps(
    records: list[dict],
    manifest_rows: list[dict[str, str]],
    data_root: Path,
    output: Path,
    retained_grid: int,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw, ImageOps

    tile = 336
    label_height = 34
    columns = min(4, len(records))
    rows = math.ceil(len(records) / columns)
    canvas = Image.new("RGB", (columns * tile, rows * (tile + label_height)), "black")
    draw = ImageDraw.Draw(canvas)
    for position, record in enumerate(records):
        row = manifest_rows[record["dataset_index"]]
        path = data_root / row["relative_path"]
        with Image.open(path) as source:
            image = ImageOps.fit(source.convert("L"), (tile, tile)).convert("RGB")
        field = record["evidence_map"].float().reshape(
            retained_grid, retained_grid
        ).numpy()
        scale = max(float(np.percentile(np.abs(field), 95)), 1e-6)
        normalized = np.clip(field / scale, -1, 1)
        color = np.zeros((retained_grid, retained_grid, 3), dtype=np.uint8)
        color[..., 0] = np.clip(normalized, 0, 1) * 255
        color[..., 2] = np.clip(-normalized, 0, 1) * 255
        heat = Image.fromarray(color).resize((tile, tile), Image.Resampling.BILINEAR)
        alpha = Image.fromarray(
            (np.abs(normalized) * 170).astype(np.uint8)
        ).resize((tile, tile), Image.Resampling.BILINEAR)
        overlay = Image.composite(heat, image, alpha)
        x = (position % columns) * tile
        y = (position // columns) * (tile + label_height)
        canvas.paste(overlay, (x, y))
        draw.text(
            (x + 5, y + tile + 5),
            (
                f"{record.get('method', '')} "
                f"target={record['target']} device={record['nuisance']} "
                f"logit={record['logit']:.3f}"
            ),
            fill="white",
        )
    canvas.save(output)


def _visualize(
    model: RobustBinaryModel,
    patches,
    metadata: dict,
    episodes: dict,
    pair_names: dict,
    scores: dict,
    selection: dict,
    seed: int,
    shot: int,
    retained_grid: int,
    device: torch.device,
    query_chunk_size: int,
    manifest: Path,
    data_root: Path,
    output_dir: Path,
    examples_per_group: int,
) -> None:
    pair_ids = [
        pair_id
        for pair_id, names in pair_names.items()
        if names[0] == "Pneumothorax"
    ]
    if not pair_ids:
        return
    pair_id = pair_ids[0]
    details = selection[pair_id]["methods"][
        "evidence_field_frozen_adapter"
    ]
    candidate = details["candidate"]
    values = scores[(candidate, pair_id, seed, shot, "test")][0]
    current = episodes[(pair_id, seed, "test")]
    targets = current["targets"].flatten()
    nuisance = current["nuisance"].flatten()
    query_count = current["query"].shape[1]
    chosen = []
    for device_value in (0, 1):
        eligible = torch.where(targets.bool() & nuisance.eq(device_value))[0]
        ranked = eligible[torch.argsort(values[eligible], descending=True)]
        chosen.extend(ranked[:examples_per_group].tolist())
    if not chosen:
        raise ValueError("no positive Pneumothorax queries are available to visualize")
    records = []
    with torch.inference_mode():
        for flat_index in chosen:
            episode_index, query_index = divmod(flat_index, query_count)
            positive_panels = _gather(
                patches,
                current["positive"][
                    episode_index : episode_index + 1, :, : 2 * shot
                ],
                metadata,
                retained_grid,
                device,
            )
            negative_panels = _gather(
                patches,
                current["negative"][
                    episode_index : episode_index + 1, :, : 2 * shot
                ],
                metadata,
                retained_grid,
                device,
            )
            query = _gather(
                patches,
                current["query"][
                    episode_index : episode_index + 1,
                    query_index : query_index + 1,
                ],
                metadata,
                retained_grid,
                device,
            )
            positive, negative, positive_mask, negative_mask = _selected(
                positive_panels,
                negative_panels,
                current,
                episode_index,
                episode_index + 1,
                shot,
                False,
            )
            positive, positive_mask = _compact(positive, positive_mask)
            negative, negative_mask = _compact(negative, negative_mask)
            logits, field = evidence_field_score(
                positive,
                negative,
                query,
                positive_mask,
                negative_mask,
                details["tau_support"],
                details["tau_query"],
                adapter=model,
                query_chunk_size=query_chunk_size,
            )
            records.append(
                {
                    "dataset_index": int(
                        current["query"][episode_index, query_index]
                    ),
                    "episode_index": episode_index,
                    "query_index": query_index,
                    "target": int(targets[flat_index]),
                    "nuisance": int(nuisance[flat_index]),
                    "logit": float(logits[0, 0]),
                    "evidence_map": field[0, 0].cpu(),
                }
            )
    torch.save(
        {
            "candidate": candidate,
            "seed": seed,
            "shot": shot,
            "records": records,
        },
        output_dir / "evidence_maps.pt",
    )
    rows = _open_manifest(manifest)
    _render_maps(
        records,
        rows,
        data_root.expanduser(),
        output_dir / "evidence_maps.png",
        retained_grid,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--rad-cache", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/iera/evidence_field_pilot_v1"),
    )
    parser.add_argument("--adapter-rho", type=float, default=0.7)
    parser.add_argument("--retained-grid", type=int, default=14)
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--primary-shot", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--targets",
        nargs="+",
        help="Restrict the retry to saved pairs with these target names",
    )
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument(
        "--tau-supports", type=float, nargs="+", default=(0.05, 0.1, 0.2)
    )
    parser.add_argument(
        "--tau-queries", type=float, nargs="+", default=(0.05, 0.1, 0.2)
    )
    parser.add_argument("--near-tie-auroc", type=float, default=0.005)
    parser.add_argument("--episode-batch-size", type=int, default=4)
    parser.add_argument("--query-chunk-size", type=int, default=1)
    parser.add_argument("--visual-examples-per-group", type=int, default=4)
    parser.add_argument("--visual-review", choices=("pending", "pass", "fail"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.primary_shot not in args.shots:
        parser.error("primary-shot must be included in shots")
    if (args.manifest is None) != (args.data_root is None):
        parser.error("manifest and data-root must be supplied together")
    if args.near_tie_auroc < 0:
        parser.error("near-tie-auroc cannot be negative")
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    saved = torch.load(args.episodes, map_location="cpu", weights_only=False)
    signature = saved["signature"]
    if not set(args.seeds).issubset(set(signature["seeds"])):
        raise ValueError("requested seeds are absent from the saved episodes")
    if args.episodes_per_seed > signature["episodes"]:
        raise ValueError("saved episode bank is smaller than requested")
    pair_names = {
        pair_id: names
        for pair_id, names in saved["pairs"].items()
        if args.targets is None or names[0] in args.targets
    }
    if not pair_names:
        raise ValueError("no saved episode pair matches --targets")
    episode_sets = {
        key: {
            name: (
                value[: args.episodes_per_seed]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == signature["episodes"]
                else value
            )
            for name, value in current.items()
        }
        for key, current in saved["episodes"].items()
        if key[0] in pair_names and key[1] in args.seeds
    }
    patches, metadata = load_patch_cache(
        args.rad_cache,
        signature["manifest_sha256"],
        expected_model=RAD_DINO_MODEL,
        access_mode="stream",
    )
    if args.retained_grid > int(metadata["pool_grid"]):
        raise ValueError("retained grid exceeds the cached token grid")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_signature = {
        "manifest_sha256": signature["manifest_sha256"],
        "retained_grid": args.retained_grid,
        "shots": list(args.shots),
        "episodes_per_seed": args.episodes_per_seed,
        "tau_supports": list(args.tau_supports),
        "tau_queries": list(args.tau_queries),
        "adapter_rho": args.adapter_rho,
        "adapter_dir": str(args.adapter_dir),
        "query_chunk_size": args.query_chunk_size,
        "targets": (
            None if args.targets is None else sorted(set(args.targets))
        ),
    }
    scores = {}
    models = {}
    for seed in args.seeds:
        checkpoint = args.adapter_dir / (
            f"model_adapter_only_rho_{args.adapter_rho:g}_seed_{seed:03d}.pt"
        )
        saved_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved_model.get("method") not in (None, "adapter_only"):
            raise ValueError(f"{checkpoint} is not an adapter-only checkpoint")
        model = RobustBinaryModel(
            int(metadata["shape"][-1]),
            adapter_dim=int(
                saved_model["state_dict"]["support_down.weight"].shape[0]
            ),
        ).to(device)
        model.load_state_dict(saved_model["state_dict"])
        model.eval().requires_grad_(False)
        models[seed] = model
        score_path = args.output_dir / f"scores_seed_{seed:03d}.pt"
        if score_path.exists():
            completed = torch.load(score_path, map_location="cpu", weights_only=False)
            if completed.get("signature") != score_signature:
                raise ValueError(f"{score_path} uses different pilot arguments")
            scores.update(completed["scores"])
            print(f"reusing evidence-field scores for seed={seed}", flush=True)
            continue
        seed_scores = {}
        for pair_id in sorted(pair_names):
            for partition in ("validate", "test"):
                current = episode_sets[(pair_id, seed, partition)]
                for shot in args.shots:
                    computed = _score(
                        model,
                        patches,
                        metadata,
                        current,
                        shot,
                        args.episode_batch_size,
                        args.retained_grid,
                        device,
                        list(args.tau_supports),
                        list(args.tau_queries),
                        args.query_chunk_size,
                    )
                    for candidate, tensors in computed.items():
                        seed_scores[
                            (candidate, pair_id, seed, shot, partition)
                        ] = tensors
                print(
                    f"finished evidence fields, pair={pair_id}, "
                    f"{partition}, seed={seed}",
                    flush=True,
                )
        temporary = score_path.with_suffix(".tmp")
        torch.save(
            {"signature": score_signature, "scores": seed_scores}, temporary
        )
        temporary.replace(score_path)
        scores.update(seed_scores)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection = _select(
        scores,
        episode_sets,
        pair_names,
        list(args.seeds),
        args.primary_shot,
        args.near_tie_auroc,
    )
    candidate_rows = _metric_rows(
        scores,
        episode_sets,
        pair_names,
        list(args.seeds),
        list(args.shots),
    )
    per_seed = _selected_rows(candidate_rows, selection)
    summaries = _summaries(per_seed)
    review_path = args.output_dir / "visual_review.json"
    if args.visual_review is not None:
        visual_review = args.visual_review
    elif review_path.exists():
        visual_review = json.loads(
            review_path.read_text(encoding="utf-8")
        ).get("status", "pending")
    else:
        visual_review = "pending"
    if visual_review not in {"pending", "pass", "fail"}:
        raise ValueError("visual_review.json has an invalid status")
    review_path.write_text(
        json.dumps(
            {
                "status": visual_review,
                "criterion": (
                    "positive evidence must not primarily highlight chest tubes"
                ),
                "artifact": "evidence_maps.png",
                "update": (
                    "rerun with --visual-review pass or --visual-review fail "
                    "after inspecting the overlay"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.manifest is not None:
        manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
        if manifest_hash != signature["manifest_sha256"]:
            raise ValueError("manifest does not match the saved episodes")
        _visualize(
            models[args.seeds[0]],
            patches,
            metadata,
            episode_sets,
            pair_names,
            scores,
            selection,
            args.seeds[0],
            args.primary_shot,
            args.retained_grid,
            device,
            args.query_chunk_size,
            args.manifest,
            args.data_root,
            args.output_dir,
            args.visual_examples_per_group,
        )
    if (
        visual_review == "pass"
        and not (args.output_dir / "evidence_maps.png").is_file()
    ):
        raise ValueError(
            "visual review cannot pass before evidence_maps.png is generated"
        )
    decision = _decision(
        per_seed,
        args.primary_shot,
        args.retained_grid,
        int(metadata.get("native_grid", metadata["pool_grid"])),
        visual_review,
    )
    _write(args.output_dir / "candidate_metrics.csv", candidate_rows)
    _write(args.output_dir / "per_seed_metrics.csv", per_seed)
    _write(args.output_dir / "summary_metrics.csv", summaries)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "stage": "scoring_only_evidence_field",
                "counterfactual_field_training_started": False,
                "backbone": metadata,
                "episodes": str(args.episodes),
                "adapter_dir": str(args.adapter_dir),
                "adapter_rho": args.adapter_rho,
                "adapter_frozen": True,
                "query_frozen": True,
                "fixed_reference": "unadapted Rad-DINO binary ProtoNet",
                "naive_dense_adapter": "same frozen adapter",
                "retained_grid": args.retained_grid,
                "shots": args.shots,
                "seeds": args.seeds,
                "targets": args.targets,
                "episodes_per_seed": args.episodes_per_seed,
                "tau_supports": args.tau_supports,
                "tau_queries": args.tau_queries,
                "temperatures_learned": False,
                "selection_partition": "validate",
                "selection_shot": args.primary_shot,
                "near_tie_auroc": args.near_tie_auroc,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"evidence-field results written to {args.output_dir}; "
        f"decision={decision['status']}"
    )


if __name__ == "__main__":
    main()
