"""Stage-1 scoring-only falsification for Stable Region Witnesses."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.residuals.metrics import _auc, select_temperature, select_threshold

from .dual_head import support_adapter
from .evidence_field_diagnostic import _open_manifest, _render_maps
from .falsification import _gather, _selected, _write
from .patch_cache import RAD_DINO_MODEL, load_patch_cache
from .robust_metrics import evaluate, normalized_sms
from .robust_model import RobustBinaryModel
from .stable_witness import (
    border_maximum,
    certified_witness_scores,
    compact_support_images,
    dn4_hard_knn_score,
    relational_descriptor,
)


TUNED_FAMILIES = (
    "raw_witness",
    "relational_witness",
    "anchor_plus_relational_witness",
)


def _candidate(
    family: str,
    fraction: float | None = None,
    gamma: float | None = None,
) -> str:
    fields = [family]
    if fraction is not None:
        fields.append(f"r={fraction:g}")
    if gamma is not None:
        fields.append(f"gamma={gamma:g}")
    return "|".join(fields)


def _fields(candidate: str) -> dict[str, str]:
    return {
        item.split("=", 1)[0]: item.split("=", 1)[1]
        for item in candidate.split("|")[1:]
    }


def _family(candidate: str) -> str:
    return candidate.split("|", 1)[0]


def _anchor_field(
    model: RobustBinaryModel,
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    positive = support_adapter(positive, model)
    negative = support_adapter(negative, model)
    positive_prototype = model._prototype(positive, positive_mask)
    negative_prototype = model._prototype(negative, negative_mask)
    query = F.normalize(query.float(), dim=-1)
    return (
        torch.einsum("bqtd,bd->bqt", query, positive_prototype)
        - torch.einsum("bqtd,bd->bqt", query, negative_prototype)
    )


def _local_scores(
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    fractions: list[float],
    token_chunk_size: int,
    query_chunk_size: int,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    positive = compact_support_images(positive, positive_mask)
    negative = compact_support_images(negative, negative_mask)
    raw = certified_witness_scores(
        positive,
        negative,
        query,
        fractions,
        token_chunk_size=token_chunk_size,
        query_chunk_size=query_chunk_size,
    )
    relational = certified_witness_scores(
        relational_descriptor(positive),
        relational_descriptor(negative),
        relational_descriptor(query),
        fractions,
        token_chunk_size=token_chunk_size,
        query_chunk_size=query_chunk_size,
    )
    dn4 = dn4_hard_knn_score(
        positive,
        negative,
        query,
        neighbours=3,
        query_chunk_size=query_chunk_size,
    )
    result = {"dn4_hard_knn": dn4}
    for fraction in fractions:
        result[_candidate("raw_witness", fraction)] = raw[fraction]
        result[_candidate("relational_witness", fraction)] = relational[
            fraction
        ]
    return result


def _score(
    model: RobustBinaryModel,
    patches,
    metadata: dict,
    episodes: dict,
    shot: int,
    retained_grid: int,
    device: torch.device,
    fractions: list[float],
    gammas: list[float],
    token_chunk_size: int,
    query_chunk_size: int,
) -> tuple[dict[str, tuple[torch.Tensor, ...]], dict[str, torch.Tensor]]:
    destinations = defaultdict(lambda: ([], [], [], [], [], []))
    field_destinations = defaultdict(list)
    model.eval()
    with torch.inference_mode():
        for episode_index in range(len(episodes["positive"])):
            start, end = episode_index, episode_index + 1
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
            anchor_logits = model(
                positive,
                negative,
                query,
                "adapter_only",
                positive_mask,
                negative_mask,
            )
            anchor_map = _anchor_field(
                model,
                positive,
                negative,
                query,
                positive_mask,
                negative_mask,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                ordinary = _local_scores(
                    positive,
                    negative,
                    query,
                    positive_mask,
                    negative_mask,
                    fractions,
                    token_chunk_size,
                    query_chunk_size,
                )
            ordinary["anchor_rho03"] = (anchor_logits, anchor_map)
            for fraction in fractions:
                relational_name = _candidate(
                    "relational_witness", fraction
                )
                local_logits, local_map = ordinary[relational_name]
                for gamma in gammas:
                    ordinary[
                        _candidate(
                            "anchor_plus_relational_witness",
                            fraction,
                            gamma,
                        )
                    ] = (
                        anchor_logits + gamma * local_logits,
                        gamma * local_map,
                    )
            panel_scores = []
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
                anchor_panel = model(
                    panel_positive,
                    panel_negative,
                    query,
                    "adapter_only",
                    panel_mask,
                    panel_mask,
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    current = _local_scores(
                        panel_positive,
                        panel_negative,
                        query,
                        panel_mask,
                        panel_mask,
                        fractions,
                        token_chunk_size,
                        query_chunk_size,
                    )
                current["anchor_rho03"] = (anchor_panel, None)
                for fraction in fractions:
                    relational_name = _candidate(
                        "relational_witness", fraction
                    )
                    local_panel = current[relational_name][0]
                    for gamma in gammas:
                        current[
                            _candidate(
                                "anchor_plus_relational_witness",
                                fraction,
                                gamma,
                            )
                        ] = (anchor_panel + gamma * local_panel, None)
                panel_scores.append(current)
            reference_zero, reference_one = model.swapped_logits(
                positive_panels[:, :, :shot],
                negative_panels[:, :, :shot],
                query,
                "uniform",
            )
            for candidate, (logits, evidence_map) in ordinary.items():
                output = destinations[candidate]
                tensors = (
                    logits,
                    panel_scores[0][candidate][0],
                    panel_scores[1][candidate][0],
                    reference_zero,
                    reference_one,
                    border_maximum(evidence_map),
                )
                for destination, tensor in zip(output, tensors):
                    destination.append(tensor.detach().cpu())
                if _family(candidate) != "anchor_plus_relational_witness":
                    field_destinations[candidate].append(
                        evidence_map.detach().to("cpu", dtype=torch.float16)
                    )
            print(
                f"scored witnesses {end:,}/{len(episodes['positive']):,}",
                flush=True,
            )
    scores = {
        candidate: tuple(torch.cat(items).flatten() for items in tensors)
        for candidate, tensors in destinations.items()
    }
    fields = {
        candidate: torch.cat(items)
        for candidate, items in field_destinations.items()
    }
    return scores, fields


def _validation_point(
    scores: dict,
    episodes: dict,
    candidate: str,
    pair_id: int,
    seeds: list[int],
    shot: int,
) -> dict:
    aurocs, sensitivities, worst, borders = [], [], [], []
    for seed in seeds:
        values = scores[(candidate, pair_id, seed, shot, "validate")]
        current = episodes[(pair_id, seed, "validate")]
        targets = current["targets"].flatten().bool()
        nuisance = current["nuisance"].flatten()
        aurocs.append(_auc(targets, values[0]))
        sensitivities.append(float(normalized_sms(*values[1:5])))
        group_aurocs = [
            _auc(targets[nuisance.eq(value)], values[0][nuisance.eq(value)])
            for value in (0, 1)
        ]
        worst.append(min(group_aurocs))
        borders.append(float(values[5].float().mean()))
    return {
        "candidate": candidate,
        "mean_auroc": statistics.mean(aurocs),
        "mean_sms_fixed_reference": statistics.mean(sensitivities),
        "mean_worst_device_auroc": statistics.mean(worst),
        "mean_border_max_fraction": statistics.mean(borders),
    }


def _select(
    scores: dict,
    episodes: dict,
    pair_id: int,
    pair_names: tuple[str, str],
    seeds: list[int],
    shot: int,
    near_tie: float,
) -> dict:
    candidates = sorted({key[0] for key in scores})
    result = {
        "pair": f"{pair_names[0]}__{pair_names[1]}",
        "selection_partition": "validate",
        "selection_shot": shot,
        "near_tie_auroc": near_tie,
        "methods": {},
    }
    fixed = ("anchor_rho03", "dn4_hard_knn")
    anchor_validation = _validation_point(
        scores,
        episodes,
        "anchor_rho03",
        pair_id,
        seeds,
        shot,
    )
    for family in (*fixed, *TUNED_FAMILIES):
        points = [
            _validation_point(
                scores, episodes, candidate, pair_id, seeds, shot
            )
            for candidate in candidates
            if _family(candidate) == family
        ]
        constraint_feasible = None
        selection_pool = points
        if family == "anchor_plus_relational_witness":
            feasible = [
                point
                for point in points
                if float(point["mean_sms_fixed_reference"]) <= 0.321
                and float(point["mean_worst_device_auroc"])
                >= float(anchor_validation["mean_worst_device_auroc"]) - 0.01
            ]
            constraint_feasible = bool(feasible)
            if feasible:
                selection_pool = feasible
        best = max(float(point["mean_auroc"]) for point in selection_pool)
        eligible = [
            point
            for point in selection_pool
            if float(point["mean_auroc"]) >= best - near_tie
        ]
        chosen = min(
            eligible,
            key=lambda point: (
                float(point["mean_sms_fixed_reference"]),
                -float(point["mean_worst_device_auroc"]),
                -float(point["mean_auroc"]),
                str(point["candidate"]),
            ),
        )
        result["methods"][family] = {
            **chosen,
            "validation_constraint_feasible": constraint_feasible,
            "candidates": sorted(
                points,
                key=lambda point: (
                    -float(point["mean_auroc"]),
                    float(point["mean_sms_fixed_reference"]),
                ),
            ),
        }
    return result


def _metric_rows(
    scores: dict,
    episodes: dict,
    pair_id: int,
    names: tuple[str, str],
    seeds: list[int],
    shot: int,
) -> list[dict]:
    rows = []
    pair = f"{names[0]}__{names[1]}"
    candidates = sorted({key[0] for key in scores})
    for candidate in candidates:
        parameters = _fields(candidate)
        for seed in seeds:
            validation = scores[
                (candidate, pair_id, seed, shot, "validate")
            ]
            validation_targets = episodes[
                (pair_id, seed, "validate")
            ]["targets"].flatten()
            temperature = select_temperature(
                validation[0][:, None],
                validation_targets[:, None],
                "multi_label",
            )
            threshold = select_threshold(
                validation[0][:, None],
                validation_targets[:, None],
                temperature,
            )
            for partition in ("validate", "test"):
                values = scores[
                    (candidate, pair_id, seed, shot, partition)
                ]
                current = episodes[(pair_id, seed, partition)]
                metrics = evaluate(
                    *values[:5],
                    current["targets"].flatten(),
                    current["nuisance"].flatten(),
                    temperature,
                    threshold,
                )
                metrics["border_max_fraction"] = float(
                    values[5].float().mean()
                )
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "partition": partition,
                            "pair": pair,
                            "target": names[0],
                            "confounder": names[1],
                            "method": _family(candidate),
                            "candidate": candidate,
                            "witness_fraction": parameters.get("r", ""),
                            "gamma": parameters.get("gamma", ""),
                            "shot": shot,
                            "seed": seed,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return rows


def _selected_rows(rows: list[dict], selection: dict) -> list[dict]:
    selected = {
        family: details["candidate"]
        for family, details in selection["methods"].items()
    }
    return [
        row
        for row in rows
        if selected.get(row["method"]) == row["candidate"]
    ]


def _summaries(rows: list[dict]) -> list[dict]:
    keys = (
        "partition",
        "pair",
        "target",
        "confounder",
        "method",
        "candidate",
        "witness_fraction",
        "gamma",
        "shot",
        "metric",
    )
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(float(row["value"]))
    result = []
    for key, values in groups.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        half = 1.96 * std / math.sqrt(len(values))
        result.append(
            {
                **dict(zip(keys, key)),
                "n_seeds": len(values),
                "mean": mean,
                "std": std,
                "ci95_low": mean - half,
                "ci95_high": mean + half,
            }
        )
    return result


def _decision(rows: list[dict], visual_review: str) -> dict:
    values = defaultdict(dict)
    for row in rows:
        if (
            row["partition"] == "test"
            and row["method"]
            in {"anchor_rho03", "anchor_plus_relational_witness"}
            and row["metric"]
            in {
                "auroc",
                "auprc",
                "sms_fixed_reference",
                "worst_nuisance_auroc",
                "support_swap_flip_rate",
                "border_max_fraction",
            }
        ):
            values[(row["method"], int(row["seed"]))][row["metric"]] = float(
                row["value"]
            )
    seeds = sorted(
        key[1]
        for key in values
        if key[0] == "anchor_plus_relational_witness"
    )
    anchor = [values[("anchor_rho03", seed)] for seed in seeds]
    full = [
        values[("anchor_plus_relational_witness", seed)] for seed in seeds
    ]
    means = lambda records, metric: statistics.mean(
        record[metric] for record in records
    )
    full_auroc = means(full, "auroc")
    full_sms = means(full, "sms_fixed_reference")
    worst_change = means(full, "worst_nuisance_auroc") - means(
        anchor, "worst_nuisance_auroc"
    )
    quantitative = (
        full_auroc >= 0.551
        and full_sms <= 0.321
        and worst_change >= -0.01
    )
    if full_auroc <= 0.541:
        status = "stop_frozen_witness_matching_falsified"
    elif not quantitative:
        status = "stop_stage1_gate_failed"
    elif visual_review == "pending":
        status = "await_witness_evidence_review"
    elif visual_review == "fail":
        status = "stop_border_or_device_dominated"
    else:
        status = "proceed_to_stage2_descriptor_training"
    return {
        "status": status,
        "stage_two_training_started": False,
        "visual_review": visual_review,
        "gate": {
            "minimum_locked_test_auroc": 0.551,
            "maximum_locked_test_sms": 0.321,
            "minimum_worst_device_change": -0.01,
            "falsification_auroc_floor": 0.541,
            "evidence_must_be_less_border_or_device_dominated": True,
        },
        "anchor_rho03": {
            metric: means(anchor, metric)
            for metric in anchor[0]
        },
        "locked_full_witness": {
            metric: means(full, metric)
            for metric in full[0]
        },
        "full_minus_anchor_auroc": full_auroc - means(anchor, "auroc"),
        "full_minus_anchor_worst_device_auroc": worst_change,
        "quantitative_gate_passed": quantitative,
        "device_overlap_fraction": "not_available_no_masks_or_detector_maps",
    }


def _save_selected_maps(
    output_dir: Path,
    fields: dict,
    scores: dict,
    episodes: dict,
    selection: dict,
    pair_id: int,
    seeds: list[int],
    shot: int,
) -> None:
    selected = {
        family: details["candidate"]
        for family, details in selection["methods"].items()
    }
    for seed in seeds:
        maps = {}
        for family, candidate in selected.items():
            field_candidate = candidate
            if family == "anchor_plus_relational_witness":
                fraction = float(_fields(candidate)["r"])
                field_candidate = _candidate(
                    "relational_witness", fraction
                )
            maps[family] = fields[
                (field_candidate, pair_id, seed, shot, "test")
            ]
        current = episodes[(pair_id, seed, "test")]
        torch.save(
            {
                "selection": selected,
                "query_indices": current["query"],
                "targets": current["targets"],
                "nuisance": current["nuisance"],
                "evidence_maps": maps,
            },
            output_dir / f"evidence_maps_seed_{seed:03d}.pt",
        )


def _render_selected(
    output_dir: Path,
    fields: dict,
    scores: dict,
    episodes: dict,
    selection: dict,
    pair_id: int,
    seed: int,
    shot: int,
    retained_grid: int,
    manifest: Path,
    data_root: Path,
    examples_per_group: int,
) -> None:
    candidate = selection["methods"][
        "anchor_plus_relational_witness"
    ]["candidate"]
    fraction = float(_fields(candidate)["r"])
    field_candidate = _candidate("relational_witness", fraction)
    maps = fields[(field_candidate, pair_id, seed, shot, "test")]
    anchor_maps = fields[("anchor_rho03", pair_id, seed, shot, "test")]
    logits = scores[(candidate, pair_id, seed, shot, "test")][0]
    anchor_logits = scores[
        ("anchor_rho03", pair_id, seed, shot, "test")
    ][0]
    current = episodes[(pair_id, seed, "test")]
    targets = current["targets"].flatten()
    nuisance = current["nuisance"].flatten()
    query_count = current["query"].shape[1]
    selected = []
    for device_value in (0, 1):
        eligible = torch.where(targets.bool() & nuisance.eq(device_value))[0]
        ranked = eligible[torch.argsort(logits[eligible], descending=True)]
        selected.extend(ranked[:examples_per_group].tolist())
    records = []
    for flat_index in selected:
        episode_index, query_index = divmod(flat_index, query_count)
        common = {
            "dataset_index": int(
                current["query"][episode_index, query_index]
            ),
            "target": int(targets[flat_index]),
            "nuisance": int(nuisance[flat_index]),
        }
        records.extend(
            (
                {
                    **common,
                    "method": "anchor",
                    "logit": float(anchor_logits[flat_index]),
                    "evidence_map": anchor_maps[
                        episode_index, query_index
                    ],
                },
                {
                    **common,
                    "method": "witness",
                    "logit": float(logits[flat_index]),
                    "evidence_map": maps[episode_index, query_index],
                },
            )
        )
    _render_maps(
        records,
        _open_manifest(manifest),
        data_root.expanduser(),
        output_dir / "locked_witness_evidence.png",
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
        default=Path("outputs/iera/stable_witness_stage1_v1"),
    )
    parser.add_argument("--adapter-rho", type=float, default=0.3)
    parser.add_argument("--retained-grid", type=int, default=37)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument(
        "--witness-fractions",
        type=float,
        nargs="+",
        default=(0.01, 0.02, 0.05, 0.1),
    )
    parser.add_argument(
        "--gammas", type=float, nargs="+", default=(0.1, 0.25, 0.5)
    )
    parser.add_argument("--near-tie-auroc", type=float, default=0.005)
    parser.add_argument("--support-token-chunk-size", type=int, default=128)
    parser.add_argument("--query-chunk-size", type=int, default=1)
    parser.add_argument("--visual-examples-per-group", type=int, default=4)
    parser.add_argument("--visual-review", choices=("pending", "pass", "fail"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.manifest is None) != (args.data_root is None):
        parser.error("manifest and data-root must be supplied together")
    if args.adapter_rho != 0.3:
        parser.error("Stage 1 requires the existing rho=.3 adapter")
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    saved = torch.load(args.episodes, map_location="cpu", weights_only=False)
    signature = saved["signature"]
    if not set(args.seeds).issubset(set(signature["seeds"])):
        raise ValueError("requested seeds are absent from saved episodes")
    if args.episodes_per_seed > signature["episodes"]:
        raise ValueError("saved episode bank is smaller than requested")
    pairs = {
        pair_id: names
        for pair_id, names in saved["pairs"].items()
        if names[0] == "Pneumothorax"
        and "Support" in names[1]
        and "Device" in names[1]
    }
    if len(pairs) != 1:
        raise ValueError("expected one Pneumothorax-Support Devices pair")
    pair_id, names = next(iter(pairs.items()))
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
        if key[0] == pair_id and key[1] in args.seeds
    }
    patches, metadata = load_patch_cache(
        args.rad_cache,
        signature["manifest_sha256"],
        expected_model=RAD_DINO_MODEL,
        expected_pool_grid=args.retained_grid,
        access_mode="stream",
    )
    if args.retained_grid != 37:
        raise ValueError("Stage 1 requires native 37x37 Rad-DINO tokens")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_signature = {
        "manifest_sha256": signature["manifest_sha256"],
        "cache_global_indices_sha256": metadata.get(
            "global_indices_sha256"
        ),
        "adapter_dir": str(args.adapter_dir),
        "adapter_rho": args.adapter_rho,
        "retained_grid": args.retained_grid,
        "shot": args.shot,
        "episodes_per_seed": args.episodes_per_seed,
        "witness_fractions": list(args.witness_fractions),
        "gammas": list(args.gammas),
        "support_token_chunk_size": args.support_token_chunk_size,
        "query_chunk_size": args.query_chunk_size,
    }
    scores, all_fields = {}, {}
    for seed in args.seeds:
        checkpoint = args.adapter_dir / (
            f"model_adapter_only_rho_{args.adapter_rho:g}_seed_{seed:03d}.pt"
        )
        saved_model = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved_model.get("method") not in (None, "adapter_only"):
            raise ValueError(f"{checkpoint} is not adapter-only")
        if saved_model.get("rho") not in (None, args.adapter_rho):
            raise ValueError(f"{checkpoint} has the wrong SMS budget")
        model = RobustBinaryModel(
            int(metadata["shape"][-1]),
            adapter_dim=int(
                saved_model["state_dict"]["support_down.weight"].shape[0]
            ),
        ).to(device)
        model.load_state_dict(saved_model["state_dict"])
        model.eval().requires_grad_(False)
        score_path = args.output_dir / f"scores_seed_{seed:03d}.pt"
        if score_path.exists():
            completed = torch.load(score_path, map_location="cpu", weights_only=False)
            if completed.get("signature") != score_signature:
                raise ValueError(f"{score_path} uses different Stage-1 arguments")
            scores.update(completed["scores"])
            all_fields.update(completed["fields"])
            print(f"reusing witness scores for seed={seed}", flush=True)
            continue
        seed_scores, seed_fields = {}, {}
        for partition in ("validate", "test"):
            current_scores, current_fields = _score(
                model,
                patches,
                metadata,
                episode_sets[(pair_id, seed, partition)],
                args.shot,
                args.retained_grid,
                device,
                list(args.witness_fractions),
                list(args.gammas),
                args.support_token_chunk_size,
                args.query_chunk_size,
            )
            for candidate, values in current_scores.items():
                seed_scores[
                    (candidate, pair_id, seed, args.shot, partition)
                ] = values
            for candidate, values in current_fields.items():
                seed_fields[
                    (candidate, pair_id, seed, args.shot, partition)
                ] = values
            print(
                f"finished Stage-1 witnesses, {partition}, seed={seed}",
                flush=True,
            )
        temporary = score_path.with_suffix(".tmp")
        torch.save(
            {
                "signature": score_signature,
                "scores": seed_scores,
                "fields": seed_fields,
            },
            temporary,
        )
        temporary.replace(score_path)
        scores.update(seed_scores)
        all_fields.update(seed_fields)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection = _select(
        scores,
        episode_sets,
        pair_id,
        names,
        list(args.seeds),
        args.shot,
        args.near_tie_auroc,
    )
    candidate_rows = _metric_rows(
        scores,
        episode_sets,
        pair_id,
        names,
        list(args.seeds),
        args.shot,
    )
    per_seed = _selected_rows(candidate_rows, selection)
    summary = _summaries(per_seed)
    _save_selected_maps(
        args.output_dir,
        all_fields,
        scores,
        episode_sets,
        selection,
        pair_id,
        list(args.seeds),
        args.shot,
    )
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
        raise ValueError("visual_review.json has invalid status")
    review_path.write_text(
        json.dumps(
            {
                "status": visual_review,
                "criterion": (
                    "locked witness evidence must be visibly less "
                    "border/chest-tube dominated than the rho=.3 anchor"
                ),
                "artifact": "locked_witness_evidence.png",
                "device_overlap": "unavailable: no masks/detector maps supplied",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.manifest is not None:
        if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != signature[
            "manifest_sha256"
        ]:
            raise ValueError("manifest does not match saved episodes")
        _render_selected(
            args.output_dir,
            all_fields,
            scores,
            episode_sets,
            selection,
            pair_id,
            args.seeds[0],
            args.shot,
            args.retained_grid,
            args.manifest,
            args.data_root,
            args.visual_examples_per_group,
        )
    if (
        visual_review == "pass"
        and not (args.output_dir / "locked_witness_evidence.png").is_file()
    ):
        raise ValueError("cannot pass visual review before rendering evidence")
    decision = _decision(per_seed, visual_review)
    _write(args.output_dir / "candidate_metrics.csv", candidate_rows)
    _write(args.output_dir / "per_seed_metrics.csv", per_seed)
    _write(args.output_dir / "summary_metrics.csv", summary)
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "stage": "stable_region_witnesses_scoring_only",
                "stage_two_training_started": False,
                "backbone": metadata,
                "descriptor_projection": "normalize((z + (z-mean3x3(z)))/sqrt(2))",
                "dn4_neighbours": 3,
                "adapter_rho": args.adapter_rho,
                "adapter_frozen": True,
                "query_frozen": True,
                "witness_fractions": args.witness_fractions,
                "gammas": args.gammas,
                "selection_partition": "validate",
                "shot": args.shot,
                "seeds": args.seeds,
                "episodes_per_seed": args.episodes_per_seed,
                "device_overlap": "not_available_no_masks_or_detector_maps",
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Stage-1 witness results written to {args.output_dir}; "
        f"decision={decision['status']}"
    )


if __name__ == "__main__":
    main()
