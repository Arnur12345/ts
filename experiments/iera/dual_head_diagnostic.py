"""Scoring-only global/local diagnostic on frozen adapters and saved episodes."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.residuals.metrics import _auc, select_temperature, select_threshold

from .dual_head import dual_scores, fused_score, support_adapter
from .falsification import _gather, _selected, _summaries, _write
from .patch_cache import RAD_DINO_MODEL, load_patch_cache
from .robust_metrics import evaluate, normalized_sms
from .robust_model import RobustBinaryModel


def _candidate_name(kind: str, weight: float, temperature: float | None = None) -> str:
    name = f"{kind}|lambda={weight:g}"
    return name if temperature is None else f"{name}|tau={temperature:g}"


def _score_branches(
    model: RobustBinaryModel,
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    temperatures: list[float],
    query_temperature: float,
) -> dict[float, dict[str, torch.Tensor]]:
    positive = support_adapter(positive, model)
    negative = support_adapter(negative, model)
    query = F.normalize(query.float(), dim=-1)
    return {
        temperature: dual_scores(
            positive,
            negative,
            query,
            positive_mask,
            negative_mask,
            temperature,
            query_temperature,
        )
        for temperature in temperatures
    }


def _diagnostic_scores(
    model: RobustBinaryModel,
    patches,
    metadata: dict,
    episodes: dict,
    shot: int,
    batch_size: int,
    retained_grid: int,
    device: torch.device,
    weights: list[float],
    temperatures: list[float],
    query_temperature: float,
) -> dict[str, tuple[torch.Tensor, ...]]:
    destinations = defaultdict(lambda: ([], [], [], [], []))
    with torch.inference_mode():
        for start in range(0, len(episodes["positive"]), batch_size):
            end = min(start + batch_size, len(episodes["positive"]))
            positive_panels = _gather(
                patches,
                episodes["positive"][start:end],
                metadata,
                retained_grid,
                device,
            )
            negative_panels = _gather(
                patches,
                episodes["negative"][start:end],
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
            ordinary = _score_branches(
                model,
                positive,
                negative,
                query,
                positive_mask,
                negative_mask,
                temperatures,
                query_temperature,
            )
            panels = []
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
                panels.append(
                    _score_branches(
                        model,
                        panel_positive,
                        panel_negative,
                        query,
                        panel_mask,
                        panel_mask,
                        temperatures,
                        query_temperature,
                    )
                )
            first_temperature = temperatures[0]
            global_logits = ordinary[first_temperature]["global"]
            current_logits = ordinary[first_temperature]["current_local"]
            global_panels = (
                panels[0][first_temperature]["global"],
                panels[1][first_temperature]["global"],
            )
            current_panels = (
                panels[0][first_temperature]["current_local"],
                panels[1][first_temperature]["current_local"],
            )
            candidates = {
                "global": (global_logits, *global_panels),
                "current_local": (current_logits, *current_panels),
            }
            for weight in weights:
                candidates[
                    _candidate_name("fused_current", weight)
                ] = (
                    fused_score(global_logits, current_logits, weight),
                    fused_score(
                        global_panels[0], current_panels[0], weight
                    ),
                    fused_score(
                        global_panels[1], current_panels[1], weight
                    ),
                )
                for temperature in temperatures:
                    selected_logits = ordinary[temperature]["selected_local"]
                    selected_panels = (
                        panels[0][temperature]["selected_local"],
                        panels[1][temperature]["selected_local"],
                    )
                    candidates[
                        _candidate_name(
                            "fused_selected", weight, temperature
                        )
                    ] = (
                        fused_score(global_logits, selected_logits, weight),
                        fused_score(
                            global_panels[0], selected_panels[0], weight
                        ),
                        fused_score(
                            global_panels[1], selected_panels[1], weight
                        ),
                    )
            for name, (logits, panel_zero, panel_one) in candidates.items():
                output = destinations[name]
                for destination, tensor in zip(
                    output,
                    (
                        logits,
                        panel_zero,
                        panel_one,
                        current_panels[0],
                        current_panels[1],
                    ),
                ):
                    destination.append(tensor.cpu())
    return {
        name: tuple(torch.cat(items).flatten() for items in values)
        for name, values in destinations.items()
    }


def _validation_key(
    candidate: str,
    pair_id: int,
    scores: dict,
    episodes: dict,
    seeds: list[int],
    shot: int,
) -> tuple[float, float]:
    aurocs, sensitivities = [], []
    for seed in seeds:
        tensors = scores[(candidate, pair_id, seed, shot, "validate")]
        targets = episodes[(pair_id, seed, "validate")]["targets"].flatten()
        aurocs.append(_auc(targets.bool(), tensors[0]))
        sensitivities.append(float(normalized_sms(*tensors[1:])))
    return statistics.mean(aurocs), -statistics.mean(sensitivities)


def _select(
    scores: dict,
    episodes: dict,
    pair_names: dict,
    seeds: list[int],
    primary_shot: int,
) -> dict[int, dict]:
    candidates = sorted({key[0] for key in scores})
    result = {}
    for pair_id, names in pair_names.items():
        current = [
            name for name in candidates if name.startswith("fused_current|")
        ]
        selected = [
            name for name in candidates if name.startswith("fused_selected|")
        ]
        chosen_current = max(
            current,
            key=lambda name: _validation_key(
                name, pair_id, scores, episodes, seeds, primary_shot
            ),
        )
        chosen_selected = max(
            selected,
            key=lambda name: _validation_key(
                name, pair_id, scores, episodes, seeds, primary_shot
            ),
        )
        current_validation = _validation_key(
            chosen_current,
            pair_id,
            scores,
            episodes,
            seeds,
            primary_shot,
        )
        selected_validation = _validation_key(
            chosen_selected,
            pair_id,
            scores,
            episodes,
            seeds,
            primary_shot,
        )
        result[pair_id] = {
            "pair": f"{names[0]}__{names[1]}",
            "selection_partition": "validate",
            "selection_shot": primary_shot,
            "fused_current": chosen_current,
            "fused_selected": chosen_selected,
            "fused_current_validation": {
                "mean_auroc": current_validation[0],
                "mean_sms": -current_validation[1],
            },
            "fused_selected_validation": {
                "mean_auroc": selected_validation[0],
                "mean_sms": -selected_validation[1],
            },
        }
    return result


def _selected_rows(
    scores: dict,
    episodes: dict,
    pair_names: dict,
    selection: dict,
    seeds: list[int],
    shots: list[int],
) -> list[dict]:
    rows = []
    for pair_id, names in pair_names.items():
        variants = {
            "global": "global",
            "current_local": "current_local",
            "fused_current": selection[pair_id]["fused_current"],
            "fused_selected": selection[pair_id]["fused_selected"],
        }
        for method, candidate in variants.items():
            fields = {
                item.split("=")[0]: item.split("=")[1]
                for item in candidate.split("|")[1:]
            }
            for seed in seeds:
                for shot in shots:
                    validation = scores[
                        (candidate, pair_id, seed, shot, "validate")
                    ]
                    validation_targets = episodes[
                        (pair_id, seed, "validate")
                    ]["targets"].flatten()
                    calibration_temperature = select_temperature(
                        validation[0][:, None],
                        validation_targets[:, None],
                        "multi_label",
                    )
                    threshold = select_threshold(
                        validation[0][:, None],
                        validation_targets[:, None],
                        calibration_temperature,
                    )
                    test_episodes = episodes[(pair_id, seed, "test")]
                    test = scores[
                        (candidate, pair_id, seed, shot, "test")
                    ]
                    metrics = evaluate(
                        *test,
                        test_episodes["targets"].flatten(),
                        test_episodes["nuisance"].flatten(),
                        calibration_temperature,
                        threshold,
                    )
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "pair": f"{names[0]}__{names[1]}",
                                "target": names[0],
                                "confounder": names[1],
                                "method": method,
                                "rho": "",
                                "lambda": fields.get("lambda", ""),
                                "patch_temperature": fields.get("tau", ""),
                                "shot": shot,
                                "seed": seed,
                                "metric": metric,
                                "value": value,
                            }
                        )
    return rows


def _decision(summary: list[dict], primary_shot: int) -> dict:
    lookup = {
        (row["pair"], row["method"]): row
        for row in summary
        if int(row["shot"]) == primary_shot and row["metric"] == "auroc"
    }
    pneumothorax_pairs = sorted(
        pair for pair, method in lookup
        if pair.startswith("Pneumothorax") and method == "fused_selected"
    )
    comparisons = []
    for pair in pneumothorax_pairs:
        selected = lookup[(pair, "fused_selected")]["mean"]
        global_score = lookup[(pair, "global")]["mean"]
        local_score = lookup[(pair, "current_local")]["mean"]
        comparisons.append(
            {
                "pair": pair,
                "fused_selected_auroc": selected,
                "global_auroc": global_score,
                "current_local_auroc": local_score,
                "gain_over_global": selected - global_score,
                "gain_over_current_local": selected - local_score,
            }
        )
    proceed = bool(comparisons) and all(
        row["gain_over_global"] >= 0.02
        and row["gain_over_current_local"] >= 0.02
        for row in comparisons
    )
    return {
        "status": (
            "proceed_to_constrained_dual_head"
            if proceed
            else "stop_or_revise_dual_head"
        ),
        "gate": (
            "fused selected-local must improve Pneumothorax test AUROC by "
            "at least 0.02 over both individual branches"
        ),
        "primary_shot": primary_shot,
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--rad-cache", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/iera/dual_head_pilot_v1"),
    )
    parser.add_argument("--adapter-rho", type=float, default=0.7)
    parser.add_argument("--retained-grid", type=int, default=14)
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--primary-shot", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--patch-temperatures",
        type=float,
        nargs="+",
        default=(0.05, 0.1, 0.2),
    )
    parser.add_argument("--query-temperature", type=float, default=0.1)
    parser.add_argument("--episode-batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.primary_shot not in args.shots:
        parser.error("primary-shot must be included in shots")
    if any(not 0 <= value <= 1 for value in args.lambdas):
        parser.error("lambdas must be in [0,1]")
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
    episode_sets = {
        key: {
            name: (
                value[: args.episodes_per_seed]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == signature["episodes"]
                else value
            )
            for name, value in episodes.items()
        }
        for key, episodes in saved["episodes"].items()
        if key[1] in args.seeds
    }
    pair_names = saved["pairs"]
    patches, metadata = load_patch_cache(
        args.rad_cache,
        signature["manifest_sha256"],
        expected_model=RAD_DINO_MODEL,
        access_mode="stream",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_signature = {
        "manifest_sha256": signature["manifest_sha256"],
        "retained_grid": args.retained_grid,
        "shots": list(args.shots),
        "episodes_per_seed": args.episodes_per_seed,
        "lambdas": list(args.lambdas),
        "patch_temperatures": list(args.patch_temperatures),
        "query_temperature": args.query_temperature,
        "adapter_rho": args.adapter_rho,
        "adapter_dir": str(args.adapter_dir),
    }
    scores = {}
    for seed in args.seeds:
        score_checkpoint = (
            args.output_dir / f"scores_seed_{seed:03d}.pt"
        )
        if score_checkpoint.exists():
            completed = torch.load(
                score_checkpoint, map_location="cpu", weights_only=False
            )
            if completed.get("signature") != score_signature:
                raise ValueError(
                    f"{score_checkpoint} uses different diagnostic arguments"
                )
            scores.update(completed["scores"])
            print(f"reusing dual-head scores for seed={seed}", flush=True)
            continue
        checkpoint = args.adapter_dir / (
            f"model_adapter_only_rho_{args.adapter_rho:g}"
            f"_seed_{seed:03d}.pt"
        )
        saved_model = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        model = RobustBinaryModel(
            int(metadata["shape"][-1]),
            adapter_dim=int(saved_model["state_dict"]["support_down.weight"].shape[0]),
            local_temperature=args.query_temperature,
        ).to(device)
        model.load_state_dict(saved_model["state_dict"])
        model.eval().requires_grad_(False)
        seed_scores = {}
        for pair_id in sorted(pair_names):
            for partition in ("validate", "test"):
                episodes = episode_sets[(pair_id, seed, partition)]
                for shot in args.shots:
                    computed = _diagnostic_scores(
                        model,
                        patches,
                        metadata,
                        episodes,
                        shot,
                        args.episode_batch_size,
                        args.retained_grid,
                        device,
                        list(args.lambdas),
                        list(args.patch_temperatures),
                        args.query_temperature,
                    )
                    for candidate, tensors in computed.items():
                        seed_scores[
                            (candidate, pair_id, seed, shot, partition)
                        ] = tensors
                print(
                    f"finished dual-head scoring, pair={pair_id}, "
                    f"{partition}, seed={seed}",
                    flush=True,
                )
        torch.save(
            {"signature": score_signature, "scores": seed_scores},
            score_checkpoint,
        )
        scores.update(seed_scores)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    selection = _select(
        scores,
        episode_sets,
        pair_names,
        list(args.seeds),
        args.primary_shot,
    )
    rows = _selected_rows(
        scores,
        episode_sets,
        pair_names,
        selection,
        list(args.seeds),
        list(args.shots),
    )
    summary = _summaries(rows)
    decision = _decision(summary, args.primary_shot)
    _write(args.output_dir / "per_seed_metrics.csv", rows)
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
                "stage": "scoring_only_dual_head",
                "backbone": metadata,
                "episodes": str(args.episodes),
                "adapter_dir": str(args.adapter_dir),
                "adapter_rho": args.adapter_rho,
                "adapter_frozen": True,
                "query_frozen": True,
                "retained_grid": args.retained_grid,
                "shots": args.shots,
                "seeds": args.seeds,
                "episodes_per_seed": args.episodes_per_seed,
                "lambdas": args.lambdas,
                "patch_temperatures": args.patch_temperatures,
                "query_temperature": args.query_temperature,
                "selection_partition": "validate",
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"dual-head results written to {args.output_dir}; "
        f"decision={decision['status']}"
    )


if __name__ == "__main__":
    main()
