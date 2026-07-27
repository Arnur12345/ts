"""Validation-only oracle headroom analysis for existing support corrections."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.residuals.metrics import _auc

from .comed import CoMeD
from .comed_run import (
    _balanced_support,
    _load_cache,
    _load_target_episodes,
    _tensor,
)


CORE_METHODS = (
    "rad_global_protonet",
    "comed_logit_residual",
    "comed_tanh_residual",
)


def _global_protonet(
    arrays,
    support: torch.Tensor,
    support_y: torch.Tensor,
    query: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    features = F.normalize(_tensor(arrays["rad"], support, device), dim=-1)
    query_features = F.normalize(
        _tensor(arrays["rad"], query, device), dim=-1
    )
    positive = features[support_y.eq(1)]
    negative = features[support_y.eq(0)]
    if len(positive) == 0 or len(negative) == 0:
        raise ValueError("support-label control removed one class")
    positive_prototype = F.normalize(positive.mean(0), dim=-1)
    negative_prototype = F.normalize(negative.mean(0), dim=-1)
    return (
        query_features @ positive_prototype
        - query_features @ negative_prototype
    )


def _comed_score(
    model,
    arrays,
    class_id,
    support,
    support_y,
    query,
    device,
    consistent_scale,
):
    support_logit = _tensor(
        arrays["prior"][:, class_id], support, device
    )
    support_prior = (
        torch.tanh(support_logit / 2)
        if consistent_scale
        else support_logit
    )
    return model(
        _tensor(arrays["rad"], support, device),
        support_y.float().mul(2).sub(1),
        _tensor(arrays["nuisance"], support, device),
        support_prior,
        _tensor(arrays["rad"], query, device),
        _tensor(arrays["prior"][:, class_id], query, device),
    )


def _external_scores(
    score_dirs,
    seeds,
    pair_id,
    shot,
    episode_count,
    query_count,
):
    result = {}
    expected = episode_count * query_count
    for score_dir in score_dirs:
        for seed in seeds:
            path = score_dir / f"scores_seed_{seed:03d}.pt"
            if not path.exists():
                print(f"skipping absent external score file {path}", flush=True)
                continue
            saved = torch.load(path, map_location="cpu", weights_only=False)
            for key, values in saved.get("scores", {}).items():
                if (
                    not isinstance(key, tuple)
                    or len(key) < 5
                    or int(key[1]) != pair_id
                    or int(key[2]) != seed
                    or int(key[3]) != shot
                    or key[4] != "validate"
                ):
                    continue
                logits = torch.as_tensor(values[0]).flatten()
                if len(logits) < expected:
                    raise ValueError(
                        f"{path} candidate {key[0]!r} has only "
                        f"{len(logits)} scores, expected {expected}"
                    )
                method = f"{score_dir.name}:{key[0]}"
                result[(method, seed)] = logits[:expected].reshape(
                    episode_count, query_count
                )
    return result


def _add_gamma_family(
    score_store,
    method_meta,
    per_query_rows,
    source,
    base_method,
    raw_score,
    text_score,
    seed,
    episode,
    target,
    nuisance,
    gammas,
    core,
):
    correction = raw_score - text_score
    for gamma in gammas:
        candidate = f"{base_method}|gamma={gamma:g}"
        score = text_score + float(gamma) * correction
        score_store[(source, candidate, seed, episode)] = score.cpu()
        method_meta[(source, candidate)] = {
            "base_method": base_method,
            "gamma": float(gamma),
            "core": bool(core),
        }
        for query_position in range(len(score)):
            per_query_rows.append(
                {
                    "partition": "validate",
                    "support_source": source,
                    "base_method": base_method,
                    "candidate": candidate,
                    "gamma": gamma,
                    "seed": seed,
                    "episode": episode,
                    "query_position": query_position,
                    "target": int(target[query_position]),
                    "device": int(nuisance[query_position]),
                    "text_logit": float(text_score[query_position]),
                    "raw_support_logit": float(raw_score[query_position]),
                    "correction": float(correction[query_position]),
                    "score": float(score[query_position]),
                }
            )


def _episode_auc(target, score, mask=None):
    if mask is not None:
        target = target[mask]
        score = score[mask]
    return _auc(target.bool(), score.float())


def _choose(candidates, target, scores, mask, method_meta, source):
    def key(candidate):
        auc = _episode_auc(target, scores[candidate], mask)
        gamma = abs(method_meta[(source, candidate)]["gamma"])
        return auc, -gamma, candidate

    return max(candidates, key=key)


def _oracle(
    score_store,
    method_meta,
    target_episodes,
    seeds,
    source,
    core_only,
):
    candidates = sorted(
        candidate
        for current_source, candidate in method_meta
        if current_source == source
        and (
            not core_only
            or method_meta[(current_source, candidate)]["core"]
        )
    )
    if not candidates:
        return [], []
    ordinary_rows, split_rows = [], []
    for seed in seeds:
        episodes = target_episodes[(seed, "validate")]
        for episode in range(len(episodes["positive"])):
            target = episodes["targets"][episode].bool()
            nuisance = episodes["nuisance"][episode].long()
            scores = {
                candidate: score_store[
                    (source, candidate, seed, episode)
                ]
                for candidate in candidates
            }
            text = scores["text_only|gamma=0"]
            selected = _choose(
                candidates, target, scores, None, method_meta, source
            )
            selected_auc = _episode_auc(target, scores[selected])
            text_auc = _episode_auc(target, text)
            ordinary_rows.append(
                {
                    "oracle": (
                        f"{source}_{'core' if core_only else 'all'}"
                    ),
                    "seed": seed,
                    "episode": episode,
                    "selected": selected,
                    "selected_auroc": selected_auc,
                    "text_auroc": text_auc,
                    "improvement": selected_auc - text_auc,
                }
            )
            fold_improvements = []
            for selection_device in (0, 1):
                selection_mask = nuisance.eq(selection_device)
                evaluation_mask = nuisance.ne(selection_device)
                if (
                    target[selection_mask].unique().numel() < 2
                    or target[evaluation_mask].unique().numel() < 2
                ):
                    raise ValueError(
                        "split oracle needs both classes in both query halves"
                    )
                chosen = _choose(
                    candidates,
                    target,
                    scores,
                    selection_mask,
                    method_meta,
                    source,
                )
                evaluation_auc = _episode_auc(
                    target, scores[chosen], evaluation_mask
                )
                evaluation_text = _episode_auc(
                    target, text, evaluation_mask
                )
                improvement = evaluation_auc - evaluation_text
                fold_improvements.append(improvement)
                split_rows.append(
                    {
                        "oracle": (
                            f"{source}_{'core' if core_only else 'all'}"
                        ),
                        "seed": seed,
                        "episode": episode,
                        "selection_device": selection_device,
                        "evaluation_device": 1 - selection_device,
                        "selected": chosen,
                        "selection_auroc": _episode_auc(
                            target, scores[chosen], selection_mask
                        ),
                        "evaluation_auroc": evaluation_auc,
                        "text_evaluation_auroc": evaluation_text,
                        "improvement": improvement,
                    }
                )
            ordinary_rows[-1]["split_mean_improvement"] = statistics.mean(
                fold_improvements
            )
    return ordinary_rows, split_rows


def _summarize_oracle(ordinary, split):
    return {
        "episodes": len(ordinary),
        "ordinary_oracle_auroc": statistics.mean(
            row["selected_auroc"] for row in ordinary
        ),
        "ordinary_text_auroc": statistics.mean(
            row["text_auroc"] for row in ordinary
        ),
        "ordinary_improvement": statistics.mean(
            row["improvement"] for row in ordinary
        ),
        "split_oracle_auroc": statistics.mean(
            row["evaluation_auroc"] for row in split
        ),
        "split_text_auroc": statistics.mean(
            row["text_evaluation_auroc"] for row in split
        ),
        "split_improvement": statistics.mean(
            row["improvement"] for row in split
        ),
        "fraction_episodes_split_improvement_gt_0.02": statistics.mean(
            row["split_mean_improvement"] > 0.02 for row in ordinary
        ),
        "ordinary_selected_counts": dict(
            sorted(
                {
                    method: sum(row["selected"] == method for row in ordinary)
                    for method in {row["selected"] for row in ordinary}
                }.items()
            )
        ),
        "split_selected_counts": dict(
            sorted(
                {
                    method: sum(row["selected"] == method for row in split)
                    for method in {row["selected"] for row in split}
                }.items()
            )
        ),
    }


def _write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--score-dir", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4)
    )
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=(-1.0, 0.0, 0.01, 0.03, 0.1, 0.3, 1.0),
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    arrays, _, data = _load_cache(args.cache)
    target_episodes = _load_target_episodes(
        args.episodes, data, args
    )
    saved_episodes = torch.load(
        args.episodes, map_location="cpu", weights_only=False
    )
    pair_ids = [
        pair_id
        for pair_id, names in saved_episodes["pairs"].items()
        if tuple(names) == ("Pneumothorax", "Support Devices")
    ]
    if len(pair_ids) != 1:
        raise ValueError("expected one Pneumothorax-Support Devices pair")
    pair_id = pair_ids[0]
    first = target_episodes[(args.seeds[0], "validate")]
    query_count = first["query"].shape[1]
    external = _external_scores(
        args.score_dir,
        args.seeds,
        pair_id,
        args.shot,
        args.episodes_per_seed,
        query_count,
    )
    model = CoMeD(
        arrays["rad"].shape[1],
        rank=16,
        metric_mode="identity",
        use_nuisance=False,
    ).to(device).eval().requires_grad_(False)
    score_store, method_meta, per_query_rows = {}, {}, []
    class_id = data.class_names.index("Pneumothorax")
    generator = torch.Generator().manual_seed(args.seed)
    with torch.inference_mode():
        for seed in args.seeds:
            episodes = target_episodes[(seed, "validate")]
            for episode in range(len(episodes["positive"])):
                query = episodes["query"][episode].long()
                target = episodes["targets"][episode].long()
                nuisance = episodes["nuisance"][episode].long()
                text_score = _tensor(
                    arrays["prior"][:, class_id], query, device
                )
                for source in ("real", "shuffled", "cross_episode"):
                    if source == "cross_episode":
                        support_episode = (
                            episode + 1
                        ) % len(episodes["positive"])
                    else:
                        support_episode = episode
                    support = _balanced_support(
                        episodes, support_episode, args.shot
                    )
                    support_y = torch.as_tensor(
                        arrays["labels"][
                            support.numpy(), class_id
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                    if source == "shuffled":
                        permutation = torch.randperm(
                            len(support_y), generator=generator
                        ).to(device)
                        support_y = support_y[permutation]
                    raw_scores = {
                        "rad_global_protonet": _global_protonet(
                            arrays,
                            support,
                            support_y,
                            query,
                            device,
                        ),
                        "comed_logit_residual": _comed_score(
                            model,
                            arrays,
                            class_id,
                            support,
                            support_y,
                            query,
                            device,
                            False,
                        ),
                        "comed_tanh_residual": _comed_score(
                            model,
                            arrays,
                            class_id,
                            support,
                            support_y,
                            query,
                            device,
                            True,
                        ),
                    }
                    score_store[
                        (source, "text_only|gamma=0", seed, episode)
                    ] = text_score.cpu()
                    method_meta[(source, "text_only|gamma=0")] = {
                        "base_method": "text_only",
                        "gamma": 0.0,
                        "core": True,
                    }
                    for query_position in range(len(text_score)):
                        per_query_rows.append(
                            {
                                "partition": "validate",
                                "support_source": source,
                                "base_method": "text_only",
                                "candidate": "text_only|gamma=0",
                                "gamma": 0.0,
                                "seed": seed,
                                "episode": episode,
                                "query_position": query_position,
                                "target": int(target[query_position]),
                                "device": int(nuisance[query_position]),
                                "text_logit": float(
                                    text_score[query_position]
                                ),
                                "raw_support_logit": float(
                                    text_score[query_position]
                                ),
                                "correction": 0.0,
                                "score": float(text_score[query_position]),
                            }
                        )
                    for base_method, raw_score in raw_scores.items():
                        _add_gamma_family(
                            score_store,
                            method_meta,
                            per_query_rows,
                            source,
                            base_method,
                            raw_score,
                            text_score,
                            seed,
                            episode,
                            target,
                            nuisance,
                            args.gammas,
                            True,
                        )
                for (method, external_seed), values in external.items():
                    if external_seed != seed:
                        continue
                    _add_gamma_family(
                        score_store,
                        method_meta,
                        per_query_rows,
                        "real",
                        method,
                        values[episode].to(device),
                        text_score,
                        seed,
                        episode,
                        target,
                        nuisance,
                        args.gammas,
                        False,
                    )
            print(
                f"prepared oracle logits for validation seed={seed}",
                flush=True,
            )
    oracle_results = {}
    all_ordinary, all_split = [], []
    for name, source, core_only in (
        ("real_core", "real", True),
        ("real_all", "real", False),
        ("shuffled_core", "shuffled", True),
        ("cross_episode_core", "cross_episode", True),
    ):
        ordinary, split = _oracle(
            score_store,
            method_meta,
            target_episodes,
            args.seeds,
            source,
            core_only,
        )
        oracle_results[name] = _summarize_oracle(ordinary, split)
        all_ordinary.extend(ordinary)
        all_split.extend(split)
    ordinary_all = oracle_results["real_all"]
    split_all = ordinary_all["split_improvement"]
    fraction_all = ordinary_all[
        "fraction_episodes_split_improvement_gt_0.02"
    ]
    real_core = oracle_results["real_core"]
    control_split = max(
        oracle_results["shuffled_core"]["split_improvement"],
        oracle_results["cross_episode_core"]["split_improvement"],
    )
    if ordinary_all["ordinary_oracle_auroc"] < 0.65:
        status = "no_oracle_headroom_current_corrections"
    elif split_all < 0.03 or fraction_all < 0.25:
        status = "ordinary_headroom_is_query_sampling_noise"
    elif real_core["split_improvement"] <= control_split:
        status = "real_support_does_not_beat_arbitrary_perturbations"
    else:
        status = "reliability_gate_is_empirically_justified"
    decision = {
        "status": status,
        "partition": "validate_only",
        "test_accessed": False,
        "criteria": {
            "ordinary_oracle_auroc": ">= 0.65",
            "split_oracle_improvement": ">= 0.03",
            "fraction_episodes_improved_gt_0.02": ">= 0.25",
            "real_support": "must outperform shuffled/cross controls",
        },
        "results": oracle_results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write(args.output_dir / "per_query_logits.csv", per_query_rows)
    _write(args.output_dir / "ordinary_oracle.csv", all_ordinary)
    _write(args.output_dir / "split_oracle.csv", all_split)
    (args.output_dir / "oracle_summary.json").write_text(
        json.dumps(oracle_results, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "oracle_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"oracle results written to {args.output_dir}; decision={status}"
    )


if __name__ == "__main__":
    main()
