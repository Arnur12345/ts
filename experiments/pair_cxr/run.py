"""Meta-train and evaluate Few-Shot PAIR-CXR on frozen Rad-DINO tokens."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.iera.comed_run import (
    _balanced_support,
    _load_cache,
    _load_target_episodes,
    _panel_support,
    _protonet,
    _tensor,
)
from experiments.iera.episodes import split_indices, stratum_pools
from experiments.iera.oracle_headroom import _global_protonet
from experiments.iera.patch_cache import (
    RAD_DINO_MODEL,
    load_patch_cache,
)
from experiments.iera.robust_metrics import (
    normalized_sms,
    ranking_disagreement,
)
from experiments.residuals.metrics import _auc, _average_precision

from .model import PAIRRouter, intervention_loss, normalized_entropy


METHODS = (
    "vlm_text_prior",
    "rad_global_protonet",
    "pair_router_zero_shot",
    "pair_cxr_adapted",
)
HARD_DISEASES = ("Pneumothorax", "Lung Lesion", "Fracture")


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _text_queries(arrays, metadata, data, args, device):
    path = args.output_dir / "rad_text_queries.pt"
    signature = {
        "manifest_sha256": data.manifest_sha256,
        "alignment_samples": args.alignment_samples,
        "alignment_ridge": args.alignment_ridge,
        "split_seed": args.split_seed,
        "seed": args.seed,
    }
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("existing text-query alignment differs")
        return saved["queries"]
    train = split_indices(data, "train", args.split_seed)
    generator = torch.Generator().manual_seed(args.seed)
    if len(train) > args.alignment_samples:
        train = train[
            torch.randperm(len(train), generator=generator)[
                : args.alignment_samples
            ]
        ]
    rad = _tensor(arrays["rad"], train, device)
    margins = _tensor(arrays["prior"], train, device)
    rad = rad - rad.mean(0, keepdim=True)
    margins = margins - margins.mean(0, keepdim=True)
    covariance = rad.T @ rad
    covariance = covariance + args.alignment_ridge * torch.eye(
        covariance.shape[0], device=device
    )
    projection = torch.linalg.solve(covariance, rad.T @ margins)
    queries = F.normalize(projection.T, dim=-1).cpu()
    torch.save(
        {
            "signature": signature,
            "queries": queries,
            "method": (
                "ridge projection of frozen BioMedCLIP semantic margins "
                "into frozen Rad-DINO global space"
            ),
            "classes": data.class_names,
        },
        path,
    )
    return queries


def _sample_unique(
    pool: torch.Tensor,
    count: int,
    subjects: list[str],
    generator: torch.Generator,
    used: set[str],
) -> list[int]:
    selected = []
    for position in torch.randperm(len(pool), generator=generator).tolist():
        index = int(pool[position])
        subject = subjects[index]
        if subject in used:
            continue
        used.add(subject)
        selected.append(index)
        if len(selected) == count:
            return selected
    raise ValueError(f"could not sample {count} unique patients")


def _group_batch(
    data,
    pools,
    per_group,
    generator,
):
    indices, targets, devices = [], [], []
    used: set[str] = set()
    for target in (0, 1):
        for device in (0, 1):
            selected = _sample_unique(
                pools[(target, device)],
                per_group,
                data.subject_ids,
                generator,
                used,
            )
            indices.extend(selected)
            targets.extend([target] * per_group)
            devices.extend([device] * per_group)
    return (
        torch.tensor(indices, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
        torch.tensor(devices, dtype=torch.long),
    )


def _meta_tasks(data, metadata, args):
    train = split_indices(data, "train", args.split_seed)
    validation = split_indices(data, "validate", args.split_seed)
    device_id = data.class_names.index("Support Devices")
    target_ids = [
        class_id for class_id in metadata["base_target_ids"]
        if data.class_names[class_id] not in set(args.hard_diseases)
    ]
    tasks = []
    minimum = max(
        args.meta_batch_per_group,
        args.base_validation_per_group,
    )
    for target_id in target_ids:
        train_pools = stratum_pools(data, train, target_id, device_id)
        validation_pools = stratum_pools(
            data, validation, target_id, device_id
        )
        if min(len(pool) for pool in train_pools.values()) < minimum:
            continue
        if min(len(pool) for pool in validation_pools.values()) < minimum:
            continue
        tasks.append(
            {
                "target_id": target_id,
                "name": data.class_names[target_id],
                "train": train_pools,
                "validate": validation_pools,
            }
        )
    if not tasks:
        raise ValueError("no non-hard base disease supports PAIR meta-training")
    return tasks


def _patches(cache, indices, device):
    return cache[indices].to(device, non_blocking=True).float()


def _base_validation(
    model,
    cache,
    text_queries,
    tasks,
    data,
    args,
    device,
):
    aurocs, worst, entropies = [], [], []
    model.eval()
    with torch.inference_mode():
        for task_index, task in enumerate(tasks):
            generator = torch.Generator().manual_seed(
                args.seed + 90_000 + task_index
            )
            indices, targets, devices = _group_batch(
                data,
                task["validate"],
                args.base_validation_per_group,
                generator,
            )
            logits, weights = model(
                _patches(cache, indices, device),
                text_queries[task["target_id"]].to(device),
            )
            targets = targets.to(device)
            devices = devices.to(device)
            aurocs.append(_auc(targets.bool().cpu(), logits.cpu()))
            group_aurocs = []
            for value in (0, 1):
                mask = devices.eq(value)
                group_aurocs.append(
                    _auc(
                        targets[mask].bool().cpu(),
                        logits[mask].cpu(),
                    )
                )
            worst.append(min(group_aurocs))
            entropies.append(float(normalized_entropy(weights)))
    return {
        "mean_auroc": statistics.mean(aurocs),
        "worst_device_auroc": statistics.mean(worst),
        "mean_router_entropy": statistics.mean(entropies),
    }


def _meta_train(
    model,
    cache,
    text_queries,
    tasks,
    data,
    args,
    device,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.meta_learning_rate,
        weight_decay=args.meta_weight_decay,
    )
    initial = _base_validation(
        model, cache, text_queries, tasks, data, args, device
    )
    best_state = copy.deepcopy(model.state_dict())
    best_metrics = initial
    best_key = (initial["worst_device_auroc"], initial["mean_auroc"])
    best_step = 0
    curve = [{"step": 0, **initial}]
    generator = torch.Generator().manual_seed(args.seed)
    task_rng = random.Random(args.seed)
    model.train()
    for step in range(1, args.meta_steps + 1):
        task = task_rng.choice(tasks)
        indices, targets, devices = _group_batch(
            data,
            task["train"],
            args.meta_batch_per_group,
            generator,
        )
        logits, weights = model(
            _patches(cache, indices, device),
            text_queries[task["target_id"]].to(device),
        )
        loss, components = intervention_loss(
            logits,
            targets.to(device),
            devices.to(device),
            args.beta_rex,
            args.lambda_invariance,
            args.lambda_responsiveness,
            args.minimum_margin,
        )
        loss = loss + args.lambda_entropy * normalized_entropy(weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.gradient_clip
        )
        optimizer.step()
        if step % args.validation_interval == 0 or step == args.meta_steps:
            metrics = _base_validation(
                model,
                cache,
                text_queries,
                tasks,
                data,
                args,
                device,
            )
            curve.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    **{
                        name: float(value.detach())
                        for name, value in components.items()
                    },
                    **metrics,
                }
            )
            key = (
                metrics["worst_device_auroc"],
                metrics["mean_auroc"],
            )
            if key > best_key:
                best_key = key
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = metrics
                best_step = step
            print(
                f"PAIR meta step={step}, "
                f"AUROC={metrics['mean_auroc']:.4f}, "
                f"worst={metrics['worst_device_auroc']:.4f}, "
                f"entropy={metrics['mean_router_entropy']:.3f}",
                flush=True,
            )
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    return {
        "best_step": best_step,
        "best_validation": best_metrics,
        "curve": curve,
    }


def _adapt_query(
    model,
    cache,
    text_query,
    support,
    labels,
    devices,
    args,
    device,
    seed,
):
    tokens = _patches(cache, support, device)
    labels = labels.to(device)
    devices = devices.to(device)
    with torch.no_grad():
        initial_query = model.project_query(text_query.to(device))
        encoded_support = model.encode_tokens(tokens)
    query = torch.nn.Parameter(initial_query.clone())
    raw_scale = torch.nn.Parameter(model.raw_scale.detach().clone())
    bias = torch.nn.Parameter(model.bias.detach().clone())
    optimizer = torch.optim.Adam(
        (query, raw_scale, bias), lr=args.adapt_learning_rate
    )
    torch.manual_seed(seed)
    for _ in range(args.adapt_steps):
        logits, _ = model.score_encoded(
            encoded_support, query, raw_scale, bias
        )
        loss, _ = intervention_loss(
            logits,
            labels,
            devices,
            args.beta_rex,
            args.lambda_invariance,
            args.lambda_responsiveness,
            args.minimum_margin,
        )
        loss = (
            loss
            + args.lambda_query_anchor
            * (1 - F.cosine_similarity(query[None], initial_query[None])[0])
            + args.lambda_calibration_anchor
            * (
                (raw_scale - model.raw_scale.detach()).square()
                + (bias - model.bias.detach()).square()
            )
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (query, raw_scale, bias), args.gradient_clip
        )
        optimizer.step()
        with torch.no_grad():
            query.copy_(F.normalize(query, dim=-1))
    return query.detach(), raw_scale.detach(), bias.detach()


def _support_labels(arrays, class_id, support, device):
    labels = _tensor(arrays["labels"][:, class_id], support, device).long()
    devices = _tensor(
        arrays["labels"][
            :, arrays["device_id"]
        ],
        support,
        device,
    ).long()
    return labels, devices


def _metrics(logits, zero, one, reference_zero, reference_one, targets, nuisance):
    result = {
        "auroc": _auc(targets.bool(), logits),
        "auprc": _average_precision(targets.bool(), logits),
        "sms_fixed_reference": float(
            normalized_sms(zero, one, reference_zero, reference_one)
        ),
        "ranking_disagreement": ranking_disagreement(zero, one),
    }
    values = []
    for device in (0, 1):
        mask = nuisance.eq(device)
        observed = _auc(targets[mask].bool(), logits[mask])
        result[f"device_{device}_auroc"] = observed
        values.append(observed)
    result["worst_device_auroc"] = min(values)
    return result


def _score_run(
    model,
    cache,
    arrays,
    text_queries,
    episodes,
    shot,
    args,
    device,
    run_seed,
):
    class_id = int(episodes["target_id"])
    outputs = {
        method: ([], [], [], [], [])
        for method in METHODS
    }
    entropy = []
    for episode in range(len(episodes["positive"])):
        query_indices = episodes["query"][episode].long()
        query_tokens = _patches(cache, query_indices, device)
        text_logits = _tensor(
            arrays["prior"][:, class_id], query_indices, device
        )
        text_query = text_queries[class_id].to(device)
        with torch.no_grad():
            projected_query = model.project_query(text_query)
            encoded_query = model.encode_tokens(query_tokens)
            zero_logits, zero_weights = model.score_encoded(
                encoded_query, projected_query
            )
            entropy.append(float(normalized_entropy(zero_weights)))
        balanced = _balanced_support(episodes, episode, shot)
        panel_zero = _panel_support(episodes, episode, shot, 0)
        panel_one = _panel_support(episodes, episode, shot, 1)
        support_sets = (balanced, panel_zero, panel_one)
        adapted_scores = []
        for support_index, support in enumerate(support_sets):
            labels, devices = _support_labels(
                arrays, class_id, support, device
            )
            query, scale, bias = _adapt_query(
                model,
                cache,
                text_query,
                support,
                labels,
                devices,
                args,
                device,
                run_seed
                + episode * 100
                + shot * 10_000
                + support_index,
            )
            with torch.no_grad():
                adapted_scores.append(
                    model.score_encoded(
                        encoded_query, query, scale, bias
                    )[0]
                )
        reference = [
            _protonet(
                arrays["rad"],
                episodes,
                episode,
                shot,
                environment,
                query_indices,
                device,
            )
            for environment in (0, 1)
        ]
        balanced_labels, _ = _support_labels(
            arrays, class_id, balanced, device
        )
        global_main = _global_protonet(
            arrays,
            balanced,
            balanced_labels,
            query_indices,
            device,
        )
        current = {
            "vlm_text_prior": (
                text_logits,
                text_logits,
                text_logits,
                reference[0],
                reference[1],
            ),
            "rad_global_protonet": (
                global_main,
                reference[0],
                reference[1],
                reference[0],
                reference[1],
            ),
            "pair_router_zero_shot": (
                zero_logits,
                zero_logits,
                zero_logits,
                reference[0],
                reference[1],
            ),
            "pair_cxr_adapted": (
                adapted_scores[0],
                adapted_scores[1],
                adapted_scores[2],
                reference[0],
                reference[1],
            ),
        }
        for method, tensors in current.items():
            for destination, tensor in zip(outputs[method], tensors):
                destination.append(tensor.detach().cpu())
    return {
        method: tuple(torch.cat(values) for values in destinations)
        for method, destinations in outputs.items()
    }, statistics.mean(entropy)


def _evaluate_partition(
    model,
    cache,
    arrays,
    text_queries,
    target_episodes,
    partition,
    args,
    device,
):
    rows, score_cache = [], {}
    signature = {
        "partition": partition,
        "manifest_sha256": target_episodes[
            (args.seeds[0], partition)
        ].get("manifest_sha256"),
        "episodes_per_seed": args.episodes_per_seed,
        "adapt_steps": args.adapt_steps,
        "adapt_learning_rate": args.adapt_learning_rate,
        "lambda_query_anchor": args.lambda_query_anchor,
        "lambda_calibration_anchor": args.lambda_calibration_anchor,
        "beta_rex": args.beta_rex,
        "lambda_invariance": args.lambda_invariance,
        "lambda_responsiveness": args.lambda_responsiveness,
        "minimum_margin": args.minimum_margin,
    }
    for seed in args.seeds:
        for shot in args.shots:
            score_path = args.output_dir / (
                f"scores_{partition}_shot_{shot:02d}_seed_{seed:03d}.pt"
            )
            if score_path.exists():
                saved = torch.load(
                    score_path, map_location="cpu", weights_only=False
                )
                if saved.get("signature") != signature:
                    raise ValueError(
                        f"{score_path} uses another PAIR protocol"
                    )
                scores = saved["scores"]
                entropy = saved["router_entropy"]
                print(
                    f"reusing PAIR {partition}, shot={shot}, seed={seed}",
                    flush=True,
                )
            else:
                scores, entropy = _score_run(
                    model,
                    cache,
                    arrays,
                    text_queries,
                    target_episodes[(seed, partition)],
                    shot,
                    args,
                    device,
                    args.seed + seed,
                )
                torch.save(
                    {
                        "signature": signature,
                        "scores": scores,
                        "router_entropy": entropy,
                    },
                    score_path,
                )
                print(
                    f"finished PAIR {partition}, shot={shot}, seed={seed}",
                    flush=True,
                )
            episodes = target_episodes[(seed, partition)]
            targets = episodes["targets"].flatten()
            nuisance = episodes["nuisance"].flatten()
            for method, tensors in scores.items():
                score_cache[(method, seed, shot, partition)] = tensors
                metrics = _metrics(
                    *tensors, targets, nuisance
                )
                if method == "pair_router_zero_shot":
                    metrics["router_entropy"] = entropy
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "partition": partition,
                            "target": "Pneumothorax",
                            "method": method,
                            "seed": seed,
                            "shot": shot,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return rows, score_cache


def _summaries(rows):
    keys = ("partition", "target", "method", "shot", "metric")
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


def _point(summaries, partition, method, shot):
    return {
        row["metric"]: row["mean"]
        for row in summaries
        if row["partition"] == partition
        and row["method"] == method
        and row["shot"] == shot
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comed-cache", type=Path, required=True)
    parser.add_argument("--rad-cache", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4)
    )
    parser.add_argument(
        "--shots", type=int, nargs="+", default=(1, 3, 5, 10, 20)
    )
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument(
        "--hard-diseases", nargs="+", default=HARD_DISEASES
    )
    parser.add_argument("--alignment-samples", type=int, default=20_000)
    parser.add_argument("--alignment-ridge", type=float, default=10.0)
    parser.add_argument("--router-bottleneck", type=int, default=64)
    parser.add_argument("--meta-steps", type=int, default=500)
    parser.add_argument("--meta-batch-per-group", type=int, default=4)
    parser.add_argument("--base-validation-per-group", type=int, default=40)
    parser.add_argument("--meta-learning-rate", type=float, default=1e-3)
    parser.add_argument("--meta-weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-interval", type=int, default=50)
    parser.add_argument("--adapt-steps", type=int, default=30)
    parser.add_argument("--adapt-learning-rate", type=float, default=0.03)
    parser.add_argument("--beta-rex", type=float, default=0.1)
    parser.add_argument("--lambda-invariance", type=float, default=0.3)
    parser.add_argument("--lambda-responsiveness", type=float, default=0.3)
    parser.add_argument("--minimum-margin", type=float, default=0.2)
    parser.add_argument("--lambda-entropy", type=float, default=0.01)
    parser.add_argument("--lambda-query-anchor", type=float, default=1.0)
    parser.add_argument(
        "--lambda-calibration-anchor", type=float, default=0.1
    )
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--sms-budget", type=float, default=0.30)
    parser.add_argument("--primary-shot", type=int, default=10)
    parser.add_argument("--minimum-validation-gain", type=float, default=0.02)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.primary_shot not in args.shots:
        parser.error("primary-shot must be present in shots")
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays, metadata, data = _load_cache(args.comed_cache)
    arrays["device_id"] = data.class_names.index("Support Devices")
    patch_cache, patch_metadata = load_patch_cache(
        args.rad_cache,
        data.manifest_sha256,
        expected_model=RAD_DINO_MODEL,
        expected_pool_grid=14,
        access_mode="stream",
    )
    if int(patch_metadata["shape"][-1]) != arrays["rad"].shape[1]:
        raise ValueError("Rad-DINO global and patch feature widths differ")
    text_queries = _text_queries(
        arrays, metadata, data, args, device
    )
    tasks = _meta_tasks(data, metadata, args)
    model = PAIRRouter(
        arrays["rad"].shape[1], args.router_bottleneck
    ).to(device)
    model_path = args.output_dir / "pair_router.pt"
    model_signature = {
        "manifest_sha256": data.manifest_sha256,
        "router_bottleneck": args.router_bottleneck,
        "hard_diseases": list(args.hard_diseases),
        "meta_steps": args.meta_steps,
        "meta_batch_per_group": args.meta_batch_per_group,
        "base_validation_per_group": args.base_validation_per_group,
        "meta_learning_rate": args.meta_learning_rate,
        "meta_weight_decay": args.meta_weight_decay,
        "beta_rex": args.beta_rex,
        "lambda_invariance": args.lambda_invariance,
        "lambda_responsiveness": args.lambda_responsiveness,
        "minimum_margin": args.minimum_margin,
        "lambda_entropy": args.lambda_entropy,
        "seed": args.seed,
    }
    if model_path.exists():
        saved_model = torch.load(
            model_path, map_location="cpu", weights_only=False
        )
        if saved_model.get("signature") != model_signature:
            raise ValueError(
                "existing PAIR router uses another protocol; "
                "choose a new output directory"
            )
        model.load_state_dict(saved_model["state_dict"])
        training = saved_model["training"]
        print("reusing meta-trained PAIR router", flush=True)
    else:
        training = _meta_train(
            model,
            patch_cache,
            text_queries,
            tasks,
            data,
            args,
            device,
        )
        torch.save(
            {
                "signature": model_signature,
                "state_dict": model.state_dict(),
                "training": training,
                "base_tasks": [task["name"] for task in tasks],
                "hard_diseases": args.hard_diseases,
            },
            model_path,
        )
    model.eval().requires_grad_(False)
    target_episodes = _load_target_episodes(
        args.episodes, data, args
    )
    available_supports = min(
        int(episodes["positive"].shape[2])
        for episodes in target_episodes.values()
    )
    if max(args.shots) > available_supports:
        raise ValueError(
            f"requested {max(args.shots)}-shot but stored panels contain "
            f"only {available_supports} candidates per environment"
        )
    validation_rows, _ = _evaluate_partition(
        model,
        patch_cache,
        arrays,
        text_queries,
        target_episodes,
        "validate",
        args,
        device,
    )
    summaries = _summaries(validation_rows)
    text = _point(
        summaries,
        "validate",
        "vlm_text_prior",
        args.primary_shot,
    )
    adapted = _point(
        summaries,
        "validate",
        "pair_cxr_adapted",
        args.primary_shot,
    )
    gate_passed = (
        adapted["auroc"]
        >= text["auroc"] + args.minimum_validation_gain
        and adapted["sms_fixed_reference"] <= args.sms_budget
        and adapted["worst_device_auroc"]
        >= text["worst_device_auroc"]
    )
    rows = list(validation_rows)
    test_accessed = False
    if gate_passed:
        test_rows, _ = _evaluate_partition(
            model,
            patch_cache,
            arrays,
            text_queries,
            target_episodes,
            "test",
            args,
            device,
        )
        rows.extend(test_rows)
        summaries = _summaries(rows)
        test_accessed = True
    _write(args.output_dir / "per_seed_metrics.csv", rows)
    _write(args.output_dir / "summary_metrics.csv", summaries)
    if not gate_passed:
        status = "stop_pair_cxr_support_adaptation_not_safe"
    else:
        test = _point(
            summaries,
            "test",
            "pair_cxr_adapted",
            args.primary_shot,
        )
        status = (
            "pair_cxr_reaches_0.70_target"
            if test["auroc"] >= 0.70
            else "pair_cxr_promising_but_below_0.70"
        )
    decision = {
        "status": status,
        "frozen_encoder": True,
        "hard_diseases_excluded_from_meta_training": args.hard_diseases,
        "base_tasks": [task["name"] for task in tasks],
        "primary_shot": args.primary_shot,
        "validation_gate": {
            "passed": gate_passed,
            "minimum_auroc_gain_over_text": args.minimum_validation_gain,
            "sms_budget": args.sms_budget,
            "text": text,
            "adapted": adapted,
        },
        "test_accessed": test_accessed,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "training.json").write_text(
        json.dumps(training, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"PAIR-CXR results written to {args.output_dir}; "
        f"decision={status}"
    )


if __name__ == "__main__":
    main()
