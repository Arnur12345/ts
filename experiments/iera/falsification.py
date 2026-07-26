"""Run repaired-detector falsification baselines on identical support swaps."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import multiprocessing as mp
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.residuals.data import load_config, load_dataset
from experiments.residuals.metrics import _auc, select_temperature, select_threshold

from .detector_diagnostic import _pool
from .episodes import (
    PILOT_PAIRS,
    eligible_directed_pairs,
    generate_pair_episodes,
    patient_counts,
    split_indices,
    stratum_pools,
)
from .labels import restore_raw_target_status
from .patch_cache import MODEL, RAD_DINO_MODEL, load_patch_cache
from .robust_metrics import evaluate
from .robust_model import RobustBinaryModel, project_direction
from .robust_support import (
    balanced_choices,
    environment_choices,
    nuisance_probability,
    select_supports,
)


LEARNED_METHODS = ("rex", "adapter_only", "anchor_only", "full_iera")
CONSTRAINED_METHODS = ("adapter_only", "anchor_only", "full_iera")


def _ids(data, names: tuple[str, str]) -> tuple[int, int]:
    return data.class_names.index(names[0]), data.class_names.index(names[1])


def _attach_choices(episodes: dict, support_count: int, seed: int) -> None:
    count = len(episodes["positive"])
    episodes["random_positive_env"] = environment_choices(
        count,
        support_count,
        nuisance_probability(episodes["patient_counts"], 1),
        seed + 11,
    )
    episodes["random_negative_env"] = environment_choices(
        count,
        support_count,
        nuisance_probability(episodes["patient_counts"], 0),
        seed + 23,
    )
    episodes["balanced_positive_env"] = balanced_choices(
        count, support_count, seed + 37
    )
    episodes["balanced_negative_env"] = balanced_choices(
        count, support_count, seed + 41
    )


def _gather(
    patches,
    indices: torch.Tensor,
    metadata: dict,
    retained_grid: int,
    device: torch.device,
) -> torch.Tensor:
    values = patches[indices].to(device, non_blocking=True)
    return _pool(values, int(metadata["pool_grid"]), retained_grid)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _meta_split(data, indices: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    train, validation = [], []
    for index in indices.tolist():
        subject = data.subject_ids[index]
        bucket = int.from_bytes(
            hashlib.sha256(
                f"iera-robust-meta|{seed}|{subject}".encode()
            ).digest()[:8],
            "big",
        ) % 10_000
        (validation if bucket >= 8500 else train).append(index)
    return (
        torch.tensor(train, dtype=torch.long),
        torch.tensor(validation, dtype=torch.long),
    )


def _selected(
    positive_panels: torch.Tensor,
    negative_panels: torch.Tensor,
    episodes: dict,
    start: int,
    end: int,
    shot: int,
    balanced: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    support_count = 2 * shot
    prefix = "balanced" if balanced else "random"
    positive, positive_mask = select_supports(
        positive_panels,
        episodes[f"{prefix}_positive_env"][start:end],
        support_count,
    )
    negative, negative_mask = select_supports(
        negative_panels,
        episodes[f"{prefix}_negative_env"][start:end],
        support_count,
    )
    return positive, negative, positive_mask, negative_mask


def _mean_difference(
    positive: torch.Tensor,
    negative: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    tokens = torch.cat((positive, negative), dim=2)
    mask = torch.cat((positive_mask, negative_mask), dim=2)
    weights = mask[..., None, None].to(tokens.dtype)
    center = (tokens * weights).sum(dim=(2, 3))
    denominator = (
        mask.sum(dim=2).clamp_min(1).to(tokens.dtype) * tokens.shape[3]
    )
    center = center / denominator[..., None]
    return F.normalize(center[:, 1] - center[:, 0], dim=-1)


def _cheap_scores(
    patches,
    metadata: dict,
    episodes: dict,
    shot: int,
    batch_size: int,
    retained_grid: int,
    device: torch.device,
    methods: tuple[str, ...],
    text_direction: torch.Tensor | None = None,
) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    model = RobustBinaryModel(metadata["shape"][-1]).to(device).eval()
    values = {
        method: ([], [], [], [], [])
        for method in methods
    }
    with torch.inference_mode():
        for start in range(0, len(episodes["positive"]), batch_size):
            end = min(start + batch_size, len(episodes["positive"]))
            positive_panels = _gather(
                patches, episodes["positive"][start:end], metadata,
                retained_grid, device,
            )
            negative_panels = _gather(
                patches, episodes["negative"][start:end], metadata,
                retained_grid, device,
            )
            query = _gather(
                patches, episodes["query"][start:end], metadata,
                retained_grid, device,
            )
            panel_positive = positive_panels[:, :, :shot]
            panel_negative = negative_panels[:, :, :shot]
            reference_zero, reference_one = model.swapped_logits(
                panel_positive, panel_negative, query, "uniform"
            )
            for method in methods:
                balanced = method == "nuisance_balanced"
                positive, negative, positive_mask, negative_mask = _selected(
                    positive_panels, negative_panels, episodes,
                    start, end, shot, balanced,
                )
                transformed_query = query
                transformed_panel_positive = panel_positive
                transformed_panel_negative = panel_negative
                if method == "mean_difference_projection":
                    direction = _mean_difference(
                        positive, negative, positive_mask, negative_mask
                    )
                    positive = project_direction(positive, direction)
                    negative = project_direction(negative, direction)
                    transformed_query = project_direction(query, direction)
                    transformed_panel_positive = project_direction(
                        panel_positive, direction
                    )
                    transformed_panel_negative = project_direction(
                        panel_negative, direction
                    )
                elif method == "text_direction_orthogonalization":
                    if text_direction is None:
                        raise ValueError("text method requires a text direction")
                    direction = text_direction.to(device)
                    positive = project_direction(positive, direction)
                    negative = project_direction(negative, direction)
                    transformed_query = project_direction(query, direction)
                    transformed_panel_positive = project_direction(
                        panel_positive, direction
                    )
                    transformed_panel_negative = project_direction(
                        panel_negative, direction
                    )
                logits = model(
                    positive, negative, transformed_query, "uniform",
                    positive_mask, negative_mask,
                )
                panel_zero, panel_one = model.swapped_logits(
                    transformed_panel_positive,
                    transformed_panel_negative,
                    transformed_query,
                    "uniform",
                )
                output = values[method]
                for destination, tensor in zip(
                    output,
                    (
                        logits, panel_zero, panel_one,
                        reference_zero, reference_one,
                    ),
                ):
                    destination.append(tensor.cpu())
    return {
        method: tuple(torch.cat(items).flatten() for items in method_values)
        for method, method_values in values.items()
    }


def _learned_scores(
    model: RobustBinaryModel,
    method: str,
    patches,
    metadata: dict,
    episodes: dict,
    shot: int,
    batch_size: int,
    retained_grid: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    values = ([], [], [], [], [])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(episodes["positive"]), batch_size):
            end = min(start + batch_size, len(episodes["positive"]))
            positive_panels = _gather(
                patches, episodes["positive"][start:end], metadata,
                retained_grid, device,
            )
            negative_panels = _gather(
                patches, episodes["negative"][start:end], metadata,
                retained_grid, device,
            )
            query = _gather(
                patches, episodes["query"][start:end], metadata,
                retained_grid, device,
            )
            positive, negative, positive_mask, negative_mask = _selected(
                positive_panels, negative_panels, episodes,
                start, end, shot, False,
            )
            logits = model(
                positive, negative, query, method,
                positive_mask, negative_mask,
            )
            panel_positive = positive_panels[:, :, :shot]
            panel_negative = negative_panels[:, :, :shot]
            panel_zero, panel_one = model.swapped_logits(
                panel_positive, panel_negative, query, method
            )
            reference_zero, reference_one = model.swapped_logits(
                panel_positive, panel_negative, query, "uniform"
            )
            for destination, tensor in zip(
                values,
                (
                    logits, panel_zero, panel_one,
                    reference_zero, reference_one,
                ),
            ):
                destination.append(tensor.cpu())
    return tuple(torch.cat(items).flatten() for items in values)


def _score_to_rows(
    scores: dict,
    episode_sets: dict,
    pair_names: dict,
    shots: list[int],
    seeds: list[int],
) -> list[dict]:
    rows = []
    methods = sorted({key[0] for key in scores})
    pair_ids = sorted({key[1] for key in scores})
    for method in methods:
        for pair_id in pair_ids:
            for seed in seeds:
                for shot in shots:
                    validation = scores[(method, pair_id, seed, shot, "validate")]
                    validation_targets = episode_sets[
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
                    test = scores[(method, pair_id, seed, shot, "test")]
                    test_episodes = episode_sets[(pair_id, seed, "test")]
                    metrics = evaluate(
                        *test,
                        test_episodes["targets"].flatten(),
                        test_episodes["nuisance"].flatten(),
                        temperature,
                        threshold,
                    )
                    target, confounder = pair_names[pair_id]
                    for metric, value in metrics.items():
                        rows.append(
                            {
                                "pair": f"{target}__{confounder}",
                                "target": target,
                                "confounder": confounder,
                                "method": method,
                                "rho": "",
                                "shot": shot,
                                "seed": seed,
                                "metric": metric,
                                "value": value,
                            }
                        )
    return rows


def _cheap_worker(
    cache_path: Path,
    manifest_hash: str,
    episode_sets: dict,
    pair_names: dict,
    shots: list[int],
    seeds: list[int],
    retained_grid: int,
    batch_size: int,
    device_name: str,
    methods: tuple[str, ...],
    checkpoint: Path,
    text_direction_path: Path | None = None,
    expected_model: str | None = None,
    signature: dict | None = None,
) -> None:
    device = torch.device(device_name)
    try:
        patches, metadata = load_patch_cache(
            cache_path, manifest_hash, expected_model=expected_model,
            access_mode="shared",
        )
    except RuntimeError:
        patches, metadata = load_patch_cache(
            cache_path, manifest_hash, expected_model=expected_model,
            access_mode="stream",
        )
    text_direction = None
    if text_direction_path is not None:
        saved = torch.load(text_direction_path, map_location="cpu", weights_only=False)
        text_direction = saved["direction"]
        if len(text_direction) != metadata["shape"][-1]:
            raise ValueError("text direction and patch cache widths differ")
    scores = {}
    for pair_id in sorted(pair_names):
        for seed in seeds:
            for partition in ("validate", "test"):
                episodes = episode_sets[(pair_id, seed, partition)]
                for shot in shots:
                    computed = _cheap_scores(
                        patches, metadata, episodes, shot, batch_size,
                        retained_grid, device, methods, text_direction,
                    )
                    for method, tensors in computed.items():
                        scores[(method, pair_id, seed, shot, partition)] = tensors
                print(
                    f"finished {metadata['model']}, pair {pair_id}, "
                    f"{partition}, seed {seed}",
                    flush=True,
                )
    rows = _score_to_rows(scores, episode_sets, pair_names, shots, seeds)
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"metadata": metadata, "rows": rows, "signature": signature}
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(checkpoint)


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict]) -> list[dict]:
    keys = ("pair", "target", "confounder", "method", "rho", "shot", "metric")
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


def _prepare_evaluation(data, args) -> tuple[dict, dict]:
    partitions = {
        name: split_indices(data, name, args.split_seed)
        for name in ("validate", "test")
    }
    episode_sets, pair_names = {}, {}
    candidate_per_environment = 2 * max(args.shots)
    pair_id = 0
    for names in PILOT_PAIRS:
        target, confounder = _ids(data, names)
        counts = {
            partition: patient_counts(
                data, stratum_pools(data, indices, target, confounder)
            )
            for partition, indices in partitions.items()
        }
        if not all(
            min(values.values()) >= args.min_stratum_patients
            for values in counts.values()
        ):
            continue
        pair_names[pair_id] = names
        for seed in args.seeds:
            for partition_index, (partition, indices) in enumerate(
                partitions.items()
            ):
                episode_seed = (
                    args.seed + pair_id * 1_000_000
                    + partition_index * 100_000 + seed
                )
                generated = generate_pair_episodes(
                    data, indices, target, confounder, args.episodes,
                    candidate_per_environment, args.queries_per_stratum,
                    episode_seed,
                    min_stratum_patients=args.min_stratum_patients,
                )
                _attach_choices(
                    generated, 2 * max(args.shots), episode_seed + 50_000
                )
                episode_sets[(pair_id, seed, partition)] = generated
        pair_id += 1
    if not pair_names:
        raise ValueError("no evaluation pair satisfies the stratum requirement")
    return episode_sets, pair_names


def _prepare_training(data, config: dict, args, run_seed: int) -> tuple[list[dict], list[dict], list[dict]]:
    all_train = split_indices(data, "train", args.split_seed)
    train_indices, validation_indices = _meta_split(
        data, all_train, args.split_seed
    )
    evaluation_labels = {
        data.class_names.index(name)
        for names in getattr(args, "evaluation_pair_names", PILOT_PAIRS)
        for name in names
        if name in data.class_names
    }
    base_ids = [
        data.class_names.index(name)
        for name in config["class_partitions"]["base"]
        if name in data.class_names
        and data.class_names.index(name) not in evaluation_labels
    ]
    validation_minimum = max(
        2 * args.train_shot + args.queries_per_stratum, 10
    )
    train_pairs = eligible_directed_pairs(
        data, train_indices, base_ids, args.min_stratum_patients, base_ids
    )
    validation_pairs = eligible_directed_pairs(
        data, validation_indices, base_ids, validation_minimum, base_ids
    )
    pairs = sorted(set(train_pairs) & set(validation_pairs))
    if not pairs:
        raise ValueError(
            "no base-only pairs support the meta-train/base-validation split"
        )
    random.Random(run_seed).shuffle(pairs)
    pairs = pairs[: min(args.max_train_pairs, len(pairs))]
    candidate_per_environment = 2 * args.train_shot
    train_bank, validation_bank, pair_metadata = [], [], []
    episodes_per_pair = max(
        2, math.ceil(min(args.max_train_steps, 120) / len(pairs))
    )
    for pair_index, (target, confounder) in enumerate(pairs):
        train = generate_pair_episodes(
            data, train_indices, target, confounder, episodes_per_pair,
            candidate_per_environment, args.queries_per_stratum,
            run_seed + 10_000 + pair_index,
            min_stratum_patients=args.min_stratum_patients,
        )
        validation = generate_pair_episodes(
            data, validation_indices, target, confounder,
            args.base_validation_episodes, candidate_per_environment,
            args.queries_per_stratum, run_seed + 50_000 + pair_index,
            min_stratum_patients=validation_minimum,
        )
        _attach_choices(
            train, 2 * args.train_shot, run_seed + 70_000 + pair_index
        )
        _attach_choices(
            validation, 2 * args.train_shot,
            run_seed + 90_000 + pair_index,
        )
        train_bank.append(train)
        validation_bank.append(validation)
        pair_metadata.append(
            {
                "target": data.class_names[target],
                "confounder": data.class_names[confounder],
            }
        )
    return train_bank, validation_bank, pair_metadata


def _training_batch(
    patches,
    metadata: dict,
    episodes: dict,
    episode_index: int,
    shot: int,
    retained_grid: int,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    start, end = episode_index, episode_index + 1
    positive_panels = _gather(
        patches, episodes["positive"][start:end], metadata,
        retained_grid, device,
    )
    negative_panels = _gather(
        patches, episodes["negative"][start:end], metadata,
        retained_grid, device,
    )
    query = _gather(
        patches, episodes["query"][start:end], metadata,
        retained_grid, device,
    )
    positive, negative, positive_mask, negative_mask = _selected(
        positive_panels, negative_panels, episodes, start, end, shot, False
    )
    return (
        positive,
        negative,
        positive_mask,
        negative_mask,
        query,
        episodes["targets"][start:end].to(device),
        episodes["nuisance"][start:end].to(device),
        positive_panels[:, :, :shot],
        negative_panels[:, :, :shot],
    )


def _constraint(
    model: RobustBinaryModel,
    method: str,
    panel_positive: torch.Tensor,
    panel_negative: torch.Tensor,
    query: torch.Tensor,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    panel_zero, panel_one = model.swapped_logits(
        panel_positive, panel_negative, query, method
    )
    with torch.no_grad():
        reference_zero, reference_one = model.swapped_logits(
            panel_positive, panel_negative, query, "uniform"
        )
    method_shift = (panel_one - panel_zero).abs().mean()
    reference_shift = (reference_one - reference_zero).abs().mean()
    violation = method_shift - rho * reference_shift.detach()
    return violation, method_shift, reference_shift


def _base_validation(
    model: RobustBinaryModel,
    method: str,
    patches,
    metadata: dict,
    bank: list[dict],
    args,
    device: torch.device,
    rho: float | None,
) -> dict[str, float]:
    losses, worst_aurocs, ratios = [], [], []
    model.eval()
    with torch.inference_mode():
        for episodes in bank:
            logits_runs, target_runs, nuisance_runs = [], [], []
            method_shifts, reference_shifts = [], []
            for start in range(0, len(episodes["positive"]), args.episode_batch_size):
                end = min(
                    start + args.episode_batch_size, len(episodes["positive"])
                )
                positive_panels = _gather(
                    patches, episodes["positive"][start:end], metadata,
                    args.retained_grid, device,
                )
                negative_panels = _gather(
                    patches, episodes["negative"][start:end], metadata,
                    args.retained_grid, device,
                )
                query = _gather(
                    patches, episodes["query"][start:end], metadata,
                    args.retained_grid, device,
                )
                positive, negative, positive_mask, negative_mask = _selected(
                    positive_panels, negative_panels, episodes,
                    start, end, args.train_shot, False,
                )
                logits = model(
                    positive, negative, query, method,
                    positive_mask, negative_mask,
                )
                targets = episodes["targets"][start:end].to(device)
                losses.append(
                    float(F.binary_cross_entropy_with_logits(logits, targets))
                )
                logits_runs.append(logits.flatten().cpu())
                target_runs.append(targets.flatten().cpu())
                nuisance_runs.append(
                    episodes["nuisance"][start:end].flatten()
                )
                if rho is not None:
                    violation, method_shift, reference_shift = _constraint(
                        model, method,
                        positive_panels[:, :, : args.train_shot],
                        negative_panels[:, :, : args.train_shot],
                        query, rho,
                    )
                    del violation
                    method_shifts.append(float(method_shift))
                    reference_shifts.append(float(reference_shift))
            logits = torch.cat(logits_runs)
            targets = torch.cat(target_runs).bool()
            nuisance = torch.cat(nuisance_runs)
            group_aurocs = [
                _auc(targets[nuisance.eq(value)], logits[nuisance.eq(value)])
                for value in (0, 1)
            ]
            worst_aurocs.append(min(group_aurocs))
            if rho is not None:
                ratios.append(
                    statistics.mean(method_shifts)
                    / max(rho * statistics.mean(reference_shifts), 1e-12)
                )
    max_ratio = max(ratios) if ratios else 0.0
    return {
        "loss": statistics.mean(losses),
        "worst_nuisance_auroc": min(worst_aurocs),
        "max_sms_budget_ratio": max_ratio,
        "sms_budget_satisfied": float(not ratios or max_ratio <= 1.0),
    }


def _checkpoint_key(method: str, validation: dict[str, float]) -> tuple[float, float]:
    if method == "rex":
        return 1.0, validation["worst_nuisance_auroc"]
    if validation["sms_budget_satisfied"]:
        return 1.0, validation["worst_nuisance_auroc"]
    return 0.0, -validation["max_sms_budget_ratio"]


def _train_model(
    model: RobustBinaryModel,
    method: str,
    patches,
    metadata: dict,
    train_bank: list[dict],
    validation_bank: list[dict],
    args,
    device: torch.device,
    run_seed: int,
    rho: float | None,
) -> dict:
    parameters = model.configure_trainable(method)
    optimizer = torch.optim.AdamW(
        parameters, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    lagrange = args.lagrange_initial if rho is not None else 0.0
    initial = _base_validation(
        model, method, patches, metadata, validation_bank,
        args, device, rho,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_validation = initial
    best_key = _checkpoint_key(method, initial)
    best_step = 0
    curve = [{"step": 0, "lagrange": lagrange, **initial}]
    generator = torch.Generator().manual_seed(run_seed)
    choices = [
        (pair_index, episode_index)
        for pair_index, episodes in enumerate(train_bank)
        for episode_index in range(len(episodes["positive"]))
    ]
    no_improvement = 0
    model.train()
    for step in range(1, args.max_train_steps + 1):
        selected = int(torch.randint(len(choices), (1,), generator=generator))
        pair_index, episode_index = choices[selected]
        batch = _training_batch(
            patches, metadata, train_bank[pair_index], episode_index,
            args.train_shot, args.retained_grid, device,
        )
        (
            positive, negative, positive_mask, negative_mask, query,
            targets, nuisance, panel_positive, panel_negative,
        ) = batch
        logits = model(
            positive, negative, query, method,
            positive_mask, negative_mask,
        )
        element_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        classification = element_loss.mean()
        penalty = classification.new_zeros(())
        violation = classification.new_zeros(())
        if method == "rex":
            group_losses = [
                element_loss[nuisance.eq(value)].mean() for value in (0, 1)
            ]
            penalty = (group_losses[0] - group_losses[1]).square()
            loss = classification + args.rex_weight * penalty
        else:
            violation, _, _ = _constraint(
                model, method, panel_positive, panel_negative, query,
                float(rho),
            )
            penalty = F.relu(violation)
            loss = classification + lagrange * penalty
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        optimizer.step()
        if rho is not None:
            lagrange = min(
                args.lagrange_max,
                max(
                    0.0,
                    lagrange
                    + args.lagrange_learning_rate * float(violation.detach()),
                ),
            )
        if step % args.validation_interval == 0 or step == args.max_train_steps:
            validation = _base_validation(
                model, method, patches, metadata, validation_bank,
                args, device, rho,
            )
            curve.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "train_classification": float(classification.detach()),
                    "train_penalty": float(penalty.detach()),
                    "lagrange": lagrange,
                    **validation,
                }
            )
            candidate_key = _checkpoint_key(method, validation)
            improved = (
                candidate_key[0] > best_key[0]
                or (
                    candidate_key[0] == best_key[0]
                    and candidate_key[1]
                    > best_key[1] + args.early_stopping_min_delta
                )
            )
            if improved:
                best_key = candidate_key
                best_state = copy.deepcopy(model.state_dict())
                best_validation = validation
                best_step = step
                no_improvement = 0
            else:
                no_improvement += 1
            print(
                f"training {method}, rho={rho}, seed={run_seed}: "
                f"{step}/{args.max_train_steps}, "
                f"worst={validation['worst_nuisance_auroc']:.4f}, "
                f"budget={'yes' if validation['sms_budget_satisfied'] else 'no'}",
                flush=True,
            )
            if no_improvement >= args.early_stopping_patience:
                break
            model.train()
    model.load_state_dict(best_state)
    model.eval()
    return {
        "method": method,
        "rho": rho,
        "best_step": best_step,
        "best_validation": best_validation,
        "final_lagrange": lagrange,
        "curve": curve,
    }


def _model_rows(
    model: RobustBinaryModel,
    method: str,
    rho: float | None,
    patches,
    metadata: dict,
    episode_sets: dict,
    pair_names: dict,
    args,
    device: torch.device,
    seed: int,
    display_method: str | None = None,
) -> list[dict]:
    rows = []
    for pair_id, names in sorted(pair_names.items()):
        for shot in args.shots:
            validation_episodes = episode_sets[
                (pair_id, seed, "validate")
            ]
            validation = _learned_scores(
                model, method, patches, metadata, validation_episodes,
                shot, args.episode_batch_size, args.retained_grid, device,
            )
            validation_targets = validation_episodes["targets"].flatten()
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
            test_episodes = episode_sets[(pair_id, seed, "test")]
            test = _learned_scores(
                model, method, patches, metadata, test_episodes,
                shot, args.episode_batch_size, args.retained_grid, device,
            )
            metrics = evaluate(
                *test,
                test_episodes["targets"].flatten(),
                test_episodes["nuisance"].flatten(),
                temperature,
                threshold,
            )
            for metric, value in metrics.items():
                rows.append(
                    {
                        "pair": f"{names[0]}__{names[1]}",
                        "target": names[0],
                        "confounder": names[1],
                        "method": display_method or method,
                        "rho": "" if rho is None else rho,
                        "shot": shot,
                        "seed": seed,
                        "metric": metric,
                        "value": value,
                    }
                )
            print(
                f"finished {display_method or method}, rho={rho}, {names[0]}, "
                f"{shot}-shot, seed={seed}",
                flush=True,
            )
    return rows


def _learned_worker(
    cache_path: Path,
    manifest_hash: str,
    training_banks: dict,
    training_pairs: dict,
    episode_sets: dict,
    pair_names: dict,
    args,
    methods: tuple[str, ...],
    rhos: list[float],
    stage_dir: Path,
    device_name: str,
) -> None:
    device = torch.device(device_name)
    try:
        patches, metadata = load_patch_cache(
            cache_path, manifest_hash, expected_model=RAD_DINO_MODEL,
            access_mode="shared",
        )
    except RuntimeError:
        patches, metadata = load_patch_cache(
            cache_path, manifest_hash, expected_model=RAD_DINO_MODEL,
            access_mode="stream",
        )
    progress_path = stage_dir / "progress.pt"
    if progress_path.exists():
        progress = torch.load(
            progress_path, map_location="cpu", weights_only=False
        )
    else:
        progress = {
            "completed": [], "rows": [], "training": [],
            "signature": {
                "manifest_sha256": manifest_hash,
                "cache_model": metadata["model"],
                "retained_grid": args.retained_grid,
                "proposal_grid": args.proposal_grid,
                "shots": list(args.shots),
                "rhos": list(rhos),
                "seeds": list(args.seeds),
                "episodes": args.episodes,
                "train_shot": args.train_shot,
                "max_train_steps": args.max_train_steps,
                "methods": list(methods),
            },
        }
    expected_signature = {
        "manifest_sha256": manifest_hash,
        "cache_model": metadata["model"],
        "retained_grid": args.retained_grid,
        "proposal_grid": args.proposal_grid,
        "shots": list(args.shots),
        "rhos": list(rhos),
        "seeds": list(args.seeds),
        "episodes": args.episodes,
        "train_shot": args.train_shot,
        "max_train_steps": args.max_train_steps,
        "methods": list(methods),
    }
    if progress.get("signature") != expected_signature:
        raise ValueError(
            "existing learned progress uses different protocol arguments; "
            "use another --output-dir"
        )
    completed = {tuple(item) for item in progress["completed"]}
    for seed in args.seeds:
        run_seed = args.seed + seed
        train_bank, validation_bank = training_banks[seed]
        pair_metadata = training_pairs[seed]
        baseline_key = ("binary_protonet_random", None, seed)
        if baseline_key not in completed:
            baseline = RobustBinaryModel(
                int(metadata["shape"][-1]),
                adapter_dim=args.support_adapter_dim,
                alpha_max=args.alpha_max,
                local_temperature=args.local_temperature,
                proposal_grid=args.proposal_grid,
            ).to(device).eval()
            progress["rows"].extend(
                _model_rows(
                    baseline, "uniform", None, patches, metadata,
                    episode_sets, pair_names, args, device, seed,
                    display_method="binary_protonet_random",
                )
            )
            progress["completed"].append(baseline_key)
            completed.add(baseline_key)
            torch.save(progress, progress_path)
            del baseline
        runs = []
        if "rex" in methods:
            runs.append(("rex", None))
        for method in methods:
            if method != "rex":
                runs.extend((method, rho) for rho in rhos)
        for method, rho in runs:
            key = (method, rho, seed)
            if key in completed:
                print(
                    f"reusing {method}, rho={rho}, seed={seed}", flush=True
                )
                continue
            _set_seed(run_seed)
            model = RobustBinaryModel(
                int(metadata["shape"][-1]),
                adapter_dim=args.support_adapter_dim,
                alpha_max=args.alpha_max,
                local_temperature=args.local_temperature,
                proposal_grid=args.proposal_grid,
            ).to(device)
            training = _train_model(
                model, method, patches, metadata, train_bank,
                validation_bank, args, device, run_seed, rho,
            )
            rho_label = "none" if rho is None else f"{rho:g}"
            model_path = stage_dir / (
                f"model_{method}_rho_{rho_label}_seed_{seed:03d}.pt"
            )
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "method": method,
                    "rho": rho,
                    "seed": seed,
                    "metadata": metadata,
                    "retained_grid": args.retained_grid,
                    "proposal_grid": args.proposal_grid,
                    "training": training,
                    "base_pairs": pair_metadata,
                },
                model_path,
            )
            progress["rows"].extend(
                _model_rows(
                    model, method, rho, patches, metadata, episode_sets,
                    pair_names, args, device, seed,
                )
            )
            progress["training"].append(
                {
                    **{k: v for k, v in training.items() if k != "curve"},
                    "seed": seed,
                    "training_seed": run_seed,
                    "base_pairs": pair_metadata,
                    "model": str(model_path),
                    "curve": training["curve"],
                }
            )
            progress["completed"].append(key)
            torch.save(progress, progress_path)
            completed.add(key)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    torch.save(
        {"metadata": metadata, **progress},
        stage_dir / "rows_learned.pt",
    )


def _pareto(rows: list[dict]) -> list[dict]:
    by_point = defaultdict(dict)
    for row in rows:
        if row["metric"] not in {"auroc", "sms_fixed_reference"}:
            continue
        key = (
            row["pair"], row["target"], row["confounder"], row["method"],
            row["rho"], row["shot"], row["seed"],
        )
        by_point[key][row["metric"]] = float(row["value"])
    averaged = defaultdict(lambda: defaultdict(list))
    for key, metrics in by_point.items():
        if len(metrics) != 2:
            continue
        group = key[:-1]
        for metric, value in metrics.items():
            averaged[group][metric].append(value)
    points = []
    for key, metrics in averaged.items():
        points.append(
            {
                **dict(
                    zip(
                        (
                            "pair", "target", "confounder", "method",
                            "rho", "shot",
                        ),
                        key,
                    )
                ),
                "auroc": statistics.mean(metrics["auroc"]),
                "sms_fixed_reference": statistics.mean(
                    metrics["sms_fixed_reference"]
                ),
            }
        )
    baselines = {
        (point["pair"], point["shot"]): point
        for point in points
        if point["method"] == "binary_protonet_random"
    }
    for point in points:
        baseline = baselines.get((point["pair"], point["shot"]))
        if baseline is None:
            point["auroc_change"] = float("nan")
            point["sms_reduction"] = float("nan")
        else:
            point["auroc_change"] = point["auroc"] - baseline["auroc"]
            point["sms_reduction"] = 1.0 - (
                point["sms_fixed_reference"]
                / max(baseline["sms_fixed_reference"], 1e-12)
            )
        competitors = [
            other for other in points
            if other["pair"] == point["pair"]
            and other["shot"] == point["shot"]
        ]
        point["pareto"] = not any(
            other["auroc"] >= point["auroc"]
            and other["sms_fixed_reference"]
            <= point["sms_fixed_reference"]
            and (
                other["auroc"] > point["auroc"]
                or other["sms_fixed_reference"]
                < point["sms_fixed_reference"]
            )
            for other in competitors
        )
    return points


def _adapter_decision(rows: list[dict], primary_shot: int = 3) -> dict:
    values = defaultdict(dict)
    for row in rows:
        if (
            row["method"] not in {"adapter_only", "full_iera"}
            or int(row["shot"]) != primary_shot
            or row["metric"] not in {"auroc", "sms_fixed_reference"}
        ):
            continue
        key = (row["pair"], str(row["rho"]), int(row["seed"]))
        values[(row["method"], *key)][row["metric"]] = float(row["value"])
    comparisons = []
    keys = sorted(
        {
            key[1:] for key in values
            if key[0] == "adapter_only"
        }
        & {
            key[1:] for key in values
            if key[0] == "full_iera"
        }
    )
    grouped = defaultdict(list)
    for pair, rho, seed in keys:
        adapter = values[("adapter_only", pair, rho, seed)]
        full = values[("full_iera", pair, rho, seed)]
        if len(adapter) < 2 or len(full) < 2:
            continue
        grouped[(pair, rho)].append(
            (
                full["auroc"] - adapter["auroc"],
                full["sms_fixed_reference"]
                - adapter["sms_fixed_reference"],
            )
        )
    clear = False
    for (pair, rho), deltas in sorted(grouped.items()):
        auroc = [item[0] for item in deltas]
        sms = [item[1] for item in deltas]
        mean = statistics.mean(auroc)
        std = statistics.stdev(auroc) if len(auroc) > 1 else 0.0
        low = mean - 1.96 * std / math.sqrt(len(auroc))
        row = {
            "pair": pair,
            "rho": rho,
            "n_seeds": len(deltas),
            "full_minus_adapter_auroc": mean,
            "auroc_delta_ci95_low": low,
            "full_minus_adapter_sms": statistics.mean(sms),
        }
        row["clear_full_improvement"] = (
            low > 0 and row["full_minus_adapter_sms"] <= 0
        )
        clear = clear or row["clear_full_improvement"]
        comparisons.append(row)
    return {
        "primary_test": "adapter_only_vs_full_iera",
        "primary_shot": primary_shot,
        "status": (
            "retain_evidence_ratio_mechanism"
            if clear
            else "prefer_simpler_constrained_adapter"
        ),
        "rule": (
            "retain IERA only when the paired 95% CI for AUROC improvement "
            "is above zero at no larger mean fixed-reference SMS"
        ),
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-labels", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/mimic_cxr_protocol_v1.json"),
    )
    parser.add_argument("--rad-cache", type=Path, required=True)
    parser.add_argument("--biomed-cache", type=Path)
    parser.add_argument("--text-direction", type=Path)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/iera/falsification_v1"),
    )
    parser.add_argument(
        "--stage", choices=("cheap", "learned", "sweep"), default="cheap"
    )
    parser.add_argument("--retained-grid", type=int, default=14)
    parser.add_argument("--proposal-grid", type=int, default=4)
    parser.add_argument("--shots", type=int, nargs="+", default=(1, 3, 5, 10))
    parser.add_argument("--rhos", type=float, nargs="+", default=[0.7])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=tuple(range(10))
    )
    parser.add_argument("--queries-per-stratum", type=int, default=1)
    parser.add_argument("--min-stratum-patients", type=int, default=50)
    parser.add_argument("--episode-batch-size", type=int, default=8)
    parser.add_argument("--train-shot", type=int, default=3)
    parser.add_argument("--max-train-pairs", type=int, default=12)
    parser.add_argument("--base-validation-episodes", type=int, default=25)
    parser.add_argument("--max-train-steps", type=int, default=300)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument(
        "--early-stopping-min-delta", type=float, default=1e-4
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--rex-weight", type=float, default=1.0)
    parser.add_argument("--lagrange-initial", type=float, default=1.0)
    parser.add_argument("--lagrange-learning-rate", type=float, default=0.1)
    parser.add_argument("--lagrange-max", type=float, default=100.0)
    parser.add_argument("--support-adapter-dim", type=int, default=16)
    parser.add_argument("--alpha-max", type=float, default=0.25)
    parser.add_argument("--local-temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if (args.biomed_cache is None) != (args.text_direction is None):
        parser.error("biomed-cache and text-direction must be supplied together")
    if args.stage == "sweep" and args.rhos == [0.7]:
        args.rhos = [0.9, 0.8, 0.7, 0.5, 0.3]
    if any(not 0 < rho <= 1 for rho in args.rhos):
        parser.error("every rho must be in (0,1]")
    if (
        args.retained_grid != 14
        or not 0 < args.proposal_grid <= args.retained_grid
        or args.train_shot <= 0
        or args.max_train_steps <= 0
        or args.base_validation_episodes < 1
    ):
        parser.error(
            "this protocol requires retained-grid 14 and positive training sizes"
        )
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    data = load_dataset(args.embeddings, args.manifest)
    restore_raw_target_status(data, args.raw_labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episode_path = args.output_dir / "episodes.pt"
    episode_signature = {
        "manifest_sha256": data.manifest_sha256,
        "shots": list(args.shots),
        "episodes": args.episodes,
        "seeds": list(args.seeds),
        "queries_per_stratum": args.queries_per_stratum,
        "min_stratum_patients": args.min_stratum_patients,
        "seed": args.seed,
        "split_seed": args.split_seed,
    }
    if episode_path.exists():
        saved_episodes = torch.load(
            episode_path, map_location="cpu", weights_only=False
        )
        if saved_episodes.get("signature") != episode_signature:
            raise ValueError(
                "existing episodes.pt has different protocol arguments; "
                "use another --output-dir"
            )
        episode_sets = saved_episodes["episodes"]
        pair_names = saved_episodes["pairs"]
        print(f"reusing identical episodes from {episode_path}", flush=True)
    else:
        episode_sets, pair_names = _prepare_evaluation(data, args)
        torch.save(
            {
                "signature": episode_signature,
                "episodes": episode_sets,
                "pairs": pair_names,
            },
            episode_path,
        )
    stage_dir = args.output_dir / args.stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "cheap":
        workers = [
            (
                args.rad_cache,
                (
                    "binary_protonet_random",
                    "nuisance_balanced",
                    "mean_difference_projection",
                ),
                None,
                RAD_DINO_MODEL,
                stage_dir / "rows_rad_dino.json",
            )
        ]
        if args.biomed_cache is not None:
            workers.append(
                (
                    args.biomed_cache,
                    ("text_direction_orthogonalization",),
                    args.text_direction,
                    MODEL,
                    stage_dir / "rows_biomedclip_text.json",
                )
            )
        context = mp.get_context("spawn")
        rows = []
        cache_metadata = []
        for cache_path, methods, text_path, expected_model, checkpoint in workers:
            worker_signature = {
                **episode_signature,
                "cache": str(cache_path),
                "methods": list(methods),
                "retained_grid": args.retained_grid,
            }
            if not checkpoint.exists():
                process = context.Process(
                    target=_cheap_worker,
                    args=(
                        cache_path, data.manifest_sha256, episode_sets,
                        pair_names, list(args.shots), list(args.seeds),
                        args.retained_grid, args.episode_batch_size,
                        str(device), methods, checkpoint, text_path,
                        expected_model,
                        worker_signature,
                    ),
                )
                process.start()
                process.join()
                if process.exitcode != 0:
                    raise RuntimeError(
                        "baseline worker failed with exit code "
                        f"{process.exitcode}"
                    )
            completed = json.loads(checkpoint.read_text(encoding="utf-8"))
            if completed.get("signature") != worker_signature:
                raise ValueError(
                    f"{checkpoint} uses different protocol arguments; "
                    "use another --output-dir"
                )
            rows.extend(completed["rows"])
            cache_metadata.append(completed["metadata"])
    else:
        config = load_config(args.config)
        args.evaluation_pair_names = list(pair_names.values())
        training_banks, training_pairs = {}, {}
        for seed in args.seeds:
            train, validation, pairs = _prepare_training(
                data, config, args, args.seed + seed
            )
            training_banks[seed] = (train, validation)
            training_pairs[seed] = pairs
        methods = (
            LEARNED_METHODS if args.stage == "learned"
            else CONSTRAINED_METHODS
        )
        context = mp.get_context("spawn")
        process = context.Process(
            target=_learned_worker,
            args=(
                args.rad_cache, data.manifest_sha256, training_banks,
                training_pairs, episode_sets, pair_names, args, methods,
                list(args.rhos), stage_dir, str(device),
            ),
        )
        process.start()
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(
                f"learned worker failed with exit code {process.exitcode}"
            )
        completed = torch.load(
            stage_dir / "rows_learned.pt",
            map_location="cpu", weights_only=False,
        )
        rows = completed["rows"]
        cache_metadata = [completed["metadata"]]
    summary = _summaries(rows)
    _write(stage_dir / "per_seed_metrics.csv", rows)
    _write(stage_dir / "summary_metrics.csv", summary)
    if args.stage == "learned":
        cheap_path = args.output_dir / "cheap" / "per_seed_metrics.csv"
        if cheap_path.exists():
            with cheap_path.open(newline="", encoding="utf-8") as handle:
                cheap_rows = [
                    row for row in csv.DictReader(handle)
                    if row["method"] != "binary_protonet_random"
                ]
            comparison = cheap_rows + rows
            _write(stage_dir / "comparison_per_seed_metrics.csv", comparison)
            _write(
                stage_dir / "comparison_summary_metrics.csv",
                _summaries(comparison),
            )
    if args.stage in {"learned", "sweep"}:
        pareto = _pareto(rows)
        if pareto:
            _write(stage_dir / "pareto.csv", pareto)
        decision = _adapter_decision(rows)
        training = completed["training"]
        decision["base_validation_all_feasible"] = all(
            bool(item["best_validation"]["sms_budget_satisfied"])
            for item in training
            if item["method"] in CONSTRAINED_METHODS
        )
        (stage_dir / "decision.json").write_text(
            json.dumps(decision, indent=2) + "\n", encoding="utf-8"
        )
    (stage_dir / "experiment.json").write_text(
        json.dumps(
            {
                "stage": args.stage,
                "methods": sorted({row["method"] for row in rows}),
                "pairs": pair_names,
                "shots": args.shots,
                "seeds": args.seeds,
                "episodes": args.episodes,
                "retained_grid": args.retained_grid,
                "proposal_grid": args.proposal_grid,
                "rhos": args.rhos if args.stage != "cheap" else [],
                "caches": cache_metadata,
                "episode_policy": (
                    "identical candidate panels and queries; random supports "
                    "follow empirical nuisance prevalence, oracle supports are balanced"
                ),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"{args.stage} results written to {stage_dir}")


if __name__ == "__main__":
    main()
