"""Adapt Rad-DINO's final blocks while retaining the global rho=.3 anchor."""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.residuals.metrics import _auc, select_temperature, select_threshold

from .falsification import _selected, _write
from .patch_cache import RAD_DINO_MODEL, StreamingPatchCache
from .representation_cache import _layer_output
from .robust_metrics import evaluate, normalized_sms
from .robust_model import RobustBinaryModel


VARIANTS = ("last1", "last2", "lora2")


class RadDinoTail(nn.Module):
    def __init__(self, layers, layernorm) -> None:
        super().__init__()
        self.layers = nn.ModuleList(copy.deepcopy(list(layers)))
        self.layernorm = copy.deepcopy(layernorm)
        self.variant = "frozen"

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden.float()
        for layer in self.layers:
            hidden = _layer_output(layer, hidden)
        return F.normalize(self.layernorm(hidden)[:, 1:].float(), dim=-1)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = copy.deepcopy(base).requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = float(alpha) / rank

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.base(values) + self.scale * self.up(self.down(values))


def _inject_lora(module: nn.Module, rank: int, alpha: float) -> int:
    replacements = []
    for name, child in module.named_modules():
        if (
            isinstance(child, nn.Linear)
            and name.rsplit(".", 1)[-1] in {"query", "key", "value"}
        ):
            replacements.append((name, child))
    for name, child in replacements:
        parent_name, _, attribute = name.rpartition(".")
        parent = module.get_submodule(parent_name) if parent_name else module
        setattr(parent, attribute, LoRALinear(child, rank, alpha))
    if not replacements:
        raise ValueError("no query/key/value projections found for LoRA")
    return len(replacements)


def configure_tail(
    source_layers,
    source_layernorm,
    variant: str,
    lora_rank: int,
    lora_alpha: float,
) -> RadDinoTail:
    if variant not in VARIANTS:
        raise ValueError(f"unknown representation variant {variant!r}")
    tail = RadDinoTail(source_layers, source_layernorm)
    tail.variant = variant
    tail.requires_grad_(False)
    if variant == "last1":
        tail.layers[-1].requires_grad_(True)
        tail.layernorm.requires_grad_(True)
    elif variant == "last2":
        tail.requires_grad_(True)
    else:
        _inject_lora(tail, lora_rank, lora_alpha)
    return tail


def _load_cache(root: Path) -> tuple[StreamingPatchCache, dict]:
    import numpy as np

    metadata = json.loads(
        (root / "representation_cache.json").read_text(encoding="utf-8")
    )
    if not metadata.get("complete"):
        raise ValueError("representation cache is incomplete")
    shape = tuple(metadata["shape"])
    token_path = root / metadata["tokens"]
    index_path = root / metadata["global_indices"]
    indices = np.fromfile(index_path, dtype=np.int64)
    if len(indices) != shape[0]:
        raise ValueError("representation cache indices have wrong length")
    return StreamingPatchCache(
        token_path, shape, global_indices=indices
    ), metadata


def _tail_images(
    tail: RadDinoTail,
    hidden: torch.Tensor,
    device: torch.device,
    image_batch_size: int,
    grad: bool,
) -> torch.Tensor:
    leading = hidden.shape[:-2]
    flattened = hidden.flatten(0, -3)
    outputs = []
    context = torch.enable_grad if grad else torch.no_grad
    with context():
        for start in range(0, len(flattened), image_batch_size):
            current = flattened[start : start + image_batch_size].to(
                device, non_blocking=True
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                outputs.append(tail(current))
    result = torch.cat(outputs)
    return result.reshape(*leading, result.shape[-2], result.shape[-1])


def _anchor_logits(
    anchor: RobustBinaryModel,
    positive: torch.Tensor,
    negative: torch.Tensor,
    query: torch.Tensor,
    positive_mask: torch.Tensor | None = None,
    negative_mask: torch.Tensor | None = None,
    method: str = "adapter_only",
) -> torch.Tensor:
    return anchor(
        positive,
        negative,
        query,
        method,
        positive_mask,
        negative_mask,
    )


def _features(
    cache,
    tail: RadDinoTail,
    reference: RadDinoTail,
    episodes: dict,
    start: int,
    end: int,
    shot: int,
    device: torch.device,
    image_batch_size: int,
    grad: bool,
) -> tuple[torch.Tensor, ...]:
    positive_hidden = cache[
        episodes["positive"][start:end, :, : 2 * shot]
    ]
    negative_hidden = cache[
        episodes["negative"][start:end, :, : 2 * shot]
    ]
    query_hidden = cache[episodes["query"][start:end]]
    positive = _tail_images(
        tail, positive_hidden, device, image_batch_size, grad
    )
    negative = _tail_images(
        tail, negative_hidden, device, image_batch_size, grad
    )
    query = _tail_images(tail, query_hidden, device, image_batch_size, grad)
    with torch.no_grad():
        reference_positive = _tail_images(
            reference, positive_hidden, device, image_batch_size, False
        )
        reference_negative = _tail_images(
            reference, negative_hidden, device, image_batch_size, False
        )
        reference_query = _tail_images(
            reference, query_hidden, device, image_batch_size, False
        )
    return (
        positive,
        negative,
        query,
        reference_positive,
        reference_negative,
        reference_query,
    )


def _episode_forward(
    cache,
    tail: RadDinoTail,
    reference: RadDinoTail,
    anchor: RobustBinaryModel,
    episodes: dict,
    start: int,
    end: int,
    shot: int,
    device: torch.device,
    image_batch_size: int,
    grad: bool,
) -> tuple[torch.Tensor, ...]:
    (
        positive_panels,
        negative_panels,
        query,
        reference_positive,
        reference_negative,
        reference_query,
    ) = _features(
        cache,
        tail,
        reference,
        episodes,
        start,
        end,
        shot,
        device,
        image_batch_size,
        grad,
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
    logits = _anchor_logits(
        anchor,
        positive,
        negative,
        query,
        positive_mask,
        negative_mask,
    )
    panels = []
    reference_panels = []
    for environment in (0, 1):
        panels.append(
            _anchor_logits(
                anchor,
                positive_panels[:, environment : environment + 1, :shot],
                negative_panels[:, environment : environment + 1, :shot],
                query,
            )
        )
        with torch.no_grad():
            reference_panels.append(
                _anchor_logits(
                    anchor,
                    reference_positive[
                        :, environment : environment + 1, :shot
                    ],
                    reference_negative[
                        :, environment : environment + 1, :shot
                    ],
                    reference_query,
                    method="uniform",
                )
            )
    return logits, panels[0], panels[1], reference_panels[0], reference_panels[1]


def _base_validation(
    cache,
    tail,
    reference,
    anchor,
    bank: list[dict],
    args,
    device,
) -> dict:
    pair_aurocs, pair_sms = [], []
    tail.eval()
    for episodes in bank:
        logits, targets = [], []
        panel_zero, panel_one = [], []
        reference_zero, reference_one = [], []
        with torch.inference_mode():
            for start in range(len(episodes["positive"])):
                values = _episode_forward(
                    cache,
                    tail,
                    reference,
                    anchor,
                    episodes,
                    start,
                    start + 1,
                    args.train_shot,
                    device,
                    args.image_batch_size,
                    False,
                )
                logits.append(values[0].flatten().cpu())
                targets.append(episodes["targets"][start].flatten())
                panel_zero.append(values[1].flatten().cpu())
                panel_one.append(values[2].flatten().cpu())
                reference_zero.append(values[3].flatten().cpu())
                reference_one.append(values[4].flatten().cpu())
        pair_aurocs.append(_auc(torch.cat(targets).bool(), torch.cat(logits)))
        pair_sms.append(
            float(
                normalized_sms(
                    torch.cat(panel_zero),
                    torch.cat(panel_one),
                    torch.cat(reference_zero),
                    torch.cat(reference_one),
                )
            )
        )
    return {
        "mean_auroc": statistics.mean(pair_aurocs),
        "worst_pair_auroc": min(pair_aurocs),
        "max_sms": max(pair_sms),
        "sms_feasible": max(pair_sms) <= args.sms_budget,
    }


def _checkpoint_key(validation: dict) -> tuple[float, float]:
    if validation["sms_feasible"]:
        return 1.0, validation["mean_auroc"]
    return 0.0, -validation["max_sms"]


def _train(
    cache,
    tail,
    reference,
    anchor,
    train_bank,
    validation_bank,
    args,
    device,
    seed,
    learning_rate,
) -> dict:
    parameters = [parameter for parameter in tail.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=args.weight_decay
    )
    initial = _base_validation(
        cache, tail, reference, anchor, validation_bank, args, device
    )
    best_state = copy.deepcopy(tail.state_dict())
    best_validation = initial
    best_key = _checkpoint_key(initial)
    best_step = 0
    lagrange = args.lagrange_initial
    curve = [{"step": 0, "lagrange": lagrange, **initial}]
    choices = [
        (pair_index, episode_index)
        for pair_index, episodes in enumerate(train_bank)
        for episode_index in range(len(episodes["positive"]))
    ]
    generator = random.Random(args.seed + seed)
    tail.train()
    if tail.variant == "last1":
        tail.layers[0].eval()
    for step in range(1, args.max_train_steps + 1):
        pair_index, episode_index = generator.choice(choices)
        episodes = train_bank[pair_index]
        values = _episode_forward(
            cache,
            tail,
            reference,
            anchor,
            episodes,
            episode_index,
            episode_index + 1,
            args.train_shot,
            device,
            args.image_batch_size,
            True,
        )
        targets = episodes["targets"][episode_index : episode_index + 1].to(
            device
        )
        classification = F.binary_cross_entropy_with_logits(values[0], targets)
        sms = normalized_sms(values[1], values[2], values[3], values[4])
        violation = sms - args.sms_budget
        loss = classification + lagrange * F.relu(violation)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.gradient_clip)
        optimizer.step()
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
                cache,
                tail,
                reference,
                anchor,
                validation_bank,
                args,
                device,
            )
            curve.append(
                {
                    "step": step,
                    "classification": float(classification.detach()),
                    "train_sms": float(sms.detach()),
                    "lagrange": lagrange,
                    **validation,
                }
            )
            key = _checkpoint_key(validation)
            if key > best_key:
                best_key = key
                best_state = copy.deepcopy(tail.state_dict())
                best_validation = validation
                best_step = step
            print(
                f"representation seed={seed}, step={step}, "
                f"val_auc={validation['mean_auroc']:.4f}, "
                f"sms={validation['max_sms']:.3f}",
                flush=True,
            )
            tail.train()
            if tail.variant == "last1":
                tail.layers[0].eval()
    tail.load_state_dict(best_state)
    tail.eval()
    return {
        "learning_rate": learning_rate,
        "best_step": best_step,
        "best_validation": best_validation,
        "final_lagrange": lagrange,
        "curve": curve,
    }


def _score(
    cache,
    tail,
    reference,
    anchor,
    episodes,
    shot,
    args,
    device,
) -> tuple[torch.Tensor, ...]:
    destinations = ([], [], [], [], [])
    tail.eval()
    with torch.inference_mode():
        for start in range(len(episodes["positive"])):
            values = _episode_forward(
                cache,
                tail,
                reference,
                anchor,
                episodes,
                start,
                start + 1,
                shot,
                device,
                args.image_batch_size,
                False,
            )
            for destination, tensor in zip(destinations, values):
                destination.append(tensor.cpu())
    return tuple(torch.cat(items).flatten() for items in destinations)


def _rows(scores, episodes, names, seeds, variants) -> list[dict]:
    rows = []
    for variant in variants:
        for seed in seeds:
            validation = scores[(variant, seed, "validate")]
            validation_targets = episodes[(seed, "validate")]["targets"].flatten()
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
                current = episodes[(seed, partition)]
                metrics = evaluate(
                    *scores[(variant, seed, partition)],
                    current["targets"].flatten(),
                    current["nuisance"].flatten(),
                    temperature,
                    threshold,
                )
                for metric, value in metrics.items():
                    rows.append(
                        {
                            "partition": partition,
                            "pair": f"{names[0]}__{names[1]}",
                            "method": variant,
                            "seed": seed,
                            "shot": 3,
                            "metric": metric,
                            "value": value,
                        }
                    )
    return rows


def _selection(rows: list[dict], sms_budget: float) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        if row["partition"] == "validate" and row["metric"] in {
            "auroc", "sms_fixed_reference"
        }:
            grouped[(row["method"], row["metric"])].append(float(row["value"]))
    points = []
    for method in sorted({key[0] for key in grouped}):
        points.append(
            {
                "method": method,
                "validation_auroc": statistics.mean(
                    grouped[(method, "auroc")]
                ),
                "validation_sms": statistics.mean(
                    grouped[(method, "sms_fixed_reference")]
                ),
            }
        )
    adapted = [point for point in points if point["method"] != "frozen_anchor"]
    feasible = [point for point in adapted if point["validation_sms"] <= sms_budget]
    pool = feasible or adapted
    chosen = max(pool, key=lambda point: point["validation_auroc"])
    return {
        "selection_partition": "validate",
        "sms_budget": sms_budget,
        "feasible_candidate_exists": bool(feasible),
        "selected": chosen,
        "candidates": points,
    }


def _summaries(rows: list[dict]) -> list[dict]:
    keys = ("partition", "pair", "method", "shot", "metric")
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


def _decision(rows: list[dict], selection: dict) -> dict:
    selected = selection["selected"]["method"]
    grouped = defaultdict(list)
    for row in rows:
        if row["partition"] == "test" and row["method"] in {
            "frozen_anchor", selected
        } and row["metric"] in {"auroc", "sms_fixed_reference"}:
            grouped[(row["method"], row["metric"])].append(float(row["value"]))
    adapted_auroc = statistics.mean(grouped[(selected, "auroc")])
    adapted_sms = statistics.mean(grouped[(selected, "sms_fixed_reference")])
    frozen_auroc = statistics.mean(grouped[("frozen_anchor", "auroc")])
    success = adapted_auroc >= 0.56 and adapted_sms <= 0.30
    return {
        "status": (
            "representation_adaptation_promising"
            if success
            else "dataset_protocol_or_backbone_likely_limiting"
        ),
        "selected_variant": selected,
        "frozen_anchor_auroc": frozen_auroc,
        "adapted_auroc": adapted_auroc,
        "adapted_sms": adapted_sms,
        "auroc_change": adapted_auroc - frozen_auroc,
        "target": "AUROC >= 0.56 with SMS <= 0.30",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=VARIANTS)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4))
    parser.add_argument("--train-shot", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=150)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--sms-budget", type=float, default=0.30)
    parser.add_argument("--lagrange-initial", type=float, default=1.0)
    parser.add_argument("--lagrange-learning-rate", type=float, default=0.1)
    parser.add_argument("--lagrange-max", type=float, default=100.0)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if any(variant not in VARIANTS for variant in args.variants):
        parser.error(f"variants must be chosen from {VARIANTS}")
    started = time.perf_counter()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    cache, metadata = _load_cache(args.cache)
    bank = torch.load(
        args.cache / metadata["episode_bank"],
        map_location="cpu",
        weights_only=False,
    )
    if not set(args.seeds).issubset(set(bank["signature"]["seeds"])):
        raise ValueError("requested seeds are absent from representation cache")
    from transformers import AutoModel

    source = AutoModel.from_pretrained(RAD_DINO_MODEL)
    source_layers = source.encoder.layer[-2:]
    source_layernorm = source.layernorm
    reference_template = RadDinoTail(
        source_layers, source_layernorm
    ).eval().requires_grad_(False)
    del source
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pair_id, names = next(iter(bank["locked"]["pairs"].items()))
    locked = {}
    for seed in args.seeds:
        for partition in ("validate", "test"):
            episodes = bank["locked"]["episodes"][(pair_id, seed, partition)]
            limit = bank["signature"]["locked_episode_count"]
            locked[(seed, partition)] = {
                name: (
                    value[:limit]
                    if isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] >= limit
                    else value
                )
                for name, value in episodes.items()
            }
    scores = {}
    for seed in args.seeds:
        anchor_path = args.adapter_dir / (
            f"model_adapter_only_rho_0.3_seed_{seed:03d}.pt"
        )
        saved_anchor = torch.load(
            anchor_path, map_location="cpu", weights_only=False
        )
        anchor = RobustBinaryModel(
            int(metadata["shape"][-1]),
            adapter_dim=int(
                saved_anchor["state_dict"]["support_down.weight"].shape[0]
            ),
        ).to(device)
        anchor.load_state_dict(saved_anchor["state_dict"])
        anchor.eval().requires_grad_(False)
        reference = copy.deepcopy(reference_template).to(device)
        for partition in ("validate", "test"):
            scores[("frozen_anchor", seed, partition)] = _score(
                cache,
                reference,
                reference,
                anchor,
                locked[(seed, partition)],
                3,
                args,
                device,
            )
        for variant in args.variants:
            model_path = args.output_dir / f"model_{variant}_seed_{seed:03d}.pt"
            if model_path.exists():
                completed = torch.load(
                    model_path, map_location="cpu", weights_only=False
                )
                tail = configure_tail(
                    source_layers,
                    source_layernorm,
                    variant,
                    args.lora_rank,
                    args.lora_alpha,
                ).to(device)
                tail.load_state_dict(completed["state_dict"])
                training = completed["training"]
                print(f"reusing {variant}, seed={seed}", flush=True)
            else:
                tail = configure_tail(
                    source_layers,
                    source_layernorm,
                    variant,
                    args.lora_rank,
                    args.lora_alpha,
                ).to(device)
                training = _train(
                    cache,
                    tail,
                    reference,
                    anchor,
                    bank["base"][seed]["train"],
                    bank["base"][seed]["validate"],
                    args,
                    device,
                    seed,
                    (
                        args.lora_learning_rate
                        if variant == "lora2"
                        else args.learning_rate
                    ),
                )
                torch.save(
                    {
                        "state_dict": tail.state_dict(),
                        "variant": variant,
                        "seed": seed,
                        "training": training,
                    },
                    model_path,
                )
            for partition in ("validate", "test"):
                scores[(variant, seed, partition)] = _score(
                    cache,
                    tail,
                    reference,
                    anchor,
                    locked[(seed, partition)],
                    3,
                    args,
                    device,
                )
            del tail
            torch.cuda.empty_cache() if device.type == "cuda" else None
        del reference, anchor
    variants = ("frozen_anchor", *args.variants)
    rows = _rows(scores, locked, names, list(args.seeds), variants)
    summary = _summaries(rows)
    selection = _selection(rows, args.sms_budget)
    decision = _decision(rows, selection)
    _write(args.output_dir / "per_seed_metrics.csv", rows)
    _write(args.output_dir / "summary_metrics.csv", summary)
    _write(args.output_dir / "candidate_metrics.csv", selection["candidates"])
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "stage": "representation_adaptation_pilot",
                "backbone": RAD_DINO_MODEL,
                "frozen_prefix_blocks": metadata["prefix_blocks"],
                "variants": args.variants,
                "seeds": args.seeds,
                "classification": "base-class episodic BCE",
                "stability": "adaptive fixed-reference normalized SMS constraint",
                "sms_budget": args.sms_budget,
                "local_matchers_or_heads": False,
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"representation results written to {args.output_dir}; "
        f"decision={decision['status']}"
    )


if __name__ == "__main__":
    main()
