from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .prepare_data import CHEXPERT_LABELS


METHODS = (
    "protonet",
    "random_subspace",
    "global_subspace",
    "oracle_subspace",
)


def choose_device(torch, requested: str):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_class_split(
    class_names: list[str], split_seed: int, split_json: Path | None
) -> dict[str, list[str]]:
    if split_json:
        with split_json.open(encoding="utf-8") as handle:
            split = json.load(handle)
    else:
        labels = [name for name in CHEXPERT_LABELS if name in class_names]
        if len(labels) != 14:
            missing = sorted(set(CHEXPERT_LABELS).difference(labels))
            raise ValueError(
                "The 8/3/3 protocol needs all 14 MIMIC-CXR labels. "
                f"Missing from embeddings: {missing}"
            )
        random.Random(split_seed).shuffle(labels)
        split = {"base": labels[:8], "validation": labels[8:11], "test": labels[11:14]}
    expected = {"base": 8, "validation": 3, "test": 3}
    for key, count in expected.items():
        if key not in split or len(split[key]) != count:
            raise ValueError(f"Split field {key!r} must contain {count} classes")
    flattened = split["base"] + split["validation"] + split["test"]
    if len(set(flattened)) != 14:
        raise ValueError("Class split must contain 14 unique labels")
    unknown = sorted(set(flattened).difference(class_names))
    if unknown:
        raise ValueError(f"Split contains labels absent from embeddings: {unknown}")
    return split


def take_features(features, indices, device):
    source_indices = indices.to(features.device)
    selected = features.index_select(0, source_indices.reshape(-1))
    return selected.reshape(*indices.shape, features.shape[-1]).to(
        device=device, dtype=__import__("torch").float32, non_blocking=True
    )


def make_episode_plan(
    torch,
    indices_by_class: dict[int, object],
    class_ids: list[int],
    oracle_size: int,
    episodes: int,
    shots: int,
    queries: int,
    seed: int,
):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    oracle_rows = []
    support_rows = []
    query_rows = []
    for class_id in class_ids:
        available = indices_by_class[class_id]
        required = oracle_size + shots + queries
        if len(available) < required:
            raise ValueError(
                f"Class id {class_id} has {len(available)} images; need at least {required} "
                f"for oracle={oracle_size}, {shots}-shot, query={queries}."
            )
        shuffled = available[torch.randperm(len(available), generator=generator)]
        oracle = shuffled[:oracle_size]
        episode_pool = shuffled[oracle_size:]
        class_support = []
        class_query = []
        for _ in range(episodes):
            picked = episode_pool[
                torch.randperm(len(episode_pool), generator=generator)[: shots + queries]
            ]
            class_support.append(picked[:shots])
            class_query.append(picked[shots:])
        oracle_rows.append(oracle)
        support_rows.append(torch.stack(class_support))
        query_rows.append(torch.stack(class_query))

    oracle_indices = torch.stack(oracle_rows)  # [N, O]
    support_indices = torch.stack(support_rows, dim=1)  # [E, N, K]
    query_indices = torch.stack(query_rows, dim=1)  # [E, N, Q]
    if any(
        torch.isin(query_indices, oracle_indices[class_index]).any().item()
        for class_index in range(len(class_ids))
    ):
        raise AssertionError("Oracle/query leakage detected")
    return oracle_indices, support_indices, query_indices


def fit_oracle_bases(torch, features, oracle_indices, max_rank: int, device):
    bases = []
    for class_indices in oracle_indices:
        samples = take_features(features, class_indices, device)
        samples = samples - samples.mean(dim=0, keepdim=True)
        _, _, vh = torch.linalg.svd(samples, full_matrices=False)
        bases.append(vh[:max_rank].T.contiguous())
    return torch.stack(bases)  # [N, D, R]


def fit_global_basis(
    torch,
    features,
    indices_by_class: dict[int, object],
    base_ids: list[int],
    max_rank: int,
    device,
    samples_per_class: int,
    chunk_size: int,
    seed: int,
):
    dimension = features.shape[-1]
    covariance = torch.zeros((dimension, dimension), dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    used_counts: dict[int, int] = {}
    for class_id in base_ids:
        indices = indices_by_class[class_id]
        if samples_per_class > 0 and len(indices) > samples_per_class:
            indices = indices[
                torch.randperm(len(indices), generator=generator)[:samples_per_class]
            ]
        used_counts[class_id] = len(indices)
        total = torch.zeros(dimension, dtype=torch.float32, device=device)
        for start in range(0, len(indices), chunk_size):
            batch = take_features(features, indices[start : start + chunk_size], device)
            total += batch.sum(dim=0)
        mean = total / len(indices)
        class_covariance = torch.zeros_like(covariance)
        for start in range(0, len(indices), chunk_size):
            batch = take_features(features, indices[start : start + chunk_size], device)
            centered = batch - mean
            class_covariance.addmm_(centered.T, centered)
        covariance += class_covariance / max(len(indices) - 1, 1)
    covariance /= len(base_ids)
    _, eigenvectors = torch.linalg.eigh(covariance)
    return eigenvectors[:, -max_rank:].flip(1).contiguous(), used_counts


def random_bases(torch, classes: int, dimension: int, rank: int, seed: int, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrices = torch.randn(classes, dimension, rank, generator=generator)
    bases = [torch.linalg.qr(matrix, mode="reduced").Q for matrix in matrices]
    return torch.stack(bases).to(device)


def distances(torch, query, prototypes, basis=None, beta: float = 1.0):
    delta = query[:, :, None, :] - prototypes[:, None, :, :]
    squared = delta.square().sum(dim=-1)
    if basis is None:
        return squared
    if basis.ndim == 2:
        projected = torch.einsum("eqnd,dr->eqnr", delta, basis)
    else:
        projected = torch.einsum("eqnd,ndr->eqnr", delta, basis)
    return (squared - beta * projected.square().sum(dim=-1)).clamp_min_(0)


def metrics(torch, distance, targets, classes: int) -> dict[str, float]:
    predictions = distance.argmin(dim=-1).reshape(-1)
    scores = -distance.reshape(-1, classes)
    targets = targets.reshape(-1)
    accuracy = (predictions == targets).float().mean()
    f1_values = []
    auc_values = []
    for class_index in range(classes):
        positive = targets == class_index
        predicted_positive = predictions == class_index
        tp = (positive & predicted_positive).sum().float()
        fp = (~positive & predicted_positive).sum().float()
        fn = (positive & ~predicted_positive).sum().float()
        f1_values.append((2 * tp / (2 * tp + fp + fn).clamp_min(1)))

        positive_scores = scores[positive, class_index]
        negative_scores = scores[~positive, class_index]
        comparisons = (positive_scores[:, None] > negative_scores[None, :]).float()
        comparisons += 0.5 * (
            positive_scores[:, None] == negative_scores[None, :]
        ).float()
        auc_values.append(comparisons.mean())
    return {
        "accuracy": float(accuracy.item()),
        "macro_f1": float(torch.stack(f1_values).mean().item()),
        "macro_auroc": float(torch.stack(auc_values).mean().item()),
    }


def evaluate_stage(
    torch,
    features,
    indices_by_class,
    class_ids,
    global_basis,
    ranks,
    args,
    seed,
    stage,
    device,
):
    stage_offset = 0 if stage == "validation" else 1_000_000
    max_shots = max(args.shots)
    oracle_indices, support_indices, query_indices = make_episode_plan(
        torch,
        indices_by_class,
        class_ids,
        args.oracle_size,
        args.episodes,
        max_shots,
        args.queries,
        seed + stage_offset,
    )
    support = take_features(features, support_indices, device)
    query = take_features(features, query_indices, device)
    query = query.reshape(args.episodes, len(class_ids) * args.queries, -1)
    targets = (
        torch.arange(len(class_ids), device=device)
        .view(1, len(class_ids), 1)
        .expand(args.episodes, len(class_ids), args.queries)
    )

    max_rank = max(ranks)
    oracle_basis = fit_oracle_bases(
        torch, features, oracle_indices, max_rank, device
    )
    random_basis = random_bases(
        torch,
        len(class_ids),
        features.shape[-1],
        max_rank,
        seed + stage_offset + 17,
        device,
    )
    rows = []
    for shots in args.shots:
        # Nested supports: 1-shot is a subset of 3-shot, which is a subset of
        # 5-shot. All shot settings use the exact same episode queries.
        prototypes = support[:, :, :shots].mean(dim=2)
        proto_metrics = metrics(
            torch, distances(torch, query, prototypes), targets, len(class_ids)
        )
        rows.append(
            {
                "stage": stage,
                "shots": shots,
                "seed": seed,
                "method": "protonet",
                "rank": 0,
                "beta": 0.0,
                **proto_metrics,
            }
        )
        for rank in ranks:
            for method, basis in (
                ("random_subspace", random_basis[:, :, :rank]),
                ("global_subspace", global_basis[:, :rank]),
                ("oracle_subspace", oracle_basis[:, :, :rank]),
            ):
                for beta in args.betas:
                    result = metrics(
                        torch,
                        distances(torch, query, prototypes, basis, beta),
                        targets,
                        len(class_ids),
                    )
                    rows.append(
                        {
                            "stage": stage,
                            "shots": shots,
                            "seed": seed,
                            "method": method,
                            "rank": rank,
                            "beta": beta,
                            **result,
                        }
                    )
    return rows


def select_hyperparameters(
    rows: list[dict[str, object]],
    ranks: list[int],
    betas: list[float],
    shot_values: list[int],
):
    selected = {}
    for shots in shot_values:
        selected[shots] = {"protonet": {"rank": 0, "beta": 0.0}}
        for method in METHODS[1:]:
            means = {}
            for rank in ranks:
                for beta in betas:
                    values = [
                        float(row["macro_auroc"])
                        for row in rows
                        if row["stage"] == "validation"
                        and row["shots"] == shots
                        and row["method"] == method
                        and row["rank"] == rank
                        and row["beta"] == beta
                    ]
                    means[(rank, beta)] = statistics.mean(values)
            rank, beta = max(
                means, key=lambda setting: (means[setting], -setting[0], -setting[1])
            )
            selected[shots][method] = {"rank": rank, "beta": beta}
    return selected


def summarize(rows: list[dict[str, object]], selected, shot_values: list[int]):
    summary = []
    for shots in shot_values:
        for method in METHODS:
            setting = selected[shots][method]
            chosen = [
                row
                for row in rows
                if row["stage"] == "test"
                and row["shots"] == shots
                and row["method"] == method
                and row["rank"] == setting["rank"]
                and row["beta"] == setting["beta"]
            ]
            result: dict[str, object] = {
                "shots": shots,
                "method": method,
                "rank": setting["rank"],
                "beta": setting["beta"],
            }
            for metric in ("accuracy", "macro_f1", "macro_auroc"):
                values = [float(row[metric]) for row in chosen]
                result[f"{metric}_mean"] = statistics.mean(values)
                result[f"{metric}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is missing. Install with: pip install -e '.[gpu]'") from error

    args.shots = sorted(set(args.shots if isinstance(args.shots, list) else [args.shots]))
    args.betas = sorted(set(float(beta) for beta in args.betas))
    if any(beta <= 0 or beta > 1 for beta in args.betas):
        raise ValueError("Every beta must be in the interval (0, 1]")
    device = choose_device(torch, args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    try:
        payload = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 2.0 compatibility
        payload = torch.load(args.embeddings, map_location="cpu")
    features = payload["features"]
    labels = payload["labels"].long()
    class_names = list(payload["class_names"])
    if not payload.get("normalized", False):
        features = torch.nn.functional.normalize(features.float(), dim=-1).half()
    if not args.keep_features_cpu:
        features = features.to(device, non_blocking=True)

    split = build_class_split(class_names, args.split_seed, args.split_json)
    name_to_id = {name: index for index, name in enumerate(class_names)}
    indices_by_class = {
        class_id: torch.nonzero(labels == class_id, as_tuple=False).flatten()
        for class_id in range(len(class_names))
    }
    base_ids = [name_to_id[name] for name in split["base"]]
    required_novel = args.oracle_size + max(args.shots) + args.queries
    insufficient = {
        name: len(indices_by_class[name_to_id[name]])
        for stage in ("validation", "test")
        for name in split[stage]
        if len(indices_by_class[name_to_id[name]]) < required_novel
    }
    if insufficient:
        raise ValueError(
            f"Novel classes need at least {required_novel} images each "
            f"(oracle + support + query). Insufficient counts: {insufficient}. "
            "Use the full dataset, change the class split, or lower --oracle-size for a pilot."
        )
    ranks = sorted(set(args.ranks))
    max_rank = max(ranks)
    if max_rank >= args.oracle_size:
        raise ValueError("Every rank must be smaller than --oracle-size")

    print(f"Computing class-balanced global base subspace on {device} ...", flush=True)
    global_basis, global_counts = fit_global_basis(
        torch,
        features,
        indices_by_class,
        base_ids,
        max_rank,
        device,
        args.base_samples_per_class,
        args.chunk_size,
        args.split_seed,
    )

    all_rows: list[dict[str, object]] = []
    with torch.inference_mode():
        for seed in args.seeds:
            for stage in ("validation", "test"):
                class_ids = [name_to_id[name] for name in split[stage]]
                print(
                    f"{stage}: seed {seed}, shots={args.shots}, "
                    f"{args.episodes} episodes",
                    flush=True,
                )
                all_rows.extend(
                    evaluate_stage(
                        torch,
                        features,
                        indices_by_class,
                        class_ids,
                        global_basis,
                        ranks,
                        args,
                        seed,
                        stage,
                        device,
                    )
                )

    selected = select_hyperparameters(all_rows, ranks, args.betas, args.shots)
    summary = summarize(all_rows, selected, args.shots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed_all_settings.csv", all_rows)
    write_csv(args.output_dir / "test_selected_summary.csv", summary)
    with (args.output_dir / "experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": split,
                "selected_rank_and_beta_by_validation_macro_auroc": selected,
                "seeds": args.seeds,
                "episodes": args.episodes,
                "shots": args.shots,
                "queries_per_class": args.queries,
                "oracle_size_per_class": args.oracle_size,
                "base_samples_used": {
                    class_names[class_id]: count for class_id, count in global_counts.items()
                },
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print("\nValidation-selected test results (mean +/- sd over seeds)")
    for row in summary:
        print(
            f"{int(row['shots'])}-shot  {row['method']:18s} "
            f"r={int(row['rank']):>2d} beta={float(row['beta']):.2f}  "
            f"AUROC {row['macro_auroc_mean']:.4f} +/- {row['macro_auroc_std']:.4f}  "
            f"F1 {row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}  "
            f"Acc {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}"
        )
    print(f"\nSaved results to {args.output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GPU-vectorized ProtoNet versus affine-subspace few-shot evaluation."
    )
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings/biomedclip.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/first_experiment"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-features-cpu", action="store_true")
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--oracle-size", type=int, default=512)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--betas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 0.75]
    )
    parser.add_argument(
        "--base-samples-per-class",
        type=int,
        default=4096,
        help="Equal cap for fast global PCA; 0 uses every base image.",
    )
    parser.add_argument("--chunk-size", type=int, default=8192)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
