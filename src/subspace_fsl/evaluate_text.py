from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable

from .evaluate import (
    build_class_split,
    choose_device,
    fit_oracle_bases,
    make_episode_plan,
    metrics,
    take_features,
    write_csv,
)


METHODS = (
    "protonet",
    "protonet_text",
    "protonet_shuffled_text",
    "oracle_subspace",
    "oracle_subspace_text",
    "oracle_subspace_shuffled_text",
)


def load_torch_file(torch, path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch 2.0 compatibility
        return torch.load(path, map_location="cpu")


def refined_centers(torch, support_mean, text_features, alpha: float):
    blended = (1.0 - alpha) * support_mean + alpha * text_features.unsqueeze(0)
    return torch.nn.functional.normalize(blended, dim=-1)


def distance_components(torch, query, centers, basis):
    delta = query[:, :, None, :] - centers[:, None, :, :]
    squared = delta.square().sum(dim=-1)
    projected = torch.einsum("eqnd,ndr->eqnr", delta, basis)
    cumulative_projection = projected.square().cumsum(dim=-1)
    return squared, cumulative_projection


def hybrid_distance(squared, cumulative_projection, rank: int, beta: float):
    return (squared - beta * cumulative_projection[..., rank - 1]).clamp_min_(0)


def result_row(stage, shots, seed, method, rank, beta, alpha, values):
    return {
        "stage": stage,
        "shots": shots,
        "seed": seed,
        "method": method,
        "rank": rank,
        "beta": beta,
        "alpha": alpha,
        **values,
    }


def evaluate_stage(
    torch,
    visual_features,
    text_features,
    indices_by_class,
    class_ids,
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
    support = take_features(visual_features, support_indices, device)
    query = take_features(visual_features, query_indices, device)
    query = query.reshape(args.episodes, len(class_ids) * args.queries, -1)
    targets = (
        torch.arange(len(class_ids), device=device)
        .view(1, len(class_ids), 1)
        .expand(args.episodes, len(class_ids), args.queries)
    )

    oracle_basis = fit_oracle_bases(
        torch, visual_features, oracle_indices, max(args.ranks), device
    )
    correct_text = text_features[class_ids]
    # Three-way tasks have two possible cyclic derangements. Alternating them
    # across seeds guarantees that shuffled text never remains on its class.
    shift = 1 + seed % (len(class_ids) - 1)
    shuffled_text = correct_text.roll(shifts=shift, dims=0)

    rows = []
    for shots in args.shots:
        support_mean = support[:, :, :shots].mean(dim=2)
        plain_centers = refined_centers(torch, support_mean, correct_text, 0.0)
        plain_squared, plain_projection = distance_components(
            torch, query, plain_centers, oracle_basis
        )
        rows.append(
            result_row(
                stage,
                shots,
                seed,
                "protonet",
                0,
                0.0,
                0.0,
                metrics(torch, plain_squared, targets, len(class_ids)),
            )
        )

        for rank in args.ranks:
            for beta in args.betas:
                distance = hybrid_distance(
                    plain_squared, plain_projection, rank, beta
                )
                rows.append(
                    result_row(
                        stage,
                        shots,
                        seed,
                        "oracle_subspace",
                        rank,
                        beta,
                        0.0,
                        metrics(torch, distance, targets, len(class_ids)),
                    )
                )

        for text_method, oracle_method, task_text in (
            (
                "protonet_text",
                "oracle_subspace_text",
                correct_text,
            ),
            (
                "protonet_shuffled_text",
                "oracle_subspace_shuffled_text",
                shuffled_text,
            ),
        ):
            for alpha in args.alphas:
                centers = refined_centers(torch, support_mean, task_text, alpha)
                squared, projection = distance_components(
                    torch, query, centers, oracle_basis
                )
                rows.append(
                    result_row(
                        stage,
                        shots,
                        seed,
                        text_method,
                        0,
                        0.0,
                        alpha,
                        metrics(torch, squared, targets, len(class_ids)),
                    )
                )
                for rank in args.ranks:
                    for beta in args.betas:
                        distance = hybrid_distance(squared, projection, rank, beta)
                        rows.append(
                            result_row(
                                stage,
                                shots,
                                seed,
                                oracle_method,
                                rank,
                                beta,
                                alpha,
                                metrics(torch, distance, targets, len(class_ids)),
                            )
                        )
    return rows


def select_hyperparameters(rows, shot_values):
    selected = {}
    for shots in shot_values:
        selected[shots] = {}
        for method in METHODS:
            method_rows = [
                row
                for row in rows
                if row["stage"] == "validation"
                and row["shots"] == shots
                and row["method"] == method
            ]
            settings = sorted(
                {
                    (int(row["rank"]), float(row["beta"]), float(row["alpha"]))
                    for row in method_rows
                }
            )
            means = {}
            for setting in settings:
                rank, beta, alpha = setting
                values = [
                    float(row["macro_auroc"])
                    for row in method_rows
                    if row["rank"] == rank
                    and row["beta"] == beta
                    and row["alpha"] == alpha
                ]
                means[setting] = statistics.mean(values)
            rank, beta, alpha = max(
                settings,
                key=lambda setting: (
                    means[setting],
                    -setting[2],
                    -setting[1],
                    -setting[0],
                ),
            )
            selected[shots][method] = {
                "rank": rank,
                "beta": beta,
                "alpha": alpha,
            }
    return selected


def selected_test_rows(rows, shots, method, setting):
    return [
        row
        for row in rows
        if row["stage"] == "test"
        and row["shots"] == shots
        and row["method"] == method
        and row["rank"] == setting["rank"]
        and row["beta"] == setting["beta"]
        and row["alpha"] == setting["alpha"]
    ]


def summarize(rows, selected, shot_values):
    summary = []
    for shots in shot_values:
        for method in METHODS:
            setting = selected[shots][method]
            chosen = selected_test_rows(rows, shots, method, setting)
            result = {"shots": shots, "method": method, **setting}
            for metric_name in ("accuracy", "macro_f1", "macro_auroc"):
                values = [float(row[metric_name]) for row in chosen]
                result[f"{metric_name}_mean"] = statistics.mean(values)
                result[f"{metric_name}_std"] = (
                    statistics.stdev(values) if len(values) > 1 else 0.0
                )
            summary.append(result)
    return summary


def semantic_sanity(rows, selected, shot_values):
    summary = []
    for shots in shot_values:
        for family, correct, shuffled in (
            ("protonet", "protonet_text", "protonet_shuffled_text"),
            (
                "oracle_subspace",
                "oracle_subspace_text",
                "oracle_subspace_shuffled_text",
            ),
        ):
            correct_rows = selected_test_rows(
                rows, shots, correct, selected[shots][correct]
            )
            shuffled_rows = selected_test_rows(
                rows, shots, shuffled, selected[shots][shuffled]
            )
            correct_by_seed = {
                int(row["seed"]): float(row["macro_auroc"]) for row in correct_rows
            }
            shuffled_by_seed = {
                int(row["seed"]): float(row["macro_auroc"]) for row in shuffled_rows
            }
            seeds = sorted(set(correct_by_seed).intersection(shuffled_by_seed))
            deltas = [
                correct_by_seed[seed] - shuffled_by_seed[seed] for seed in seeds
            ]
            correct_mean = statistics.mean(correct_by_seed[seed] for seed in seeds)
            shuffled_mean = statistics.mean(shuffled_by_seed[seed] for seed in seeds)
            delta_mean = statistics.mean(deltas)
            summary.append(
                {
                    "shots": shots,
                    "family": family,
                    "correct_text_auroc_mean": correct_mean,
                    "shuffled_text_auroc_mean": shuffled_mean,
                    "paired_delta_mean": delta_mean,
                    "paired_delta_std": (
                        statistics.stdev(deltas) if len(deltas) > 1 else 0.0
                    ),
                    "semantic_signal": delta_mean > 0.0,
                }
            )
    return summary


def run(args: argparse.Namespace) -> None:
    try:
        import torch
    except ImportError as error:
        raise SystemExit("PyTorch is missing. Install the project GPU dependencies.") from error

    args.shots = sorted(set(args.shots))
    args.ranks = sorted(set(args.ranks))
    args.alphas = sorted(set(float(value) for value in args.alphas))
    args.betas = sorted(set(float(value) for value in args.betas))
    if any(value < 0 or value > 1 for value in args.alphas + args.betas):
        raise ValueError("Alpha and beta values must lie in [0, 1]")

    device = choose_device(torch, args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    visual_payload = load_torch_file(torch, args.embeddings)
    text_payload = load_torch_file(torch, args.text_embeddings)
    visual_features = visual_payload["features"]
    visual_labels = visual_payload["labels"].long()
    class_names = list(visual_payload["class_names"])
    if not visual_payload.get("normalized", False):
        visual_features = torch.nn.functional.normalize(
            visual_features.float(), dim=-1
        ).half()
    if not args.keep_features_cpu:
        visual_features = visual_features.to(device, non_blocking=True)

    text_names = list(text_payload["class_names"])
    text_name_to_id = {name: index for index, name in enumerate(text_names)}
    missing_text = sorted(set(class_names).difference(text_names))
    if missing_text:
        raise ValueError(f"Missing text embeddings for classes: {missing_text}")
    text_features = text_payload["features"].float()
    text_features = torch.stack(
        [text_features[text_name_to_id[name]] for name in class_names]
    )
    text_features = torch.nn.functional.normalize(text_features, dim=-1).to(device)
    if text_features.shape[-1] != visual_features.shape[-1]:
        raise ValueError(
            f"Visual/text dimensions differ: {visual_features.shape[-1]} vs "
            f"{text_features.shape[-1]}"
        )
    visual_model = visual_payload.get("model")
    text_model = text_payload.get("model")
    if visual_model and text_model and visual_model != text_model:
        raise ValueError(f"Visual and text encoders differ: {visual_model} vs {text_model}")

    split = build_class_split(class_names, args.split_seed, args.split_json)
    name_to_id = {name: index for index, name in enumerate(class_names)}
    indices_by_class = {
        class_id: torch.nonzero(visual_labels == class_id, as_tuple=False).flatten()
        for class_id in range(len(class_names))
    }
    required = args.oracle_size + max(args.shots) + args.queries
    insufficient = {
        name: len(indices_by_class[name_to_id[name]])
        for stage in ("validation", "test")
        for name in split[stage]
        if len(indices_by_class[name_to_id[name]]) < required
    }
    if insufficient:
        raise ValueError(
            f"Novel classes need at least {required} images each. "
            f"Insufficient counts: {insufficient}"
        )
    if max(args.ranks) >= args.oracle_size:
        raise ValueError("Every rank must be smaller than --oracle-size")

    all_rows = []
    with torch.inference_mode():
        for seed in args.seeds:
            for stage in ("validation", "test"):
                print(
                    f"{stage}: seed {seed}, shots={args.shots}, "
                    f"{args.episodes} episodes",
                    flush=True,
                )
                class_ids = [name_to_id[name] for name in split[stage]]
                all_rows.extend(
                    evaluate_stage(
                        torch,
                        visual_features,
                        text_features,
                        indices_by_class,
                        class_ids,
                        args,
                        seed,
                        stage,
                        device,
                    )
                )

    selected = select_hyperparameters(all_rows, args.shots)
    test_summary = summarize(all_rows, selected, args.shots)
    sanity_summary = semantic_sanity(all_rows, selected, args.shots)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "per_seed_all_settings.csv", all_rows)
    write_csv(args.output_dir / "test_selected_summary.csv", test_summary)
    write_csv(args.output_dir / "semantic_sanity_summary.csv", sanity_summary)
    with (args.output_dir / "experiment.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "split": split,
                "selected_hyperparameters_by_validation_macro_auroc": selected,
                "descriptions": dict(
                    zip(text_names, text_payload.get("descriptions", text_names))
                ),
                "seeds": args.seeds,
                "episodes": args.episodes,
                "shots": args.shots,
                "queries_per_class": args.queries,
                "oracle_size_per_class": args.oracle_size,
                "alphas": args.alphas,
                "betas": args.betas,
                "ranks": args.ranks,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    print("\nValidation-selected test results (mean +/- sd over seeds)")
    for row in test_summary:
        print(
            f"{int(row['shots'])}-shot  {row['method']:32s} "
            f"r={int(row['rank']):>2d} beta={float(row['beta']):.2f} "
            f"alpha={float(row['alpha']):.2f}  "
            f"AUROC {row['macro_auroc_mean']:.4f} +/- {row['macro_auroc_std']:.4f}"
        )
    print("\nCorrect-text minus shuffled-text AUROC")
    for row in sanity_summary:
        conclusion = "semantic signal" if row["semantic_signal"] else "no semantic signal"
        print(
            f"{int(row['shots'])}-shot  {row['family']:18s} "
            f"delta={row['paired_delta_mean']:+.4f} ({conclusion})"
        )
    print(f"\nSaved results to {args.output_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Frozen-BioMedCLIP episodic text/prototype/oracle evaluation."
    )
    parser.add_argument(
        "--embeddings", type=Path, default=Path("data/embeddings/biomedclip.pt")
    )
    parser.add_argument(
        "--text-embeddings",
        type=Path,
        default=Path("data/embeddings/biomedclip_text.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/text_experiment")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--keep-features-cpu", action="store_true")
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--shots", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--queries", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--oracle-size", type=int, default=256)
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75]
    )
    parser.add_argument(
        "--betas",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
    )
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    run(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
