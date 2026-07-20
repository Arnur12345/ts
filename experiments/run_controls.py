"""Run only the support/text diagnostic controls on rotating class splits."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .control_decision import write_decision
from .control_episodes import batch, load_or_create, write_episode_ids
from .controls import METHODS
from .episodes import load_dataset
from .metrics import evaluate


TEMPERATURES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _outputs(method, data, block, shot, device):
    result = []
    for run in block["runs"]:
        support_images, support_reports, labels, query_images, query_labels, query_indices = batch(data, run, shot, device)
        logits = method(support_images, support_reports, labels, query_images)
        result.append((logits.detach().cpu().reshape(-1, 3), query_labels, query_indices.reshape(-1)))
    return result


def _temperature(outputs) -> float:
    logits = torch.cat([item[0] for item in outputs])
    labels = torch.cat([item[1] for item in outputs])
    return min(TEMPERATURES, key=lambda value: F.cross_entropy(logits / value, labels).item())


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row["value"]))
    result = []
    for group, values in grouped.items():
        result.append({
            **dict(zip(keys, group)), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0, "n": len(values),
        })
    return result


def _prediction_rows(method, source, temperature, fold, seed, shot, block, logits, labels, indices, data):
    probabilities = (logits / temperature).softmax(1)
    predictions = probabilities.argmax(1).tolist()
    logits, probabilities = logits.tolist(), probabilities.tolist()
    labels, indices = labels.tolist(), indices.tolist()
    subjects = data["subject_ids"].tolist()
    dicoms = data["dicom_ids"]
    names, class_ids = block["class_names"], block["class_ids"]
    for row, (target, index) in enumerate(zip(labels, indices)):
        episode, position = divmod(row, 3)
        predicted = predictions[row]
        yield {
            "method": method, "temperature_source": source, "temperature": temperature,
            "fold": fold, "seed": seed, "shot": shot,
            "episode_id": f"f{fold:02d}-test_novel-s{seed:02d}-e{episode:04d}",
            "query_position": position, "query_index": index,
            "query_subject_id": subjects[index], "query_dicom_id": dicoms[index],
            "true_class_id": class_ids[target], "true_class_name": names[target],
            "predicted_class_id": class_ids[predicted], "predicted_class_name": names[predicted],
            "class_0": names[0], "class_1": names[1], "class_2": names[2],
            "logit_0": logits[row][0], "logit_1": logits[row][1], "logit_2": logits[row][2],
            "probability_0": probabilities[row][0], "probability_1": probabilities[row][1], "probability_2": probabilities[row][2],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/biomedclip_pairs_7000.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/biomedclip_pairs_7000.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/controls_v2"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    data = load_dataset(args.embeddings, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_or_create(
        args.output_dir / "control_episodes.pt", data, args.manifest,
        episode_count=args.episodes, seeds=tuple(range(10)), fold_count=args.folds,
    )
    write_episode_ids(args.output_dir / "episode_ids.csv.gz", episodes, data)

    seed_rows, class_rows = [], []
    prediction_path = args.output_dir / "per_query_predictions.csv.gz"
    with gzip.open(prediction_path, "wt", newline="", encoding="utf-8") as prediction_handle:
        prediction_writer = None
        for fold in episodes["folds"]:
            for shot in (1, 3, 5):
                temperatures = {}
                for source in ("text_only", "visual_protonet", "protonet_text"):
                    validation = _outputs(METHODS[source][0], data, fold["validation_novel"], shot, device)
                    temperatures[source] = _temperature(validation)

                for method_name, (method, temperature_source) in METHODS.items():
                    temperature = temperatures[temperature_source]
                    print(f"fold {fold['fold']}, {shot}-shot, {method_name}, T={temperature}", flush=True)
                    test_outputs = _outputs(method, data, fold["test_novel"], shot, device)
                    for run_position, (logits, labels, indices) in enumerate(test_outputs):
                        seed = episodes["seeds"][run_position]
                        overall, per_class = evaluate(logits / temperature, labels, fold["test_novel"]["class_names"])
                        for metric, value in overall.items():
                            seed_rows.append({
                                "method": method_name, "fold": fold["fold"], "shot": shot, "seed": seed,
                                "temperature_source": temperature_source, "temperature": temperature,
                                "metric": metric, "value": value,
                            })
                        for class_result in per_class:
                            for metric in ("auroc", "auprc", "f1", "accuracy", "nll", "calibration_error"):
                                class_rows.append({
                                    "method": method_name, "fold": fold["fold"], "shot": shot, "seed": seed,
                                    "class": class_result["class"], "metric": metric, "value": class_result[metric],
                                })
                        rows = _prediction_rows(
                            method_name, temperature_source, temperature, fold["fold"], seed, shot,
                            fold["test_novel"], logits, labels, indices, data,
                        )
                        if prediction_writer is None:
                            first = next(rows)
                            prediction_writer = csv.DictWriter(prediction_handle, fieldnames=list(first), lineterminator="\n")
                            prediction_writer.writeheader()
                            prediction_writer.writerow(first)
                        prediction_writer.writerows(rows)

    _write(args.output_dir / "per_seed_metrics.csv", seed_rows)
    _write(args.output_dir / "per_fold_metrics.csv", _summaries(seed_rows, ("method", "fold", "shot", "temperature_source", "temperature", "metric")))
    _write(args.output_dir / "overall_metrics.csv", _summaries(seed_rows, ("method", "shot", "metric")))
    _write(args.output_dir / "per_class_per_seed.csv", class_rows)
    _write(args.output_dir / "per_class_metrics.csv", _summaries(class_rows, ("method", "shot", "class", "metric")))
    decision = write_decision(args.output_dir, seed_rows)

    split_metadata = []
    for fold in episodes["folds"]:
        names = data["class_names"]
        split_metadata.append({
            "fold": fold["fold"],
            "base": [names[i] for i in fold["base_class_ids"]],
            "validation_novel": fold["validation_novel"]["class_names"],
            "test_novel": fold["test_novel"]["class_names"],
        })
    (args.output_dir / "experiment.json").write_text(json.dumps({
        "embeddings": str(args.embeddings), "manifest": str(args.manifest),
        "episodes_per_seed": args.episodes, "seeds": episodes["seeds"], "shots": [1, 3, 5],
        "ways": 3, "queries_per_class": 1, "folds": split_metadata,
        "temperature_selected_on": "validation_novel NLL",
        "shared_temperature_controls": {
            "protonet_text_shuffled_text": "protonet_text",
            "visual_protonet_permuted_support_labels": "visual_protonet",
            "visual_protonet_duplicated_support": "visual_protonet",
        },
        "decision": decision["decision"],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"reports and {decision['decision']} decision written to {args.output_dir}")


if __name__ == "__main__":
    main()
