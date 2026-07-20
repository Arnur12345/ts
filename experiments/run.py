from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from . import lp_text, proker, protonet_text, tcla, text_only_zero_shot, tip_adapter_f, visual_protonet
from .episodes import batch, load_dataset, load_or_create
from .metrics import evaluate


METHODS = [
    ("text_only_zero_visual_shot", text_only_zero_shot.predict, True),
    ("visual_protonet", visual_protonet.predict, False),
    ("protonet_text", protonet_text.predict, True),
    ("lp_text", lp_text.predict, True),
    ("tip_adapter_f", tip_adapter_f.predict, True),
    ("proker", proker.predict, True),
    ("tcla_final_layer", tcla.predict, True),
]
TEMPERATURES = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)


def _predict(method, data, episodes, partition, shot, shuffled, device):
    outputs = []
    for run in episodes[partition]["runs"]:
        support_images, support_reports, labels, query_images, query_labels = batch(data, run, shot, device)
        logits = method(support_images, support_reports, labels, query_images, shuffled=shuffled)
        outputs.append((logits.detach().cpu().reshape(-1, 3), query_labels))
    return outputs


def _temperature(validation_outputs) -> float:
    logits = torch.cat([item[0] for item in validation_outputs])
    labels = torch.cat([item[1] for item in validation_outputs])
    return min(TEMPERATURES, key=lambda value: F.cross_entropy(logits / value, labels).item())


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(float(row["value"]))
    result = []
    for group, values in grouped.items():
        result.append(
            {
                **dict(zip(keys, group)),
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all embedding-only few-shot experiments.")
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/biomedclip_pairs_7000.pt"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/biomedclip_pairs_7000.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiments"))
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")
    data = load_dataset(args.embeddings, args.manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    episodes = load_or_create(
        args.output_dir / "subset_episodes.pt", data, args.manifest, episode_count=args.episodes
    )
    overall_seed_rows, class_seed_rows = [], []

    for shot in (1, 3, 5):
        for base_name, method, supports_shuffle in METHODS:
            for shuffled in ((False, True) if supports_shuffle else (False,)):
                name = f"{base_name}_shuffled_text" if shuffled else base_name
                print(f"running {name}, {shot}-shot", flush=True)
                validation = _predict(method, data, episodes, "validation_novel", shot, shuffled, device)
                temperature = _temperature(validation)
                test = _predict(method, data, episodes, "test_novel", shot, shuffled, device)
                class_names = episodes["test_novel"]["class_names"]

                for seed, (logits, labels) in enumerate(test):
                    overall, per_class = evaluate(logits / temperature, labels, class_names)
                    for metric, value in overall.items():
                        overall_seed_rows.append(
                            {"method": name, "shot": shot, "seed": seed, "temperature": temperature, "metric": metric, "value": value}
                        )
                    for class_row in per_class:
                        for metric in ("auroc", "auprc", "f1", "accuracy", "nll", "calibration_error"):
                            class_seed_rows.append(
                                {"method": name, "shot": shot, "seed": seed, "class": class_row["class"], "metric": metric, "value": class_row[metric]}
                            )

    _write(args.output_dir / "per_seed_metrics.csv", overall_seed_rows)
    _write(args.output_dir / "overall_metrics.csv", _summaries(overall_seed_rows, ("method", "shot", "temperature", "metric")))
    _write(args.output_dir / "per_class_per_seed.csv", class_seed_rows)
    _write(args.output_dir / "per_class_metrics.csv", _summaries(class_seed_rows, ("method", "shot", "class", "metric")))
    (args.output_dir / "experiment.json").write_text(
        json.dumps(
            {
                "embeddings": str(args.embeddings),
                "manifest": str(args.manifest),
                "episodes_per_seed": args.episodes,
                "seeds": episodes["seeds"],
                "shots": [1, 3, 5],
                "ways": 3,
                "queries_per_class": 1,
                "temperature_selected_on": "validation_novel NLL",
                "note": "TCLA is the final-layer ablation because intermediate features were not saved.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"reports written to {args.output_dir}")


if __name__ == "__main__":
    main()
