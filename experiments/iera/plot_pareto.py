"""Plot pathology-specific AUROC-SMS frontiers from a completed sweep."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


LABELS = {
    "binary_protonet_random": "Random ProtoNet",
    "nuisance_balanced": "Balanced oracle",
    "mean_difference_projection": "Mean projection",
    "text_direction_orthogonalization": "Text orthogonalization",
    "rex": "REx",
    "adapter_only": "Adapter only",
    "full_iera": "Full IERA",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pareto", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shot", type=int, default=3)
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit("Install matplotlib to render the Pareto figure") from error

    with args.pareto.open(newline="", encoding="utf-8") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if int(row["shot"]) == args.shot
        ]
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair"]].append(row)
    if not by_pair:
        raise ValueError(f"no {args.shot}-shot rows in {args.pareto}")

    figure, axes = plt.subplots(
        1, len(by_pair), figsize=(6 * len(by_pair), 4.5), squeeze=False
    )
    colors = {
        "adapter_only": "#0072B2",
        "full_iera": "#D55E00",
    }
    markers = ("o", "s", "^", "v", "P", "X", "D")
    for axis, (pair, pair_rows) in zip(axes[0], sorted(by_pair.items())):
        grouped = defaultdict(list)
        for row in pair_rows:
            grouped[row["method"]].append(row)
        for method_index, (method, method_rows) in enumerate(sorted(grouped.items())):
            points = sorted(
                method_rows, key=lambda row: float(row["sms_reduction"])
            )
            x = [100 * float(row["sms_reduction"]) for row in points]
            y = [100 * float(row["auroc_change"]) for row in points]
            constrained = method in {"adapter_only", "full_iera"}
            axis.plot(
                x,
                y,
                marker=markers[method_index % len(markers)],
                linestyle="-" if constrained else "none",
                color=colors.get(method),
                label=LABELS.get(method, method),
            )
            if constrained:
                for x_value, y_value, row in zip(x, y, points):
                    axis.annotate(
                        f"ρ={float(row['rho']):g}",
                        (x_value, y_value),
                        xytext=(4, 4),
                        textcoords="offset points",
                        fontsize=8,
                    )
        axis.axhline(0, color="0.65", linewidth=1)
        axis.axvline(0, color="0.65", linewidth=1)
        axis.set_title(pair.replace("__", " – "))
        axis.set_xlabel("SMS reduction vs random ProtoNet (%)")
        axis.set_ylabel("AUROC change vs random ProtoNet (points)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(4, len(labels)))
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
