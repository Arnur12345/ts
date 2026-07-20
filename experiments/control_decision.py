"""Turn the predeclared controls into one of the requested decisions."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


def _paired(rows: list[dict], baseline: str, control: str, metric: str = "auroc", shot: int = 5) -> dict:
    selected = {
        (row["fold"], row["seed"], row["method"]): float(row["value"])
        for row in rows if row["metric"] == metric and row["shot"] == shot and row["method"] in (baseline, control)
    }
    seeds = sorted({seed for _, seed, _ in selected})
    differences = []
    for seed in seeds:
        folds = sorted({fold for fold, row_seed, _ in selected if row_seed == seed})
        differences.append(statistics.mean(selected[fold, seed, baseline] - selected[fold, seed, control] for fold in folds))
    mean = statistics.mean(differences)
    error = 2.262 * statistics.stdev(differences) / math.sqrt(len(differences))
    return {"baseline": baseline, "control": control, "metric": metric, "shot": shot, "n": len(differences), "mean_drop": mean, "ci95": [mean - error, mean + error]}


def _mean(rows: list[dict], method: str, metric: str, shot: int = 5) -> float:
    values = [float(row["value"]) for row in rows if row["method"] == method and row["metric"] == metric and row["shot"] == shot]
    return statistics.mean(values)


def write_decision(output_dir: Path, rows: list[dict]) -> dict:
    permuted = _paired(rows, "visual_protonet", "visual_protonet_permuted_support_labels")
    duplicated = _paired(rows, "visual_protonet", "visual_protonet_duplicated_support")
    clean = ("text_only", "visual_protonet", "protonet_text")
    clean_aurocs = {method: _mean(rows, method, "auroc") for method in clean}
    best = max(clean_aurocs, key=clean_aurocs.get)
    text_ece = _mean(rows, "protonet_text", "calibration_error")

    supports_genuine = permuted["ci95"][0] > 0 and duplicated["ci95"][0] > 0
    supports_barely_matter = all(
        max(map(abs, evidence["ci95"])) < 0.01 for evidence in (permuted, duplicated)
    )
    text_best_poorly_calibrated = best == "protonet_text" and text_ece > 0.05
    if supports_genuine:
        decision = "abandon_text_dominance_claim"
        reason = "Permuting labels hurts and independent supports beat duplicated supports with positive paired 95% confidence bounds."
    elif supports_barely_matter:
        decision = "proceed_with_support_grounding"
        reason = "Both support interventions change 5-shot AUROC by less than the preregistered 0.01 margin."
    elif text_best_poorly_calibrated:
        decision = "focus_on_semantic_anchoring_and_calibration"
        reason = "ProtoNet+text has the highest clean 5-shot AUROC and ECE exceeds 0.05."
    else:
        decision = "controls_inconclusive"
        reason = "The intervention results do not satisfy any complete decision branch."

    report = {
        "decision": decision,
        "reason": reason,
        "preregistered_rules": {
            "barely_matters_auroc_margin": 0.01,
            "poor_calibration_ece": 0.05,
            "confidence_level": 0.95,
            "paired_unit": "seed-level difference averaged across folds",
        },
        "five_shot_evidence": {"permuted_labels": permuted, "duplicated_support": duplicated, "clean_auroc": clean_aurocs, "protonet_text_ece": text_ece},
    }
    (output_dir / "decision.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "decision.md").write_text(
        f"# Control decision\n\n**{decision}**\n\n{reason}\n\n"
        f"- Permuted-label AUROC drop: {permuted['mean_drop']:.4f} (95% CI {permuted['ci95'][0]:.4f}, {permuted['ci95'][1]:.4f})\n"
        f"- Independent-minus-duplicated AUROC: {duplicated['mean_drop']:.4f} (95% CI {duplicated['ci95'][0]:.4f}, {duplicated['ci95'][1]:.4f})\n"
        f"- Best clean method: {best} (AUROC {clean_aurocs[best]:.4f})\n"
        f"- ProtoNet+text ECE: {text_ece:.4f}\n",
        encoding="utf-8",
    )
    return report
