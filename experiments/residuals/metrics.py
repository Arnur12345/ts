"""Calibration and metrics for softmax and independent sigmoid episodes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


TEMPERATURES = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
THRESHOLDS = tuple(value / 20 for value in range(2, 19))


def _auc(target: torch.Tensor, score: torch.Tensor) -> float:
    score, order = torch.sort(score)
    target = target[order].float()
    positives, negatives = target.sum(), (1 - target).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    _, groups, counts = torch.unique_consecutive(score, return_inverse=True, return_counts=True)
    starts = counts.cumsum(0) - counts
    ranks = starts + (counts + 1) / 2
    rank_sum = ranks[groups][target.bool()].sum()
    return float((rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def _average_precision(target: torch.Tensor, score: torch.Tensor) -> float:
    score, order = torch.sort(score, descending=True)
    target = target[order].float()
    if target.sum() == 0:
        return float("nan")
    _, counts = torch.unique_consecutive(score, return_counts=True)
    ends = counts.cumsum(0) - 1
    true_positive = target.cumsum(0)[ends]
    precision = true_positive / (ends + 1)
    recall_gain = torch.diff(torch.cat((torch.zeros(1), true_positive / target.sum())))
    return float((precision * recall_gain).sum())


def _ece(probability: torch.Tensor, target: torch.Tensor, bins: int = 15) -> float:
    result = torch.tensor(0.0)
    for low, high in zip(torch.linspace(0, 1, bins + 1)[:-1], torch.linspace(0, 1, bins + 1)[1:]):
        mask = (probability > low) & (probability <= high)
        if mask.any():
            result += mask.float().mean() * (probability[mask].mean() - target[mask].float().mean()).abs()
    return float(result)


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def select_temperature(logits: torch.Tensor, targets: torch.Tensor, regime: str) -> float:
    if regime == "single_label":
        return min(TEMPERATURES, key=lambda value: float(F.cross_entropy(logits / value, targets.long())))
    known = targets.ge(0)
    return min(
        TEMPERATURES,
        key=lambda value: float(F.binary_cross_entropy_with_logits(logits[known] / value, targets[known])),
    )


def select_threshold(logits: torch.Tensor, targets: torch.Tensor, temperature: float) -> float:
    probability = torch.sigmoid(logits / temperature)
    scores = []
    for threshold in THRESHOLDS:
        class_f1 = []
        for class_id in range(targets.shape[1]):
            known = targets[:, class_id].ge(0)
            target = targets[known, class_id].bool()
            prediction = probability[known, class_id].ge(threshold)
            tp = (target & prediction).sum()
            fp = (~target & prediction).sum()
            fn = (target & ~prediction).sum()
            class_f1.append(float(2 * tp / (2 * tp + fp + fn).clamp_min(1)))
        scores.append(_mean(class_f1))
    return THRESHOLDS[max(range(len(scores)), key=scores.__getitem__)]


def evaluate_single(logits: torch.Tensor, targets: torch.Tensor, class_names: list[str]):
    probability = logits.softmax(1).clamp(1e-7, 1 - 1e-7)
    prediction = probability.argmax(1)
    per_class = []
    for class_id, name in enumerate(class_names):
        target = targets.eq(class_id)
        predicted = prediction.eq(class_id)
        tp, fp = (target & predicted).sum(), (~target & predicted).sum()
        fn, tn = (target & ~predicted).sum(), (~target & ~predicted).sum()
        per_class.append(
            {
                "class": name,
                "auroc": _auc(target, probability[:, class_id]),
                "auprc": _average_precision(target, probability[:, class_id]),
                "f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
                "accuracy": float((tp + tn) / len(targets)),
                "nll": float(-(target * probability[:, class_id].log() + (~target) * (1 - probability[:, class_id]).log()).mean()),
                "calibration_error": _ece(probability[:, class_id], target),
            }
        )
    overall = {
        "auroc": _mean([row["auroc"] for row in per_class]),
        "auprc": _mean([row["auprc"] for row in per_class]),
        "macro_f1": _mean([row["f1"] for row in per_class]),
        "accuracy": float(prediction.eq(targets).float().mean()),
        "nll": float(-probability[torch.arange(len(targets)), targets].log().mean()),
        "calibration_error": _ece(probability.max(1).values, prediction.eq(targets)),
    }
    return overall, per_class, probability


def evaluate_multilabel(
    logits: torch.Tensor, targets: torch.Tensor, class_names: list[str], threshold: float
):
    probability = torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)
    per_class = []
    for class_id, name in enumerate(class_names):
        known = targets[:, class_id].ge(0)
        target = targets[known, class_id].bool()
        score = probability[known, class_id]
        predicted = score.ge(threshold)
        tp, fp = (target & predicted).sum(), (~target & predicted).sum()
        fn, tn = (target & ~predicted).sum(), (~target & ~predicted).sum()
        per_class.append(
            {
                "class": name,
                "auroc": _auc(target, score),
                "auprc": _average_precision(target, score),
                "f1": float(2 * tp / (2 * tp + fp + fn).clamp_min(1)),
                "accuracy": float((tp + tn) / len(target)),
                "nll": float(-(target * score.log() + (~target) * (1 - score).log()).mean()),
                "calibration_error": _ece(score, target),
            }
        )
    known = targets.ge(0)
    prediction = probability.ge(threshold)
    overall = {
        "auroc": _mean([row["auroc"] for row in per_class]),
        "auprc": _mean([row["auprc"] for row in per_class]),
        "macro_f1": _mean([row["f1"] for row in per_class]),
        "accuracy": float(prediction[known].eq(targets[known].bool()).float().mean()),
        "nll": float(F.binary_cross_entropy(probability[known], targets[known])),
        "calibration_error": _ece(probability[known], targets[known].bool()),
    }
    return overall, per_class, probability
