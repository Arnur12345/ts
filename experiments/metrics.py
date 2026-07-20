from __future__ import annotations

import torch


def _auc(target: torch.Tensor, score: torch.Tensor) -> float:
    score, order = torch.sort(score)
    target = target[order].float()
    positives, negatives = target.sum(), (1 - target).sum()
    if positives == 0 or negatives == 0:
        return float("nan")
    _, groups, counts = torch.unique_consecutive(score, return_inverse=True, return_counts=True)
    starts = counts.cumsum(0) - counts
    average_ranks = starts + (counts + 1) / 2
    rank_sum = average_ranks[groups][target.bool()].sum()
    return ((rank_sum - positives * (positives + 1) / 2) / (positives * negatives)).item()


def _average_precision(target: torch.Tensor, score: torch.Tensor) -> float:
    score, order = torch.sort(score, descending=True)
    target = target[order].float()
    if target.sum() == 0:
        return float("nan")
    _, counts = torch.unique_consecutive(score, return_counts=True)
    ends = counts.cumsum(0) - 1
    true_positives = target.cumsum(0)[ends]
    precision = true_positives / (ends + 1)
    recall_gain = torch.diff(torch.cat([torch.zeros(1), true_positives / target.sum()]))
    return (precision * recall_gain).sum().item()


def _ece(confidence: torch.Tensor, correct: torch.Tensor, bins: int = 15) -> float:
    value = 0.0
    for low, high in zip(torch.linspace(0, 1, bins + 1)[:-1], torch.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            value += mask.float().mean() * (confidence[mask].mean() - correct[mask].float().mean()).abs()
    return float(value)


def evaluate(logits: torch.Tensor, labels: torch.Tensor, class_names: list[str]):
    logits, labels = logits.cpu(), labels.cpu()
    probability = logits.softmax(1).clamp(1e-7, 1 - 1e-7)
    prediction = probability.argmax(1)
    per_class = []
    for class_id, class_name in enumerate(class_names):
        target = labels.eq(class_id)
        predicted = prediction.eq(class_id)
        tp, fp = (target & predicted).sum(), (~target & predicted).sum()
        fn, tn = (target & ~predicted).sum(), (~target & ~predicted).sum()
        f1 = (2 * tp / (2 * tp + fp + fn).clamp_min(1)).item()
        binary_nll = -(target * probability[:, class_id].log() + (~target) * (1 - probability[:, class_id]).log()).mean()
        per_class.append(
            {
                "class": class_name,
                "auroc": _auc(target, probability[:, class_id]),
                "auprc": _average_precision(target, probability[:, class_id]),
                "f1": f1,
                "accuracy": ((tp + tn) / len(labels)).item(),
                "nll": binary_nll.item(),
                "calibration_error": _ece(probability[:, class_id], target),
            }
        )
    overall = {
        "auroc": sum(row["auroc"] for row in per_class) / len(per_class),
        "auprc": sum(row["auprc"] for row in per_class) / len(per_class),
        "macro_f1": sum(row["f1"] for row in per_class) / len(per_class),
        "accuracy": prediction.eq(labels).float().mean().item(),
        "nll": -probability[torch.arange(len(labels)), labels].log().mean().item(),
        "calibration_error": _ece(probability.max(1).values, prediction.eq(labels)),
    }
    return overall, per_class
