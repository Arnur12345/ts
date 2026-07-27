"""Numerical sanity check for the fixed CoMeD support correction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .comed import CoMeD
from .comed_run import (
    _balanced_support,
    _load_cache,
    _load_target_episodes,
    _panel_support,
    _tensor,
)


def _summary(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "mean": float(tensor.mean()),
        "std": float(tensor.std(unbiased=False)),
        "median": float(tensor.median()),
        "p05": float(torch.quantile(tensor, 0.05)),
        "p95": float(torch.quantile(tensor, 0.95)),
        "min": float(tensor.min()),
        "max": float(tensor.max()),
    }


def _panel_diagnostic(
    model,
    arrays,
    class_id,
    support,
    query,
    device,
) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor]:
    support_z = F.normalize(_tensor(arrays["rad"], support, device), dim=-1)
    query_z = F.normalize(_tensor(arrays["rad"], query, device), dim=-1)
    support_prior = _tensor(
        arrays["prior"][:, class_id], support, device
    )
    query_prior = _tensor(arrays["prior"][:, class_id], query, device)
    support_y_pm = _tensor(
        arrays["labels"][:, class_id], support, device
    ).mul(2).sub(1)
    support_phi = model.transform(support_z)
    query_phi = model.transform(query_z)
    disease_kernel = support_phi @ support_phi.T
    noise = F.softplus(model.log_noise) + 1e-4
    eye = torch.eye(
        len(support), device=device, dtype=disease_kernel.dtype
    )
    kernel = disease_kernel + noise * eye
    eigenvalues = torch.linalg.eigvalsh(kernel)
    condition = eigenvalues[-1] / eigenvalues[0].clamp_min(1e-30)
    residual = support_y_pm - support_prior
    chol = torch.linalg.cholesky(kernel + 1e-5 * eye)
    alpha = torch.cholesky_solve(residual[:, None], chol)
    correction = (query_phi @ support_phi.T @ alpha).squeeze(-1)
    finite = all(
        torch.isfinite(value).all()
        for value in (
            kernel,
            eigenvalues,
            residual,
            alpha,
            correction,
        )
    )
    return (
        {
            "condition_number": float(condition),
            "kernel_min_eigenvalue": float(eigenvalues[0]),
            "kernel_max_eigenvalue": float(eigenvalues[-1]),
            "alpha_norm": float(alpha.norm()),
            "residual_norm": float(residual.norm()),
            "support_prior_std": float(
                support_prior.std(unbiased=False)
            ),
            "query_prior_std": float(query_prior.std(unbiased=False)),
            "correction_std": float(correction.std(unbiased=False)),
            "correction_abs_mean": float(correction.abs().mean()),
            "correction_to_prior_std": float(
                correction.std(unbiased=False)
                / query_prior.std(unbiased=False).clamp_min(1e-8)
            ),
            "noise": float(noise),
            "finite": bool(finite),
        },
        support_prior.detach().cpu(),
        support_y_pm.detach().cpu(),
        query_prior.detach().cpu(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4)
    )
    parser.add_argument("--episodes-per-seed", type=int, default=100)
    parser.add_argument("--shot", type=int, default=3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    arrays, _, data = _load_cache(args.cache)
    episodes = _load_target_episodes(args.episodes, data, args)
    model = CoMeD(
        dim=arrays["rad"].shape[1],
        rank=16,
        metric_mode="identity",
        use_nuisance=False,
    ).to(device).eval().requires_grad_(False)
    target_id = data.class_names.index("Pneumothorax")
    rows = []
    support_priors, support_targets, query_priors, query_targets = [], [], [], []
    failures = []
    with torch.inference_mode():
        for seed in args.seeds:
            current = episodes[(seed, "validate")]
            for episode in range(len(current["positive"])):
                query = current["query"][episode].long()
                panels = {
                    "balanced": _balanced_support(
                        current, episode, args.shot
                    ),
                    "device_poor": _panel_support(
                        current, episode, args.shot, 0
                    ),
                    "device_rich": _panel_support(
                        current, episode, args.shot, 1
                    ),
                }
                for panel_name, support in panels.items():
                    try:
                        metrics, prior, target_pm, query_prior = (
                            _panel_diagnostic(
                                model,
                                arrays,
                                target_id,
                                support,
                                query,
                                device,
                            )
                        )
                    except RuntimeError as error:
                        failures.append(
                            {
                                "seed": seed,
                                "episode": episode,
                                "panel": panel_name,
                                "error": str(error),
                            }
                        )
                        continue
                    rows.append(
                        {
                            "seed": seed,
                            "episode": episode,
                            "panel": panel_name,
                            **metrics,
                        }
                    )
                    if panel_name == "balanced":
                        support_priors.append(prior)
                        support_targets.append(target_pm)
                        query_priors.append(query_prior)
                        query_targets.append(
                            current["targets"][episode].flatten()
                        )
    if not rows:
        raise RuntimeError("all CoMeD sanity-check panels failed")
    support_prior = torch.cat(support_priors)
    support_target = torch.cat(support_targets)
    query_prior = torch.cat(query_priors)
    query_target = torch.cat(query_targets).mul(2).sub(1)
    prior_centered = query_prior - query_prior.mean()
    target_centered = query_target - query_target.mean()
    prior_target_correlation = float(
        (prior_centered * target_centered).sum()
        / (
            prior_centered.norm().clamp_min(1e-8)
            * target_centered.norm().clamp_min(1e-8)
        )
    )
    numeric_fields = (
        "condition_number",
        "kernel_min_eigenvalue",
        "kernel_max_eigenvalue",
        "alpha_norm",
        "residual_norm",
        "support_prior_std",
        "query_prior_std",
        "correction_std",
        "correction_abs_mean",
        "correction_to_prior_std",
        "noise",
    )
    summaries = {
        field: _summary([float(row[field]) for row in rows])
        for field in numeric_fields
    }
    all_finite = all(bool(row["finite"]) for row in rows)
    condition_stable = summaries["condition_number"]["max"] < 1e8
    eigenvalue_stable = summaries["kernel_min_eigenvalue"]["min"] > 1e-8
    numerically_stable = (
        not failures and all_finite and condition_stable and eigenvalue_stable
    )
    decision = {
        "status": (
            "numerically_stable_close_comed_branch"
            if numerically_stable
            else "numerical_problem_found_fix_before_interpretation"
        ),
        "partition": "validate_only",
        "panels_checked": len(rows),
        "cholesky_failures": failures,
        "all_finite": all_finite,
        "text_prior_vs_pm_labels": {
            "support_prior": _summary(support_prior.tolist()),
            "support_pm_label": _summary(support_target.tolist()),
            "query_prior": _summary(query_prior.tolist()),
            "query_pm_label": _summary(query_target.tolist()),
            "query_prior_pm_label_correlation": prior_target_correlation,
        },
        "kernel_and_correction": summaries,
        "numerical_criteria": {
            "condition_number_max_below": 1e8,
            "kernel_min_eigenvalue_above": 1e-8,
            "all_values_finite": True,
            "zero_cholesky_failures": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "sanity_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "sanity_check.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"CoMeD sanity check written to {args.output_dir}; "
        f"decision={decision['status']}"
    )


if __name__ == "__main__":
    main()
