# IERA experiments

The current experimental order follows `addition.md`. The constrained IERA
result is not interpreted until a credible Pneumothorax detector exists.

## Experiment 1: establish the detector

The old classifier used only a positive prototype:

```text
z(q) = sim(q, p+)
```

The repaired binary ProtoNet uses both support classes:

```text
z(q) = sim(q, p+) - sim(q, p-)
```

`model.py` now uses this binary readout for frozen ProtoNet, learned uniform,
unanchored IERA, and Anchored IERA. The standalone detector diagnostic compares
the old positive-only score against the repaired score without any learned
projection, IERA, SMS regularization, calibration, or threshold tuning.

The factorial is:

- positive-only versus binary ProtoNet;
- retained 4x4 versus 14x14 patch tokens;
- BioMedCLIP versus the CXR-specialized RAD-DINO backbone;
- 1, 3, 5, and 10 shots over ten episode seeds.

Configuration selection uses validation AUROC. The selected configuration is
reported once on the test partition, and is considered credible only if its
test AUROC 95% interval is above 0.5.

### Build BioMedCLIP native 14x14 tokens

```bash
PYTHONPATH=. python3 -m experiments.iera.patch_cache \
  --encoder biomedclip \
  --data-root ~/data/mimic-cxr-jpg-2.1.0 \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --output-dir outputs/iera/patch_cache_biomedclip_14x14 \
  --pool-grid 14 \
  --batch-size 128 \
  --device cuda
```

### Build RAD-DINO tokens retained at 14x14

RAD-DINO is a chest-X-ray-specific DINOv2 encoder. Its native 37x37 feature map
is pooled once to 14x14 for storage; the diagnostic can then pool it to 4x4.
Its released checkpoint was self-supervised on MIMIC-CXR among other datasets,
so results must be labeled as potentially transductive backbone evidence rather
than external validation. The released training-image list should be audited
against the evaluation studies before publication.

```bash
PYTHONPATH=. python3 -m experiments.iera.patch_cache \
  --encoder rad-dino \
  --data-root ~/data/mimic-cxr-jpg-2.1.0 \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --output-dir outputs/iera/patch_cache_rad_dino_14x14 \
  --pool-grid 14 \
  --batch-size 32 \
  --device cuda
```

Both caches are resumable and must remain in separate output directories.

### Run the no-IERA factorial

```bash
PYTHONPATH=. python3 -m experiments.iera.detector_diagnostic \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --raw-labels ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --cache biomedclip=outputs/iera/patch_cache_biomedclip_14x14 \
  --cache rad_dino=outputs/iera/patch_cache_rad_dino_14x14 \
  --output-dir outputs/iera/detector_factorial_v1 \
  --grids 4 14 \
  --shots 1 3 5 10 \
  --primary-shot 3 \
  --episodes 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --queries-per-stratum 1 \
  --min-stratum-patients 50 \
  --episode-batch-size 16 \
  --device cuda
```

Outputs are `per_seed_metrics.csv`, `summary_metrics.csv`,
`paired_head_deltas.csv`, `decision.json`, `experiment.json`, and the exact
episode indices.

## Experiments 2 and 3: falsification and SMS sweep

`falsification.py` uses the repaired positive-minus-negative detector for every
method. RAD-DINO is retained at 14x14 and the query readout is frozen and
localized. IERA's quadratic support-evidence proposal is explicitly pooled to
4x4 by default, but its anchored prototype and the detector's query readout
remain in the unprojected 14x14 RAD-DINO space. The text baseline is the only
exception to the backbone: it must use
BioMedCLIP because a BioMedCLIP text vector is not geometrically meaningful in
RAD-DINO space. It still uses the exact same episode indices and seeds.

All three commands below use one output root. The first command creates
`episodes.pt`; later stages refuse to proceed if their episode arguments differ.
Each learned run is checkpointed after a method/rho/seed combination, so
rerunning the same command resumes completed work.

### Build the fixed BioMedCLIP device direction

```bash
PYTHONPATH=. python3 -m experiments.iera.text_direction \
  --output outputs/iera/biomedclip_device_text_direction.pt \
  --device cuda
```

### 1. Cheap baselines

This runs random binary ProtoNet, nuisance-balanced oracle sampling,
mean-difference projection, and text-direction orthogonalization.

```bash
PYTHONPATH=. python3 -m experiments.iera.falsification \
  --stage cheap \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --raw-labels ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --rad-cache outputs/iera/patch_cache_rad_dino_14x14 \
  --biomed-cache outputs/iera/patch_cache_biomedclip_14x14 \
  --text-direction outputs/iera/biomedclip_device_text_direction.pt \
  --output-dir outputs/iera/falsification_v1 \
  --retained-grid 14 \
  --proposal-grid 4 \
  --shots 1 3 5 10 \
  --episodes 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --episode-batch-size 8 \
  --device cuda
```

### 2. Learned ablations at rho=0.7

This runs REx, constrained adapter only, bounded anchor only, and full Anchored
IERA. REx has no SMS loss. The other three use adaptive dual ascent and select
the highest worst-nuisance base-validation AUROC satisfying the SMS constraint.

```bash
PYTHONPATH=. python3 -m experiments.iera.falsification \
  --stage learned \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --raw-labels ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --rad-cache outputs/iera/patch_cache_rad_dino_14x14 \
  --output-dir outputs/iera/falsification_v1 \
  --retained-grid 14 \
  --proposal-grid 4 \
  --shots 1 3 5 10 \
  --rhos 0.7 \
  --episodes 100 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --train-shot 3 \
  --base-validation-episodes 25 \
  --max-train-steps 300 \
  --episode-batch-size 4 \
  --device cuda
```

### 3. Five-seed SMS-budget sweep

This stage trains only adapter-only and full IERA. Random ProtoNet is rescored;
the first five nuisance-balanced, REx, and cheap-baseline results are reused as
fixed reference points. The BioMedCLIP text baseline remains in the comparison
table but is excluded from the RAD-DINO Pareto plot because its SMS scale is not
commensurate.

```bash
PYTHONPATH=. python3 -m experiments.iera.falsification \
  --stage sweep \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --raw-labels ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --rad-cache outputs/iera/patch_cache_rad_dino_14x14 \
  --output-dir outputs/iera/falsification_v1 \
  --retained-grid 14 \
  --proposal-grid 4 \
  --shots 1 3 5 10 \
  --rhos 0.9 0.8 0.7 0.5 0.3 \
  --episodes 100 \
  --seeds 0 1 2 3 4 \
  --train-shot 3 \
  --base-validation-episodes 25 \
  --max-train-steps 300 \
  --episode-batch-size 4 \
  --device cuda
```

Render the primary three-shot figure:

```bash
PYTHONPATH=. python3 -m experiments.iera.plot_pareto \
  --pareto outputs/iera/falsification_v1/sweep/pareto.csv \
  --output outputs/iera/falsification_v1/sweep/pareto_3shot.pdf \
  --shot 3
```

Each stage writes per-seed and summary metrics in its own subdirectory. Learned
stages additionally write model checkpoints, training progress, `pareto.csv`,
and `decision.json`. The primary decision retains IERA only when full IERA has
a positive paired AUROC 95% interval versus adapter-only at no larger mean
fixed-reference SMS; otherwise it selects the simpler constrained adapter.

## Dual-head scoring-only gate

Before training a dual evidence head, isolate the scoring change. This pilot
loads the already trained rho=0.7 adapter for each seed and freezes it. Query
features, Rad-DINO, the 14x14 cache, support choices, and all validation/test
episodes remain unchanged. Validation three-shot episodes choose lambda from
`0, 0.25, 0.5, 0.75, 1` and support-patch temperature from
`0.05, 0.1, 0.2`; test episodes are never used for selection.

```bash
PYTHONPATH=. python3 -m experiments.iera.dual_head_diagnostic \
  --episodes outputs/iera/falsification_v1/episodes.pt \
  --rad-cache outputs/iera/patch_cache_rad_dino_14x14 \
  --adapter-dir outputs/iera/falsification_v1/learned \
  --adapter-rho 0.7 \
  --output-dir outputs/iera/dual_head_pilot_v1 \
  --retained-grid 14 \
  --shots 1 3 5 10 \
  --primary-shot 3 \
  --seeds 0 1 2 3 4 \
  --episodes-per-seed 100 \
  --lambdas 0 0.25 0.5 0.75 1 \
  --patch-temperatures 0.05 0.1 0.2 \
  --query-temperature 0.1 \
  --episode-batch-size 4 \
  --device cuda
```

The four reported variants are global, current local, validation-selected
global/current-local fusion, and validation-selected global/selected-local
fusion. Continue to constrained dual-head training only when `decision.json`
reports `proceed_to_constrained_dual_head`; otherwise revise or stop without
changing resolution or retraining the adapter.
