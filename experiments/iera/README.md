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

## Later experiments

Only if Experiment 1 establishes a credible detector:

1. run the cheap falsification baselines and component ablations from
   `addition.md`, especially constrained adapter without IERA;
2. sweep the SMS constraint to produce the AUROC-SMS Pareto frontier;
3. report raw SMS, learned-uniform-reference-normalized SMS, ranking
   instability, and threshold flip rate;
4. run explicit-negative versus blank-as-negative sensitivity analysis and
   external validation.

The previous adaptive Anchored-IERA implementation remains in `run.py` for this
later stage. It uses exact evaluated normalized SMS during training, adaptive
dual ascent, a support-only residual adapter, frozen query projection, at least
25 base-validation episodes per pair, and constrained checkpoint selection.
