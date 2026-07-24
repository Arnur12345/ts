# IERA pilot

This folder implements the go/no-go pilot from `test-6.pdf` using BioMedCLIP
patch tokens and the native multi-label manifest produced for PAIR-FSL.

The PDF specifies the positive/negative log-mean-exp evidence ratio and the
cross-environment soft minimum, but does not define the final prototype or
query readout. The minimal readout implemented here is documented explicitly:

1. evidence-ratio attention weights target-positive support patches;
2. their weighted mean is the target prototype;
3. the prototype is compared directly with query patches using a smooth local
   maximum, without support-conditioned query attention;
4. a learned scale converts the local match to a binary logit.

Every patch from the source radiograph is excluded when estimating that
support image's evidence; with one shot, the other environment supplies the
independent positive bank. The frozen encoder is never updated. A small linear projection, evidence temperatures,
scale, and bounded anchor gate are meta-trained on
base-only target/confounder pairs that exclude the pilot target labels.
The positive prototype baseline runs directly in frozen BioMedCLIP patch
space. Learned uniform and unanchored IERA use independent initializations;
Anchored IERA is initialized from the best same-seed learned-uniform
checkpoint.

The pilot evaluates:

- frozen BioMedCLIP ProtoNet;
- learned-uniform ProtoNet with the same projection and local readout;
- unanchored two-environment IERA;
- Anchored IERA with explicit support-consistency training.

Anchored IERA interpolates from the learned-uniform prototype toward IERA's
evidence-weighted proposal. A support-dependent gate is bounded by
`--alpha-max 0.25`, preventing attention from replacing the stable prototype.
Its hinge-only objective penalizes sensitivity only when normalized SMS exceeds
`--invariance-budget 0.7` times the fixed learned-uniform reference. Training
SMS is exactly the reported quantity: mean absolute panel-logit shift divided
by pooled panel-logit standard deviation. A projected dual-ascent Lagrange
multiplier adapts during training instead of remaining fixed. There is no
additional raw invariance penalty. The learned-uniform projection and query
head remain frozen. An anchored-only bottleneck residual adapter modifies
support tokens, while query tokens remain in the original frozen space.

The four fixed stress pairs are pneumothorax/support devices,
edema/cardiomegaly, pleural effusion/atelectasis, and
pneumonia/consolidation. Every episode contains all four `(c,d)` strata,
verified target-positive and target-negative supports in both environments,
fixed queries, and no repeated patient. Blank/uncertain targets are never
converted to negatives. A pair is skipped unless every stratum has at least
`--min-stratum-patients` patients in both validation and test; exact counts are
written to `eligibility.json`.

Because MIMIC's official validation/test partitions are too small for the
required four-way co-label strata, the pilot creates a deterministic 70/15/15
patient split over the full cache using `--split-seed 2026`. A subject and all
of their studies occur in exactly one partition.

Reported metrics include AUROC, AUPRC, nuisance-specific and worst-nuisance
performance, false-positive activation on `c- d+`, Brier score, ECE, and a
preregistered decision file. SMS is computed from uncalibrated logits and also
normalized by the method's pooled panel-logit standard deviation; support-swap
flip rate and panel error metrics use the validation-selected calibrated
threshold. Calibration can no longer suppress the primary SMS value.

## 1. Use the existing low-resolution patch cache

Use `outputs/iera/patch_cache_4x4`; it remains valid and must not be rebuilt for
this repair. Do not build 14x14/512x512 tokens unless the 30% constraint first
becomes feasible on every same-seed base-validation run.

## 2. Run the pilot

```bash
PYTHONPATH=. python3 -m experiments.iera.run \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --patch-cache outputs/iera/patch_cache_4x4 \
  --raw-labels ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --output-dir outputs/iera/adaptive_anchor_v3 \
  --shots 1 3 5 10 \
  --episodes 100 \
  --queries-per-stratum 1 \
  --seeds 0 1 2 3 4 \
  --max-train-steps 1000 \
  --validation-interval 25 \
  --early-stopping-patience 5 \
  --early-stopping-episodes-per-pair 25 \
  --alpha-max 0.25 \
  --support-adapter-dim 16 \
  --invariance-budget 0.7 \
  --lagrange-initial 1.0 \
  --lagrange-learning-rate 0.05 \
  --lagrange-max 100 \
  --device cuda
```

Each value in `--seeds` now controls an independent model initialization,
meta-training run, validation episode set, and test episode set. Models are
saved separately by method and seed. The existing patch cache remains valid
after these fixes and does not need to be rebuilt.

Anchored IERA starts from and measures its budget against the fixed best
learned-uniform checkpoint from the same seed. Checkpoint selection uses at
least 25 episodes per pair from a separate patient-disjoint set of base-class
episodes, never either evaluated pair. Among checkpoints satisfying the SMS
budget on every validation pair, it selects the highest worst-nuisance AUROC,
not the lowest combined loss. `training_curves.csv` records all selection
diagnostics, and every saved model contains the restored selected checkpoint.
The training curves also record the signed constraint violation and adaptive
Lagrange multiplier.

For a fast server smoke test, use `--shots 1 3 --episodes 2 --seeds 0
--max-train-steps 25 --validation-interval 5 --early-stopping-patience 2`.
The 25 validation episodes per pair remain enforced. Results are saved as `per_seed_metrics.csv`,
`summary_metrics.csv`, `decision.json`, `experiment.json`, the learned model,
and exact episode indices.

`--raw-labels` is mandatory because the earlier residual manifest normalized
blank labels to zero. IERA reloads the original table and treats only explicit
0/1 values as verified target status; blank and -1 values are excluded.

`decision.json` requires normalized SMS at or below 0.70 times learned uniform
on both eligible pairs—a real 30% reduction—while allowing at most a 0.01
decrease in ordinary or worst-nuisance AUROC. It also records whether every
same-seed base-validation run found a budget-feasible checkpoint. High
resolution is blocked when that field is false. Only when it is true and
Pneumothorax still misses the evaluation budget should 512x512 inputs with at
least 14x14 retained patch tokens be tested.
