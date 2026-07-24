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
Each IERA ablation is initialized and trained independently. The positive
prototype baseline runs directly in frozen BioMedCLIP patch space.

The pilot evaluates:

- frozen BioMedCLIP ProtoNet;
- learned-uniform ProtoNet with the same projection and local readout;
- unanchored two-environment IERA;
- Anchored IERA with explicit support-consistency training.

Anchored IERA interpolates from the learned-uniform prototype toward IERA's
evidence-weighted proposal. A support-dependent gate is bounded by
`--alpha-max 0.25`, preventing attention from replacing the stable prototype.
Its objective adds normalized support-panel consistency and penalizes
sensitivity above `--invariance-budget 0.7` times the fixed, independently
trained learned-uniform reference. No other architectural controls run during
this rescue cycle.

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

## 1. Build the patch cache

The existing global embedding cache cannot support patch attention. Build a
7x7 pooled patch-token cache aligned to the same 167,183-row manifest:

```bash
PYTHONPATH=. python3 -m experiments.iera.patch_cache \
  --data-root ~/data/mimic-cxr-jpg-2.1.0 \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --output-dir outputs/iera/patch_cache \
  --pool-grid 7 \
  --batch-size 128 \
  --device cuda
```

The cache is written incrementally as a float16 memory map and flushed every
20 batches. Its expected size is approximately 12 GB for 49 x 768 tokens; the
exact width depends on the BioMedCLIP visual trunk. Progress metadata is saved
with every flush, so rerunning the identical command resumes an interrupted
cache instead of starting again.

## 2. Run the pilot

```bash
PYTHONPATH=. python3 -m experiments.iera.run \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --patch-cache outputs/iera/patch_cache \
  --raw-labels ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --output-dir outputs/iera/anchored_v1 \
  --shots 1 3 5 10 \
  --episodes 100 \
  --queries-per-stratum 1 \
  --seeds 0 1 2 3 4 \
  --max-train-steps 1000 \
  --validation-interval 25 \
  --early-stopping-patience 5 \
  --alpha-max 0.25 \
  --invariance-weight 1.0 \
  --invariance-budget 0.7 \
  --device cuda
```

Each value in `--seeds` now controls an independent model initialization,
meta-training run, validation episode set, and test episode set. Models are
saved separately by method and seed. The existing patch cache remains valid
after these fixes and does not need to be rebuilt.

Every learned method is trained independently. Anchored IERA's budget uses the
fixed best learned-uniform checkpoint from the same seed. Checkpoint selection uses a
separate patient-disjoint set of base-class episodes, never either evaluated
pair. `training_curves.csv` records train/validation loss, and every saved model
contains the restored best checkpoint and its best step rather than the final
optimization state.

For a fast server smoke test, use `--shots 1 3 --episodes 2 --seeds 0
--max-train-steps 25 --validation-interval 5 --early-stopping-patience 2`. Results are saved as `per_seed_metrics.csv`,
`summary_metrics.csv`, `decision.json`, `experiment.json`, the learned model,
and exact episode indices.

`--raw-labels` is mandatory because the earlier residual manifest normalized
blank labels to zero. IERA reloads the original table and treats only explicit
0/1 values as verified target status; blank and -1 values are excluded.

`decision.json` requires Anchored IERA to beat the learned-uniform baseline on
normalized SMS while retaining ordinary and worst-nuisance AUROC on both
eligible pairs. Failure to reduce Pneumothorax SMS after this direct constraint
is reported as evidence of pathology-versus-device non-identifiability.
