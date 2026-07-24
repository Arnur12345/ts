# IERA pilot

This folder implements the go/no-go pilot from `test-6.pdf` using BioMedCLIP
patch tokens and the native multi-label manifest produced for PAIR-FSL.

The PDF specifies the positive/negative log-mean-exp evidence ratio and the
cross-environment soft minimum, but does not define the final prototype or
query readout. The minimal readout implemented here is documented explicitly:

1. evidence-ratio attention weights target-positive support patches;
2. their weighted mean is the target prototype;
3. the same evidence ratio weights query patches;
4. a learned-scale cosine similarity gives the binary logit.

Exact positive patch self-matches are masked. The frozen encoder is never
updated. A small linear projection and the five positive scalar parameters
`tau`, `tau_attention`, `tau_query`, `beta`, and `gamma` are meta-trained on
base target/confounder pairs that exclude all four pilot targets.

The pilot evaluates:

- standard positive prototype;
- full two-environment IERA;
- `E=1`;
- no negative bank;
- mean across environments instead of the robust soft minimum.

The four fixed stress pairs are pneumothorax/support devices,
edema/cardiomegaly, pleural effusion/atelectasis, and
pneumonia/consolidation. Every episode contains all four `(c,d)` strata,
verified target-positive and target-negative supports in both environments,
fixed queries, and no repeated patient. Blank/uncertain targets are never
converted to negatives.

Because MIMIC's official validation/test partitions are too small for the
required four-way co-label strata, the pilot creates a deterministic 70/15/15
patient split over the full cache using `--split-seed 2026`. A subject and all
of their studies occur in exactly one partition.

Reported metrics include AUROC, AUPRC, nuisance-specific and worst-nuisance
performance, support-mixture sensitivity (SMS), false-positive activation on
`c- d+`, Brier score, ECE, and a preregistered decision file.

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
  --output-dir outputs/iera/pilot_v1 \
  --shots 1 3 5 10 \
  --episodes 100 \
  --queries-per-stratum 1 \
  --seeds 0 1 2 3 4 \
  --train-steps 100 \
  --device cuda
```

For a fast server smoke test, use `--shots 1 3 --episodes 2 --seeds 0
--train-steps 2`. Results are saved as `per_seed_metrics.csv`,
`summary_metrics.csv`, `decision.json`, `experiment.json`, the learned model,
and exact episode indices.

`--raw-labels` is mandatory because the earlier residual manifest normalized
blank labels to zero. IERA reloads the original table and treats only explicit
0/1 values as verified target status; blank and -1 values are excluded.
