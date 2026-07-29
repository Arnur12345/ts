# TRACE kill tests

This folder implements only the two falsification experiments prescribed by
`trace.md`; it does not implement the full temporal atom dictionary or
likelihood decoder.

1. `shrinkage_lda` estimates one full covariance matrix from every unlabeled
   RAD-DINO global embedding belonging to a main-training patient. Ridge is
   selected on the saved validation episodes, and the locked estimator is
   evaluated on the saved test episodes.
2. The temporal pilot orders consecutive studies using MIMIC `StudyDate` and
   `StudyTime`, retains only training-partition patients, and creates another
   patient-disjoint fit/validation/test split inside that partition. A small
   integer translation registers 14x14 RAD-DINO token grids. A linear
   classifier tests disease-change identifiability; a canonical onset-oriented
   residual atom tests whether the signal improves the unchanged 3-shot
   episodes.

The target-specific temporal atom is a diagnostic oracle, not a valid
novel-class model. If the pilot passes, the next implementation must train on
base diseases only and mask the held-out disease name and synonyms.

Run:

```bash
PYTHONPATH=. python3 -m experiments.trace.run \
  --stage both \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --raw-labels ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz \
  --metadata-csv ~/data/mimic-cxr-jpg-2.1.0/mimic-cxr-2.0.0-metadata.csv.gz \
  --rad-global outputs/iera/rad_dino_linear_probe_v1/rad_dino_global.float32.npy \
  --rad-global-metadata outputs/iera/rad_dino_linear_probe_v1/rad_dino_global.json \
  --episodes outputs/iera/falsification_v1/episodes.pt \
  --rad-cache outputs/iera/patch_cache_rad_dino_14x14 \
  --output-dir outputs/trace/kill_tests_v1 \
  --shots 1 3 5 10 \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --retained-grid 14 \
  --pleural-review pending \
  --device cuda
```

`--pleural-review pending` intentionally prevents a numeric result from
silently claiming anatomical localization. Review the saved transition
heatmaps and examples, then rerun with `pass` or `fail`; the transition feature
caches are reused.

