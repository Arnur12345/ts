# PAIR-FSL residual experiments

This folder implements the pilot in `proposal_residuals.md` with frozen
BioMedCLIP image features. It deliberately does not use support or query
reports, patch routing, generative counterfactuals, or end-to-end training.

## What is evaluated

The same model is evaluated under two protocols:

- `single_label`: studies with exactly one certain positive label; 3-way
  softmax classification proves that the residual representation is useful in
  a conventional few-shot setting.
- `multi_label`: native comorbid labels are retained; each novel disease is an
  independent one-vs-rest target scored with a sigmoid. Uncertain targets are
  masked rather than silently treated as negative.

Both protocols are patient-disjoint inside every episode. The 1/3/5-shot
supports are nested prefixes, and all methods reuse the same saved episodes.
For each `K`, the runner evaluates balanced `K+/K-` support and realistic
`K+/M-` support (`M=20` by default).

Implemented arms:

1. `positive_prototype`
2. `global_negative_centroid`
3. `random_residual`
4. `metadata_matched_residual`
5. `full_embedding_matched_residual`
6. `anatomy_matched_residual`
7. `shuffled_anatomy_match`

The anatomy matcher projects out the episode's preliminary disease direction
before nearest-control retrieval. This is the frozen-feature approximation to
matching in a disease-insensitive space. Metadata matching uses AP/PA view and,
when supplied, sex and age. The shuffled arm breaks the association between
matching features and control embeddings.

Temperature and the multi-label threshold are selected once on
`validation_novel`; test-novel results never select hyperparameters. Outputs
include per-seed, per-class, aggregate, and compressed per-query predictions.

## Build the native multi-label cache

The old `biomedclip_pairs_7000.pt` cache contains only exactly-one-positive
studies, so it is suitable only for `--regime single_label`. Build a new cache
from the full `study_manifest.csv.gz` to run both regimes:

```bash
PYTHONPATH=. python3 -m experiments.residuals.build_cache \
  --data-root ~/data/mimic-cxr-jpg-2.1.0 \
  --study-manifest ~/data/mimic-cxr-jpg-2.1.0/protocols/mimic-cxr-fsl-v1/study_manifest.csv.gz \
  --output-manifest outputs/residuals/multilabel_manifest.csv \
  --output outputs/residuals/biomedclip_multilabel.pt
```

If a MIMIC patient table is available, add
`--patients /path/to/patients.csv.gz`; this enables age/sex matching in addition
to view matching. Use `--selection-only` first to inspect the retained cohort.
Certain studies with zero, one, or multiple positive labels are retained so
target-negative controls and real comorbid cases remain available.

## Run both protocols

```bash
PYTHONPATH=. python3 -m experiments.residuals.run \
  --embeddings outputs/residuals/biomedclip_multilabel.pt \
  --manifest outputs/residuals/multilabel_manifest.csv \
  --output-dir outputs/residuals/pair_fsl_v1 \
  --regime both \
  --shots 1 3 5 \
  --abundant-controls 20 \
  --episodes 500 \
  --seeds 0 1 2 3 4 \
  --device cuda
```

For a short pipeline check, add `--episodes 2 --queries-per-class 1`. A server
run can be resumed only when its episode-generation settings are unchanged;
otherwise use a new output directory.

The decisive comparison is
`anatomy_matched_residual > random_residual > positive_prototype`, especially
at one shot. `shuffled_anatomy_match` should erase the matched-control gain.
