# MIMIC-CXR few-shot protocol v1

This protocol creates a fixed 14-label, class-disjoint and patient-disjoint
benchmark for medical vision-language few-shot learning. It is designed for
paired support-dependence experiments: every method receives byte-identical
episode files, support images, labels, query images, and class ordering.

## Fixed design

| Item | Protocol |
|---|---|
| Label source | Official `mimic-cxr-2.0.0-chexpert.csv.gz` |
| Image metadata | Official `mimic-cxr-2.0.0-metadata.csv.gz` |
| Patient split | Official `mimic-cxr-2.0.0-split.csv.gz` |
| Classes | All 14 CheXpert labels |
| Class partitions | 8 base / 3 validation-novel / 3 test-novel |
| Episode size | 3-way, 1-/3-/5-shot, 15 queries per class |
| Repetitions | 500 episodes for each of 5 seeds |
| Views | Frontal PA or AP; one image per study, PA preferred |
| Label policy | Exactly one certain positive; any `-1` study is excluded |
| Reuse | Samples may recur in different episodes, never within an episode |

The 14 targets are the official CheXpert label set. Strictly speaking, “No
Finding” is not a pathology and “Support Devices” is not a disease, but both
are retained because they are part of the official 14-label target space.

### Class partitions

- Base (official train patients): Atelectasis, Pleural Effusion,
  Consolidation, Pleural Other, Pneumothorax, Enlarged Cardiomediastinum,
  Lung Opacity, Lung Lesion.
- Validation-novel (official validate patients): Edema, Support Devices,
  Pneumonia.
- Test-novel (official test patients): No Finding, Fracture, Cardiomegaly.

The assignment comes from shuffling the canonical label order with
`random.Random(2026).shuffle`, then taking the first 8, next 3, and final 3
classes. It was fixed independently of model results and is recorded in
`configs/mimic_cxr_protocol_v1.json`. It must not be changed after inspecting
test results. A future alternative assignment is a new named protocol, not an
edit to v1.

## Why the eligibility policy is single-label

A 3-way softmax episode assumes one ground-truth class per image. MIMIC-CXR is
natively multi-label, so directly placing co-morbid images in a 3-way episode
can make two episode classes simultaneously correct. Protocol v1 avoids that
ambiguity by retaining studies with exactly one certain positive among the 14
labels. Empty label cells are treated as zero, and studies containing any
uncertain (`-1`) label are excluded. The full canonical study manifest retains
all normalized labels and exclusion reasons for auditing.

This defines a clean classification benchmark; it is not a prevalence study
and should not be used to estimate clinical disease frequencies. A later
multi-label protocol should use per-class binary tasks and needs a separate
name and analysis.

## Leakage controls

There are two levels of patient separation:

1. Base, validation-novel, and test-novel use official train, validate, and
   test patients, respectively. The builder fails if one subject appears in
   more than one official split.
2. Within each episode, every support and query item comes from a different
   patient, including across its three classes. Support and query therefore
   cannot be alternate studies or views from the same patient.

One canonical frontal image is selected for each study. If a study has PA and
AP images, PA is preferred; ties are resolved by lexicographic DICOM ID.
Lateral views are excluded.

The 1-, 3-, and 5-shot settings are nested prefixes of one saved five-shot
support list. All three settings use the same query list. Thus shot-scaling
comparisons change only the additional supports, not the episode or queries.

## Hyperparameter firewall

- Base data may be used for fitting or meta-training.
- Validation-novel episodes are the only novel-class data allowed for model
  selection, early stopping, temperature selection, prompt choice, thresholds,
  or any other hyperparameter decision.
- Test-novel data must not inform method design or parameter selection. Freeze
  the complete method after validation, then run all five saved test seeds.
- Report every attempted validation configuration. Do not choose a different
  setting per test seed.
- Class descriptions are treated as part of the method and must be frozen
  before opening test results.

For paired method comparisons, join results on `episode_uid`, query
`dicom_id`, and shot. Never let an evaluator resample its own episodes.
Support-label permutation, duplication, and replacement controls should be
stored as deterministic transformations of these episode files; the original
files remain unchanged.

## Saved artifacts

The builder writes:

- `protocol_config.json`: resolved, frozen protocol configuration;
- `study_manifest.csv.gz`: every canonical frontal study, all 14 normalized
  labels, eligibility, and exclusion reason;
- `protocol_samples.csv.gz`: compact model-facing samples restricted to each
  class's assigned official split and tagged with its protocol partition;
- `episodes/<partition>/seed_NNN.jsonl`: saved episodes;
- `episodes/index.csv`: episode-file checksums and counts;
- `build_summary.json`: counts used for the dataset table and feasibility audit;
- `protocol.lock.json`: source and artifact SHA-256 hashes;
- `validation_summary.json`: result of full episode validation.

Each JSONL record stores a five-shot support list with `shot_rank`, a shared
query list with `query_index`, episode-local labels, stable subject/study/DICOM
IDs, view, and image path relative to the MIMIC-CXR root.

## Required reporting

Before model experiments, archive `build_summary.json`, `protocol.lock.json`,
and `episodes/index.csv`. Report, for every class and official split, eligible
study and unique-patient counts. If any configured pool has fewer than 20
eligible patients (5 supports + 15 queries), the builder stops; do not silently
reduce query count or change label handling. Any protocol revision should be
versioned and reported.
