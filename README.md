# MIMIC-CXR few-shot protocol

This repository currently implements the fixed dataset protocol for the first
MIMIC-CXR experiments. Model implementations can consume these artifacts in
the next stage without performing their own sampling.

The protocol provides:

- all 14 official CheXpert targets;
- fixed 8 base / 3 validation-novel / 3 test-novel classes;
- nested 1-, 3-, and 5-shot, 3-way episodes;
- 15 queries per class;
- 500 episodes for each of five seeds;
- official patient-disjoint train/validate/test pools plus episode-level
  patient disjointness;
- saved JSONL episodes and SHA-256 checksums shared by every method;
- a validation-only model-selection policy.

The full scientific specification is in
[`docs/mimic_cxr_protocol.md`](docs/mimic_cxr_protocol.md), and the frozen
machine-readable configuration is
[`configs/mimic_cxr_protocol_v1.json`](configs/mimic_cxr_protocol_v1.json).

## Install

Python 3.10 or newer is sufficient; protocol generation has no third-party
runtime dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e . --no-build-isolation
```

## Build on the GPU server

Given the dataset layout in the project description:

```bash
mimic-build-protocol \
  --data-root ~/data/mimic-cxr-jpg-2.1.0 \
  --output-dir ~/data/mimic-cxr-jpg-2.1.0/protocols/mimic-cxr-fsl-v1
```

The command verifies that every selected JPG exists, checks source schemas and
official patient separation, writes all episodes once, hashes them, and then
validates every saved support/query item. It refuses to overwrite a non-empty
output directory.

For a metadata-only dry run when the JPG tree is unavailable, add
`--skip-image-check`. Do not use that flag for the final archived protocol.

## Validate an existing build

```bash
mimic-validate-protocol \
  ~/data/mimic-cxr-jpg-2.1.0/protocols/mimic-cxr-fsl-v1
```

## Consume an episode

Each JSONL record contains the maximum five-shot support and one shared query
set. Select nested supports by rank:

```python
import json
from mimic_cxr_protocol.protocol import support_for_shot

with open("episodes/test_novel/seed_000.jsonl") as handle:
    episode = json.loads(next(handle))

support_1 = support_for_shot(episode, 1)
support_3 = support_for_shot(episode, 3)
support_5 = support_for_shot(episode, 5)
queries = episode["query"]
```

Model code should join image paths to the dataset root and must not resample,
reorder, or replace items except in an explicitly named counterfactual arm.
Use `protocol_samples.csv.gz`, not the broader audit rows in
`study_manifest.csv.gz`, as the only model-facing sample pool.

## Test

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests build two complete synthetic protocols, prove byte-identical episode
generation, validate nested shots and patient separation, and verify checksum
tamper detection.

## BioMedCLIP image-report embeddings

Build a 7,000-pair, single-label stratified subset and embed each original JPG
and its original MIMIC report:

```bash
pip install -e ".[embedding]" --no-build-isolation

mimic-embed-pairs \
  --data-root ~/data/mimic-cxr-jpg-2.1.0 \
  --protocol-dir ~/data/mimic-cxr-jpg-2.1.0/protocols/mimic-cxr-fsl-v1 \
  --subset-manifest outputs/biomedclip_pairs_7000.csv \
  --output outputs/biomedclip_pairs_7000.pt
```

The subset uses at most one study per patient. Scarce classes are retained and
unused quota is redistributed. Use `--selection-only` to inspect the selected
manifest before loading BioMedCLIP.

The saved report vectors come from the radiology report associated with each
study, not from prompts such as “a picture with pneumonia.” Query reports often
state the diagnosis, so they must not be provided to an image-only test-time
classifier; use them only in an explicitly labeled report-aware experiment.

## Run all embedding experiments

```bash
PYTHONPATH=. python3 -m experiments.run \
  --embeddings outputs/biomedclip_pairs_7000.pt \
  --manifest outputs/biomedclip_pairs_7000.csv \
  --output-dir outputs/experiments
```

See [`experiments/README.md`](experiments/README.md) for method definitions and
the generated overall, per-seed, and per-class metric files.

For the diagnostic rerun with nested patient-disjoint episodes, rotating class
splits, saved per-query predictions, and no new architectures, run:

```bash
PYTHONPATH=. python3 -m experiments.run_controls \
  --embeddings outputs/biomedclip_pairs_7000.pt \
  --manifest outputs/biomedclip_pairs_7000.csv \
  --output-dir outputs/controls_v2
```
