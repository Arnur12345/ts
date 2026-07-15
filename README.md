# MIMIC-CXR ProtoNet vs. affine subspaces

This is a small, training-free pipeline for the first experiment:

`ProtoNet vs Random Subspace vs Global Base Subspace vs Oracle Subspace`

BioMedCLIP is frozen. Images are resized once, embeddings are extracted once, and all 500-episode evaluations use fast tensor operations.

## Important input detail

The recommended inputs are the official `mimic-cxr-2.0.0-metadata.csv.gz` and `mimic-cxr-2.0.0-chexpert.csv.gz`. The metadata provides image IDs and views; the CheXpert file provides the 14 study-level targets. The loader reconstructs paths under `files/p10/...` automatically. The Kaggle `Untitled.csv`/augmented CSV schema is also supported, but it is not required. Do not create ground truth with report keyword matching.

The copy in this workspace is treated only as a schema sample. Counts and oracle feasibility are checked later against the full downloaded dataset.

The preparation step:

- expands only the `AP` and `PA` lists (lateral images are ignored);
- joins each path to its label using the `s<study_id>` path component;
- drops uncertain studies by default;
- retains studies with exactly one positive label;
- removes duplicate paths and corrupt/missing files;
- letterboxes each image to 224 x 224 without cropping the chest;
- writes class counts so the retained subset is auditable.

## Setup

Python 3.10+ is required. Install PyTorch using the command appropriate for your CUDA version, then install the project:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gpu]"
```

## 1. Clean and resize

After extracting the Kaggle archive, point `--data-root` at the directory containing `files/p10/...`:

```bash
mimic-prepare \
  --input-csv data/raw/official_data_iccv_final/mimic-cxr-2.0.0-metadata.csv.gz \
  --labels-csv data/raw/official_data_iccv_final/mimic-cxr-2.0.0-chexpert.csv.gz \
  --data-root data/raw/official_data_iccv_final \
  --output-dir data/processed \
  --size 224
```

The metadata and label files keep their `2.0.0` filenames even when downloaded from the PhysioNet 2.1.0 release page. Both plain `.csv` and compressed `.csv.gz` inputs are supported directly.

Outputs are `data/processed/manifest.csv`, `prepare_summary.json`, and the resized image cache. The default keeps every valid image. `--max-per-class 5000` is available for a quicker pilot, but the episodic evaluator is already class-balanced and does not need oversampling.

If you intentionally want uncertain `-1` labels treated as negatives, pass `--uncertain-policy negative`. The safer default is to drop those studies.

## 2. Extract BioMedCLIP embeddings once

```bash
mimic-embed \
  --manifest data/processed/manifest.csv \
  --output data/embeddings/biomedclip.pt \
  --batch-size 256 \
  --workers 8
```

CUDA inference uses `torch.inference_mode`, FP16 autocast, pinned-memory loading, channels-last input, and TF32 matrix multiplication. Increase the batch size until GPU memory is well utilized. Model weights download on the first run only.

## 3. Run the first experiment

```bash
mimic-evaluate \
  --embeddings data/embeddings/biomedclip.pt \
  --split-json configs/mimic_split_seed_2026.json \
  --episodes 500 \
  --shots 1 3 5 \
  --queries 1 \
  --seeds 0 1 2 3 4 \
  --oracle-size 512 \
  --ranks 1 2 4 8 \
  --betas 0.1 0.25 0.5 0.75
```

The evaluator uses 8 base, 3 validation, and 3 held-out test classes. The included split is a deterministic random split, not a claim about the unpublished exact class assignment in Mahawar et al. Replace the JSON if you obtain their original split.

Rank and hybrid weight beta are selected using mean validation macro AUROC separately for each shot setting, then frozen for test reporting. Results are written to:

- `outputs/first_experiment/per_seed_all_settings.csv`
- `outputs/first_experiment/test_selected_summary.csv`
- `outputs/first_experiment/experiment.json`

Each 3-way episode has nested 1/3/5-shot supports and one shared query per class. The oracle pool contains 512 additional labeled images per novel class and is sampled before support/query generation, so it is disjoint by construction.

The four distances are:

```text
ProtoNet:          ||z - mu_c||^2
Hybrid subspace:   ||z - mu_c||^2 - beta ||B_c^T (z - mu_c)||^2
```

For the global method, a single basis is learned from within-class residuals of base classes. Each base covariance is normalized before averaging, so abundant classes do not dominate. The default samples at most 4,096 images per base class for speed; use `--base-samples-per-class 0` to use all base images. Random and oracle methods use class-specific bases. Oracle PCA uses only the separate oracle pool; the affine center `mu_c` always comes from the 5-shot support set.

If the embedding tensor does not fit in GPU memory, add `--keep-features-cpu`; episode tensors and linear algebra still run on the selected device.

## Verify

```bash
python -m unittest discover -s tests -v
```

The tests check single-label filtering, oracle/support/query disjointness, and a complete synthetic 8/3/3 run.

## Text-conditioned experiment

Text means one fixed description per class, not patient radiology reports. Cache the 14 normalized BioMedCLIP text embeddings once:

```bash
mimic-embed-text \
  --descriptions configs/mimic_class_descriptions.json \
  --output data/embeddings/biomedclip_text.pt
```

Then compare ProtoNet, ProtoNet + text, oracle subspace, oracle subspace + text, and their shuffled-text controls:

```bash
mimic-evaluate-text \
  --embeddings data/embeddings/biomedclip.pt \
  --text-embeddings data/embeddings/biomedclip_text.pt \
  --split-json configs/mimic_split_seed_2026.json \
  --output-dir outputs/text_experiment \
  --episodes 500 \
  --seeds 0 1 2 3 4 \
  --shots 1 3 5 \
  --queries 1 \
  --oracle-size 256 \
  --ranks 1 2 4 8 \
  --alphas 0 0.1 0.25 0.5 0.75 \
  --betas 0 0.1 0.25 0.5 0.75 1
```

Rank, alpha, and beta are selected using validation macro AUROC separately for every method and shot setting. The test query sets are shared across nested 1/3/5-shot supports. Shuffled text uses a class derangement with no fixed labels. Inspect `test_selected_summary.csv` for the main results and `semantic_sanity_summary.csv` for paired correct-minus-shuffled AUROC. A non-positive shuffled-control delta means there is no evidence that any text gain is semantic.
