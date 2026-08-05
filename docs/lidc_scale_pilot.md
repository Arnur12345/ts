# LIDC scale-perception pilot

This pipeline isolates 2D physical-scale perception from nodule detection. It
selects 60 four-reader LIDC nodules (one per scan), takes the median reader
largest-axial diameter as ground truth, and chooses the axial slice with the
largest 50%-consensus mask area.

Each source slice is sampled on physical grids at 0.50, 0.75, and 1.00 mm/px,
then emitted as a fixed 448×448 crop. All arms use the same lung window
(center −600 HU, width 1500 HU), cubic kernel, output center, lossless PNG
encoding, and fixed-pixel marker ring. The ring localizes the nodule but is not
a physical reference: its radius and stroke are constant in pixels. Condition
C adds a 10 mm scale bar whose pixel length is computed from the target
spacing. Growth composites place two 448px panels in a padded 896×896 canvas,
preventing anisotropic resize in fixed-square model processors.

## Server setup and stimulus build

`pylidc` ships the annotation database but needs the DICOM files. The builder
overrides pylidc's DICOM root only inside its process, so it does not create or
modify `~/.pylidcrc`. It also restores the removed `np.int`, `np.float`, and
`np.bool` dtype aliases in-process because pylidc 0.2.3 still uses them; the
server's NumPy installation is not downgraded or modified.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[lidc-scale,lidc-models,lidc-frontier]"

lidc-scale-build \
  --data-root /data/lidc/raw \
  --output-dir outputs/lidc-scale-pilot-v1
```

The build refuses to overwrite a non-empty output directory. It writes 180
single-image acquisitions, 180 no-growth acquisition pairs, SHA-256 hashes,
all physical metadata, and `audit/contact_sheet_20.png`.

Stop here and inspect the contact sheet. Confirm that the red ring is centered
on the same nodule, the lung window is correct, anatomy is not transposed, and
the expected nodule footprint changes monotonically with spacing. Record the
audit decision before model inference. Do not treat `built_unreviewed` as a
completed dataset review:

```bash
lidc-scale-audit outputs/lidc-scale-pilot-v1 \
  --reviewer YOUR_NAME \
  --decision pass \
  --notes "All 20 markers and windows checked."
```

The audit command verifies the 20 distinct source images against their hashes,
records the contact-sheet hash and timestamp, and changes the build status to
`audit_passed` or `audit_failed`. It refuses to overwrite a prior review.

## Freeze requests

```bash
lidc-scale-requests outputs/lidc-scale-pilot-v1
```

This creates 4,860 full-factorial requests:

- 60 nodules × 3 acquisitions/pairs × 3 questions × 3 prompt conditions × 3 paraphrases;
- condition A: marker only, no spacing text;
- condition B: marker plus exact spacing text;
- condition C: exact spacing text plus rendered 10 mm scale bar.

It also creates 54 frontier requests as six complete blocks. Each block keeps
all three acquisition variants and all three paraphrases, so acquisition and
paraphrase variance remain identifiable. Increase `--frontier-blocks` if the
API budget permits; nine blocks (81 calls) cover the 3×3 question/condition
grid more cleanly.

## Local models

Both local runs use greedy decoding, a fixed seed, and token-level JSON-schema
constraints. MedGemma is gated on Hugging Face, so accept its license and set
`HF_TOKEN` on the server before loading it.

```bash
lidc-scale-run-local \
  --model qwen3-vl-2b \
  --requests outputs/lidc-scale-pilot-v1/requests.jsonl \
  --output outputs/lidc-scale-pilot-v1/results/qwen3-vl-2b.jsonl

lidc-scale-run-local \
  --model medgemma-4b \
  --requests outputs/lidc-scale-pilot-v1/requests.jsonl \
  --output outputs/lidc-scale-pilot-v1/results/medgemma-4b.jsonl
```

The runner appends one flushed JSONL row per request and skips completed
request IDs on restart. Use `--limit` only for a smoke test.

## Frontier model

The default API arm uses Gemini structured output. Set `GEMINI_API_KEY`, then
run the 54-call complete-block subset. Confirm that the default model name is
still available in your account immediately before the run; if it changes,
pass `--model-id` and preserve the exact returned model identifier with the
artifact.

```bash
lidc-scale-run-gemini \
  --requests outputs/lidc-scale-pilot-v1/frontier_requests.jsonl \
  --output outputs/lidc-scale-pilot-v1/results/gemini-frontier.jsonl \
  --model-id gemini-3.5-flash
```

The API call uses temperature 0, the same fixed seed, PNG bytes, and a JSON
response schema. It retries transient failures but does not write a failed row,
so restarting cannot silently turn a complete nine-call variance block into an
incomplete one.

## Score

```bash
lidc-scale-score \
  outputs/lidc-scale-pilot-v1/results/qwen3-vl-2b.jsonl \
  outputs/lidc-scale-pilot-v1/results/medgemma-4b.jsonl \
  outputs/lidc-scale-pilot-v1/results/gemini-frontier.jsonl \
  --output outputs/lidc-scale-pilot-v1/results/metrics.json
```

The report includes absolute-diameter MAE and bias, mean per-nodule standard
deviation across acquisitions, Q2 accuracy and acquisition flip rate, Q3
false-growth rate, invalid JSON rate, and acquisition/paraphrase variance with
their ratio for every model/question/condition cell.

## Frozen choices and interpretation

- The ground-truth diameter is the median of four reader measurements, not the
  diameter of the consensus mask on one chosen slice.
- PNG is used instead of JPEG, removing compression quality as a nuisance
  variable.
- Only in-plane spacing changes; the axial slice and source pixels are shared.
- Bare-condition failure that vanishes with spacing text is not enough for the
  intended mechanism claim. Q3 under condition B is the load-bearing test.
- A result is scale-specific only when acquisition variance materially exceeds
  paraphrase variance. Report both, including incomplete groups, before making
  that claim.
