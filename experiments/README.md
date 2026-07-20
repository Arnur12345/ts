# Embedding experiments

Each method is implemented in its own short file. All methods use the same
saved subset episodes and never read query reports.

- `text_only_zero_shot.py`: zero visual-support classification from real
  support-report prototypes.
- `visual_protonet.py`: visual prototypes only.
- `protonet_text.py`: visual and report prototypes.
- `lp_text.py`: learned visual linear probe plus class-wise report logits.
- `tip_adapter_f.py`: fine-tuned visual cache plus report logits.
- `proker.py`: proximal kernel ridge correction.
- `tcla.py`: final-layer TCLA ablation. Full TCLA needs intermediate image
  layers and prompt ensembles, which are absent from the saved embeddings.

`Tip-Adapter-F` fine-tunes cached embeddings without image augmentation because
only frozen embeddings were saved. `TCLA-final-layer` uses the final-layer mean
prototype approximation. Both names are recorded explicitly in the outputs.

Run from the repository root:

```bash
PYTHONPATH=. python3 -m experiments.run \
  --embeddings outputs/biomedclip_pairs_7000.pt \
  --manifest outputs/biomedclip_pairs_7000.csv \
  --output-dir outputs/experiments
```

The runner creates fixed 3-way, 1/3/5-shot subset episodes with one query per
class, 500 episodes and five seeds. Temperature calibration is selected using
validation-novel classes only. It writes:

- `overall_metrics.csv`: mean and standard deviation across seeds for AUROC,
  AUPRC, macro F1, accuracy, NLL, and 15-bin calibration error;
- `per_class_metrics.csv`: the same one-vs-rest metrics for every test class;
- `per_seed_metrics.csv` and `per_class_per_seed.csv`: raw seed-level results;
- `subset_episodes.pt`: identical episodes reused by every method;
- `experiment.json`: experiment metadata.

Every text-aware method also gets a deranged, shuffled-text control. These
experiments use report prototypes derived from the labeled support studies;
they are therefore “zero visual-shot,” not conventional prompt-only zero-shot.

## Control-only rerun

Run this on the GPU server from `~/ts`; it does not train a new architecture:

```bash
PYTHONPATH=. python3 -m experiments.run_controls \
  --embeddings outputs/biomedclip_pairs_7000.pt \
  --manifest outputs/biomedclip_pairs_7000.csv \
  --output-dir outputs/controls_v2
```

This run uses ten seeds, 500 episodes per seed, and five rotating 8-base /
3-validation-novel / 3-test-novel splits. The five test folds cover all 14
classes. Each episode stores one query and an ordered five-support set, so the
same query is used at every shot and `S1` is nested in `S3`, which is nested in
`S5`. Every episode contains 18 distinct patients at five-shot.
Because these controls operate on frozen embeddings, the eight base classes
define each split but are not used for an additional training stage.

Only these diagnostics run:

- report-prototype text-only;
- visual ProtoNet with independent supports;
- ProtoNet + report prototypes;
- ProtoNet + shuffled reports, using the unshuffled temperature;
- visual ProtoNet with cyclically permuted support labels;
- visual ProtoNet with its first support duplicated instead of independent
  3/5-shot supports.

The support-label and duplicated-support controls reuse the visual ProtoNet
temperature. `episode_ids.csv.gz` records subject, DICOM, support, and query
IDs. `per_query_predictions.csv.gz` records raw logits, calibrated
probabilities, targets, predictions, and episode IDs. Aggregate CSVs retain all
requested metrics. `decision.md` and `decision.json` apply the preregistered
paired-control rules; inspect the evidence before accepting the branch.
