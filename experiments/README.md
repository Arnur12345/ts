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
