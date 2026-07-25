The feedback is harsh but mostly correct. **A 2/6 assessment is fair for the current evidence**, and it gives us a much better experimental order. The project is not dead; the framing and controlled support-swap protocol still have value.

FewSTAB and MetaCoCo confirm that spurious correlations in few-shot episodes are already an established problem. Our defensible distinction is narrower: **hold every query fixed and intervene only on support composition**, thereby isolating adaptation-time support sensitivity rather than general support–query distribution shift. [FewSTAB](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/10418.pdf) and [MetaCoCo](https://proceedings.iclr.cc/paper_files/paper/2024/hash/6d7a9f292360193eb530d693f7941c73-Abstract-Conference.html) must appear prominently in the introduction. ProtoCLIP is also important adjacent work, although it is zero-shot rather than episodic few-shot learning. [ProtoCLIP](https://arxiv.org/abs/2604.18444)

## One additional critical issue I found

The current “ProtoNet” is not actually a standard binary ProtoNet. In [model.py](/Users/arnurartykbay/projects/paper/wacv/experiments/iera/model.py:173), both frozen ProtoNet and learned-uniform construct only a **positive prototype**:

\[
z(q)=\operatorname{sim}(q,p^+).
\]

The negative supports are ignored by these classifiers. A proper binary ProtoNet should use:

\[
z(q)=\operatorname{sim}(q,p^+)-\operatorname{sim}(q,p^-).
\]

This may be a major reason Pneumothorax is near chance. Generic chest anatomy can dominate positive similarity unless it is contrasted against a negative prototype. We should fix this before concluding that the backbone cannot detect Pneumothorax.

## What the reviewer is completely right about

- Pneumothorax AUROC around 0.48 makes its invariance result clinically uninterpretable.
- Training directly toward a 30% SMS reduction makes the achieved reduction expected, not independently surprising.
- The meaningful result is the **AUROC–SMS trade-off**, not whether one arbitrary budget passes.
- Native \(14\times14\) ViT patch tokens should be tested immediately; \(4\times4\) is too coarse.
- Cheap baselines could eliminate the need for IERA.
- Within-method normalized SMS can be improved by increasing logit dispersion.
- IERA needs component ablations because unanchored IERA itself is not invariant.
- Five seeds are inadequate for confident comparative conclusions.
- The explicit-negative label policy creates a selected cohort and needs sensitivity analysis.

## Important nuance

Balanced support sampling is not a complete replacement for the proposed problem. It requires nuisance labels, adequate examples from every environment, and control over support selection at deployment. Our method aims to remain robust when the available supports are naturally imbalanced. Nevertheless, balanced sampling is an essential **oracle baseline**.

Similarly, text orthogonalization may fail when “support devices” are visually diverse or entangled with genuine pathology, but that makes it a powerful falsification baseline—not something we can omit.

## Revised experimental order

### Experiment 1: establish a real Pneumothorax detector

Run a small factorial diagnostic:

- Current positive-only head versus proper \(p^+-p^-\) binary ProtoNet.
- \(4\times4\) versus native \(14\times14\) patch tokens.
- BioMedCLIP plus one CXR-specialized backbone.
- No SMS regularization and no IERA initially.

The immediate objective is not \(0.70\) as an arbitrary magic number, but clearly above-chance, statistically stable Pneumothorax discrimination. If no reasonable baseline learns Pneumothorax, the pair cannot support the main experiment.

### Experiment 2: falsification baselines

Using the repaired detector, compare:

1. Random support sampling.
2. Nuisance-balanced support sampling.
3. Mean-difference nuisance projection.
4. Text-direction orthogonalization.
5. Group-DRO or REx.
6. Constrained support adapter without IERA.
7. Anchoring without the support adapter.
8. Full Anchored IERA.

The most important ablation is **constrained adapter without IERA**. If it matches full IERA, remove the evidence-ratio mechanism and simplify the paper.

### Experiment 3: Pareto frontier

Sweep:

\[
\rho\in\{0.9,0.8,0.7,0.5,0.3\},
\qquad
\operatorname{SMS}_{A}\leq\rho\operatorname{SMS}_{U}.
\]

Plot:

\[
\text{AUROC change versus SMS reduction}
\]

for each pathology–confounder pair. The likely scientific result is that:

- Edema–Cardiomegaly has a favorable frontier.
- Pneumothorax–Devices has a sharp frontier caused by genuine feature entanglement.

That is potentially more interesting than claiming one universally successful model.

### Experiment 4: repair the metric

Report four complementary quantities:

- Raw logit SMS.
- SMS normalized using a **fixed learned-uniform reference scale**, rather than each method’s own standard deviation.
- Query-ranking instability, such as Kendall disagreement under support swaps.
- Threshold-based flip rate.

Every stability result must remain paired with AUROC to rule out constant prediction collapse.

### Experiment 5: label-policy and external validation

- Report every four-stratum patient count.
- Compare explicit-negative-only against a documented blank-as-negative sensitivity analysis.
- Manually inspect a small sample if feasible.
- Replicate on CheXpert.
- Increase to ten training seeds for the final experiment.

## Bottom line

We should accept this review, with one modification: **fix the binary classifier head before blaming only patch resolution**.

The next run should not contain ranking distillation or another elaborate module. First determine whether a true positive-minus-negative ProtoNet with native ViT tokens produces a credible Pneumothorax signal. Then test whether IERA survives cheap baselines and component ablations. If it does, the Pareto-frontier paper could be strong; if it does not, SMS and the fixed-query support-intervention protocol may still survive as the main contribution.