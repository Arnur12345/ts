Fix these now, in this order:

1. Change the anchored loss to hinge-only:

\[
L=L_{\text{cls}}+\lambda\max(0,L_{\text{SMS}}-0.7L_{\text{SMS}}^{\text{uniform}})
\]

Remove the additional raw invariance penalty.

2. Initialize Anchored IERA from the trained `learned_uniform` checkpoint instead of training from scratch.

3. Initially freeze the projection/query head; train only evidence attention and anchor parameters. Optionally unfreeze later with 10× lower learning rate.

4. Increase early-stopping validation from 2 to at least 25 episodes per pair.

5. Select the highest worst-nuisance AUROC checkpoint that satisfies the SMS budget—not the lowest combined loss.

6. Allow ≤0.01 AUROC loss in the decision rule.

Run this controlled version before changing resolution. If Pneumothorax remains weak, then move from 4×4 to at least 14×14 retained patch tokens using 512×512 inputs.