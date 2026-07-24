Do one controlled repair cycle before abandoning the idea:

1. **Fix evaluation first**
   - Make the decision require both eligible pairs, not three impossible pairs.
   - Compute support-swap predictions using the calibrated threshold, not raw logit `0`.
   - Keep normalized SMS as the primary sensitivity metric.

2. **Train properly**
   - Replace 50 fixed steps with validation-based early stopping, likely allowing 500–2,000 steps.
   - Use separate base-class episodes for early stopping—do not select checkpoints using the evaluated novel pairs.
   - Save the best checkpoint and training curves, not merely the final batch loss.

3. **Run only four methods**
   - Positive ProtoNet.
   - Full IERA.
   - `iera_no_negatives`.
   - `iera_mean_env`.

   Do not add more architectural components yet.

4. **Use this decision**
   - If full IERA reduces SMS and improves worst-nuisance AUROC consistently: continue.
   - If `no_negatives` wins: remove subtraction and reformulate the paper as **invariant support-evidence selection**, not “explain-away.”
   - If `mean_env` wins: the soft-min robustness operator is hurting; simplify it.
   - If every learned method still increases SMS: abandon the present mechanism and redesign it.

My current expectation is that **`iera_no_negatives` is your best rescue path**. It nearly met the Edema criterion already, whereas explicit negative subtraction appears unstable. First establish the effect on MIMIC; only afterward expand to CheXpert/NIH for external validation.