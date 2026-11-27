# DROID earlier-state masking results

The audit verified 25 evidence payloads, 114 input frames, 56 regenerated masked-frame hashes and 84 answer scores. The original annotated condition matched all 28 archived predictions and passed the 0.05 logit tolerance with a maximum absolute difference of 0.03125. This run used an RTX 4080 and the archived natural-history baseline used an RTX 4080 SUPER. The three masking conditions were compared in the same new run.

| Condition | Robust pairs | Correct branch-order answers |
|---|---:|---:|
| Original annotated inputs | 1/7 | 14/28 |
| Earlier target region masked | 0/7 | 14/28 |
| Equal-area non-target region masked | 1/7 | 14/28 |

The target mask corrected three branch-order answers and regressed three. It recovered no robust pairs and lost the wide-white-drawer pair solved by both comparators. On that pair, the closed branch changed from “Closed” to “Open” in both option orders. The non-target mask corrected one answer and regressed one on other pairs. The masking rule failed the exploratory criterion.

The result shows that obscuring the earlier target region in the first two images was insufficient to improve paired accuracy. Spatial masking also changes object context and may leave evidence visible outside the rectangle or in the third history frame. The result does not identify how much the failure comes from state recognition, temporal selection or prompt bias.

## Research decision

Retire the mask as an improvement candidate. The remaining targeted diagnosis is a same-scene terminal-only comparison. Use the exact three archived final-state frames and score them both alone and with the shared four recent frames. Replay the existing ten-frame annotated arm on the same GPU. Score both option orders and report the input lengths. That diagnostic distinguishes failure on the visible terminal state from failure after adding the remaining input context. Its shorter inputs serve as capability controls. Keep these scenes outside any confirmation set and avoid further selection tuning on them.

[Audit JSON](../../evidence/audits/droid_state_interference_2026-11-27_audit.json) · [Evidence ZIP](../../evidence/archives/HERMES_droid_state_interference_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-droid-state-interference.ipynb)

The [terminal-state diagnostic](<2026-11-27 droid terminal context protocol.md>) is frozen for the same seven pairs.
