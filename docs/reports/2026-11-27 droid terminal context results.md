# DROID terminal-state context results

The audit verified 23 evidence payloads, 114 input frame hashes and 84 choice scores. All 28 full-history predictions and logits replayed exactly from the masking run on the same reported RTX 4080 GPU model. The three conditions share the same terminal images, scorer and options. A robust pair requires both branches correct in both option orders.

| Input | Robust pairs | Correct branch-order answers |
|---|---:|---:|
| Three terminal frames | **6/7** | **27/28** |
| Same terminal frames plus four recent frames | 5/7 | 26/28 |
| Full ten-frame annotated history | 1/7 | 14/28 |

Adding the shared recent frames gained one robust pair and lost two. At the branch-order level, one answer was corrected and two regressed. Prepending the three earlier history frames to that seven-frame sequence lost four robust pairs, gained none and changed 12 correct answers to incorrect. This is the largest measured loss in the three-step comparison despite retaining identical terminal images.

The model can recognize the terminal state in isolation for most of these scenes. The results establish sensitivity to the additional context under the tested input construction. Adding frames also changed token positions and sequence length. This comparison cannot distinguish those changes from the content of the earlier images. The earlier mask altered part of that content and yielded no robust gains. It may leave competing-state cues visible.

## Research decision

Run a fixed-length counterfactual before designing an automatic method. Keep the ten positions, all three terminal frames and the four shared recent frames unchanged. Swap the first two verified initial-state frames between paired branches. The other branch starts in the recipient branch's terminal state, so the swap changes whether those two frames support the competing state or agree with the terminal state. The corresponding swap in the opposite direction uses the same images. Check the splice visually and retain the third early frame. Compare with the unchanged ten-frame baseline and score both option orders. Report changes in per-option margins and robust pairs.

An improvement under the swap would motivate a readout that distinguishes earlier states from the queried terminal state. The swap also changes scene details from another episode, so additional matched controls and fresh scenes would be required for a claim specific to state conflict. Keep all reviewed footage out of confirmation.

[Audit JSON](../../evidence/audits/droid_terminal_context_2026-11-27_audit.json) · [Evidence ZIP](../../evidence/archives/HERMES_droid_terminal_context_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-droid-terminal-context.ipynb)

The [fixed-length swap protocol](<2026-11-27 droid earlier state swap protocol.md>) records the frozen substitutions and replay check.
