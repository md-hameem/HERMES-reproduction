# DROID state capability audit

The pinned LLaVA-OneVision 7B model completed six site-separated open/closed pairs. The audit verified 22 archive payloads and 12 frame receipts and recomputed 96 choice scores. A robust pair requires both branches correct in both option orders.

| Condition | Robust pairs | Correct branch-order answers |
|---|---:|---:|
| Aligned uniform, cue visible | 4/6 | 22/24 |
| Off-grid uniform, cue absent | 0/6 | 12/24 |
| Off-grid annotated, cue restored | **4/6** | **22/24** |
| Cue only | 4/6 | 21/24 |

Aligned uniform and the annotation-selected off-grid condition presented identical pixel tensors and reproduced all choice logits exactly. Off-grid uniform presented identical neutral inputs across the paired branches. It predicted “Closed” all 24 times and therefore reached 50% branch-order accuracy while solving zero contrastive pairs. That condition verifies the absence of visual state information by construction.

The restored-cue arm met the four-of-six capability threshold fixed before GPU evaluation. Its two failures were on the GuptaLab and IRIS pairs. In each, one branch changed its prediction when the options were reversed. Those same pairs also failed with the cue alone, which limits the explanation that neutral context by itself caused the failure. The six pairs represent one narrow object-state contrast and the timeline used a single real frame with synthetic neutral filler. The result is a controlled capability finding.

## Research decision

Proceed to a natural-history capability test before designing another selector. Freeze fresh, source-separated recordings with visible state transitions, naturally occurring distractor frames and a common real recent continuation whose content cannot reveal the earlier state. Keep both temporal states of an object visible in history when possible. Then compare coarse uniform against an annotation-selected history at the same ten-frame budget, with exact repeat checks, both option orders and no tuning on model results. Have a second reviewer check the target states and recent-context construction. The decision is whether supplying verified evidence can improve robust answers when the model must select the queried state among plausible alternatives.

[Audit JSON](../../evidence/audits/droid_state_control_2026-11-27_audit.json) · [Evidence ZIP](../../evidence/archives/HERMES_droid_state_control_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-droid-state-control.ipynb) · [Frozen protocol](<2026-11-27 droid state capability protocol.md>)

The [natural-history follow-up](<2026-11-27 droid natural history protocol.md>) freezes seven real pairs for the matched comparison.
