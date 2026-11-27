# TwinPast scene-boundary GPU audit

The GPU run reproduced 64 archived answer score vectors with zero maximum absolute difference. Its recorded retrieval replay discrepancy was zero. The audit verified 29 payloads, matched all 16 branches to the frozen CPU selections and recomputed 96 choice scores. Each method used six historical frames, four matched recent frames and the same direct scorer in both option orders. A robust pair requires all four answers to be correct.

| Method | Robust pairs | Correct branch-order answers | Median sum of measured stages per branch |
|---|---:|---:|---:|
| Coarse uniform | **1/8** | 16/32 | 1.60 s |
| Local uniform | 0/8 | 14/32 | 4.52 s |
| Scene-boundary ranking and pre-cut sampling | 0/8 | **17/32** | 6.34 s |

The boundary method recovered seven branch-order answers and regressed six versus coarse uniform. It lost the only robust pair, `twinpast_0255`, whose decisive evidence appears at the start of the clip. Against local uniform it corrected four branch-order answers and regressed one, with no robust change. The development gate required a robust recovery without losing a robust baseline pair, so the method did not advance.

The CPU test showed a gain in final annotated interval coverage from 0/12 to 8/12 branches. This GPU run shows that the gain did not translate into robust paired answers on these clips. For example, boundary sampling made both option orders correct for one guitar-case branch, but both remained wrong on its counterpart. Coverage is a timestamp measure and cannot establish whether the selected frames contain usable evidence or whether the model successfully reads it.

The boundary scan decoded 1,996 distinct candidate frame indices summed across branches. The timing figures allocate shared work and average the two answer-order measurements within each branch. They describe this run on an RTX 4080 SUPER. They are not independently measured end-to-end latencies.

## Research decision

Retire the frozen boundary rule as a proposed method and retain the result as an ablation. The matched local refinement, observation readout and boundary rule have each failed to add a robust pair on inspected development footage. Further tuning on those clips would compound selection bias.

Before another automatic policy, freeze an independent evaluation set with source groups, visible terminal states, paired contrasting answers and identical recent context. First compare coarse uniform with an annotation-selected history control at the same ten-frame budget. Verify exact baseline replay and score both option orders. Require the control to recover pairs on this new set before investing in a fresh policy. If it cannot, focus the thesis on characterizing the evidence-to-answer failure and agree with the advisor on the remaining original-method requirement before committing to a submission claim.

[Audit JSON](../../evidence/audits/twinpast_boundary_gpu_2026-11-27.json) · [Evidence ZIP](../../evidence/archives/HERMES_twinpast_boundary_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-twinpast-boundary.ipynb) · [CPU coverage audit](<2026-11-27 twinpast boundary coverage analysis.md>)

The [DROID follow-up](<2026-11-27 droid state capability protocol.md>) freezes source-verifiable states for a controlled capability check before natural-stream evaluation.
