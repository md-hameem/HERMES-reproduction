# TwinPast matched refinement audit

The AutoDL LLaVA-OneVision-7B run completed on 27 November 2026. It scored eight development pairs after screening their media against prior clips and checking shared recent frames. A robust pair requires both video branches to be correct in both answer orders.

| History policy | Robust pairs | Correct branch-order answers | Selected a frame in an annotated interval |
|---|---:|---:|---:|
| Coarse uniform, 1 FPS | 1/8 | 16/32 | 2/12 annotated branches |
| Local uniform, 4 FPS candidate pool | 0/8 | 14/32 | 4/12 |
| RGB-change guided, 4 FPS candidate pool | 0/8 | 16/32 | 6/12 |

Each answer used six historical and four shared recent frames. The two local policies searched the same three windows and used the same 39 decoded history candidates per branch. The audit rechecked the frame selections, matched prompt and token counts, and recomputed predictions from 96 saved score vectors. The combined run logged 384 shared retrieval forwards and 96 answer forwards. RGB-change scoring incurs additional image processing.

The wider 1 FPS pool intersected annotations on 8/12 annotated branches. The selected 4 FPS windows intersected 6/12. The change rule selected an in-interval frame on all six branches reached by the fine pool, yet neither local method solved a robust pair. On three of those six branches, the change rule answered correctly in both orders. Interval overlap measures timing and leaves visual sufficiency unresolved.

The executed notebook includes the local-wheel setup and lacks saved output from the model-download cell. The archive verifies the recorded model revision, environment probe, embedded source hashes and completed checkpoints. It does not independently document each weight transfer. The dataset lacks original source IDs, so the sampled duplicate screen cannot establish source-level independence.

## Research decision

Stop tuning this RGB-change rule and keep the result as a development ablation. The prior 2/7 recovery with annotation-selected history is evidence of capability on an inspected set. A publishable method still needs an automatic gain against a matched baseline.

Before spending more GPU time, review the existing clips and separate three cases: the selected windows miss the event, the windows contain the event but the chosen frames miss the defining state, or the answer fails despite visible evidence. Use the review to pick one intervention. If window misses dominate, test a coverage-oriented candidate policy before ranking. If selected frames already show the defining state, test the readout on those same frames. Freeze the chosen intervention and evaluate on independent sources, excluding the inspected development footage.

[Evidence archive](../../evidence/archives/HERMES_twinpast_refinement_evidence.zip) · [Audit JSON](../../evidence/audits/twinpast_refinement_2026-11-27_audit.json) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-twinpast-refinement.ipynb)

The [visual follow-up](<2026-11-27 twinpast refinement visual review.md>) distinguishes window misses from cases where the selected frames already show the target state.
