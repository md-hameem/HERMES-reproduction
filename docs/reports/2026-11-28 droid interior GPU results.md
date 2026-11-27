# DROID event-relative interior evidence on GPU

Annotation-guided access improved coverage from five to seven clips in the CPU review. On the pinned 7B model, the best fixed window answered more clips correctly in both option orders. [CPU coverage audit](<2026-11-28 droid interior coverage results.md>)

| Attention condition | Correct in both orders | Correct branch-order answers |
|---|---:|---:|
| Unrestricted native | 4/7 | 10/14 |
| Fixed center-three | 6/7 | 13/14 |
| Last-three recency | 5/7 | 12/14 |
| Annotation-guided interior | 5/7 | 11/14 |

The annotation-guided mask recovered zero clips from the fixed window and lost the microwave question about the pink object. The two clips whose annotated slot fell outside the center window had identical predictions under both conditions in both option orders. Each condition answered one of those two clips correctly in both orders. On the four clips whose endpoints showed the opposite state, the fixed and annotated conditions both answered 3/4 correctly in both orders.

The RTX 4080 run yielded 56 choice scores with the same ten-frame input and token positions. All 14 full causal-mask parity checks passed. The largest option-logit difference was 0.015625 against the 0.05 tolerance. The audit matched the notebook and embedded source to the ZIP, rechecked all 70 frame receipts, and recomputed the outcomes from the logits. [GPU audit](../../evidence/audits/droid_interior_gpu_2026-11-28_audit.json) · [Executed notebook](../../notebooks/executed/2026-11-28_hermes-droid-interior.ipynb) · [Evidence ZIP](../../evidence/archives/HERMES_droid_interior_evidence.zip)

## Research decision

Keep center-three as the reference for this screen and close the current annotated-window candidate. The task labels identify frames that show the event and state together. A model may also infer the answer from surrounding actions, and masked readout can still receive information through contextualized visual tokens. This result leaves open which cues supported the center-window answers.

Before another GPU run, look for matched clips from new sources with the same question, closely matched endpoint or center evidence, and opposite answers at the queried event. Review both branches and record why the short fixed windows should be ambiguous. Proceed to an annotated capability check only if that CPU review produces a useful matched set. Keep the present run as a negative result for the Neurocomputing study.
