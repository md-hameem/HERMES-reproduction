# TwinPast GPU diagnostic and next decision

The frozen nine-pair experiment completed on an RTX 4090. Earlier footage changed the answers in the correct direction on some pairs, but only **3 of 9 pairs** were correct on both branches under both option orders. The planned advancement rule required at least five. Keep the automatic correction method on hold.

[Independent audit](../../evidence/audits/twinpast_diagnostic_2026-11-23_audit.json) · [Evidence archive](../../evidence/archives/HERMES_twinpast_diagnostic_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-diagnostic.ipynb) · [Frozen decision rule](../../paper/neurocomputing/submission_phase_plan.md#1-test-whether-the-backbone-uses-earlier-evidence)

## Saved evidence and controls

The audit checked all 24 ZIP payloads, the embedded source and manifest, and all 72 score rows against nine checkpoints and the frozen media records. The executed notebook matches the distributed source with one added AutoDL network-acceleration cell. The first-pair preflight passed. The maximum recorded native-cache score difference was 0.015625 in each order, with the top option preserved. The causal history arms use the same four recent-frame hashes and answer prompt within each option order. This is a verification of the saved execution and outputs.

| Measure | Canonical | Reversed |
| --- | ---: | ---: |
| Both history branches correct | 4/9 | 3/9 |
| Correct history-arm branch answers | 13/18 | 12/18 |
| Correct recent-only branch answers | 9/18 | 9/18 |
| Correct question-only branch answers | 9/18 | 9/18 |
| Recoveries / regressions vs recent-only | 5 / 1 | 3 / 0 |
| Identical predictions across A and B history arms | 5/9 | 6/9 |
| Incorrect answer reversals | 0 | 0 |
| Positive answer-margin shift | 8/9 | 9/9 |
| Median directional logit change | +0.546875 | +0.265625 |

The two branches have different correct answers. An identical-input control gives both branches the same prediction, so its pair accuracy is zero by construction. The gain in branch accuracy and the directional shifts are descriptive. The nine pairs were enriched for visible contrasts, and their annotated history frames were supplied to the model. The run establishes sensitivity to supplied history on this sample and says little about whether a streaming system can find that evidence automatically.

All three robust pairs were Attribute and Identity Recall cases. A fourth case from that task, `twinpast_0214`, worked in canonical order only. None of the four Historical State Recall pairs reversed correctly in either order. Their directional shifts span −0.015625 to +0.546875, compared with +0.0625 to +3.65625 across the five identity/attribute pairs and both orders. This is an exploratory pattern with few examples. Source-group independence remains unverified. Keep all 12 inspected pairs out of confirmation.

## Next experiment

This follow-up has completed. The same three pairs passed under both option orders, and no historical-state pair passed both orders with six frames. [History-only analysis](<2026-11-23 twinpast history-only analysis.md>)

Run the [standalone history-only AutoDL notebook](../../notebooks/autodl/hermes-twinpast-history-only.ipynb) in the same container to reuse the 18 verified clips. Save the executed notebook and evidence ZIP when it finishes.

Run one additional capability diagnostic on the same selected pairs. Score the existing two annotated earlier frames alone using the same prompts, model and two option orders. This tests whether the sampled history can support the answer when the recent window is absent. If the four Historical State Recall cases remain failures, add a denser annotation-constrained history-only diagnostic on those cases, matching the sample count and relative sampling rule between A and B. Preserve temporal order and keep each branch within its prefix. Log both-branch correctness, signed margins and the observed number of recoveries. Where available, have a human reviewer verify that sampled frames show the requested event or transition.

- If the same two earlier frames work alone, investigate why the answer changes when recent context is added. A later matched-position and token-count control would be needed to isolate recent-content interference from sequence-layout effects.
- If only the denser history works, prioritize the sampling and evidence representation problem before testing a selection policy.
- If visibly sufficient denser evidence still fails, limit claims for this backbone and consider one small compatible-backbone capability check before spending on a larger method run.

Treat this follow-up as development. A successful capability control does not retroactively pass the original gate. Any new method and confirmatory evaluation need a separate frozen protocol.
