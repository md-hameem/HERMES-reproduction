# TwinPast history-only result and next decision

The RTX 4090 follow-up passed preflight and completed all nine development pairs. Removing the recent window preserved the same **3/9** pairs that were correct on both branches under both option orders. There were **zero robust recoveries and zero losses**. Adding four annotated frames to the original two also failed to produce a robust success on any of the four Historical State Recall pairs.

[Archive](../../evidence/archives/HERMES_twinpast_history_only_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-history-only.ipynb) · [Audit](../../evidence/audits/twinpast_history_only_2026-11-23_audit.json)

## What the run established

The audit matched the notebook source to the distributed version and verified all 32 payloads and 52 score rows. It checked the prior logits, prompt hashes and branch labels against the original archive. A separate local decode reproduced the native RGB hashes for all 84 recorded frame slots across the two conditions, including repeated frames kept in both conditions. The native-cache preflight had zero saved score difference under both orders.

| Condition | Correct branches, canonical | Correct branches, reversed | Both branches correct under both orders |
| --- | ---: | ---: | ---: |
| Two earlier plus four recent, prior run | 13/18 | 12/18 | 3/9 |
| Same two earlier frames alone | 13/18 | 13/18 | 3/9 |
| Six earlier frames alone, four unresolved historical-state pairs | 4/8 | 4/8 | 0/4 |

The same three Attribute and Identity Recall pairs passed both orders with and without the recent window. On the four historical-state pairs, two earlier frames produced one correct reversal in each option order, on different pairs. The denser condition produced none in either order. The six-frame contact sheets show the book opening versus closing, the door closing versus remaining open against a stool, and the box ending closed versus visibly propped open. The bottle contrast is also apparent, although the stream itself is a smaller visual detail. This was a single inspection of contact sheets. It does not certify visual sufficiency for every frame or prove the model's failure mechanism.

Removing the recent window also moved the question to an earlier sequence position. Within the dense test, adding frames changed both temporal coverage and token count. Scope the conclusion to these inputs, this backbone and these scoring conditions. The nine development pairs were selected using annotated evidence. Keep all 12 inspected TwinPast pairs out of confirmation.

## Next experiment

This check has completed. The 7B checkpoint recovered two of four historical-state pairs and retained all three positive controls. [7B results and follow-up](<2026-11-23 twinpast 7b backbone analysis.md>)

The [stronger-backbone AutoDL notebook](../../notebooks/autodl/hermes-twinpast-backbone.ipynb) pins LLaVA-OneVision Qwen2 7B and embeds the archived frames' identities, the original prompts and the 0.5B scores.

Run one stronger-backbone capability check before changing memory or retrieval. Use one compatible larger video-language checkpoint, preferably within the same model family. Start with a single GPU smoke pair to establish that the pinned checkpoint and six-frame path fit. Then replay the exact six earlier frames on the four historical-state pairs and the exact two earlier frames on the three successful attribute/identity pairs as positive controls. Preserve chronological order, shared option semantics, both label orders and the same comparative scoring rule. Record preprocessing, token layouts, precision and checkpoint differences so the result is interpreted as transfer across systems.

Proposed development criterion to freeze before the run. Require at least two of four state pairs and two of three positive-control pairs to pass both branches in both option orders before pursuing historical-state correction on that backbone. If the valid test misses this criterion, do not escalate this branch to a large benchmark. Reassess whether an attribute/identity-specific research mechanism has a distinct contribution and enough fresh evidence. A passing stronger-backbone result would establish headroom on these selected pairs. The original 0.5B advancement gate remains failed, and an automatic method still requires matched baselines and source-disjoint confirmation.
