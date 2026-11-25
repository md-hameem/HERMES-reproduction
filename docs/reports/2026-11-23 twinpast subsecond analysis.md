# TwinPast sub-second evidence result

The annotation-guided capability check **passed its development criterion**. Native-timestamp evidence recovered **2/7 robust pairs** versus **0/7** for uniform history at the same ten-frame answer budget. All 28 archived uniform score vectors replayed exactly.

[Archive](../../evidence/archives/HERMES_twinpast_subsecond_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-subsecond.ipynb) · [Independent audit](../../evidence/audits/twinpast_subsecond_2026-11-23_audit.json)

| Input | Robust pairs | Correct branch-order answers |
| --- | ---: | ---: |
| Six uniform history frames plus four shared recent frames | 0/7 | 11/28 |
| Native annotated evidence plus uniform fillers where needed and four shared recent frames | 2/7 | 19/28 |

A robust pair requires both branches correct under both option orders. The audit verified 38 ZIP payloads, 56 score rows, identical answer token counts and the exact uniform replay. The two robust recoveries were the laptop lid open/closed contrast (`twinpast_0001`) and the screwdriver exposed/covered contrast (`twinpast_0037`). Contact sheets show the contrasting states. This is a single inspection of the exported sheets.

Both recovered pairs had at least one annotated-frame match per branch in the previous 1 FPS pool. Native timestamp selection supplied more frames within their annotated spans and changed the history input composition. None of the three pairs whose evidence intervals the 1 FPS pool missed on both branches became robust. The cabinet pair improved only in reversed order despite a visible open/closed contrast on the contact sheet. The result supports testing finer selection around useful spans. It does not show that 1 FPS misses alone explain the endpoint selector's failure.

## Next experiment

The [matched refinement AutoDL notebook](../../notebooks/autodl/hermes-twinpast-refinement.ipynb) freezes three question-relevant historical windows, 39 decoded sub-second candidates, six selected historical frames and four shared recent frames. It compares fixed local offsets with frames bracketing the largest adjacent RGB change. That change score is a proxy and can respond to camera motion. The fresh-pair sample is screened for detected footage overlap. Original-source provenance remains unresolved.

Freeze one **annotation-blind local refinement** experiment. Use the same low-rate relevance scores to choose a small number of earlier windows. Within those windows, compare a budgeted sampler that concentrates on visual state changes with uniform sub-second sampling. Match the number of refined frames inspected, the six-history plus four-recent answer budget, window access, and decoding. Keep ordinary 1 FPS uniform history as a practical reference.

Use new development pairs and keep the already evaluated pairs as debugging controls. Freeze the sampling and selection rules before scoring, and use annotations only after predictions are saved. Report robust pair changes, scanning and encoding cost, and failures on static endpoints versus transitions. Advance only if local refinement improves paired answers over equal-cost uniform refinement. Original-source provenance and independent confirmation remain open.
