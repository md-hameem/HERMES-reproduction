# TwinPast 7B backbone result and next decision

The pinned LLaVA-OneVision Qwen2 7B replay passed the predeclared development criterion. It recovered **2/4 historical-state pairs**, up from **0/4** on 0.5B, and preserved **3/3 positive controls**. A pair counted only when both branches were correct under both option orders. The criterion required at least two successes in each cohort.

[Evidence archive](../../evidence/archives/HERMES_twinpast_backbone_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-backbone.ipynb) · [Independent audit](../../evidence/audits/twinpast_backbone_2026-11-23_audit.json)

| Cohort and input | 0.5B | 7B |
| --- | ---: | ---: |
| Historical state, six earlier frames | 0/4 | 2/4 |
| Attribute/identity positive controls, two earlier frames | 3/3 | 3/3 |
| All seven selected pairs | 3/7 | 5/7 |

The recovered pairs were the bottle-stream contrast (`twinpast_0094`) and the closed versus propped-open box (`twinpast_0259`). The book and stool/door cases still failed to answer both branches correctly. The audit independently reproduced all 28 score rows and verified 24 ZIP payloads, the notebook source, and the exact frame indices, RGB hashes, prompt hashes and language-model token counts against the 0.5B archive. The six-frame GPU preflight preserved the top option with a maximum native-cache score difference of 0.015625 in each order. Peak allocated GPU memory in the saved run was **16.53 GiB** on the RTX 4090.

The stronger checkpoint can use the supplied historical evidence on some of these development pairs. The two checkpoints differ in more than parameter count, and the successful controls were selected for success on 0.5B. Recent frames were absent and evidence was selected from published annotations. These results establish capability on the screened cases. The causal streaming setting, automatic retrieval, method novelty, and gains on new sources still require evidence.

## Next experiment

The fixed-recent experiment is complete and passed the criterion below. [Audited result and automatic-selection plan](<2026-11-23 twinpast 7b fixed-recent analysis.md>)

Run the [fixed-recent AutoDL notebook](../../notebooks/autodl/hermes-twinpast-fixed-recent.ipynb). Using the same instance lets it reuse the verified model weights and clips.

Use the cached 7B model and the same seven pairs. Add the four audited recent frames to each archived history window. For each pair and option order, both history arms must use the same recent tensors, prompt, token layout, and chronology. Retain a fresh recent-only control and replay the archived history-only input to check that the earlier result reproduces. Save scores for both branches under both option orders.

Freeze the following development decision before running. Continue toward automatic selection if both previously recovered state pairs still pass both orders with recent context and at least two of the three positive controls remain robust. Report any newly recovered state pair separately. If this criterion is missed and the history-only result reproduces, test whether adding recent content or moving the question changes the result before designing retrieval. If the replay itself differs, resolve the technical mismatch first.

If the fixed-recent test passes, freeze one annotation-blind selector and one matched correction comparison. Use chronological and query-relevance baselines, the same selected frames with direct decoding, and a closest published comparator. Keep source annotations and the other branch out of the deployable path. Keep all 12 inspected TwinPast pairs and overlapping sources out of later confirmation.
