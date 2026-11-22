# Q1 experiment audit of target binding

**Verdict for an improvement claim: BLOCKED (CRITICAL ISSUES).** The completed run is reproducible as a within-benchmark negative result. A fresh validation verified 395 archive payloads, 300 questions, 1,800 selector outcomes and the archived notebook's run provenance. This audit applied the six checks from the uploaded [Q1 Experiment Auditor](https://chatgpt.com/skills?collection=created&skill_id=6b02e0ab3ec4819192d3e1ce0b1d7757).

| Severity | Category | Exact issue | Required fix |
|---|---|---|---|
| Critical | Statistical rigor | The primary estimate was −1.86 percentage points versus mean aggregation (95% video-bootstrap interval −4.40 to +0.37, two wins and seven losses). The tested rule also fell behind recent-only on the same 269 eligible questions. | Preserve the negative result. Freeze any replacement method before testing on independent data and report paired effects with source-aware uncertainty. |
| Critical | Ablation completeness | The purported two-score rule picks the same window as binding alone on all 269 questions. The binding margin changes sign with reversed A/B labels in 815 of 1,264 eligible windows. | Verify which option the visual evidence actually supports. Use a matched older distractor and test option order explicitly. |
| Major | Baseline fairness | All selectors share the answering backbone and answer frame budget. Mean and minimum also share the verifier calls. Comparing either with recent-only brings extra scans and probes. No faithful close published comparator was run. | Match the candidate access and computation of the closest comparator. Keep recent-only as the practical baseline and report the compute trade-off. |
| Major | Data leakage and confounders | Logged windows obey the causal cutoff. Original source overlap is unverified, and StreamingBench informed the choice of the next method. | Use this set for development and a separately verified source-disjoint set for confirmation. Keep paired examples together when applicable. |
| Major | Efficiency and overhead | Median summed component time was 920.1 ms for the minimum rule and 248.2 ms for recent-only. These measurements exclude decoding video, extracting features, and maintaining the stream. End-to-end latency and peak memory for this comparison are missing. | Measure end-to-end latency, including p95, and peak resource use for the surviving method and matched baselines on the same hardware. |
| None found | Numeric and visual consistency | The saved exports match the checkpoint reconstruction. Counts and the primary interval agree with the dated analysis. Visual-support labels remain pending. | Recheck manuscript tables and captions against the final exported evidence once the figures exist. |

## Minimum missing work

1. Label the blinded 32-question review. Check candidate-option support, wrong-target or wrong-setting distractors, and whether the evidence applies at query time. [Worksheet](../../results/target_binding_2026-11-22/visual_support_review.csv)
2. Hold recent frames fixed. Compare two human-verified older frames with two matched distractor frames under the same token budget. Use the human selections only to test the mechanism.
3. If step two passes, freeze an automatic method and compare it with the closest method and recent-only on independent data, using a second backbone for the planned generalization test.

[Archive audit](../../evidence/audits/target_binding_2026-11-22_audit.json) · [CPU verifier replay](../../results/target_binding_2026-11-22/verifier_replay.json) · [Frozen results](<2026-11-22 target binding analysis.md>)
