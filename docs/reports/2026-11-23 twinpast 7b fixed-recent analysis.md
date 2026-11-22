# TwinPast 7B fixed-recent result and next decision

The fixed-recent experiment **passed the development criterion**. Both historical-state pairs recovered on 7B, `twinpast_0094` and `twinpast_0259`, were still correct on both branches in both option orders after adding four identical recent frames. All three positive controls also passed. The other two historical-state pairs remained unresolved.

[Evidence archive](../../evidence/archives/HERMES_twinpast_fixed_recent_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-fixed-recent.ipynb) · [Independent audit](../../evidence/audits/twinpast_fixed_recent_2026-11-23_audit.json)

| Condition | Robust historical-state pairs | Robust positive controls | Robust total |
| --- | ---: | ---: | ---: |
| 7B history-only replay | 2/4 | 3/3 | 5/7 |
| 7B history plus four fixed recent frames | 2/4 | 3/3 | 5/7 |
| 7B recent-only | 0/4 | 0/3 | 0/7 |

Recent-only gave 6/14 correct branch answers in each order. Its joint pair accuracy is zero by construction because both branches have identical inputs and different gold answers. The archive audit verified 24 payloads, 70 score rows, matched prompt and frame hashes, and exact token-count differences. All 28 history-only score vectors reproduced exactly under the predeclared 0.05 absolute logit tolerance. The ten-frame smoke test passed at 2,058 language-model tokens. Peak GPU memory allocated in that test was 17.24 GiB.

The result establishes historical-evidence headroom in the presence of recent context on selected cases. It uses annotation-selected history. The run does not establish automatic retrieval, an improved memory policy, or generalization.

## Next experiment

The first automatic-selection screen has completed. Uniform and relevance each passed 3/7 pairs. [Automatic-selection audit and decision](<2026-11-23 twinpast automatic selection analysis.md>)

The [automatic-selection AutoDL notebook](../../notebooks/autodl/hermes-twinpast-automatic.ipynb) freezes a 13-frame candidate cap, up to six older frames for answering and a 0.5-logit margin threshold. It uses both label orders for each two-frame relevance probe and loads gold and evidence intervals only after predictions are checkpointed.

Build one bounded annotation-blind development screen on the cached 7B system. Keep both branches independent and permit only timestamped frames from the queried branch before query time. Freeze candidate sampling, retrieval scoring, tie breaks, and override parameters before inference. Hide target labels, evidence intervals and counterpart frames from the policy. Use those fields only in scoring and post-run audits.

Start at the demonstrated input budget of up to six older frames plus four recent frames. Compare four recent frames alone, a matched chronological or uniform history selector, and a matched query-relevance selector. Feed the exact same retrieved frames to direct decoding and the candidate answer-change rule. Keep selected-frame count, prompt, option mappings and candidate access matched for the selector and decoding comparisons. Report candidate scans, all model calls, recoveries, regressions, latency and retained state. A shorter final input is a separate budget experiment.

The candidate rule is to override a recent answer only if the retrieved earlier evidence changes the canonical score margin in favor of a specific alternative. Check whether an added temporal-validity control provides benefit beyond direct decode and relevance retrieval before making a novelty claim. Compare the actual mechanism with the closest executable published work. Use the seven cases to reject weak candidates and record exact failure modes. A claimed improvement requires a frozen method and untouched source groups, with all 12 inspected TwinPast pairs and overlapping sources excluded.
