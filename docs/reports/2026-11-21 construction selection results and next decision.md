# Construction history selection results

The greedy two-frame selector tied both simpler two-frame rules at **26/40** and cost more than selecting the latest two. It also scored one answer below all-history construction, with a median additional **2.91 ms** in the measured scoring plus construction path. This result does not support advancing the current selector as the proposed improvement.

| Arm | Correct | Median scoring plus construction |
|---|---:|---:|
| Greedy selected two | 26/40 | 33.21 ms |
| Latest two | 26/40 | 25.21 ms |
| Attention-mass two | 26/40 | 33.22 ms |
| No history at matched positions | 25/40 | 23.58 ms |
| All history | 27/40 | 29.98 ms |
| Compact recent-four | 25/40 | 38.20 ms |

All six arms had the same median answer cache of **9.34 MiB**. These timings describe the prepared control paths. Shared reference preparation, query extraction and answer time are logged separately. The attention-mass arm is charged the shared scorer including the greedy work, so the recorded time does not establish standalone attention-mass cost. Each arm was measured once per question. No live-system speedup follows from the table.

The selector changed the frame subset versus each simple rule on **26 of 36** questions with more than two available older frames. It lowered the first-layer reconstruction error versus each simple rule on 26 of those questions, using a numerical comparison tolerance of (10^{-12}). That improvement in the proxy did not translate into aggregate answer gains. Against either simple rule, it recovered 209/q0 and lost 208/q4. All-history construction answered both correctly. The run reuses the eight development videos, so the matched controls narrow the hypothesis without providing independent accuracy evidence.

The greedy scorer took a median **7.86 ms**. Per-question comparisons show **7.82 ms** more scoring-plus-construction time than latest-two and **2.91 ms** more than all-history. Latest-two alone saved a median **4.65 ms** of construction versus all-history and lost one answer with no recoveries. Keep the simpler rule as a construction-budget control.

The audit verified 117 archive payloads, 40 question checkpoints, 240 answers, all 80 reference replays and each answer's cache placement and decode positions. The supplied notebook was byte-identical to the maintained source notebook with empty outputs. The completed execution is documented by the ZIP checkpoints and closed logs. Measured post-model run time was 67.60 seconds.

## Decision

Retire this first-layer matching rule as the main method. Avoid another selector or frame-budget sweep on these same questions. The original empirical finding remains that the benefit of older history can differ between recent-cache construction and direct answer readout.

Freeze three practical arms for the next source-separated comparison: **compact recent-four**, **latest-two contextual construction with short readout**, and **all-history contextual construction with short readout**. Include masked contextual readout on a fixed efficiency subset if needed. Select source groups before running, keep development clips out of the estimate and report paired video-level wins, regressions and uncertainty. Evaluate all-history versus compact as the primary accuracy comparison at equal answer-cache bytes. Use latest-two to test how much construction cost can be saved. Include the shared history maintenance and actual end-to-end costs before making an efficiency claim.

If the contextual arms improve on compact recent-four across sources, compare with the closest prior methods and determine whether the construction/readout policy makes a defensible original contribution. If they do not, keep the selector as a negative thesis result and use independent regressions to choose the next hypothesis. The current evidence does not satisfy the original-method requirement.

[Audited tables](../../results/construction_selection_2026-11-21/) · [Audit and hashes](../../evidence/audits/construction_selection_2026-11-21_audit.json) · [Run log and evidence](../../evidence/archives/HERMES_construction_selection_evidence.zip)
