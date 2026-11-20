# Stage attribution results and next experiment

The completed AutoDL run contains 200 main answers across 40 development questions from eight videos, plus four answers for one separate long-prefix question. The saved-data audit verified all 271 payloads, all 204 checkpoints, the prediction and comparison tables, the recorded cache positions and mask traces, and 81/81 exact choice replays. The run reused all eight feature caches. The 56.19 seconds reported after model load covers this reused-feature experiment. It does not measure deployment cost or a fresh container.

| Main condition | History during recent encoding | History during answering | Correct |
|---|---|---|---:|
| A | Visible | Visible | 21/40 |
| B | Hidden | Visible | 24/40 |
| C | Visible | Hidden | 27/40 |
| D | Hidden | Hidden | 25/40 |
| Compact recent-4 | Absent | Absent | 25/40 |

**Direct readout is the strongest finding.** A and C share the same encoded cache, physical storage, positions and input groups. Hiding direct access to older tokens recovered seven answers and lost one, for a net gain of six answers. With history already blocked during recent encoding, B → D had four recoveries and three losses. The separate 204/q3 long-prefix case also flipped from incorrect in A and B to correct in C and D. This is one additional case outside the main denominator.

**The recent-only baseline is hard to beat.** C gained three questions and lost one versus compact recent-4. The net +2 answers come from +2 on video 208, +1 on video 209 and −1 on video 203. Leave out video 208 and the net difference is zero. D tied compact in aggregate with one win, one loss and four changed choices. C retained all older cache slots physically, so its measured accuracy says nothing about savings from eviction. The experiment used questions selected for development from previously studied videos. These paired outcomes are hypothesis-generating.

**Construction needs a matched control.** A → B recovered three answers and lost none. D → C recovered two and lost none (208/q4 and 209/q0). The original ingestion group straddled the history/recent boundary on 36/40 main questions, so A/C and B/D can differ in the size of the forward call that encodes the first recent tokens. The previous partition diagnostic changed no choices across 19 answered contrasts on selected questions. That does not substitute for matching the boundary in this run. The A → C comparison already holds construction fixed.

The successful run used the revised checks from the executed notebook. On the two-frame fixture, the strict FP32 comparison used a physically independent reference. The FP16 test changed hidden history while keeping the shape fixed and measured zero recent-cache difference. The archived runner also requires identical logits upon changing hidden values in the same-shape masked readout. Across the nine native/explicit and masked/compact FP16 comparisons, all tested top tokens agreed, maximum absolute logit difference was 0.03125, and 44 entries exceeded the earlier elementwise bound. Those differences were recorded. They were not pass/fail criteria for the successful FP16 run. The successful evidence includes the actual runner source, and the upload is preserved unchanged. The underlying GPU computation cannot be repeated from the ZIP alone.

## Decision for the paper

1. Run one matched-boundary control on the 40 development questions. Build contextual recent K/V by restoring H and encoding R in the same groups used by B/D, once with H visible and once with H blocked. Keep the readout mask, positions, features and answer settings fixed. Retain A, C and compact recent-4 as references. Check all changed questions, especially 208/q4 and 209/q0.
2. Evaluate **a practical recent readout policy** against compact recent-4 and a strong fixed-history or decoupled-cache comparator at a matched query budget. One candidate to screen is to keep contextualized recent K/V and make history available only on a query-triggered fallback. Freeze the rule before confirmation. Test whether the rule detects cases where masking harms performance, including 212/q3, without using gold answers to trigger.
3. Continue toward a research article only if the candidate beats the strong simple baseline on source-separated confirmation data and the contribution is distinct from the closest prior methods. Report paired video-level uncertainty and separate cache construction, query time and storage costs. If the compact baseline matches the candidate, stop claiming an accuracy improvement and reconsider the method before renting more GPU time.

[Audited tables](../../results/stage_attribution_2026-11-21/) · [Audit output](../../evidence/audits/stage_attribution_2026-11-21_audit.json) · [Executed notebook](../../notebooks/executed/2026-11-21_hermes-stage-attribution.ipynb)
