# TwinPast automatic-selection result

The annotation-blind 7B run completed all seven development pairs. **Uniform and relevance selection each passed 3/7 pairs.** The history-control correction rule passed 1/7. It recovered one branch-order answer and regressed six relative to direct decoding of the exact same retrieved frames. Retire this correction rule at the tested setting, and keep uniform history as the practical comparison.

[Evidence archive](../../evidence/archives/HERMES_twinpast_automatic_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-automatic.ipynb) · [Independent audit](../../evidence/audits/twinpast_automatic_2026-11-23_audit.json)

| Method | Robust pairs | Correct branch-order answers | Recoveries / regressions vs recent |
| --- | ---: | ---: | ---: |
| Recent only | 0/7 | 12/28 | 0 / 0 |
| Latest history | 1/7 | 15/28 | 5 / 2 |
| Uniform history | 3/7 | 18/28 | 7 / 1 |
| Relevance, direct decoding | 3/7 | 19/28 | 8 / 1 |
| Margin-only correction | 3/7 | 19/28 | 7 / 0 |
| History-control correction | 1/7 | 14/28 | 2 / 0 |

A robust pair requires both video branches to be correct under both option orders. Counts across branch and order are descriptive. The two label orders are repeated views of the same input. Identical recent input on the two branches makes recent-only robust accuracy zero by construction. The history arms share the same final frame count, while the recent-only reference uses a shorter input. The annotated 5/7 result used different historical windows and only two earlier frames on positive controls, so it is a capability reference.

The audit reproduced all six decision paths from logged logits, including the candidate ranking, option canonicalization, override counts and the post-run overlap table. It verified 24 archive payloads and counted **304 relevance-probe calls** among **408** forward calls. Relevance probes used about **17.0 seconds of summed forward time** on this run. This excludes other work and is not an end-to-end standalone latency estimate.

## Why the next test is about the candidate window

Uniform recovered the bottle-stream state pair in both orders. Relevance recovered none of the four historical-state pairs even though it selected more frames inside the bottle's annotated intervals than uniform. None of the automatic arms recovered the box state pair in both orders. Each branch of those recovered cases had only one or two exact curated frames in the capped candidate pool, and each relevance selection retained one. Annotation overlap alone therefore cannot determine whether a selection preserves the particular visual transition. A policy may also fail when sufficient cues are present because the ordering or extra frames change the model's answer.

## Next experiment

Run the [candidate-pool AutoDL notebook](../../notebooks/autodl/hermes-twinpast-candidate-pool.ipynb). It freezes the 1 FPS pool and the original capped pool, the same six-plus-four input budget, and direct forced-choice scoring. Advancement requires at least one dense-only robust pair, zero capped-only pairs and successful curated replay.

Run one matched **candidate-pool capability check** on the four historical-state pairs before tuning any correction threshold. Keep the same 7B model, four recent frames, six older-frame budget, two option orders and direct forced-choice decoding. Compare a deterministic annotation-guided selection within the current capped pool with the same selection rule in a denser causal pool. Define the rule before inference to spread picks across eligible evidence intervals, then fill unused slots chronologically from the same pool. Replay the previously successful curated histories as a reference.

Treat the annotation-guided selections as diagnostic controls. If denser sampling recovers a state case that the capped pool misses, redesign candidate coverage and test one annotation-blind selector against uniform on fresh development sources. If both candidates fail despite replaying the curated success, inspect the event endpoints and context composition before renting more GPU time. Keep the current correction gate out of the confirmation run. A paper claim still needs an original mechanism, a convincing matched advantage and independent sources outside all 12 inspected TwinPast pairs and their source groups.
