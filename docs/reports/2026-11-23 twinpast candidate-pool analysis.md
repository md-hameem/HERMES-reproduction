# TwinPast candidate-pool capability result

The candidate-pool test **passed its development criterion**. The capped pool produced **0/4** robust historical-state pairs. The denser pool produced **2/4**, recovering the bottle-stream (`twinpast_0094`) and box-state (`twinpast_0259`) cases with no capped-only successes. Curated replay reproduced all 16 archived score vectors exactly.

[Archive](../../evidence/archives/HERMES_twinpast_candidate_pool_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-23_hermes-twinpast-candidate-pool.ipynb) · [Independent audit](../../evidence/audits/twinpast_candidate_pool_2026-11-23_audit.json)

| Input | Older candidates per branch | Earlier frames sent to 7B | Robust pairs |
| --- | ---: | ---: | ---: |
| Curated replay | Archived references | 6 | 2/4 |
| Capped 0.5 FPS pool, annotation-guided | 13 | 6 | 0/4 |
| Uncapped 1 FPS pool, same selection rule | 26–48 | 6 | 2/4 |

Each condition used the same four recent frames and two option orders. A pair was robust when both branches were correct in both orders. The audit verified 19 ZIP payloads and 48 score rows, rederived frame selections from saved timestamps, matched prompts and input token counts to the earlier experiment, and reproduced the frozen decision.

In the two recovered pairs, the dense arm selected six interval-matched earlier frames on each branch. The capped arm selected two or three. This establishes headroom from a broader candidate pool when guided by evidence annotations. The test changed both sampling rate and the candidate cap. It does not establish their separate effects, the exact visual cause of each recovery, or an automatic or cost-neutral improvement. The book and stool/door pairs remained unresolved across both option orders.

## Next experiment

The exploratory endpoint comparison has completed. All methods scored zero robust pairs, and six annotated branches had no evidence frame in the 1 FPS pool. [Temporal-window audit and follow-up](<2026-11-23 twinpast temporal windows analysis.md>)

The [temporal endpoint AutoDL notebook](../../notebooks/autodl/hermes-twinpast-temporal-windows.ipynb) implements an exploratory version of the comparison. The pinned annotations contain public pair IDs and file paths, with no mapping to original source groups. It freezes 12 new HSR candidates, excludes the 12 inspected pairs, screens sampled visual overlap against them and between accepted pairs, checks exact recent-frame equality, and scores up to eight accepted pairs. This provides fresh-pair development evidence. Verified source-disjoint confirmation remains a separate milestone. [TwinPast Appendix B](https://arxiv.org/html/2610.41983v2)

Run the next annotation-blind selector on a fresh, source-grouped development set with 1 FPS causal candidates. Exclude the 12 inspected TwinPast pairs and overlapping sources, and keep a separate untouched confirmation split. Freeze candidate windows, search and storage caps, six older plus four recent frames, and both option orders before evaluating. Record any source groups whose provenance cannot be verified.

Test whether keeping **both endpoints of selected temporal windows** preserves state-change evidence better than ranking individual frames. Use the same question-only window relevance scores and candidate budget for both score-based arms. Spend six historical slots on the first and last frames of three non-overlapping high-scoring windows. Compare against uniform history and individual-frame top-k on the same dense pool. Use direct forced-choice decoding in all arms. Count the extra candidate encoding, retained state and search cost. A baseline at the old sampling rate can measure the practical effect of the broader candidate setup separately.

This is a method hypothesis. Advance only if it preserves more paired historical-state answers than both matched dense baselines at a practical cost, then check originality against the closest published mechanism and evaluate the frozen method on independent sources. Keep the previous history-control correction out of this experiment.
