# TwinPast development review

The StreamingBench assistant review now has a validated structured worksheet for all 147 windows. A full-size recheck confirmed that Q03 and Q17 show the decisive evidence in both an earlier window and the latest one. This was a recheck by the same reviewer. The original eight-case, same-clip pilot threshold remains unmet. [Structured StreamingBench review](../../results/target_binding_2026-11-22/visual_support_assistant_review.csv)

## What the TwinPast CPU screen established

- **12 frozen pairs.** Eleven have two strictly earlier sampled frames and four recent frames. One pair, `twinpast_0196`, has only one annotated slot after excluding the slot shared with the recent window. It remains recorded in the frozen set with an exclusion reason.
- **44/44 sampled recent frame pairs match exactly at decoded RGB resolution** across the eleven checked pairs. The selected earlier samples are inside both branches’ published evidence intervals.
- **Nine pairs have a clear visual answer contrast** on the sampled frames. Examples include a beanie versus an antler hat, a robot versus a desktop tower, and a door closing versus staying obstructed. I recorded the visual answers before comparing with the benchmark letters. All nine agreed on both branches. Letter order in an oblique view and the endpoint of a door trial remained uncertain in two pairs.

This is a single-reviewer development screen. The nine clear pairs can test whether our decoder responds to a controlled change in history. The visual agreement is not model accuracy or an estimate for all TwinPast questions. Preserve the 12 frozen IDs as development cases and keep them out of confirmation. [Frozen sample](../../data/twinpast/historical_correction_development.json) · [Media checks](../../results/twinpast_development_2026-11-23/media_check.json) · [Visual decisions](../../results/twinpast_development_2026-11-23/pair_visual_screen.csv) · [Reference comparison](../../results/twinpast_development_2026-11-23/visual_audit.json)

## Next test

Run a small paired GPU diagnostic. For each of the nine clear pairs, hold the same four verified recent frames, question, choices and decoder fixed. Evaluate those recent frames alone, then prepend earlier evidence from branch A or branch B in chronological order. Both earlier-evidence arms use two earlier frames and identical token and position budgets. Check whether the top answer reverses in the correct direction, whether the option logit difference changes with the history, and whether reversing the answer labels changes the conclusion. Report pair accuracy, single-branch accuracy and unchanged answers, with the two uncertain inputs reported separately if evaluated.

This is a cross-branch intervention. The StreamingBench same-clip distractor study remains a distinct experiment. A successful development run would justify building an automatic correction policy and testing it on fresh source groups. [TwinPast paper](https://arxiv.org/abs/2610.41983v2)
