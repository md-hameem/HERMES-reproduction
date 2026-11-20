# Selecting older frames for recent-cache construction

Proposed on 21 November 2026. The candidate is a training-free, bounded history selector for constructing a recent video KV cache. The answer decoder receives the prompt and the recent four frames' contextualized K/V. The selected older frames participate only in recent-frame encoding. This is an experimental proposal.

The [AutoDL development notebook](../../notebooks/autodl/hermes-construction-selection.ipynb) implements the six-arm pilot. Reuse the previous container and select **Restart Kernel → Run All**. The result ZIP is `HERMES_construction_selection_evidence.zip`.

## Why this follows from the results

On the same 40 development questions, the matched comparison gained two correct answers when the model could use older history to encode recent frames while older slots were hidden at readout. Making all those slots visible during answering produced a net loss in the contextual arm. Physical removal then preserved all 40 saved masked answers. The short path scored 27/40 versus 25/40 for compact recent-four, which used the same query KV bytes and was faster after extraction. These observations motivate testing whether a small subset of older frames can preserve useful context at lower construction cost. They do not identify which earlier frames should be selected or establish a gain on independent sources. [Matched construction audit](<../../docs/reports/2026-11-21 matched-boundary results and next decision.md>) · [Short query-cache audit](<../../docs/reports/2026-11-21 short query-cache results and next decision.md>).

## Proposed rule

Use the last up to four recent frames and up to 16 older candidates from the existing causal window. Keep their original logical positions. On the first decoder layer, let recent visual queries attend to the prompt and causally available recent tokens. Compute the reference attention output using all available older frames as well. Starting with no older frames selected, greedily choose the older frame that makes the selected attention output closest to the full-history output, averaging squared difference over recent queries and heads. Repeat once for a provisional two-frame budget. Break ties in favor of the nearer frame. Only visual features and pre-query caches are used for selection.

Compute each frame's log normalizer and normalized value mean once. Combine those per-frame statistics with the prompt/recent statistics using log-sum-exp to evaluate candidate subsets without rerunning the decoder for every frame. This reproduces the candidate subset's first-layer attention output for fixed projected queries, keys and values. It is a proxy for the downstream answer. It can preserve irrelevant effects from the full-history target and can miss information that matters in deeper layers.

Restore the prompt and the selected history K/V, preserve their logical positions, and encode the recent frames once using the same recent groups across arms. Preserve the original starting positions even when some older slots are physically absent. Build the answer cache from the prompt and causal recent K/V at those same positions. Log the selected source frame IDs, scorer time, cache assembly, recent encoding, query extraction and answering time. Existing older K/V may already contain information from earlier frames, so the budget refers to direct construction inputs.

The score borrows the idea of comparing attention outputs. [DropKV](https://arxiv.org/abs/2610.22130) already uses output perturbation for token eviction. [FrameBlend](https://arxiv.org/abs/2610.29229) already re-prefills selected tokens using query-relevant context, including a HERMES integration. [DSCache §4.3](https://arxiv.org/html/2605.01858v1#S4.SS3) builds an independent instant cache and combines it with a cumulative cache at inference. The distinction to test here is selecting a few earlier **frame groups for encoding current frames**, with a visual-only score and physically short readout. A literature comparison and matched experimental evidence are still required to establish a distinct contribution.

## Development experiment

Run one small notebook on the existing features and 40 development questions. Fix the two-frame budget for the first pass. Save each answer and the changed cases.

| Arm | Older inputs to recent encoding | Purpose |
|---|---|---|
| No older inputs | None, at the original recent positions | Position-matched lower construction budget |
| Latest two | The nearest two older frames | Simple recency baseline |
| Attention-mass two | Two older frames with largest first-layer recent attention mass | Same-budget scoring ablation |
| Selected two | Greedy attention-output matching | Candidate |
| All older inputs | Entire available older window | Existing short contextual reference |
| Compact recent-four | None, using its ordinary compact positions | Strong practical baseline |

Keep the answer cache budget, model, questions, visual frame slots and greedy answer procedure fixed across arms. Compact recent-four retains its original position policy. Use the null cases with zero older inputs, confirm source timestamps and positions, and validate grouped statistics against direct FP32 masked attention. Compare total scorer plus recent-encoding cost with the all-history arm. Report shared history cache build separately so any construction savings can be scoped precisely.

In this pilot the measured shared preparation includes the original reference encode used to obtain the exact historical prefix. The scorer cost includes the diagnostics and both ranking rules. The attention-mass baseline is charged that shared cost, so timing versus a standalone attention-mass implementation remains open. The notebook measures one timing per question and arm. It requires all-history and compact recent-four to match the 80 previous choices and saves a checkpoint before stopping on a replay mismatch.

Continue only if the selective rule adds value over the equal-budget recency and attention-mass baselines, either by better paired answers or a measurable construction advantage at comparable quality. Check whether compact recent-four already gives the better accuracy/cost trade-off. If the pilot supports the method, freeze the algorithm, comparison rules and a source-separated evaluation manifest before viewing confirmation outcomes. Report video-paired uncertainty and regressions, and compare against the nearest practical prior method. The original-method and Q1 publication claims remain open.
