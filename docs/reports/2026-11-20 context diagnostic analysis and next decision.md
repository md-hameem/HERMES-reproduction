# Context diagnostic: controlled recovery and next research gate

Received and audited 20 November 2026. **The seven-case run completed. Exact replay and chunk-partition controls passed at the answer-choice level. Altering earlier context at fixed input length and fixed recent-frame positions recovers three full-prefix failures. We now have reproducible, intervention-sensitive examples, but not a deployable new method or a publication-ready improvement.**

## Integrity, completion and preservation

The original [ZIP](../../evidence/archives/HERMES_context_diagnostic_evidence.zip) SHA256 is `fee3165e323c0f7106688a37cb185ec390d4f831e9debdfbf0352bd72881b774`; the [executed notebook](../../notebooks/executed/2026-11-20_hermes-context-diagnostic.ipynb) SHA256 is `26f2ec995b044b7d1595c73945eebc0285675882a7f93bd7719b53dfa6bf2055`. Both were moved out of the repository root without changing their bytes. Full extraction is ignored under `evidence/local/context_diagnostic_2026-11-20/`; [compact results](../../results/context_diagnostic_2026-11-20/) preserve original exports and independently reconstructed comparisons.

The [independent audit](../../scripts/validate_context_diagnostic_evidence.py) verifies all **140 payloads in 141 ZIP members**, 84 checkpoints, config/source/model and reference identities, original annotation/gold/parser, repeated source-slot maps, temporal boundaries, exact ingestion groups, cache lengths, protected-tail positions, statuses and all exported tables. Archived source is parsed, never executed. Five audit tests reject changed tail/source/position/chunk mappings, invalid correctness/status and damaged identities, and recompute complete coverage.

All seven case blocks completed: **74 answered records, five undefined candidate-support skips and five configured-cap skips** for the long 208/q4 full-prefix arms. Five historical feature sets were verified/reused and no answer checkpoints were reused. GPU preflight reports bitwise feature-ingestion agreement and zero maximum native-attention error before/after pruning on the checked prefix. The 13-cell execution record has no retained errors and emits the final ZIP hash. The intermediate ZIP hash in the live log precedes final closed-log repackaging, as intended.

The archive contains 27 original-prefix JPEG review pages and three original-resolution candidate PNGs. Raw video/feature tensors remain absent, so their numerical computation and GPU inference cannot be rerun locally from this package. This is a saved-evidence audit, not an independent GPU reproduction.

## What the controls establish

**Replay:** all **26 comparable historical answer choices match**. There are 28 replay rows: 204/q3 full prefix was unrun in the earlier 8K-cap screen, and 208/q4 remains capped, so those two are not historical choice comparisons.

**Chunking:** full-prefix original versus framewise matches on six answered questions; original versus fixed-four also matches on six; recent-20 original versus fixed-four matches on seven. Thus **19 answered partition contrasts change zero choices**. They support answer-level partition stability in these cases. They do not prove identical internal tensors or exclude every numerical effect elsewhere.

**Null controls:** 208/q0 has only four sampled frames. Its opening/local replacements contain the original input unchanged. Those four identical-input contrast rows remain unchanged and are not four extra independent successes or failures.

## Content-replacement outcomes

| Condition | Correct / answered | Paired interpretation |
|---|---:|---|
| Full prefix, original/fixed-four/framewise | 1/6 each | Identical predicted choices across partitions |
| Full prefix, opening-frame replacement | 4/6 | Three recoveries, zero correctness losses against fixed-four on the same six questions |
| Full prefix, local-frame replacement | 2/6 | Two recoveries and one loss on the same six questions |
| Recent 20, original/fixed-four | 2/7 each | Identical choices across those partitions |
| Recent 20, opening-frame replacement | 4/7 | Two recoveries, zero correctness losses |
| Recent 20, local-frame replacement | 2/7 | One recovery and one loss |
| Analyst candidate support | 2/2 | Only the two preselected 204 examples; no certified-oracle/general accuracy claim |

This is not a claim of “improving accuracy from 16.7% to 66.7%.” The six questions were selected using observed outcomes, and the replacement arms repeat pre-query filler in counterfactual sequences while preserving the last four input frames. They are diagnostic interventions, not an evaluated deployable streaming selection algorithm. Repeated arms/contrasts are correlated, and the three full-prefix recoveries come from only **two source videos**. No population effect or significance claim follows.

### Three recoverable full-prefix examples

| Question | Gold | Full prefix | Opening replacement | Local replacement | Candidate support |
|---|---|---|---|---|---|
| 204/q0, yellow block | C | A | C | C | C from sampled [6,7] |
| 203/q2, hand reaching for green block | C | A | C | A | Not defined |
| 204/q3, “Alphabet B transform to Box” | B | A | B | B | B from sampled [69] |

All replacement comparisons retain the same token count, recent source frames, recent token positions, initial prompt, model, question and no-compression policy. The full-prefix treatment has no eviction. Consequently, prior token loss, fewer input slots or a mere shift of recent frames to earlier absolute positions cannot explain these particular paired replacement recoveries. Earlier content/repetition influences the output under those controls. The exact internal path remains unisolated: old information may alter recent-token construction, question-time access, attention normalization or other interactions.

Original-resolution inspection confirms the yellow block in [204/q0 sampled frame 7, 14 seconds](../../results/context_diagnostic_2026-11-20/figures/sample_204_q0_frame_7.png) and the requested overlay in [204/q3 sampled frame 69, 138 seconds](../../results/context_diagnostic_2026-11-20/figures/sample_204_q3_frame_69.png). The current model answers both selected candidate inputs correctly. This establishes useful capability headroom for these examples, with analyst-selected support and selection limits explicit. The [203/q2 last sampled page](../../results/context_diagnostic_2026-11-20/figures/sample_203_q2_last_page.jpg) shows the current hand/green-block action, but no support-only experiment or original-resolution candidate export was made for that question.

### Filler identity and temporal shortcuts remain important

The opening frame is **not universally blank**. For example, 203's opening shows a multicolored toy structure, whereas 204 opens black. Thus “black padding solves the problem” is unsupported. Local filler can repeat a salient cue or distractor; repetition may change attention normalization. Different fillers give different results, so neither should be called neutral without additional evidence.

209/q0 stays correct with opening replacement but becomes wrong with local replacement (D to A). Removing earlier source events while retaining the correct label therefore does not itself demonstrate that the model remembered or grounded those events. Some success may follow recent visual cues or option/language priors. This case preserves the need for visual/temporal dependence checks and demonstrates that one blanket replacement rule can harm useful answers.

202/q0 and 208/q0 remain wrong. 208/q4 is wrong in every answered visual arm, while question-only is correct; its full-prefix arms remain unrun. These are unresolved input/perception/grounding/counting cases, not evidence that all remaining errors can be repaired by historical-context suppression.

## Novelty boundary: a mechanism baseline comes next

Targeted primary-method checks were refreshed during this analysis; implementations were not audited or reproduced. [DSCache §4.3](https://arxiv.org/html/2605.01858v1) already constructs recent instant caches independently of historical caches and combines them at inference; Appendix E also studies periodic approximation. [SelectStream §§3.3–3.4](https://arxiv.org/html/2606.16353v2) already restricts and calibrates historical evidence supplied alongside current observations. [StreamScout §3.2](https://arxiv.org/html/2609.00291v1) already uses progressively richer query-time views. Protecting recent perception or choosing a recent/history view is therefore not sufficient originality.

**Decision:** stop adding filler variants to the same seven questions. Move to one bounded stage-attribution and feasible-baseline experiment, then choose or reject one proposed improvement. The [next protocol](../../paper/neurocomputing/stage_attribution_protocol.md) separates history's role in constructing recent representations from its direct visibility during question/answer decoding. A DSCache-inspired factor is a comparator, not our invention or a faithful full DSCache reproduction.

Run the stage matrix on all 40 existing development questions at a fixed recent-20 budget, retain regressions and compare the same question identities. Keep a focused longer-prefix 204/q3 check separate because its recent-20 input already succeeds. Do not combine different subsets into a single favorable headline. The next runner is not implemented by this evidence intake. No full 498-video run, broad hyperparameter search, new training campaign or journal submission is justified yet.

If the nearest simple decoupling/readout baseline explains the recoveries, credit it and seek a precise residual limitation before claiming a contribution. If a distinct intervention survives matched simple controls and closest-method/code review, freeze it and reserve independent data for confirmation. Existing cases remain development evidence. Neurocomputing remains the working target; originality and submission readiness remain open.

## Timing

The recorded post-model phase is **75.330 seconds**, including preflight, verified feature work, arm replay and review/export. It excludes setup/download/model loading/final packaging. Five feature sets were reused. Arm timing scopes differ with input length and partitioning, and this is not a repeated full-system efficiency study or a fresh-container runtime guarantee.

## Subsequent implementation

The [stage-attribution notebook](../../notebooks/autodl/hermes-stage-attribution.ipynb) is now implemented with [instructions](../STAGE_ATTRIBUTION.md), fourteen local tests and pending complete GPU execution. It implements the 200-answer main matrix and four separate optional 204/q3 answers. This later code availability does not change the measured outcomes or establish an original improvement in this report.
