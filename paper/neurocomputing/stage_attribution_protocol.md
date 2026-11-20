# Next experiment: separate recent-cache construction from historical readout

Plan dated 20 November 2026, following the [audited context result](<../../docs/reports/2026-11-20 context diagnostic analysis and next decision.md>). The separate [stage notebook](../../notebooks/autodl/hermes-stage-attribution.ipynb) now implements this design with [instructions](../../docs/STAGE_ATTRIBUTION.md) and fourteen local tests; complete GPU execution is pending. This is a bounded attribution/baseline gate before selecting at most one candidate method, not a new paper contribution by itself.

## Why this is now justified

Three selected full-prefix failures recover after earlier-content replacement at fixed length/recent positions; 26 historical replays and 19 answered chunk-partition contrasts preserve all choices. Two analyst-selected support inputs answer correctly. Filler replacement changes both the construction of recent representations and historical content available to the question. It therefore does not locate the harmful stage. Repeating padding variations on the same cases has diminishing research value.

## Fixed input and four-factor cells

On **all 40 existing development questions**, select the last up to 20 original sampled frames. Split them into older H (up to 16 frames) and recent R (last up to four). Use the same features/model/precision/greedy decoding. All features remain pre-query. Construct H identically from the initial prompt in every cell; no eviction, generated-answer history or auxiliary semantic retrieval is introduced.

| Cell | Does R encoding attend H? | Can question/answer tokens attend H directly? | Role |
|---|---|---|---|
| A | Yes | Yes | Fresh recent-20 reference |
| B | No | Yes | Independent-recent-cache construction comparator |
| C | Yes | No | Remove direct old-history access while preserving its influence already mixed into R |
| D | No | No | Independent recent information with original logical positions retained |

Retain the **same physical H/R cache slots and budget** for answer-time comparisons; an attention mask changes access, not reported storage. Sink/initial prompt tokens remain available. For B/D, encode R against only initial prompt/sink state while assigning R the same logical positions it had after H, then assemble H/R for answering with correct rotary positions. Do not simply concatenate independently encoded keys with incompatible phases. Native-mask/position and source-cache immutability preflight is mandatory, and no-op cell A must reproduce the tested recent-20 path.

The implemented equivalent keeps H physically present and applies an H-blocking causal mask during R encoding. GPU preflight compares its R K/V against physically independent construction at the same logical positions. This avoids unnecessary cache reassembly without changing the intended information visibility. H states are sliced from A's causally encoded cache and held identical across cells. The original partition is preserved for A so it can replay the earlier reference.

The revised preflight uses FP32 math attention to check structural equivalence on the two-frame fixture, then restores the model's original precision and checks FP16 agreement with separately declared tolerances. The run records per-layer differences and requires both passes, exact zero attention on blocked positions, matching tested argmaxes and the existing per-question replay. The initial FP16 tolerance failure occurred before QA and supplies no stage-attribution outcomes. [Validation and rerun instructions](../../docs/STAGE_ATTRIBUTION.md).

Report transient caches, feature buffers, complete query time and peak allocation. Equal answer-cache slots do not imply equal total resource cost, especially when independent construction or snapshots require extra work/state.

For C/D, mask H for the question prefill **and every generated answer token at every layer**. Masking only the first step is insufficient. Preserve causality among question/generation tokens. H can still influence C indirectly through R: hiding direct attention is not erasing all historical information. Store explicit logical/physical position maps, cache lengths and mask policy.

Add **one compact recent-4 reference**, because D's preserved position gap from the sink differs from ordinary fresh recent-4. Thus the main matrix plans **5 × 40 = 200 answers**, subject to implementation preflight and declared failures. Do not interpret A/B/C/D differences as semantic effects until attention/position equivalence is verified. Contextless earliest questions with no H are null controls, not independent history-intervention examples.

## Longer-prefix example and scope

204/q3 is already correct in recent-20 but wrong with its full 71-frame prefix. A single separate A–D check at the same feasible full-prefix length can localize that example's stage; keep these four answers and their resource costs separate from the 40-question bounded-window table. Do not turn it into a whole-dataset long-context sweep. Candidate-support and source visibility are already observed for two 204 examples and need not be repeatedly rerun to inflate evidence.

## Baselines, novelty and outcome rules

Cell B is **inspired by the independent recent-cache idea in [DSCache §4.3](https://arxiv.org/html/2605.01858v1)**. It is not a faithful implementation of DSCache's complete position-agnostic encoding, cumulative history policy, sink and approximate-cache design. A published-method superiority claim requires a compatible full-method/code baseline, explicit adaptations and matched resources. Generic independent recent caching is not our original method.

Compare all cells on identical questions, log both choice/correctness changes, and retain every regression. Attribute as follows:

- B helps relative to A with stable positions/masks: recent-cache construction is a candidate harmful path; test whether existing decoupling already explains the benefit.
- C helps relative to A: direct historical readout is a candidate harmful path; a blanket history mask may sacrifice useful historical reasoning, so assess corresponding losses.
- D helps while B/C do not: interactions matter; do not assign one stage prematurely.
- Compact recent-4 differs from D: position/sink-distance effects remain relevant and must be accounted for.
- Results fail replay/no-op equivalence: stop scientific interpretation and fix the implementation.
- No consistent recoverable benefit beyond existing/simple controls: stop this candidate branch rather than add an adaptive gate around a null result.

This protocol may identify a useful existing baseline and nothing original. If so, credit it and identify a specific unresolved failure before proposing a mechanism. Before a new candidate, write its changed computation, information assumptions, memory/latency costs and distinction from DSCache, SelectStream, StreamScout and nearest relevant methods. A confidence switch, recent/history split or calibrated injection is not automatically novel.

At most one candidate should follow this matrix, with its rule frozen before independent confirmation. No gold-dependent arm selection in deployment, no tuning until significance appears, and no claim that the inspected 40 questions are held-out evidence. Expand to an external compatible source-disjoint setting only after a distinct mechanism and baseline feasibility are established. Preserve the old AutoDL assets and avoid paying for an idle GPU during source/method review.
