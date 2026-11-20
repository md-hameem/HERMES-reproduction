# Next diagnostic after the completed causal-scorer screen

Decision dated 20 November 2026. FIFO, HERMES and fresh causal scoring each score 24/40; fresh scoring changes one wrong choice and no correctness outcomes. This protocol follows the [audited analysis](<../../docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>). It is not a report of controls already executed. The separate [diagnostic notebook](../../notebooks/autodl/hermes-input-diagnostic.ipynb) now implements these arms with [run instructions](../../docs/INPUT_DIAGNOSTIC.md), eighteen local tests and pending fresh AutoDL execution.

The [adversarial research plan](research_execution_plan.md) supplies the broader execution and publication gates. Its updated literature check identifies [MemInStream](https://arxiv.org/abs/2611.15640) as direct prior work for evidence-solvability/necessity controls. This diagnostic is necessary for selecting our method; we do not claim the control design as novel.

## Preserve and reuse existing assets

Keep `/root/autodl-tmp/hermes_causal_pilot/videos/`, `features/`, the model cache and the original evidence ZIP if the container/data volume still exists. The ZIP contains feature hashes and metadata, not the feature tensors or videos. Before reuse, verify files against their saved manifests and confirm the same pinned model, processor, source/sampling settings and original feature chunk identities. Do not overwrite the archived runner or invalidate the completed pilot by editing its outputs.

Use a separate work/output namespace for additional treatments. Do not change the completed notebook just to add controls to its historical results. A follow-up runner must explicitly support verified reuse of the old feature cache rather than silently recompute a changed feature set. Avoid keeping a GPU idle while reviewing the CPU evidence.

## Stage A: inspect seven first-priority questions

The [generated queue](../../results/causal_pilot_2026-11-20/next_review_queue.csv) contains 18 questions. Copy it to a separately named working annotation sheet before editing; the audit regenerates the queue. Start with these seven:

| Question | Why first | Needed observation |
|---|---|---|
| 204/q0, 202/q0, 208/q0 | All methods wrong before any eviction. | Locate valid support in the exact sampled prefix; check temporal/part wording, detail and gold ambiguity. This is not a prior-eviction problem. |
| 212/q3, 218/q3 | HERMES/causal correct, FIFO wrong. | Determine whether old visual support is necessary or whether options/recent frames/language priors explain success. |
| 204/q3, 219/q3 | FIFO correct, HERMES/causal wrong. | Locate the actual support before query time and assess distraction, summary/readout or history effects without assuming the cause. |

Annotate exact support intervals/frame indices, sampling visibility, whether the evidence is sufficient, modality dependence, annotation ambiguity and confidence. For text overlays, inspect all relevant original sampled frames and the source sequence around their occurrence. Sparse contact sheets can miss a short overlay. If future frames are inspected to audit annotation timing, flag this separately and never feed them to a treatment answering the earlier question.

## Stage B: matched capability and input controls on the same 40 questions

Keep every question, choices, gold convention, model/processor revision, precision and greedy answer policy unchanged. These controls diagnose what information is useful; they are not deployable unknown-future-query retention methods or an original algorithm. Log intentional input/history changes explicitly. Use exact cached per-frame features only after verifying the feature layout and hash; do not substitute equally named unverified files.

| Control | Input and purpose | Boundary |
|---|---|---|
| Question-only | Initial system prompt plus current question/options, no visual input. Reset per query. | Correctness suggests a potential language/option shortcut, not proof that the original answer ignored video. Compare paired changes. |
| Recent 4 frames | The last up to four sampled frames available before the original query, freshly encoded into the LM without prior history or compression. | Tests short current evidence and old-context distraction. This changes input extent and dialogue history; do not attribute improvements solely to eviction. |
| Recent 20 frames | The last up to twenty sampled frames, about a 4K visual-token input at this layout, with fresh LM state and no compression. | Provides a bounded recent-information control. Equal GPU KV counts do not imply identical contextual information to FIFO's continually updated states. |
| Native uncompressed prefix, feasible subset | Original prefix without eviction/summary, retaining chronological ingestion and valid positions. Respect configured context/VRAM with room for the question and generated answer. | Mark infeasible questions as not run; never truncate silently or call HERMES 50K a native baseline. Compare on exactly the shared feasible question set. |

The first three controls would add 120 answers if all 40 questions execute, with additional native-prefix answers only where feasible. This is a proposed matrix, not a runtime guarantee. Start with two examples to validate input construction and measure real throughput/memory. Shared feature preparation and replay timers stay separate. Do not fabricate an aggregate score by mixing different question subsets.

## Stage C: verified-support oracle on a few failures

After Stage A annotations, choose a few well-supported failures, preferably covering early error and a post-eviction example. Present only verified pre-query support frames using a documented prompt and input order. Limit input within the model's validated context. This uses human support knowledge as an upper-bound diagnostic and is not a deployable proposed method.

If the current model still fails despite sufficient visible support, inspect annotation, input processing and perception/grounding. A few tests on a stronger compatible checkpoint can be considered after a VRAM/storage/time probe; do not start a broad multi-model run automatically. If oracle support recovers the answer, investigate whether the original policy encoded and then removed, summarized or failed to use that cue. Source-token survival must distinguish directly retained tokens from summary ancestry; a compressed summary is not equivalent to total deletion of the old evidence.

## Decision rules for an original contribution

- **Recoverable retention/readout failure:** support is valid and sampled, a capability reference answers correctly, and a controlled cache/readout intervention consistently changes the answer. Investigate a precise memory mechanism and compare its full details with nearest existing methods before coding a claimed contribution.
- **Recent input beats long context:** investigate distraction, temporal/part grounding or a precisely defined readout problem. A fixed window or generic retrieval trigger already has precedents and is not automatically original.
- **Question-only explains successes:** reassess the need for long memory in the selected cases. Include visual-dependence checks and diversify evaluation before making a memory claim.
- **Support absent or annotation temporally invalid:** this is not evidence for a better eviction policy. Separate sampling/annotation cases and define any new research problem accordingly.
- **No repeatable recoverable effect:** stop developing that candidate and reconsider the problem/data/backbone. Do not spend on a full-benchmark grid merely to search for a favorable result.

If a history-feedback hypothesis survives the support checks, test it separately by forking identical state and changing only generated/corrected/wrong/question-only past-answer conditioning before subsequent ingestion. The completed pilot did not perform this intervention. Keep it separate from scorer changes to preserve attribution.

## Publication gate

The next round should produce a failure-attribution decision memo, not a submission claim. The Neurocomputing regular-paper route still needs a distinct neural-inference contribution, close prior comparisons, appropriate matched validation and generalization at the claimed scope. All existing outcomes are development evidence. After a method is frozen, use a justified independent confirmation plan and preserve negative results.
