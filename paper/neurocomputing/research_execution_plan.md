# Research execution plan and adversarial review

Decision date: 20 November 2026. Working venue: **Neurocomputing**, assuming the user's reference to “neuroscience” means the journal previously selected. This plan does not propose biological neuroscience claims. It extends the [diagnostic protocol](next_diagnostic_protocol.md) and [completed pilot analysis](<../../docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>).

**Execution update:** Phase 1's mandatory input controls and feasible native arm have now completed. The [input diagnostic decision](<../../docs/reports/2026-11-20 input diagnostic analysis and next decision.md>) records 14/40 question-only, 25/40 recent-4, 21/40 recent-20 and 10/14 native, with no oracle answers. It prioritizes support-traced context/position interventions before the conditional generated-answer-feedback candidate below. The earlier candidate ordering was provisional. Originality and independent confirmation remain pending.

## Honest starting position

The engineering and reproducibility foundation is substantial; the original contribution remains missing. The full 2,495-question HERMES/FIFO difference is +0.3607 percentage points with a video-cluster interval crossing zero. The selected 40-question scorer pilot gives 24 correct answers for every method. Fresh causal scoring changes no correctness outcome and increases recorded replay time. These are informative development results, not proof that HERMES never works or that the methods are equivalent.

Three common failures precede eviction. No eviction rule can fix an error by restoring tokens that were never evicted. Fourteen common failures are not automatically fourteen recoverable memory failures. Existing 0.5B-model results cannot substantiate general claims about modern VLMs. A stronger backbone may expose useful headroom or remove the apparent failure; it is a control, not itself our contribution.

The existing literature ledger was a selective review of 51 local PDFs, not exhaustive novelty clearance. New primary-source checks below expose additional overlap. A large repository, many reviewed papers, correct attention, numerous experiments, or a named algorithm do not by themselves establish an original research result.

## Adversarial reviewer objections

| Reviewer objection | Present exposure | Evidence required to answer it |
|---|---|---|
| This is a reproduction correction | Causal scoring is a corrected implementation and yields no QA gain in this screen | A distinct mechanism beyond restoring intended behavior, with controlled downstream benefit |
| The model cannot perceive the answer | Early unpruned errors and unannotated support | Exact sampled-support inspection, native/capability references and a few verified-support tests |
| Your task does not need long memory | Recent observations or option priors may explain successes | Text-only/recent controls and evidence-age annotation; report the complete selected set |
| This is already published | Generic selective retrieval, uncertainty gates and memory tiers are crowded | Versioned full-method/code comparison against nearest work, not titles/abstracts alone |
| You tuned on the test set | All StreamingBench outcomes already informed the direction | Treat them as development evidence; freeze method before independent confirmation |
| Your gain comes from extra resources | New state/retrieval can add hidden storage and computation | Equal total budget or explicit resource trade-off; CPU/disk, auxiliary state and latency included |
| Your oracle leaks future knowledge | Support and corrected-history controls use privileged information | Label diagnostic-only arms; no future frames, answers or gold-dependent decisions in deployment |
| Your speed claim excludes expensive work | Cached-feature replay omits shared visual work | Report component and complete wall time separately; measure arrivals/lag only if claiming live streaming |
| A few favorable videos drove the result | Eight outcome-selected videos, correlated QA | Independent source groups, paired video-level uncertainty and complete regression reporting |

## Updated prior-art boundary

Primary pages checked online on 20 November 2026. These entries screen overlap; they are not a completed implementation audit or claims of independent replication. Published numbers from different backbones/protocols are not comparable to our 60% screen.

| Work | Verified overlap | Consequence for our plan |
|---|---|---|
| [MemInStream, v1, 11 Nov 2026](https://arxiv.org/abs/2611.15640), [construction/validation text](https://arxiv.org/html/2611.15640v1) | Evidence localization, evidence solvability, and controls excluding recent/post-evidence/text-only shortcuts | Failure-attribution controls are necessary research practice here, not our new contribution. Inspect its assets/license/protocol before using it as external confirmation. |
| [SelectStream, v2, 26 Sep 2026](https://arxiv.org/abs/2606.16353) | Selective latent historical evidence, current-scene visibility, adaptive windows and consolidation | “Use less history/select relevant memory” is insufficient novelty. |
| [StreamScout, v1](https://arxiv.org/abs/2609.00291) | Escalation from recent views to historical views and query-salient retrieval | A recent-first or confidence-triggered access cascade is not a cleared contribution. |
| [StreamFlow, v1](https://arxiv.org/abs/2608.10949) | Dynamics-aware memory and attention-guided latent retrieval during generation | Generic EDMI/reinjection or visual-attention-triggered recall is occupied. |
| [DSCache, v1](https://arxiv.org/abs/2605.01858) | Decoupled cumulative/instant cache construction and position handling | Fresh recent-cache construction alone is insufficient novelty. |
| [StreamClarify, v1](https://arxiv.org/abs/2611.01263) | Pre-answer uncertainty, bounded active memory and historical retrieval for clarification | A generic entropy gate plus hierarchical memory is insufficient novelty. |

These findings change the scope of our proposal, not the value of running the controls. The earlier [gap map](<../../docs/reports/HERMES research gap map.md>) retains dated hypotheses; the completed null screen and this plan take precedence when allocating new compute.

## Chosen scientific question

**Under a fixed streaming-memory budget, when does the system lose usable evidence, and can one specific inference intervention prevent that loss without equivalent cost or regressions elsewhere?**

We will first distinguish lack of input, inability to interpret input, harmful context, and loss/misuse of previously usable support. Only the last two can justify the immediate memory/readout directions below. “Novelty” remains an obligation to establish, not an assumption in this question.

## Phase 1: locate headroom before another method

1. Preserve the old container's videos/features/model cache if available and verify hashes. Otherwise reacquire pinned assets. Keep uploaded evidence immutable. Complete source-support annotation for the seven first-priority questions, then the remaining eleven queue entries. Record uncertainty and ambiguous gold rather than forcing a cause. Have the student review proposed annotations; ask an independent reviewer to check disputed and publication-critical examples where feasible.
2. Implement a separate diagnostic notebook using the [existing protocol](next_diagnostic_protocol.md). First validate two examples, input identity, history/reset behavior, context feasibility and timing. Run question-only, recent-4 and recent-20 on all 40 questions: 120 new answers. Native uncompressed-prefix QA is a separately reported feasible subset. Review sufficiently supported failures with a few human-support oracle tests. Oracle success is evidence of potential recoverability, not proof of eviction causation.
3. Keep original queries/choices/golds/precision fixed. Select recent frames strictly at or before query time. Separate deliberate history changes from eviction changes. Compare alternatives only on common question identities; retain infeasible cases in an explicit status table.
4. Produce a failure table: support available in source, sampled, interpreted by capability reference, directly retained or summarized, and used successfully at readout. Support age means the actual decisive interval, not simply time since video start or dataset query timestamp. Sparse thumbnails cannot certify visibility or absence.

**Exit:** identify a reproducible intervention-sensitive failure on more than one independent source video. This is a mechanism-screening criterion, not statistical confirmation. If none appears, stop the immediate memory-method branch and investigate input/backbone/task suitability. If early failures persist with verified support, run a small stronger compatible model probe before committing to a new algorithm. No broad multi-model grid.

## Phase 2: compare two narrow hypotheses, then choose one

### First diagnostic candidate: generated-answer feedback

Why examine it: the local guidance path conditions future retention on previous generated answers. This is a concrete dependency in our code and a cheap intervention. It is not yet a demonstrated harmful effect or a novel observation relative to every paper.

Fork identical pre-intervention cache state. Change only the past-answer conditioning used by subsequent retention guidance: original generated answer, question-only history, corrected past answer and a plausible wrong past answer. Keep frames, future questions, positions and random seeds matched. Do not change ordinary user dialogue and retention guidance simultaneously. Correction uses only an already answered question's gold as a diagnostic; it is not an implementable policy with free labels.

Measure selection divergence, verified cue survival/summary ancestry, later correctness and cost. Repeat with different plausible past-answer errors; include an unaffected-history control. A wrong past answer harming the later answer can reflect legitimate dialogue dependence, so demonstrate that the path operates through retention rather than direct answer conditioning.

If a reproducible harmful path exists, investigate a narrowly specified answer-independent or reliability-aware retention policy. First beat the simple question-only-history remedy and fixed/random gates with matched intervention counts. If deleting answer text solves everything, that is a useful baseline/fix; a complex confidence mechanism still needs independent justification. Compare full methods against SAVEMem, StreamAgent, LORE-KV and other close work in the existing ledger before claiming novelty.

### Conditional alternative: irreversible loss of a usable cue

Use only examples where sampled evidence is sufficient, a capability reference can answer, and a controlled retention/readout intervention changes the answer. Trace whether the cue is discarded, diluted into a summary, represented but not accessed, or confused with a newer state. Restore verified support only for diagnosis and account for any displaced tokens.

If the problem is loss during repeated consolidation, investigate a precise bounded residual/provenance mechanism tied to that loss. If it is harmful context, investigate a precise separation/readout operation. Both are hypotheses: ProtoKV, ReMem, FreshMem, DSCache, SelectStream and related methods create substantial overlap. Simple residual storage, recency splits, semantic retrieval, or evidence-age plots alone do not establish novelty.

**Exit:** choose at most one mechanism after demonstrating its cause and writing a nearest-work comparison that identifies the changed computation, information assumptions and cost. Abandon a candidate if it only changes attention, fails against its simplest control, requires oracle support in deployment, or has no distinction beyond existing work. The fallback is to revise the scientific question, not rename the mechanism.

## Phase 3: freeze the method and evaluate the claim

Before confirmation, write a one-page specification: equations/pseudocode, state update and readout, unchanged backbone, every parameter and its tuning source, exact causality constraints, total memory accounting, intended scope, primary comparison and rejection conditions. Give the method a name only after this specification is useful.

Minimum comparison set is claim-dependent: FIFO, current HERMES, fresh causal HERMES where scorer fidelity matters, recent-window control, the closest feasible published mechanism, and the simplest alternative to the proposed intervention. Include native QA on a valid feasible subset. A method with growing CPU storage cannot be sold as equal-total-memory to strict eviction. Failed or incompatible implementations must be disclosed; published scores on other models are not matched baselines.

On development data, test the main mechanism off/on and its simplest alternative, then one meaningful budget stress setting. After freezing, use an external compatible dataset/source-disjoint set not already used for tuning. Inspect MemInStream access, license, frame budget, question format, backbone dependence of filtering and source overlap before selecting it; OVO-Bench is another candidate subject to the same feasibility audit. Reusing a benchmark is acceptable with transparent limits, but do not call inspected data untouched. Human-selected hard cases remain diagnostics and cannot replace complete evaluation of the declared confirmation set.

Prefer a second compatible backbone for a general-method claim, beginning with a small feasibility test. If compute cannot support it, narrow the claim and disclose the limitation; two datasets/two models are our possible evidence design, not a universal journal rule. A compact existing method can be a credible paper when its mechanism is distinct and the evidence is convincing; scale alone is not the contribution.

Primary reporting: paired QA difference, both wins and losses, video/source-cluster uncertainty, source-count and dependence assumptions. Size confirmation using development discordance/cluster variability and the precision needed for the claim, rather than an invented minimum question count or searching until significance appears. Greedy repeated runs on the same examples do not increase the effective sample size. For timing use matched repeated interleaving, medians/tails as appropriate, and separate full pipeline, shared feature preparation, memory maintenance and answer cost. Log all auxiliary GPU/CPU/disk bytes and growth with stream length. Keep real arrival/lag tests optional unless making a real-time claim.

**Exit:** a distinct mechanism improves the declared accuracy/resource trade-off beyond the nearest/simple controls, with uncertainty and regressions consistent with the claimed scope, and shows appropriate independent confirmation. If the result remains inconclusive, report it as such and revise the submission claim. No fixed gain or p-value guarantees acceptance.

## Budget and schedule

Human work and rented GPU time are different resources. Planning envelope: 2–3 days for annotation/closest-work checking; one short GPU diagnostic session after the notebook passes CPU checks; 2–5 days to specify/prototype one supported mechanism; then a separate budget decision for confirmation. These are work estimates, not completion promises. Annotation and novelty clearance can take longer than inference.

For the available few rented GPU days, reserve roughly 15% for headroom/feasibility, 25% for one prototype, 40% for frozen comparisons/confirmation and 20% for recovery/repeats. This is a proposed allocation, not a monetary estimate. Measure two-video throughput and storage/context limits before converting it to hours or sample counts. Do not extrapolate 5.45 minutes of old post-model phase to downloads, new backbones or longer prefixes. Avoid simultaneous broad sweeps, repeated setup and idle GPU annotation. Keep daily checkpoint exports outside a disposable container.

After two well-controlled candidate failures, pause new mechanism development and reassess with the advisor. Do not spend the entire confirmation reserve pursuing a positive development result. A few GPU days may establish a promising mechanism; they do not guarantee a submission-ready paper or journal acceptance.

## Publication package and remaining eligibility questions

Neurocomputing's [publisher description](https://shop.elsevier.com/journals/neurocomputing/0925-2312) includes neural/learning systems and computer vision applications. Our contribution should concern neural inference/memory behavior, with original mechanism and matched evidence. Do not add biological terminology to manufacture journal fit. The current ScienceDirect scope/guide URLs returned 403 during this check; detailed format requirements still refer to the archived supplied guide and [readiness assessment](<../../docs/reports/Neurocomputing submission requirements and HERMES readiness.md>).

Once evidence supports the claim, draft a regular article around: precise failure and hypothesis; distinct method relative to nearest work; matched complete comparisons; mechanism tests/ablations; uncertainty, limits and reproducibility. Preserve the null reproduction/scorer results as motivation and context, without implying they are universal refutations. Follow the current journal template/guide, prepare required declarations/data statement and approve final authorship/submission with the supervisor. No fabricated contribution bullets, title or abstract gains.

Resolve in parallel: institution-accepted Q1 database/category/year, whether CCF B is an additional requirement, qualifying article type, and whether the deadline requires submission, acceptance or publication. Neurocomputing's name alone certifies none of these. These administrative questions do not block the diagnostic work.

## Immediate deliverables

1. A reviewed support sheet copied from the generated queue, with seven initial cases and unresolved ambiguities explicit.
2. A new resumable AutoDL diagnostic notebook implementing the 120-answer control matrix, feasible native arm and explicit optional annotated-support input. Implemented and executed as [hermes-input-diagnostic.ipynb](../../notebooks/autodl/hermes-input-diagnostic.ipynb), with eighteen implementation tests, five saved-evidence audit tests and [run instructions](../../docs/INPUT_DIAGNOSTIC.md). Verified-support QA remains pending; fixed-position context replacements and new mechanisms require separate implementation.
3. A diagnostic decision memo choosing feedback, retention/readout, input/backbone investigation, or stopping the immediate branch.
4. A full-method/code comparison against nearest work before implementing the selected proposed contribution.
5. Only after those gates: one prototype, frozen confirmation plan, and an evidence-led manuscript.
