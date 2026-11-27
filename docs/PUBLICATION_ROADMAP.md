# Neurocomputing publication roadmap

Updated 27 November 2026. Neurocomputing remains the working target for a regular article under the advisor's original-method requirement. The DROID terminal-attention feasibility run recovered three robust pairs at fixed input length and positions. The next milestone is a source-disjoint test of automatic span selection against matched recency. [Terminal-attention audit](<reports/2026-11-27 droid terminal attention results.md>)

Use the [phased Neurocomputing plan](../paper/neurocomputing/submission_phase_plan.md) for the current go/no-go criteria, confirmation design, pre-submission review and revision workflow. The previous experiment records below preserve their original decisions.

**20 November input diagnostic:** Question-only scored 14/40, recent-four 25/40, recent-20 21/40 and native 10/14. The latter matched historical baselines on the exact native subset. [Audited decision](<reports/2026-11-20 input diagnostic analysis and next decision.md>)

**Subsequent context result:** the [new analysis](<reports/2026-11-20 context diagnostic analysis and next decision.md>) audits 74 answers, all comparable replays and answer-level chunking stability. Fixed-position content replacement recovers three selected full-prefix errors; local filler also harms one answer. Two analyst support probes succeed. Move to the [stage-attribution/existing-baseline gate](../paper/neurocomputing/stage_attribution_protocol.md), not more padding variants or a full benchmark. No original method or certified-oracle result is established.

Source requirements are recorded in the [19-page guide assessment](<reports/Neurocomputing submission requirements and HERMES readiness.md>). Q1 remains mandatory. Neurocomputing is CCF C in the supplied catalogue; the user has not explicitly resolved whether B is an additional condition. The applicable Q1 database/category/year and recognized publication milestone must be checked before submission. This check does not prevent the research preparation below.

## Current decision after evidence intake

Stages 1–2 below are complete for the supplied screen. The [audited decision](<reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) finds no QA benefit from fresh causal scoring at this scope and three shared errors before any eviction. Proceed to the [failure-attribution/control protocol](../paper/neurocomputing/next_diagnostic_protocol.md) before choosing a new memory mechanism. Original-method and manuscript-result claims remain pending. Keep the initial intake instructions below for reproducibility and later packages.

## 1. Completed development screen and preserved evidence

Run [hermes-causal-scorer-pilot.ipynb](../notebooks/autodl/hermes-causal-scorer-pilot.ipynb) with the chosen settings. The default eight-video run contains **40 distinct questions answered by three methods, producing 120 answers**. Methods are FIFO, current HERMES and fresh causal HERMES at 4K. All questions in each video are replayed in order. The videos were selected using existing outcomes, so this is an exploratory development screen.

At completion, download `HERMES_causal_scorer_pilot_evidence.zip` and provide it for analysis. Preserve the `WORK` directory if further execution or evidence review is needed. The ZIP excludes model weights, raw videos and feature tensors. Do not delete those before extracting any additional required evidence. Once the evidence is safely downloaded, avoid paying for an idle GPU while CPU analysis and method design proceed.

Check `exports/completion.json`: `complete` means all planned blocks finished; `partial_time_budget` means an explicitly partial screen. A failed or partial package can still support troubleshooting, but its QA coverage must not be represented as the full planned run. A valid eight-video result requires eight complete checkpoints for each method and 40 aligned QA predictions per method.

## 2. Audit and interpret the ZIP before deciding on a method

| Audit | Required evidence |
|---|---|
| Integrity | Recompute every entry's size and SHA256 against `evidence_manifest.json`; check expected file coverage. |
| Correctness gate | Inspect `preflight.json` for feature-ingestion equivalence, native attention agreement before/after pruning, causality and source-cache preservation. A failed gate prevents a scientific treatment interpretation. |
| Run identity | Check source/model revisions, environment, settings, selected videos, full conversation order and annotation/prediction alignment. |
| Completion | Match completion status, planned/completed counts and method/video checkpoints. |
| QA effects | Recompute correct counts and paired wins/losses for causal/current/FIFO, including regressions and all-history questions as well as selected targets. |
| Cost | Separate shared video preparation from method replay, and compare memory, compression and answer costs. Reused-feature timings are not live full-system latency. |
| Failure attribution | Inspect actual video support and sampled timestamps; use contact sheets as an index only. Distinguish missed sampling, perception/reasoning, retention and readout. Predictions alone cannot locate lost evidence. |

Publish a short pilot interpretation with question-level examples and explicit limits. Eight selected videos cannot establish a general accuracy gain or support a population-level significance claim. Equal or worse accuracy is a valid finding. Do not tune a mechanism until the result is understood.

## 3. Choose the next branch from the evidence

| Pilot result | Next action | Decision boundary |
|---|---|---|
| Preflight or replay fails | Correct the implementation/environment and rerun the affected stage. | No QA-quality claim from an invalid treatment. |
| Fresh scoring recovers useful answers but increases maintenance cost | Confirm supporting examples, then investigate reducing fresh-scoring cost through contextual probe-state reuse and controlled refresh. | State reuse is a provisional candidate. Check the nearest full methods/code and compare with fresh scoring and matched refresh controls. |
| Fresh scoring changes attention/selection without convincing QA benefit | Review shared failures and a few verified-support capability controls before building more machinery. | Do not promote attention agreement into an accuracy claim. A small null screen does not rule out small effects. |
| Evidence is absent from the sampled input | Investigate sampling/detail allocation with equal input/encoding cost if this is the chosen research problem. | Changing eviction cannot recover cues that were never encoded. |
| Earlier incorrect answers appear related to later errors | Run a separate controlled history intervention on identical pre-ingestion state. | Generated/corrected/plausibly wrong/question-only histories diagnose causality. Correlation alone is insufficient. |
| Mixed improvements and regressions | Stratify by actual support age, question type and failure stage; check repeatability. | Avoid selecting a favorable subgroup after seeing results and treating it as confirmation. |

The scorer, input, context and [stage-attribution](<reports/2026-11-21 stage attribution analysis and next decision.md>) diagnostics have completed. The strongest stage contrast improves from 21/40 to 27/40 by masking direct history during answering, while compact recent-four already scores 25/40. This result motivates a matched construction control and a comparison with the simple baseline on source-separated data.

## 4. Define one original contribution before the main run

Write a one-page method specification containing the observed failure, nearest prior mechanisms, a precise algorithm, its expected causal effect, all auxiliary memory/cost and a falsifiable claim. Complete mechanism-level comparisons with the closest literature. Generic query banks, adaptive hierarchy and visual reinjection already have precedents in the [gap map](<reports/HERMES research gap map.md>).

For a state-reuse branch, relevant controls include fresh scoring, reuse without refresh, fixed-interval refresh and random refresh with matched refresh counts. Any selective refresh rule must improve the measured quality/cost trade-off beyond these controls. Account for changed history prompts, token identity, positions and invalidation after eviction/compaction. Correcting the reproduction's causal scorer alone is a baseline correction; originality remains a separate question.

Stop or revise a candidate if the matched controls explain its gains, if the nearest published method already implements it, or if useful gains disappear after valid replay. A smaller, defensible contribution is preferable to an unsupported broad claim.

## 5. Freeze a budgeted validation plan

Use actual pilot throughput and resource measurements to size the next GPU allocation. Freeze development choices before evaluating confirmation data. The existing full StreamingBench results have already been inspected and influenced the project; a different subset of that inspected benchmark is not an untouched external test.

The main comparison should include FIFO, current HERMES, fresh causal HERMES, the proposed method and the closest feasible relevant published control. Document implementation fidelity and limitations rather than call an approximate reimplementation an exact baseline. Keep sampled frames, decoding, precision, budget and prompt policies matched wherever the claim requires them.

Evaluate useful QA behavior alongside memory and cost at more than one meaningful budget if feasible. Add component/trigger ablations and repeats that address the observed failure or known instability. A second compatible backbone or an independent benchmark can support a broader claim if affordable; these are proposed evidence choices, not fixed journal requirements. Check source-video overlap before treating two benchmarks as independent.

Report paired outcomes with video-cluster uncertainty on an appropriate evaluation set, selected/exploratory subgroup labels, hardware/session differences and complete costs. Preserve negative results. A limited evaluation can support only a correspondingly limited claim. No specific improvement percentage, number of models/datasets or runtime target is invented as a Neurocomputing rule.

## 6. Draft alongside research, complete claims after validation

The [paper workspace](../paper/neurocomputing/README.md) now contains a manuscript outline and claim/evidence ledger. Background, related-work organization, task definition, baseline limitations and reproducibility details can be drafted now. The proposed algorithm, claimed contributions, abstract results, result tables and conclusion require actual method/evaluation evidence.

The intended regular paper should explain a real contribution to neural-network inference/learning systems, rather than present an unrelated combination of existing algorithms. Maintain a distinction between earlier reproduction results and the new matched treatment. Credit HERMES and all upstream implementations.

## 7. Assemble the submission package after the science is ready

Prepare editable LaTeX/Word source, an abstract of at most 250 words, 1–7 keywords, separate figure files and editable tables. Highlights are encouraged; if supplied, use 3–5 bullets of at most 85 characters each. Prepare applicable CRediT, funding, conflicts, data-availability and AI-assistance statements; author biographies/photos are requested in the supplied guide. Verify source references and resolve final author order with the supervisor.

Freeze an appropriately licensed, citable code/data/evidence release in a suitable repository. Do not redistribute third-party videos or weights without the corresponding rights; explain availability limits and lawful retrieval. Draft a cover letter describing the demonstrated contribution and journal fit, without invented results or acceptance claims.

Use the current author guide and submission system for a final format/policy check. Check the university's Q1 eligibility and required publication milestone, and applicable costs/licensing before selecting an OA option. Obtain all authors' approval and supervisor review before actual submission. The current authorization covers preparation, not a manuscript submission today.

## Current next action

The construction selector, frozen short-query confirmation and target-binding verifier did not meet their respective accuracy gates. The single-reviewer StreamingBench pass also fell below the same-clip pilot threshold. Preserve those results as development evidence. [Target-binding audit](<reports/2026-11-22 q1 audit of target binding.md>) · [Blinded review](<reports/2026-11-23 blinded visual support first pass.md>)

The DROID terminal-context comparison found a drop from 6/7 robust pairs in isolation to 1/7 with full history. At fixed input length and positions, annotation-guided terminal readout recovered three robust pairs and preserved the causal control's only robust success. Evaluate an annotation-blind selector against a matched three-frame recency window on source-disjoint cases. [Terminal-attention results](<reports/2026-11-27 droid terminal attention results.md>)
