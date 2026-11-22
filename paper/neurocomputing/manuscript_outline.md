# Manuscript planning outline

Status as of 23 November 2026. Neurocomputing is the working venue for a regular article. The current scientific question is whether time-valid earlier evidence can correct a recent answer at a fixed streaming budget. The completed reproduction and controls can support the background and diagnostic sections. The nine-pair TwinPast GPU diagnostic and any method results are pending. [Execution plan](historical_correction_execution_plan.md)

## Title, abstract and keywords

Choose the title after the actual contribution is fixed. Write the final abstract after validation, within 250 words. State the problem, actual method, tested setting, measured results and supported conclusion. Keep numerical result slots unfilled until evidence is available. Select 1–7 keywords relevant to the eventual method.

## 1. Introduction

Draft now: streaming inference constraints, why future-question relevance is hard to preserve, the target resource model and the importance of reliable evaluation. Cite existing HERMES and the closest later memory methods. The audited reproduction provides motivation with an inconclusive HERMES/FIFO difference in its tested 0.5B setting. It does not establish a universal failure of hierarchical memory.

Fill later: the precise observed failure and two or three concrete contributions supported by the new method/evaluation. Do not describe an implementation correction as a certified original algorithm.

## 2. Related work

Organize around streaming video-language inference, bounded memory versus offloaded retrieval, attention/probe scoring and maintenance cost. Use the gap ledger to identify mechanisms already implemented. Compare the eventual method with the nearest full papers and available code, including evidence limits for abstract-only checks. Distinguish proposed future work from implemented mechanisms.

## 3. Task, resource model and diagnostic foundation

Define unknown future questions, chronological QA, retained GPU KV, auxiliary states, CPU/disk storage and maintenance/readout timing. Specify source-video identities and sampling timestamps. Separate native full-prefix QA from a native attention reference on the same compressed cache.

Describe the reproduction and pilot as development evidence with their respective scope and selection rules. The default pilot contains eight outcome-selected videos, 40 distinct questions and 120 answers across three methods. The ZIP is now audited: all eight videos completed, all methods score 24/40, and causal scoring changes zero correctness outcomes relative to HERMES. This is diagnostic development evidence, not the new method's main result.

The subsequent input controls completed: question-only 14/40, recent-4 25/40, recent-20 21/40 and native 10/14, matching historical baselines on that exact native subset. Include regressions and the unpruned context-sensitive examples if relevant to the eventual method. Verified-support answers and an isolated cause remain pending; the single net recent-4 gain is not the proposed contribution.

The later context screen reproduces all 26 comparable choices and preserves choices across 19 answered partition contrasts. Fixed-position opening replacement recovers three selected full-prefix failures; another filler also causes a regression. Two analyst-selected support probes answer correctly. These motivate stage-specific attribution and an existing-method comparator, not a claim of original method or general accuracy improvement. Certified human-oracle evidence and the specific internal path remain open.

## 4. Proposed method — pending selection and validation

Supply the precise algorithm, state representation, update/invalidation rules, positional semantics, complexity and all memory costs. Explain why the mechanism addresses the observed failure and how it differs from the nearest work. Include pseudocode and a reproducible diagram. State any assumptions and failure conditions.

No mechanism or originality claim is filled in at this stage. Contextual probe-state reuse and separate history-feedback diagnostics remain conditional branches in the roadmap.

## 5. Experimental design

Record actual model/dataset revisions, hardware, dependencies, input frames, precision, decoding, prompts, budgets and seeds. Describe development/confirmation boundaries and dataset overlap. Plan controls around the claim: FIFO, current HERMES, fresh causal scoring, proposed method and a feasible nearest prior method. Add relevant ablations, repeats and budget comparisons with a measured compute plan.

The number of models/datasets must justify the scope of the final claim. It is not a fixed minimum imposed by the supplied guide. Avoid whole-pipeline speed claims from replay-only timing with shared vision features.

## 6. Results — pending actual measurements

Planned evidence products, not completed findings:

- Main QA/memory/cost table with method identities and consistent timing scopes.
- Paired wins/losses and uncertainty appropriate to the evaluation sample.
- Quality/cost curves across tested budgets, if those comparisons are executed.
- Ablations showing which component or trigger accounts for useful behavior.
- Repeatability, regressions and failure cases.
- Evidence-support examples based on actual frames and, where required, token/retention traces.

Keep the small pilot descriptive. Mark reused development outcomes and exploratory subgroups. Use only observed values and preserve raw provenance behind each figure/table.

## 7. Discussion and limitations

Explain where the method helps, where it regresses and what the experiments cannot establish. Address selected development examples, backbone capacity, sampling, positional handling, external storage, timing scope, hardware/session differences and untested domains as applicable. Report null or inconclusive results without substituting attention agreement for QA benefit.

## 8. Conclusion

Summarize the demonstrated contribution and bounded outcome. Do not imply publication readiness or broad superiority from a favorable small-screen result.

## Submission materials

Prepare applicable acknowledgements, CRediT roles, funding/conflicts, data availability and accurate AI-use disclosure. Use the required placement from the current guide, including AI disclosure before references when applicable. Verify reference identities, formal publication versions and upstream software attribution. Supply editable tables/equations, separate figures, requested author biographies/photos, supplements and optional highlights within their limits. Confirm final author approval before submission.
