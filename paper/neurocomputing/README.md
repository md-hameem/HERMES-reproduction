# Neurocomputing paper preparation

Working target selected by the user on 20 November 2026. Intended route is a regular research article with an original neural-inference improvement. The eight-video GPU screen is complete and all methods score 24/40; no validated original method or final paper title has been established. No manuscript has been submitted.

- [Publication roadmap](../../docs/PUBLICATION_ROADMAP.md) describes the steps after today's AutoDL screen.
- [Manuscript outline](manuscript_outline.md) separates draftable background from result-dependent sections.
- [Claim/evidence ledger](claim_evidence.csv) records which statements have support and which remain pending. Pending rows are research questions, not manuscript findings.
- [Journal requirements](<../../docs/reports/Neurocomputing submission requirements and HERMES readiness.md>) records the supplied guide with page references.

Use editable source and the journal's current template when beginning the manuscript. No journal template has been downloaded or compiled as part of this planning step. Do not manufacture tables, contributions, p-values or abstract results to fill the outline. Keep earlier reproduction evidence distinguishable from new same-session experiments, and credit the original HERMES authors.

Q1 is mandatory according to the user. The applicable institutional database/category/year, whether CCF B is also required, recognized article type and publication milestone remain to be resolved before submission. Preparing this workspace does not certify eligibility or acceptance.

The [completed pilot decision](<../../docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) found no correctness gain from fresh causal scoring in the selected screen. The follow-up [diagnostic protocol](next_diagnostic_protocol.md) established the input and capability controls used in the later development runs.

The [research execution plan and adversarial review](research_execution_plan.md) adds current prior-art overlap, reviewer objections, conditional method choices, budget allocation and publication gates. MemInStream already covers evidence-solvability/necessity diagnostics; these controls are research infrastructure for us, not an original contribution. The separate [input diagnostic notebook](../../notebooks/autodl/hermes-input-diagnostic.ipynb) has now completed with reused assets. The [audited analysis](<../../docs/reports/2026-11-20 input diagnostic analysis and next decision.md>) records recent-4 25/40, recent-20 21/40, question-only 14/40 and native 10/14 (matching historical baselines on that subset). Prioritize support plus controlled context/position tests. No original method has been validated.

The [context diagnostic notebook](../../notebooks/autodl/hermes-context-diagnostic.ipynb) has now completed. Its [audit and analysis](<../../docs/reports/2026-11-20 context diagnostic analysis and next decision.md>) confirm 74 answers, exact comparable replays and answer-level partition stability, plus three full-prefix recoveries after fixed-position content replacement. The next [stage-attribution protocol](stage_attribution_protocol.md) separates recent-cache construction from direct historical readout and treats existing decoupling as a comparator. Selected filler/support probes remain development evidence; no original method is established.

The [matched-boundary control](<../../docs/reports/2026-11-21 matched-boundary results and next decision.md>) retained all 200 prior choices and two construction wins. The [completed short query-cache experiment](<../../docs/reports/2026-11-21 short query-cache results and next decision.md>) reproduced all 40 masked answers at 79.7% fewer median query K/V bytes. Compact recent-four remains within two correct answers, at the same query-cache size and lower measured cost. The next step is a frozen source-separated confirmation and a closest-method comparison.

The [construction selection protocol](construction_history_selection_protocol.md) defines the new improvement candidate. The pilot compares a visual-only attention-output selector with two equal-budget rules before deciding what to freeze for source-separated confirmation.
