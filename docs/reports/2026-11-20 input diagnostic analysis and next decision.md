# Input diagnostic: audited findings and next decision

Evidence received 20 November 2026. **The input diagnostic completed successfully. It provides specific evidence of recoverable input/context sensitivity, but no original-method result and no general advantage for a four-frame window. The next experiment should isolate that sensitivity before developing a new memory policy.**

## Preservation and audit

The uploaded [ZIP](../../evidence/archives/HERMES_input_diagnostic_evidence.zip) is preserved byte-for-byte, SHA256 `f184271bd8929f2ca5dead71a1ec60e4e1b555f33f2ed66d9909e3d7e0ffa1e8`. The [executed notebook](../../notebooks/executed/2026-11-20_hermes-input-diagnostic.ipynb) is preserved byte-for-byte, SHA256 `24df662b86955295cae4aa24a559806b72021ded6ce6a0815a0eb1ce321313bf`. Both were moved out of the repository root. Full extraction is ignored under `evidence/local/input_diagnostic_2026-11-20/`; [compact exports](../../results/input_diagnostic_2026-11-20/) contain original and independently reconstructed tables plus two unchanged review images.

The [independent validator](../../scripts/validate_input_diagnostic_evidence.py) reads data and parses archived source with AST; it never executes archived Python or notebook cells. It verifies all **287 manifest payloads in 288 ZIP files**, 200 unique checkpoints, source/model/configuration identities, original questions/choices/golds/parser, selected frame indices/timestamps, fresh cache lengths, status and correctness conventions, complete CSV exports, historical comparisons and runtime log counts. Five audit tests include rejected future input, wrong feature identity, corrupted correctness/cache lengths and skipped-case counting.

The archived runner and utilities match maintained sources. The embedded notebook runner, utility source and reference match the ZIP. The notebook adds a network-helper cell and a trailing blank cell, with nine consecutively numbered code executions and no retained error output. Its final emitted archive hash matches the supplied ZIP. This records successful execution with reused assets; it does not independently establish a fresh-container installation or rerun the GPU computation locally.

The pinned 0.5B model, PyTorch 2.10.0+cu128, Transformers 4.45.0.dev0 and RTX 4090 match the intended configuration. Preflight reports bitwise cached-feature ingestion agreement, zero maximum attention error before/after forced pruning on the checked prefix, and unchanged source cache. All eight video/feature metadata sets match the previous audited pilot. Logs show eight verified historical feature reuses, 200 new case records, eight finished video blocks and no reused answer checkpoints. Actual raw videos/tensors remain outside the supplied ZIP, so their hashes and numerical inference cannot be independently recomputed locally.

## Results with the correct denominators

| Treatment | Correct | Answered | Skipped | Interpretation |
|---|---:|---:|---:|---|
| Question only | 14 | 40 | 0 | Some questions can be answered without images; correctness alone does not prove how another treatment used vision. |
| Recent 4 | 25 | 40 | 0 | 62.5%; five wins and four losses against historical HERMES. |
| Recent 20 | 21 | 40 | 0 | 52.5%; two wins and five losses against historical HERMES. |
| Native uncompressed prefix | 10 | 14 | 26 | 71.4% on its feasible subset; every historical baseline also answers 10/14, with exactly the same choices. |
| Verified support | 0 | 0 | 40 | Not requested; this is not an oracle accuracy measurement. |
| Historical FIFO / HERMES / causal | 24 each | 40 each | — | Separate earlier session; same audited question identities/features. |

All 120 mandatory answers completed. There are **134 answered cases and 66 explicit skips**, totaling 200 case records. Native skips comprise 22 configured visual-token-cap cases and four context-limit cases. Skipped native prefixes retain their full intended frame lists; there was no silent truncation. The support configuration is empty and all 40 original review rows remain pending.

Recent-4 versus recent-20 has six recent-4-only successes and two recent-20-only successes, with eleven changed choices. Recent-4 versus native on the shared 14 questions has two recent-4-only successes and one native-only success. Recent-20 and native make identical choices on those 14. [Within-run pairs](../../results/input_diagnostic_2026-11-20/within_run_pairs.csv)

Recent-4's advantage over HERMES is one net question: **+2.5 percentage points in a selected 40-question development screen**. It does not establish superiority, equivalence or a population effect. Five gains occur at 204/q0, 204/q3, 203/q2, 203/q4 and 219/q3; four regressions occur at 212/q3, 209/q0, 218/q3 and 208/q4. Gains are concentrated in a few source videos, so source diversity and paired regressions matter. Native's 71.4% must never be compared directly with HERMES's 60% across all 40. [Question casebook](../../results/input_diagnostic_2026-11-20/question_casebook.csv)

The question-only condition scores 5/15 on prospective reasoning and 0/6 on attribute recognition. Recent-4 scores 10/15 and 4/6 respectively; recent-20 scores 10/15 and 2/6. These are descriptive task counts with very small groups, not evidence of a validated specialized method. Ten questions remain wrong across historical HERMES, question-only, recent-4 and recent-20. Those require support/input/capability review; their errors are not automatically memory failures.

## What the new evidence actually changes

### A short, unpruned counterexample now exists

**204/q0, query 16 s:** all eight original sampled frames span 0–14 s. Native prefix and recent-20 both use those same eight frames and predict red (A), matching historical HERMES/FIFO/causal. Recent-4 uses sampled indices 4–7 (8–14 s) and correctly predicts yellow (C). The [complete sampled-prefix sheet](../../results/input_diagnostic_2026-11-20/figures/sample_204_q0_prefix.jpg) shows the yellow addition at sampled index 7, 14 s. Proposed support candidates are indices 6–7, showing the local transition; original-resolution inspection should confirm sufficiency/wording.

No prior eviction is involved in the eight-frame native input or the historical first-question prefix. Therefore a retention correction is not the explanation for this recovery. The comparison establishes answer sensitivity to input extent in this controlled fresh-state setting. It does **not** uniquely prove semantic distraction: removing early frames also changes absolute positions and contextualized feature states, and numerical/chunking effects need checking. A prior introductory image contains salient competing colors, but its causal role is still a hypothesis.

**203/q2, query 76 s:** recent-4 also recovers the color answer while native prefix/recent-20/HERMES fail. Native has no eviction even though this prefix is longer. This is a second source-video example worth checking, but its full-resolution decisive support has not yet been localized by this intake. Do not treat its correct short-window answer as proof of visual grounding.

### The missing overlay has been located

**204/q3, query 141 s:** sampled index **69 at 138 s** visibly reads “Alphabet B transform to Box” on the purple overlay. The [last prefix page](../../results/input_diagnostic_2026-11-20/figures/sample_204_q3_recent_page.jpg) contains that frame. Both recent controls answer gold B, while historical HERMES/causal answer A and FIFO answers B. Earlier uncertainty from the sparse twelve-thumbnail sheet is resolved for sampled visibility. This does not prove which token was discarded or what caused the historical HERMES error. In particular, the earlier introductory “Alphabet A transform to Aeroplane” is a possible competing cue, not a demonstrated cause.

The two published image observations are analyst review of saved sheets, separate from the unchanged original pending review CSV. Full-resolution source support, annotation ambiguity and actual policy trace remain to be checked. No verified-support model answer was run in this intake.

### A shorter window has a real cost

209/q0 is correct with historical HERMES, recent-20 and the feasible native prefix, but wrong with recent-4. Likewise some prospective/counting cases regress with short input. A rule that always chooses recent-4 would exchange failures, not solve the research problem. Selecting whichever treatment agrees with gold on each question would be an oracle and cannot be presented as an implementable improvement.

## Runtime and scope

The completion timer records **100.961 seconds (1.68 minutes)** after model loading. It includes preflight, verification/reading of existing features, fresh-state controls and review-page construction; it excludes setup, downloads, model loading and final packaging. Summed answered-question wall times are 9.073 s, 10.683 s, 13.355 s and 3.973 s for question-only/recent-4/recent-20/native respectively. Their denominators and input sizes differ. The feature metadata's preparation times are historical values from the prior run, not new encoder work in this reused-cache run. No repeated timing study or cross-session speed claim follows.

## Updated publication decision and immediate next experiment

**Prioritize a small support-and-context intervention experiment. Keep generated-answer feedback conditional rather than automatically next.** The new fresh-state, uncompressed examples give a more direct observation than a hypothesized history-feedback path. Corrected scorer maintenance remains deprioritized. No broad benchmark or budget sweep is justified yet.

1. Confirm original-resolution support for 204/q0, 203/q2 and 204/q3, then inspect the four recent-4 regressions. Keep 202/q0 and 208/q0 as unrecovered perception/grounding controls rather than assuming they need memory. Check modality and annotation ambiguity on prospective questions.
2. On the two color examples, hold current question, feature values, decoding, reset state and absence of eviction fixed. Compare exact full sampled prefix, recent window and reviewed support-only input. Repeat equivalent full-prefix input with another safe chunk partition to check numerical/implementation sensitivity.
3. To isolate earlier-context content from absolute positions/length, keep input slots/token counts and decisive-frame positions fixed while replacing only earlier non-support slots with documented, pre-query, noncompeting content. For 204/q0 the already sampled black opening frame is a possible filler control, subject to review. Use more than one filler/control where possible; repetition and changed visual content introduce their own confounds. Log every source-frame-to-input-slot mapping. These are counterfactual diagnostics, not natural streaming observations or a proposed deployable method.
4. Include both recovery and regression cases. If a repeatable content-dependent effect survives position/chunking controls on independent source videos, define one narrowly scoped repair and compare its complete mechanism with SelectStream, DSCache, StreamScout and other close work in the [research plan](../../paper/neurocomputing/research_execution_plan.md). Generic recent/history separation or a confidence gate remains insufficient novelty.
5. If the effect disappears or correct support is still misunderstood, investigate capability/input/annotation limits. If a later retention intervention is studied, distinguish directly retained tokens from summary ancestry. Test generated-answer feedback separately only when the relevant examples and trace support it.

The current notebook can run reviewed support cases in a new WORK directory, but it **does not implement fixed-position context replacements, alternative chunking controls or a new memory method**. Those need a separate tested follow-up runner. Preserve the existing container's features/videos/environment rather than starting another download-heavy campaign. After a distinct mechanism is selected, freeze it and confirm on suitable independent data; all current cases remain development evidence. Neurocomputing stays the working target, with original-method and submission-readiness claims pending.
