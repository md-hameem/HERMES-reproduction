# Publication readiness and conference route

Research checked on 19 November 2026, using China Standard Time for the project date. This note addresses publication readiness, conference suitability and an affordable staged plan. The user clarified that the next compute allocation is one rented GPU for only a few days. An institution-approved venue PDF will be supplied later. Venue eligibility therefore remains provisional.

## What is missing between the current evidence and an original-method submission?

### Takeaway

The project has a substantial audited experimental foundation. The central remaining scientific gap is an implemented original mechanism with a supported explanation of why it helps, followed by confirmation under a clearly specified evaluation protocol. Conference review criteria concern the contribution, soundness, significance and reproducibility. Counts of collected files and papers do not establish those properties.

### Cited Findings

- The local study covers 498 real-time visual StreamingBench videos and 2,495 questions per method. HERMES 4K scores 62.3246% and FIFO 4K 61.9639%, with a video-cluster 95% interval for the difference of −0.4810 to +1.1628 percentage points. The preserved results leave an accuracy advantage unestablished for this configuration. [Full paired results](../../../results/five_stage/full_benchmark_paired.csv)
- Existing evidence includes a second 50-video shard, four component ablations, a real-video attention diagnostic and 60 interleaved timing runs. Native-model and corrected-attention QA treatments, clean replay and an original extension remain open. The original larger-budget HERMES control still passes through the HERMES pipeline. [Current project status](../../PROJECT_STATUS.md)
- The original full accuracy aggregate reuses October predictions for samples 201–250 and November results for other shards, spanning sessions and GPU types. The dedicated timing comparison has a separate matched-session design. [Local evidence audit](../../../evidence/audits/five_stage_local_audit.json)
- ACM Multimedia asks reviewers to assess what a paper contributes, scientific rigor, reproducibility and whether evidence supports the claims. Its scope also values useful new empirical insights. This accommodates a rigorous critical study, although the user's professor separately requires an original method or improvement. [ACM MM 2026 reviewer guidelines](https://2026.acmmm.org/site/review-process-guidelines.html)
- CVPR's current reviewer guidance emphasizes technical soundness, a contribution to knowledge, novelty and potential impact. It states that failing to exceed benchmark state of the art alone does not justify rejection. There is no rule here requiring a fixed number of models, datasets or percentage-point improvement. [CVPR 2027 reviewer guidelines](https://cvpr.thecvf.com/Conferences/2027/ReviewerGuidelines)
- ARR's review form explicitly evaluates soundness and thoroughness, relevance and interest, and the information needed to reproduce the work. Main-conference recommendations also consider novelty and impact. [ARR review form](https://aclrollingreview.org/reviewform)

### Inferences

The following is an assessment of the local evidence against the cited criteria, not a venue-imposed checklist.

| Dimension | Current position | Next evidence needed |
|---|---|---|
| Reproduction and provenance | Strong saved-output audit, with explicit recovery history | One clean replay of the actual maintained experiment configuration used for the new method |
| Original contribution | No measured original method in the audited package | A precise mechanism, contrast against nearest prior art, and implementation |
| Mechanistic support | Current manual attention differs numerically and changes retention sets on one video prefix | Determine whether corrected scores change QA and whether relevant visual evidence is actually lost by current policies |
| Baselines | HERMES and matched FIFO, plus limited 50K control | Candidate versus both controls in the same session; include a corrected scorer or nearest relevant competitor when it is necessary to isolate the claim |
| Generalization | One 0.5B backbone and one fully evaluated benchmark track | Scope claims accordingly. A later small additional-backbone or untouched-benchmark confirmation would strengthen generality if affordable |
| Statistics | Existing full paired analysis and clustered intervals are available | Freeze new method settings before confirmation, report paired effects and uncertainty, and handle exploratory subgroup results as exploratory |
| Efficiency | Same-session timing and allocated/reserved memory exist | Profile loading/decoding and other currently untimed work before claiming whole-system acceleration |
| Streaming validity | Existing pipeline uses question timestamps and incremental ingestion | New retention must use only frames and questions already available at that time; preserve frame identity through pruning and positional compaction for audits |
| Reproducible artifact | Source/model pins, checkpoints and hashes are preserved | Provide exact new-method configuration, raw predictions, failure records and a runnable minimal example |

The earlier analysis found roughly 93% of video wall time outside the disjoint encoding, answering and compression-path timers. This supports adding explicit pipeline timers, not attributing all residual time to a specific cause. The available event scopes are documented in the local status report. [Timing scope documentation](../../PROJECT_STATUS.md)

### Gaps

- The institution's approved venue list and exact graduation qualification rules have not yet been supplied. A journal quartile or a prestigious conference label alone cannot establish institutional eligibility.
- The submission-versus-acceptance deadline is unknown. Review duration cannot be promised from the current evidence.
- There is no measured new method yet, so acceptance probability, final target tier and full experiment cost cannot be responsibly estimated.
- The current code and measurements cannot yet separate shared perception/sampling failures from retention-specific failures across all questions.

## Which conference route fits, and what dates are actually verified?

### Takeaway

ACM Multimedia is a natural topical match for an original streaming multimodal memory method. CVPR/ICCV/ECCV fit a strong visual-understanding or efficiency contribution, while ACL/EMNLP fit a paper centered on video-language grounding and question answering. Conference eligibility must be checked against the forthcoming approved list. Use each conference's own classification and track name rather than attaching a journal Q1 label.

### Cited Findings

- ACM Multimedia's technical call explicitly includes multimedia foundation models, multimodal fusion, multimedia reasoning and multimedia-and-language topics. Its regular track calls for new theoretical and algorithmic contributions, and it prefers contributions that are substantively multimedia or multimodal. [ACM MM 2026 technical call](https://2026.acmmm.org/site/cfp-guidelines.html)
- The CVF describes CVPR, ICCV and ECCV as major computer-vision conferences. The official ICCV page confirms the 2027 Hong Kong event, but the fetched landing page contains inconsistent event-date text and does not establish a paper-submission deadline. [CVF conference overview](https://www.thecvf.com/?page_id=100), [ICCV 2027 official page](https://iccv.thecvf.com/Conferences/2027)
- ACL 2027 includes efficient NLP methods, multimodality and language grounding to vision, and question answering. Its call welcomes original research and several other research contribution types. [ACL 2027 main call](https://2027.aclweb.org/calls/main/)
- CVPR 2027 lists paper registration on 10 November 2026 and full-paper submission on 16 November 2026, both AoE. As of 19 November 2026, this paper-submission deadline has passed. The supplementary deadline does not reopen paper submission. [CVPR 2027 official dates](https://cvpr.thecvf.com/Conferences/2027/Dates)
- ACL 2027 lists an ARR submission deadline of 4 January 2027 at 11:59 PM UTC−12. The fetched official site still lists commitment, notification and camera-ready dates as TBA. ARR reviewing and conference commitment are separate steps. [ACL 2027 official dates](https://2027.aclweb.org/), [ACL 2027 submission process](https://2027.aclweb.org/calls/main/)
- The ACM MM 2026 call disallows simultaneous review of substantially overlapping submissions, requires appropriate anonymization and describes administrative submission requirements. The actual target year's policy must be checked before submission. [ACM MM author instructions](https://2026.acmmm.org/site/author-instructions.html)

### Inferences

- A conference route makes sense if an original mechanism and its convincing results are ready before a suitable verified deadline and the institution accepts the particular venue and track.
- A journal route may better accommodate a larger mechanistic and experimental story, but it does not remove the need for novelty or robust evidence and does not guarantee a predictable acceptance date.
- ACL's January deadline is a verified option to discuss, not a reason to hurry an unvalidated method into submission. A video-language contribution must be central to its narrative.
- A workshop or Findings acceptance may have different institutional treatment from a main-conference paper. Verify the exact track with the professor before counting it toward graduation.

### Gaps

- No reliable upcoming ACM MM 2027 paper deadline was verified. Its 2027 site was inaccessible through the research tool. The 2026 scope is used only to establish topical fit.
- No authoritative ICCV 2027 submission deadline was established. Search-engine calendars and extrapolations from prior years should not be presented as confirmed dates.
- No acceptance rates or publication guarantees have been established for this project's proposed submission.

## What can be accomplished on one rented GPU for a few days?

### Takeaway

Use the next GPU rental to test one sharply defined mechanism, rather than attempting a broad new benchmark campaign. Complete the failure analysis, literature-overlap check, implementation and CPU-safe validation before renting. Commit expensive confirmation runs only after a small matched pilot provides evidence that the intervention targets the intended failure.

### Cited Findings

- The existing full evaluation is already analyzed and has informed research direction. New hyperparameter choices based on these outcomes should be documented as development choices. A later untouched evaluation is the stronger basis for confirmation. This recommendation follows the review criteria for methodological validity and supported claims. [Existing evidence scope](../../PROJECT_STATUS.md), [ARR soundness criteria](https://aclrollingreview.org/reviewform)
- Existing timing varies by video and includes substantial work outside the model events, so the small fixed workload's throughput should be measured before estimating a larger campaign. [Interleaved timing summary](../../../results/five_stage/efficiency_interleaved_summary.csv), [Timing scope](../../PROJECT_STATUS.md)
- The available source and artifact package already contains raw predictions, source/model pins, paired comparisons and checkpoint machinery that can support a controlled next experiment. [Evidence index](../../../evidence/README.md)

### Inferences

Suggested stages and decision gates, subject to the professor's list and the user's actual budget.

1. **Before GPU rental.** Select around 20–30 development videos with clear temporal-support cases and a spread of failure types. Include some HERMES-only wins, FIFO-only wins and common failures. Label these as diagnostic selections. Inspect the supporting video intervals. Determine whether evidence was sampled, represented and retained. Define one candidate mechanism and its nearest prior-art differences. Prepare all code and checkpoint-resume tests.
2. **Initial calibration.** Run two or three representative videos with current HERMES, FIFO and the candidate under the same sampling, decoding, precision, cache budget and environment. Measure actual wall time, memory and failure rate. Use that measurement to set the remaining run budget. Avoid quoting a generic number of GPU hours from a different workload.
3. **Matched development pilot.** Run the three policies on the selected videos. If the proposed mechanism specifically addresses attention approximation, include a corrected-attention reference on a smaller feasible subset. Record token/frame retention traces and both accuracy and cost. Reuse previous outputs for exploration, while retaining matched new controls for the intervention comparison.
4. **Decision gate.** Check whether the method changes the intended retention behavior, recovers the identified evidence and gives a promising quality/cost result without an unacceptable regression elsewhere. A 20-video pilot establishes feasibility and mechanism evidence. It does not establish publication-level generalization or a statistically robust full-benchmark gain. Abandon or revise the hypothesis if the mechanism is inactive or its supposed benefit is absent.
5. **Use the remaining allocation conditionally.** Freeze the selected variant and compare it with the strongest relevant controls on a larger video-disjoint confirmation set. If budget permits, reserve a small untouched second-benchmark or additional-model confirmation. Describe its actual scale transparently. Do not require two full backbones and two full benchmarks as an invented hard venue rule.
6. **After the rental.** Complete paired statistics, failure examples, ablations and a reproducibility package. Write claims at the demonstrated scope. If evidence remains weak, extend the research before submitting a method paper rather than enlarging the narrative beyond the results.

Acceptance preparation should focus on a one-sentence research claim, an explicit nearest-work comparison, an experiment that can falsify the mechanism, fair quality/cost comparisons and an artifact reviewers can inspect. Neither several incremental tweaks nor a longer literature catalogue substitutes for a well-supported contribution. The final title and venue should follow the resulting contribution.

### Gaps

- Current rental price, GPU model, usable storage and exact remaining runtime are unknown. No calendar completion promise or dollar estimate is justified.
- The original-method design is being assessed by a separate research note. This plan does not certify that a proposed adaptive-retention mechanism is novel.
- A few GPU days may establish a promising method pilot while leaving the broader evidence package for a later allocation. The current project state cannot support a promise of Q1 acceptance within that rental.
