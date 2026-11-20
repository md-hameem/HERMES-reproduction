# HERMES project status

Updated 20 November 2026 after independent verification of the completed input/capability diagnostic, following the causal-scorer pilot and five-stage AutoDL audit. This page is the current project index. Dated entries in the research log and audit retain their historical context.

All five planned execution stages have completed. The project has progressed from a 50-video reproduction to an audited full StreamingBench real-time visual benchmark comparison on 498 videos and 2,495 questions per method. The latest GPU recorded in the evidence is an RTX 4090 with 24,564 MiB. Earlier runs used the hardware recorded in the October evidence.

## Main result

| Full real-time visual benchmark | Correct | Questions | Accuracy |
|---|---:|---:|---:|
| HERMES 4K | 1,555 | 2,495 | 62.3246% |
| FIFO 4K | 1,546 | 2,495 | 61.9639% |

The observed HERMES difference is +0.3607 percentage points. The 95% video-cluster bootstrap interval is −0.4810 to +1.1628 points using 10,000 resamples. HERMES alone answers 58 questions correctly and FIFO alone answers 49 correctly. The exploratory exact discordance test gives p = 0.43944. These measurements leave an accuracy advantage and equivalence unestablished for this model and budget. The evidence covers the benchmark's real-time visual annotations. Other StreamingBench tracks and other models require separate evaluation.

## Five completed stages

| Stage | Verified outcome |
|---|---|
| Real-video attention | One 16-frame chunk, 3,136 visual tokens, 24 layers and three pseudo-queries. Current versus native attention has mean cosine 0.5593. Mean retained-token Jaccard is 0.6503 at a constraining 256-token budget. The scorer changes which tokens are retained. The later selected 40-question causal-scorer screen below measured no correctness gain over current HERMES. |
| Generalization, samples 251–300 | HERMES 4K 59.2%, FIFO 4K 58.0%, HERMES 6K 57.6%, FIFO 6K 57.2%, HERMES 50K 56.4%. All five runs contain 250 questions. The 4K paired difference is +1.2 points with a clustered interval of −1.2 to +3.6. |
| Component ablations, samples 251–300 | No smoothing 59.2%, no summary 59.2%, uniform layers 58.4%, no position reindexing 57.6%. Full HERMES scores 59.2%. All four paired intervals include zero. Equal aggregate accuracy can still include changed predictions. |
| Interleaved efficiency | Ten videos, three repetitions and two methods produce 60 video runs. Median video wall time is 165.883 s for HERMES and 164.079 s for FIFO. Total compression-path time is 54.872 s versus 2.227 s over 30 runs each. Peak allocated memory is approximately 4.689 GiB for both. |
| Full real-time visual benchmark | Ten shards per method cover all 498 videos and 2,495 questions. Samples 201–250 reuse the audited October originals and 251–300 reuse the generalization stage. The other eight shards were run for both methods. |

Timing scopes are nested. Pruning is included within pseudo-forward time, which is included within the HERMES compression path. Adding those times would double count work. Whole-video wall time includes loading, decoding, CPU work and checkpoint validation within the analyzer measurement. The answer-event table measures the complete answering function. Its median is 0.13885 s for HERMES and 0.13693 s for FIFO. The efficiency summary's median-of-video-medians uses a different aggregation. Three repetitions reuse the same ten videos.

## Evidence organization

The GitHub copy stores one canonical five-stage ZIP alongside the executed work file and compact published results. The local extraction is ignored by Git. Notebook workflows are organized by environment and the research log, figures and literature are under `docs/`. [Repository README](../README.md) · [Notebook index](../notebooks/README.md) · [Evidence index](../evidence/README.md)

| Location | Role |
|---|---|
| `evidence/archives/HERMES_AutoDL_five_stage_evidence.zip` | Original uploaded five-stage archive, 2,921 manifest payloads |
| `evidence/local/five_stage/` | Local extraction of the archive, excluded from Git to avoid a duplicate payload |
| `results/five_stage/` | Published summary tables, report, figures and provenance for GitHub readers |
| `evidence/autodl_2026-10-25/` | Original AutoDL results, logs and derived analyses with historical internal paths preserved |
| `notebooks/executed/2026-11-18_hermes-five-stage-autodl.ipynb` | Uploaded executed work file, renamed from `hermes-five-stage-autodl-cache-fix (1).ipynb` with identical bytes |
| `notebooks/autodl/hermes-five-stage-autodl-cache-fix.ipynb` | Maintained workflow with permanent shard-count, CSV text and duplicate-option validator fixes |
| `evidence/autodl_2026-10-28/` | Earlier October evidence package |
| `evidence/archives/historical_evidence_supplement.zip` | Three recovered October payloads with hashes matching the historical manifest |
| `scripts/validate_five_stage_evidence.py` | CPU audit of archive integrity, annotations, predictions, checkpoints, statistics, timing events and retained-token sets |
| `evidence/audits/five_stage_local_audit.json` | Machine-readable outcome of the local audit |

The uploaded ZIP contains Python experiment code, JSON annotations, CSV predictions and Markdown analyses. It contains no R scripts.

The five-stage archive SHA-256 is `b8bc4cf8a3b28e6854b2e76b8c0239b12d3b5598ee1a07dafc03d0326a900518`. The executed notebook SHA-256 is `90174597f4132f3b116f3ab1f0d9410b92f655d9017d39b579c741ca1c3e0745`. Both match the supplied files and the archive hash printed in the notebook. The original archive's embedded notebook is an earlier source snapshot. The uploaded executed notebook is the execution record.

Run the current local audit from the project directory with Python, NumPy, pandas and SciPy installed.

```bash
python scripts/validate_five_stage_evidence.py
```

The audit verified every archived payload, all 53 earlier manifest payloads against the existing October evidence, 25 new complete runs, 1,246 video checkpoints and their timing totals, 18 master accuracy rows, task tables, all paired comparisons, 60 interleaved timing runs, and the attention/selection summaries. Run-completion logs agree with their saved completion records. Saved source hashes match the checkout after newline normalization. This is an audit of saved measurements.

## Recovery history and interpretation limits

The executed run required live notebook and runner fixes. Shard question counts are 250, 252, 253, 250, 250, 250, 240, 250, 250 and 250. Samples 301–350 contain 48 annotated videos, with samples 316 and 332 absent. Video IDs differ from sample IDs, so shard membership follows the sample ID in the source video path.

Numeric-looking gold answers require reading prediction CSVs as text. Sample 463, question index 4, repeats the gold text in options A and C. The validator now follows inference's existing first-match gold convention, A. The three runner hashes recorded across sessions reconstruct exactly from the original embedded runner, the CSV text fix, and the duplicate-option fix. They change validation rather than inference. The working notebook sources omit some patches applied through notebook history, so the saved execution record does not demonstrate a fresh Run All replay.

The full benchmark combines October results on samples 201–250 with November measurements for the other shards. Its primary accuracy comparison therefore uses multiple sessions and GPU types. The dedicated interleaved efficiency comparison uses its own matched session. The historical 60.4% memory reduction compares the October 4K and 50K HERMES runs. Both small-budget methods use approximately the same memory in the new matched comparison.

## Remaining work

The [research gap map](<reports/HERMES research gap map.md>) and [paper review ledger](<reports/HERMES paper review ledger.csv>) now account for all 51 local PDF files through selective method/evaluation/limitations review, with per-file depth and verified hashes. The [gap register](<reports/HERMES gap register.csv>) separates occupied directions, implementation fixes and conditional research questions. This 19 November review recommends evidence localization and a small corrected-scorer QA diagnostic before selecting a new method; it does not establish a validated original contribution.

The [failure-analysis pilot](FAILURE_PILOT.md) has started with a reproducible CPU casebook. All 2,495 paired outcomes reproduce the audited totals. Thirty target questions span 30 distinct videos, with 150 full-history questions retained for replay. The [review sheet](../results/failure_pilot/evidence_review.csv) is pending because benchmark videos are absent from this checkout. The separate eight-video GPU scorer screen has now completed; its 18-case follow-up queue still needs full support annotation. The builder, source hashes, selection limitations and execution protocol are saved alongside the outputs.

The [new AutoDL notebook](../notebooks/autodl/hermes-causal-scorer-pilot.ipynb) is prepared for a fresh container. A smaller eight-video screen from samples 201–250 defaults to 120 answers across three matched methods, selective downloads, shared features, a mandatory native-attention gate and resumable checkpoints. The maintained workflow has eleven local numerical/safety/monitor tests and separate setup tests. The supplied pre-heartbeat execution completed GPU preflight and QA; the later heartbeat version remains a maintained workflow rather than the preserved execution record.

The user's first AutoDL attempt reached dependency installation and timed out downloading the NVIDIA NCCL wheel before any QA experiment. Notebook setup now separates package-index traffic from turbo proxies, extends HTTP timeouts/retries, limits concurrent downloads and adds an alternate domestic index. Three additional mocked setup tests cover routing and fallback. The recovered execution subsequently completed installation, GPU preflight and all 120 answers; the original setup failure remains part of its recovery history.

Earlier 2–3-hour rental planning ranges were unmeasured and conservative. The completed RTX 4090 screen records 327.028 seconds for its post-model-load phase, including preflight, feature preparation and replay, while excluding installation, downloads, model loading and final packaging. Shared preparation totals 176.394 seconds and all three methods together replay in 54.580 seconds. Total container rental wall time is not established. Use the measured scopes to budget further diagnostics, not as a promise for fresh setup or larger workloads.

1. Complete thesis synthesis around the audited reproduction, component diagnostics and full-benchmark result, including the null accuracy finding.
2. Follow the completed scorer screen with capability/input controls and source-evidence review. Fresh causal scoring passes preflight but adds no correctness benefit over HERMES on the selected 40 questions. An original mechanism remains to be established.
3. Establish a matched native LLaVA baseline on a feasible, explicitly defined context range if making claims against the base model. Current 50K results use the HERMES pipeline.
4. Verify a clean replay of the maintained notebook. The existing evidence establishes a recovered completed run.
5. Develop an original method or improvement as required by the advisor, after the diagnostic pilot identifies a recoverable failure. Broad EDMI-style reinjection overlaps StreamFlow and is not an established original contribution. The user confirmed on 20 November 2026 that the professor accepts CCF B. Q1 remains mandatory. The user subsequently selected Neurocomputing as the working journal target; use the supplied author guide and the current publication roadmap for preparation. The applicable Q1 database/year and required publication milestone remain unspecified; additional models and workloads depend on the eventual claim and compute budget.

All three historical archival omissions have been recovered in the supplemental ZIP with their original hashes. The old bundles still document the omissions. Raw model weights, videos and full attention tensors are outside this evidence package. Attention tables and selection sets can be checked locally, while rerunning the GPU computation requires the pinned assets and environment.

Related documents are the detailed research log, the dated technical audit and the comprehensive research report. [Research log](research_log.md) · [Technical audit](HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md) · [Technical report](HERMES_Comprehensive_Technical_Research_Report.md)

## Publication-target clarification, 20 November 2026

The user clarified that Q1 is mandatory and requested a more attainable alternative to TCSVT. The [alternative journal comparison](<reports/HERMES alternative Q1 journal shortlist.md>) records TOMM as CCF B and Neurocomputing, ESWA and EAAI as CCF C, with separate source-qualified Q1 evidence. Their relative acceptance difficulty was not established. The final venue remains open; whether C journals qualify through Q1 and the applicable quartile database/year remain unspecified.

## Neurocomputing guide intake, 20 November 2026

The supplied 19-page author guide has been archived with its original SHA256 and reviewed across all pages using rendered-page OCR, with critical clauses visually checked. The [requirements and readiness assessment](<reports/Neurocomputing submission requirements and HERMES readiness.md>) separates journal rules from our proposed evidence plan. A regular article remains the preferred route under the original-method requirement. Replication/software tracks exist, with scope and graduation eligibility still to establish. No new GPU result, manuscript submission, quartile verification or acceptance probability is established by this intake.

## Neurocomputing preparation roadmap, 20 November 2026

The user selected Neurocomputing as the working target. The supplied eight-video GPU run has now completed and been audited; it does not establish an original improvement. The [publication roadmap](PUBLICATION_ROADMAP.md) defines ZIP integrity/completion checks, paired QA and cost analysis, evidence-based method-selection branches, novelty checks, budgeted validation and manuscript preparation. The [paper workspace](../paper/neurocomputing/README.md) contains an outline and claim/evidence ledger. Today’s default screen is 40 distinct questions across three methods, producing 120 answers; it is selected development evidence. A validated original contribution remains the main scientific gap.

## Completed causal-scorer pilot and decision, 20 November 2026

The [independent audit](../evidence/audits/causal_pilot_2026-11-20_audit.json) verifies 103 payloads, 24 checkpoints and all 120 predictions. All methods score 24/40; fresh causal versus HERMES changes one wrong choice and no correctness outcomes. Native preflight maximum attention error is zero before/after forced pruning on one 16-frame prefix. HERMES and FIFO predictions match historical replay exactly at this scope. Four HERMES/FIFO correctness discordances split two wins each. Fourteen failures are shared, including three before any eviction. Support visibility and actual failure causes remain unclassified.

The [publication decision](<reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) deprioritizes probe-state reuse as the immediate main contribution and calls for the [next diagnostic protocol](../paper/neurocomputing/next_diagnostic_protocol.md): question-only, recent-4, recent-20, feasible native-prefix and a few verified-support controls. Continue Neurocomputing preparation, but do not launch a full-benchmark grid or claim a publishable new method from this null screen. The archived ZIP and executed notebook preserve the original bytes; compact tables live under `results/causal_pilot_2026-11-20/`. Human support review and original-method validation remain pending.

The [adversarial research plan](../paper/neurocomputing/research_execution_plan.md) records prior-art overlap and scientific gates. Its separate [input diagnostic notebook](../notebooks/autodl/hermes-input-diagnostic.ipynb) has now completed with reused assets; the current decision is below. Eighteen implementation tests and fourteen reused numerical/setup tests pass, with five new saved-evidence audit tests. [Run instructions](INPUT_DIAGNOSTIC.md)

## Completed input diagnostic and revised priority, 20 November 2026

The [audit](../evidence/audits/input_diagnostic_2026-11-20_audit.json) verifies 287 payloads, 200 case checkpoints, 134 answers and 66 explicit skips. Question-only scores 14/40, recent-4 25/40, recent-20 21/40, and native prefix 10/14. All three historical baselines also score 10/14 with identical choices on those native identities. The 26 native skips comprise 22 visual-token-cap cases and four context-limit cases; 40 support-oracle cases were not requested. All eight historical feature sets were verified/reused, and no answer checkpoints were reused. Recorded post-model time is 100.961 seconds, excluding setup/download/model load/final packaging.

Recent-4 has five gains and four regressions versus HERMES, not a validated general improvement. Sample 204/q0 recovers with four recent frames but fails with the full eight-frame uncompressed prefix, before eviction. Sample 203/q2 supplies another uncompressed/recent disagreement. The previously unresolved purple overlay is visible at sample 204, sampled index 69, 138 seconds. These observations prioritize support verification and controlled context/position/chunking interventions over a new retention score or automatically starting past-answer-feedback work. No unique interference cause, oracle QA result or original method is established. [Full analysis and next decision](<reports/2026-11-20 input diagnostic analysis and next decision.md>) · [Results](../results/input_diagnostic_2026-11-20/)
