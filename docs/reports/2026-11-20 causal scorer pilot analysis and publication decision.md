# Causal-scorer pilot analysis and publication decision

Audited 20 November 2026. The supplied run completed all eight annotated videos and 40 chronological questions under three methods, producing 120 answers. **The fresh causal scorer passed its native-attention check, but supplied no correctness improvement over current HERMES on this selected screen. The next research step is failure attribution and capability/input controls. An original regular-article contribution is not established by this run.**

## Evidence and organization

The original [evidence ZIP](../../evidence/archives/HERMES_causal_scorer_pilot_evidence.zip) has SHA256 `af4b7392e90bd3f433ad20182a9058d8c8027a8ede936ed5d759572be04e0361`. The uploaded [executed notebook](../../notebooks/executed/2026-11-20_hermes-causal-scorer-pilot.ipynb) has SHA256 `24620321fc4b2783e68706928e8831456579aa8e8e03a4eccbf2c5a1345686a6`. Both were moved out of the repository root with unchanged bytes. Full extraction is under ignored `evidence/local/causal_pilot_2026-11-20/`; compact original exports and independently derived tables are under [results](../../results/causal_pilot_2026-11-20/).

The [independent audit](../../scripts/validate_causal_pilot_evidence.py) verified all 103 manifest payloads in 104 archive files, 24 complete method/video checkpoints, 120 aligned predictions, the exported summaries/pairs, event timings, annotation/source/config identities, sampling boundaries and historical baseline replay. No archived Python or notebook code was executed. The [audit JSON](../../evidence/audits/causal_pilot_2026-11-20_audit.json) records the results and limitations. Five negative/control tests exercise damaged hashes, duplicate members, unsafe paths and prediction/gold mismatches.

The executed runner is the pre-heartbeat version with SHA256 `6b52ce8cc8b8ac068dfe076f1b2bc575ef318df8b4853b87acf025f524d9faa9`, matching repository revision `39be0c5`. The inference checkout is pinned to `2a1385e0abc94debb64761b0465713c799a81c1c`. The maintained notebook now includes later logging changes and must remain separate from the execution record. The record contains extra network/environment cells and nonconsecutive execution counts; it supports this completed recovered execution, rather than proving a clean fresh single-pass replay.

The ZIP was repackaged after its live log closed. The log includes an intermediate ZIP hash, while the notebook's final emitted hash matches the supplied archive. This is consistent with the two-stage packaging code. Three editor-checkpoint logs remain in the preserved original archive and are excluded from the compact reader-facing result tables.

## Numerical and QA results

The run used the pinned LLaVA-OneVision-Qwen2-0.5B snapshot, PyTorch `2.10.0+cu128`, Transformers `4.45.0.dev0`, an RTX 4090, 0.5 fps, seed 2024 and a 4K visual-memory budget. The preflight reports bitwise-identical K/V and positions for cached-feature ingestion versus the original encoder wrapper. Layerwise causal attention has maximum absolute difference **0.0** against the native reference both before pruning and after forced summary pruning. Cache lengths are 3,149 before and 269 after, across 24 layers, on one 16-frame visual prefix. Source cache preservation passed. This does not independently validate every later scored chunk or rerun the GPU computation locally.

| Method | Correct / questions | Accuracy | Selected targets correct | Total method replay | Total compression path |
|---|---:|---:|---:|---:|---:|
| FIFO 4K | 24/40 | 60.0% | 4/8 | 12.398 s | 0.248 s |
| Current HERMES 4K | 24/40 | 60.0% | 4/8 | 19.874 s | 7.679 s |
| Fresh causal HERMES 4K | 24/40 | 60.0% | 4/8 | 22.307 s | 10.051 s |

Equal totals are not identical predictions. HERMES versus FIFO changes five choices, with two HERMES-only successes and two FIFO-only successes. Fresh causal versus current HERMES changes **one choice and zero correctness outcomes**: sample 209, QA index 2 changes B to D, while gold is A. All 40 correctness indicators remain identical between those two treatments. Fresh causal versus FIFO changes four choices and has the same two wins/two losses. [Pairwise table](../../results/causal_pilot_2026-11-20/pairwise_outcomes.csv)

The new FIFO and current-HERMES predictions match the historical full-benchmark predictions exactly on all 40 selected question identities, with no choice changes. This is useful replay evidence at this scope. It does not demonstrate whole-pipeline equivalence for all videos or hardware. [Historical replay comparison](../../results/causal_pilot_2026-11-20/historical_replay_comparison.csv)

All three methods share 14 failures. The eight videos contain 28 episodic-memory-labeled and 12 working-memory-labeled questions, including 15 prospective-reasoning questions. Dataset labels do not establish actual decisive-evidence age. Representative sheets show five toy/block-building videos and three math-instruction videos; this is a narrow, outcome-selected development sample. Do not use its 60% score as a new full-benchmark estimate, attach a population significance claim, or infer equivalence from the tie.

## Cost and runtime interpretation

Across each method's replay, 99 chunk events ingest 1,292 frames and 253,232 visual tokens before compression. All post-compression cache lengths remain at most 4,013 including the 13-token prompt. Maximum allocated GPU memory is about 4.671 GiB for each method. The fresh causal treatment adds approximately **12.24% replay time** and **30.89% compression-path time** relative to current HERMES in these recorded runs. One shuffled method order per video provides a descriptive timing comparison, not a repeated efficiency study or a general latency guarantee.

Shared chunk decoding/preprocessing/vision work totals 176.394 seconds. Combined replay across all three methods totals 54.580 seconds. The completion timer reports **327.028 seconds, about 5.45 minutes**, beginning after model load and video preparation and covering preflight, feature work, contact sheets, serialization and replay. It excludes installation, dataset/model downloads, model loading and final packaging. The remaining 96.054 seconds after subtracting recorded preparation and replays includes other operations and is not a measured individual bottleneck.

This phase was much faster than the conservative pre-run planning ranges. Those earlier ranges were estimates. The observed values should inform the next experiment budget, with setup/transfer costs kept separate. The archive does not establish total rental or fresh-container wall time, and shared-feature replay must not be advertised as an end-to-end streaming acceleration over the old five-stage harness.

The selective downloader reports 369,126,256 transferred bytes. Selected ZIP members were CRC-checked during extraction and observed per-video hashes were saved. The full 2.08 GiB dataset archive was not downloaded or SHA-verified. Original videos and feature tensors are absent from the supplied evidence ZIP, so their observed hashes and numerical computation cannot be recomputed locally from this package alone.

## Failure attribution and initial visual review

Three shared wrong answers occur before any prior token eviction: sample 204/q0 at 16 seconds, sample 202/q0 at 12 seconds, and sample 208/q0 at 7 seconds. This is derived from the actual chunk/token counts and cache-length transitions, not just query time. They cannot be errors caused by prior cache eviction. It does not distinguish sampling, visual feature quality, part/temporal grounding, language priors, annotation ambiguity or readout errors. Eleven other shared failures occur after eviction, which likewise does not prove eviction caused them.

Representative contact sheets were inspected across all eight videos. For the three early questions, the sheets contain the full sampled prefix because it has fewer than twelve frames. The crane example asks about the arm's colors; its thumbnail shows a green/yellow arm with blue elsewhere in the structure, while all methods choose an option that also includes blue. This suggests part-level grounding as a hypothesis, not a proven causal explanation. The race-car and block questions require closer source-frame/annotation inspection before assigning a cause.

The sample 204 text question is especially useful to review. HERMES and causal predict A, FIFO predicts the gold B. The query is at 141 seconds, while the sparse sheet does not show every two-second frame. The early introductory overlay visible in the first-query sheet concerns a different transformation. Neither observation proves the later annotation wrong or the decisive cue absent. Locate the actual relevant overlay before query time, verify it was sampled, then test support recoverability. Do not infer evidence visibility or retention from a twelve-thumbnail overview alone.

Original `review/evidence_review.csv` entries are still pending. A generated [18-case review queue](../../results/causal_pilot_2026-11-20/next_review_queue.csv) prioritizes three early common errors and four HERMES/FIFO correctness-discordant questions, followed by eleven other shared failures. Human support annotation remains separate from this generated queue.

## Publication decision

**Continue preparation for a Neurocomputing regular article, with the scientific contribution still open. Deprioritize causal-probe reuse/refresh as the immediate main method.** The fresh causal reference works on the checked prefix, but the expected downstream QA benefit did not appear in this selected screen. Accelerating its extra maintenance alone would not resolve the missing original contribution or demonstrate a useful advantage over FIFO/current HERMES. A small null screen does not rule out benefits elsewhere; a later reuse study would need new evidence and close prior-art checks.

The next allocation should isolate recoverable headroom using the existing eight videos and cached features: question-only, recent four-frame, recent twenty-frame and feasible native uncompressed-prefix controls, followed by a few verified-support oracle cases. This can distinguish language shortcuts, excess-context distraction, input/perception limits and candidate retention failures. The [next diagnostic protocol](../../paper/neurocomputing/next_diagnostic_protocol.md) defines the conditions and stopping rules. It is a plan, not a completed experiment or an implemented new method.

Proceed toward an original memory method only when evidence shows that a useful cue was observed, can support a correct answer, and the tested memory/readout policy loses or mishandles it in a reproducible way. If visible support remains incorrectly interpreted with a feasible uncompressed or oracle input, investigate backbone/input/grounding limitations before another eviction rule. If question-only/recent controls explain the result, revise the benchmark claim rather than force a long-memory narrative.

Avoid a full 498-video rerun, broad budget grid or a new paper title around the causal scorer at this decision point. There is no validated original improvement to confirm yet. Preserve the available GPU assets, complete the narrow diagnostic, then freeze one distinct mechanism and a justified validation plan if evidence supports it. Replication/software article tracks remain possible only if editorial scope and the advisor's original-method/degree rules permit them; this run does not authorize substituting those routes.

The [paper claim ledger](../../paper/neurocomputing/claim_evidence.csv) is updated to record GPU preflight success, historical replay agreement and the observed absence of a correctness gain. Original-method, generalization, complete system-speed and submission-readiness claims remain unestablished. No manuscript submission or publication acceptance is implied.
