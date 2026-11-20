# Evidence-loss diagnostic pilot

Started 19 November 2026; completed GPU screen audited 20 November 2026. All three methods score 24/40 on eight selected videos. Fresh causal scoring passes preflight but changes no correctness outcomes relative to HERMES. Human support review remains pending. This pilot follows the [research gap map](<reports/HERMES research gap map.md>) and addresses G01–G05 and G12 in the [gap register](<reports/HERMES gap register.csv>).

## Fresh-container AutoDL notebook

Upload [hermes-causal-scorer-pilot.ipynb](../notebooks/autodl/hermes-causal-scorer-pilot.ipynb) by itself and select **Restart Kernel and Run All Cells**. The notebook contains the experimental runner, sparsely clones the pinned source revision and installs an isolated Python 3.12 environment with the audited Transformers commit and CUDA 12.8 PyTorch wheel. It does not require uploading an earlier evidence archive. A 24 GB RTX 4090 is recommended; startup checks require at least 15 GiB GPU memory and 18 GiB free data-disk space.

The first reported AutoDL attempt reached environment installation and timed out downloading `nvidia-nccl-cu12==2.27.5` with uv's 30-second timeout. This was an incomplete dependency download, before scorer preflight or QA. Setup now bypasses turbo proxies for package indexes, uses 600-second HTTP timeouts, five HTTP retries and two concurrent downloads, and falls back from Tsinghua to Aliyun. GitHub source downloads retain turbo networking and are installed from a cached local archive. Existing uv downloads and the virtual environment are preserved. Three mocked setup tests check proxy isolation, index fallback and failure reporting; the recovered AutoDL execution later completed installation and the screen, with its original notebook preserved.

The default fast screen selects **eight distinct videos from samples 201–250**, with two targets from each original outcome category. It retains all 40 questions and evaluates FIFO, current HERMES and fresh causal HERMES, for 120 answers. This is a separately recorded, smaller screen than the 30-case review set below. It reuses inspected development data and cannot support population accuracy or original-method claims.

`PER_OUTCOME=1` selects four videos and 60 answers; `2` selects eight and 120; `3` selects twelve and 180. If changing screen size after a run, use a new `WORK` directory to preserve existing human annotations. `MAX_HOURS=2` is a soft inference budget checked between complete three-method video blocks after at least two videos. It excludes installation/downloads and can overrun by one block. A partial result is explicitly labeled and packaged. Increasing the time budget preserves matching checkpoints.

For planning on an RTX 4090, reserve roughly **2–3 hours** for the default fresh-container run. These estimates have not been measured with this notebook on a GPU.

| Phase | Planning estimate |
|---|---|
| Fresh environment, model and dataset downloads | 20–45 minutes |
| Validation and eight-video, three-method experiments | 45–120 minutes |
| Four-video alternative, `PER_OUTCOME=1` | 30–60 minutes of experiments, plus setup |

Network throughput, source video lengths and GPU performance can move execution outside these ranges. The runner prints an elapsed-time-based estimate after the first complete video block. The two-hour soft budget is not a two-hour total rental guarantee. If the run stops at the budget, keep the same container/data directory and Run All again to reuse validated assets, features and matching checkpoints. Download `HERMES_causal_scorer_pilot_evidence.zip` before deleting the container, including when the run is labeled partial.

The data loader first extracts selected members through validated HTTP byte ranges and ZIP member CRC checks. In this mode, the whole-archive SHA is explicitly unverified; observed video SHA256 values support later resumption. If range requests fail, it downloads only the 2.08 GiB 201–250 archive and verifies its known full SHA256. Both public endpoints accepted range requests during the local preparation check. Network performance and availability can change.

Video chunks are decoded/preprocessed/vision-encoded once into CPU feature files and reused across methods. Native language-model ingestion still runs for each method with its own positions and cache. Mandatory GPU preflight verifies bitwise-equivalent K/V ingestion from cached features and native causal attention agreement before and after actual summary pruning. The fresh scorer propagates hidden states through real decoder layers with explicit causal masks and separate temporary cache lists. It changes only the manual scoring method on the experimental instance; original source files, answering and hierarchy policy stay intact.

Outputs include full-history predictions, paired question outcomes, target summaries, shared-preparation metadata, per-method replay times, source/asset hashes, sampled frame timestamps, sparse contact sheets and a pending evidence-review sheet. Shared feature preparation is outside method replay timings, so these timings do not establish full-system streaming latency. Contact sheets are review aids and cannot establish cue survival. Past-answer interventions and a native full-prefix QA control are not included in this first screen.

The final cell exposes `HERMES_causal_scorer_pilot_evidence.zip` for download. Atomic, identity-validated per-method/video checkpoints resume on Run All. Logs are snapshotted before manifest hashing and repackaged after the subprocess closes. Failures after model setup preserve a partial diagnostic package. Model weights, video files and feature tensors are excluded from the evidence ZIP.

Local verification now covers eleven CPU numerical/safety/monitor tests, actual summary pruning, mocked range/ZIP64 extraction, checkpoint rejection, package hashes, fresh/repeated sparse Git bootstrap and headless execution of all notebook code cells in validation mode. The supplied pre-heartbeat notebook completed its CUDA probe, numerical gate and QA on an RTX 4090. Its extra network cells and nonconsecutive counts document a recovered execution; a clean fresh single-pass replay is not established. Later heartbeat code is separate from the execution record.

Regenerate the notebook after runner changes with `python scripts/make_failure_pilot_notebook.py`. Numerical tests require the pinned Transformers environment and run with `python -m unittest discover -s scripts -p test_autodl_failure_pilot.py -v`.

### Monitoring a quiet run

The user reported more than ten minutes without visible output in the run cell. The earlier downloader reported a video only after complete extraction, leaving network attempts and initial preparation silent. This observation alone does not establish a stalled download, failed GPU or completed experiment.

The maintained runner now emits immediate `STAGE` messages, range-request/received-byte information, feature-chunk counts and a `HEARTBEAT` every 30 seconds. `WORK/progress.json` records the current stage, elapsed time and time since the last stage update. A heartbeat establishes liveness only. It does not prove byte transfer, GPU activity or scientific completion. The monitor is stopped cleanly, and diagnostic-file write failures do not terminate computation. The evidence package includes a frozen progress snapshot.

For an existing run using the earlier code, inspect `pilot_runner.py` in the process list, `logs/pilot.log`, sizes of files under `videos/` and `downloads/`, and `nvidia-smi` from a separate terminal. Compare file sizes over time to distinguish active transfer from waiting. GPU inactivity during data preparation can be normal. Do not start a second runner against the same work directory or change files used by an active run. Updated logging takes effect only in a newly launched process; a changed runner hash also changes checkpoint/feature identities, so avoid restarting a productive run merely to see the new logging.

## Question and decision

Determine whether failures contain recoverable visual evidence that a memory-policy change could preserve. Then test whether fresh causal scoring improves QA under the same compressed-cache budget. The existing attention discrepancy alone does not establish an answer-quality effect or an original method.

If evidence is missing from sampled inputs, prioritize sampling. If the backbone cannot answer with verified support, investigate capability before retention. If observed support is lost and a reference treatment recovers answers, a retention improvement has a concrete target. Keep inconclusive cases and regressions.

## Prepared artifacts

Run from the repository root with Python 3 and its standard library.

```bash
python scripts/build_failure_pilot.py
python -m unittest discover -s scripts -p test_failure_pilot.py -v
```

The builder reads the canonical evidence ZIP, validates the two exported prediction hashes against its embedded manifest, and verifies every prediction against local annotations. It preserves the historical parser and first-matching-option gold convention. The independently audited full archive is still checked by `scripts/validate_five_stage_evidence.py`.

Outputs are under [results/failure_pilot](../results/failure_pilot/).

| File | Purpose |
|---|---|
| [paired_inventory.csv](../results/failure_pilot/paired_inventory.csv) | All 2,495 question pairs, including earlier-answer context and ambiguity flags |
| [task_outcomes.csv](../results/failure_pilot/task_outcomes.csv) | Descriptive outcomes by annotated task |
| [selected_cases.csv](../results/failure_pilot/selected_cases.csv) | 30 review targets from 30 different videos |
| [selected_videos.json](../results/failure_pilot/selected_videos.json) | Complete conversations for the selected videos, 150 questions |
| [evidence_review.csv](../results/failure_pilot/evidence_review.csv) | Human evidence annotation sheet, initially pending |
| [provenance.json](../results/failure_pilot/provenance.json) | Source hashes, seed, selection rule, quotas and limitations |

Selection uses eight HERMES-only successes, eight FIFO-only successes, ten shared failures and four shared successes. Within each outcome, task round-robin and seed-2024 SHA256 ranking select cases, with at most one target per video across all outcomes. Duplicate gold options are excluded from selection and retained in the inventory. The outcome order is fixed and affects video availability for later groups. This deliberately selected diagnostic sample cannot estimate population accuracy or support significance claims. Task labels are dataset annotations, not inferred failure causes.

The generated selection is reproducible. Rerunning preserves `evidence_review.csv` and rejects a case-ID change if a review already exists. Use a separate `--output` directory for a different selection. Generated inventories are replaced by a rerun, so write observations in the review sheet.

## Review the evidence before the GPU run

The checkout contains annotations and predictions but no StreamingBench videos. Resolve each selected `video_path` on AutoDL or the original data storage. Do not assume sample IDs equal `video_id`. Preserve source checksums and actual decoded timestamps when obtaining assets. Do not infer a failure cause from question wording, task label or disagreement alone.

For each target, inspect only the video prefix available by its query time. Record one or more supporting intervals in seconds, for example `[[12.0, 16.5], [81.0, 84.0]]`. Jointly required events need multiple intervals. Flag audio dependence, ambiguous gold, absent visible support and uncertain support. A present-state question must keep its original query time because delaying it can change the answer.

Reconstruct the original sampler at 0.5 fps and the original chunking before marking support visible. `BaseVQA.load_video` rounds source fps and uses an integer frame step. The prefix loop uses `ceil(end_time * sample_fps)` sampled frames. Nominal multiples of two seconds are insufficient to prove exact visibility or the query boundary. Record frame indices and actual timestamps, including any boundary mismatch, rather than silently changing the sampler during diagnosis.

Use these review fields consistently.

| Field | Values or meaning |
|---|---|
| `review_status` | `pending`, `reviewed`, `uncertain` |
| `support_intervals_seconds_json` | JSON interval list; leave blank when unknown |
| `support_visible_at_original_sampling` | `yes`, `no`, `uncertain` |
| `support_available_before_query` | `yes`, `no`, `uncertain` |
| `requires_audio`, `gold_ambiguity` | `yes`, `no`, `uncertain` |
| `evidence_type` | Free text such as static text, action, identity, counting or distributed events |
| `failure_stage` | `unclassified`, `annotation`, `sampling`, `perception_or_reasoning`, `retention`, `readout`, `mixed` |
| `oracle_support_qa_result` | Pending until inference; record prompt, frames and result |
| `retention_trace_result` | Pending until trace exists; record support-token survival and cache stage |

Use `retention` only after verifying that useful support was encoded and lost, with an appropriate reference recovering the answer. A visible cue and an incorrect response alone cannot establish this. Raw token IDs/attention traces are not present in the saved QA predictions.

## Calibrate and run the scorer pilot

Start with two or three selected videos on the rented GPU. Log the exact checkout, model revision, dependencies, GPU, precision, sampling, positional policy and seed. Verify source assets, clean replay and complete history. Measure actual peak GPU memory, CPU memory and wall time before setting the run count. No runtime estimate is asserted here.

Replay each selected video's full chronological conversation. Running only a target question changes the cache and historical guidance. Compare FIFO 4K, current HERMES 4K and a separately implemented fresh causal scorer at 4K with identical sampled inputs and answer settings. Validate the scorer against native model attention on the **same compressed cache** before QA. This treatment must propagate query hidden states across layers, apply the correct causal mask and respect positions/token identities after compaction. Keep production inference unchanged until the separate treatment is verified.

A native full-prefix QA control is a separate capability/retention diagnostic, feasible only within context and memory limits. HERMES 50K still compresses and cannot substitute for it. A verified-support oracle input and question-only control can help diagnose recoverability and language shortcuts. Log exact support-frame inputs and prompt changes because oracle inputs change the input distribution. A recent-window control tests whether distant evidence is necessary.

Record paired answers, target and all-history regressions, support survival, compression/update time, wall time and all auxiliary memory. Pilot gains are exploratory because outcomes were selected and the benchmark has already guided development. Do not report this set as untouched confirmation.

Proceed to cached probe-state reuse only if verified causal scoring shows recoverable QA or evidence-preservation benefit worth its cost. Require later comparisons against fresh scoring, no refresh, fixed refresh and matched-count random refresh before claiming a selective rule helps. Stop this branch if only attention agreement changes without useful downstream behavior. A small null pilot is inconclusive about tiny effects but does not justify a large mechanism build.

## Keep the history-feedback experiment separate

The inventory records previous answers and correctness for both methods. That metadata identifies possible interventions; it does not demonstrate harm. For a target after a previous QA, fork the identical pre-ingestion cache and hold later frames fixed. Intervene on the past-answer text used by guidance with generated, corrected, plausible wrong and question-only history. Past gold is a diagnostic intervention, never a deployable oracle. Select a target with later ingestion/compression so the history can affect retention.

Measure guidance, selection and next-answer differences. Test question-only or confidence gating as remedies only after an error-mediated effect is observed. Compare the nearest published mechanisms before claiming originality. Do not combine this intervention with scorer refresh in the first experiment.

## Current boundary

The original 30-case review and archived eight-target review sheets remain pending. The smaller eight-video scorer screen is complete: all methods score 24/40, with no correctness change from fresh causal scoring. Representative sheets have been inspected, but no full support annotation, oracle QA or causal feedback intervention is complete. The [audited analysis](<reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) and [next diagnostic protocol](../paper/neurocomputing/next_diagnostic_protocol.md) supersede the initial scorer-pilot decision point. Confirmation still requires a frozen original method and separate evaluation beyond inspected development cases.

## Completed evidence and measured runtime

The completed [execution record](../notebooks/executed/2026-11-20_hermes-causal-scorer-pilot.ipynb) and [ZIP](../evidence/archives/HERMES_causal_scorer_pilot_evidence.zip) were preserved byte-for-byte. Run `python scripts/validate_causal_pilot_evidence.py` for the standard-library audit. It verifies 103 payloads, 24 checkpoints, all predictions/exports and historical baseline agreement without executing archived code. Original video/feature bytes are excluded from the ZIP, limiting independent numerical/data re-execution.

The recorded post-model-load phase is 327.028 seconds, including 176.394 seconds of shared preparation and 54.580 seconds of combined three-method replay, plus preflight/contact/serialization/other work. This was much shorter than the earlier planning range above. Installation, network transfer, model loading and packaging are outside that timer; total fresh-container rental time is not measured here. See [result scopes](../results/causal_pilot_2026-11-20/README.md). Do not rerun merely to obtain the latest heartbeat logging or treat selected-screen timings as full-system streaming speed.
