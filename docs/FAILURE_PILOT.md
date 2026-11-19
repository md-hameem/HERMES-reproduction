# Evidence-loss diagnostic pilot

Started 19 November 2026. The CPU preparation is complete. Video review and GPU treatments remain pending. This pilot follows the [research gap map](<reports/HERMES research gap map.md>) and addresses G01–G05 and G12 in the [gap register](<reports/HERMES gap register.csv>).

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

All 30 review entries are pending. No visual support annotation, oracle result, corrected-scorer QA gain or feedback harm has been measured. The literature map, outcome inventory and selected full-history annotations are ready for the next execution session. Confirmation will require a frozen method and separate evaluation beyond this inspected development set.
