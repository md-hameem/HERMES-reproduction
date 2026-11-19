# HERMES five-stage research report

Generated from completed, validated outputs on 2026-11-18T15:51:37.883261+00:00.

## Scope and evidence rules

- Prior samples 201–250 were reused after hash and annotation validation; none were rerun.
- New generalization and ablation runs use non-overlapping samples 251–300 at 0.5 fps, seed 2024, greedy decoding with the official repetition penalty.
- Full benchmark: 498 videos / 2,495 questions per method; primary contrast HERMES-4K versus FIFO-4K.
- The 50K configuration is a near-uncompressed HERMES control, not a native LLaVA baseline.
- Partial runs were not included in accuracy tables. Any optional 50K failure remains recorded in run status and logs.
- Exact discordance tests are exploratory; intervals resample whole videos. Repeated efficiency measurements are not independent accuracy samples.

## 1. Real-video attention

Real visual cache: 3136 visual tokens, 3149 total past tokens, 24 layers, 16 frames. Top-k=256 is constraining.

| comparison | layers | mean_cosine | min_cosine | mean_absolute_difference | max_absolute_difference |
| --- | --- | --- | --- | --- | --- |
| causal_frozen_vs_native | 72 | 0.630723 | 0.27594 | 0.000343 | 1.0 |
| current_vs_causal_frozen | 72 | 0.87844 | 0.709082 | 0.00014 | 0.999645 |
| current_vs_native | 72 | 0.559312 | 0.16429 | 0.000393 | 1.0 |

| comparison | layers | mean_jaccard | min_jaccard | mean_symmetric_difference |
| --- | --- | --- | --- | --- |
| causal_frozen_vs_native | 24 | 0.662895 | 0.186047 | 130.416667 |
| current_vs_causal_frozen | 24 | 0.942977 | 0.80212 | 15.5 |
| current_vs_native | 24 | 0.65032 | 0.169725 | 137.583333 |

These metrics test numerical and selection agreement. They do not by themselves establish that a corrected path would improve QA accuracy.

## 2. Generalization

| scope | method | videos | questions | correct | accuracy_pct |
| --- | --- | --- | --- | --- | --- |
| generalization_251_300 | HERMES_4000 | 50 | 250 | 148 | 59.2 |
| generalization_251_300 | HERMES_6000 | 50 | 250 | 144 | 57.6 |
| generalization_251_300 | FIFO_4000 | 50 | 250 | 145 | 58.0 |
| generalization_251_300 | FIFO_6000 | 50 | 250 | 143 | 57.2 |
| generalization_251_300 | HERMES_50000 | 50 | 250 | 141 | 56.4 |

| A | B | n | n_videos | accuracy_A_pct | accuracy_B_pct | difference_A_minus_B_pp | A_only_correct | B_only_correct | video_cluster_CI95_low_pp | video_cluster_CI95_high_pp | exact_discordance_p_exploratory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HERMES_4000 | FIFO_4000 | 250 | 50 | 59.2 | 58.0 | 1.2 | 7 | 4 | -1.2 | 3.6 | 0.54883 |
| HERMES_6000 | FIFO_6000 | 250 | 50 | 57.6 | 57.2 | 0.4 | 6 | 5 | -2.4 | 3.2 | 1.0 |
| HERMES_4000 | HERMES_50000 | 250 | 50 | 59.2 | 56.4 | 2.8 | 14 | 7 | -0.8 | 6.8 | 0.18925 |
| HERMES_6000 | HERMES_50000 | 250 | 50 | 57.6 | 56.4 | 1.2 | 12 | 9 | -2.4 | 4.8 | 0.66362 |

## 3. Component ablations

| scope | method | videos | questions | correct | accuracy_pct |
| --- | --- | --- | --- | --- | --- |
| ablation_251_300 | NO_SMOOTHING_4000 | 50 | 250 | 148 | 59.2 |
| ablation_251_300 | NO_SUMMARY_4000 | 50 | 250 | 148 | 59.2 |
| ablation_251_300 | UNIFORM_LAYERS_4000 | 50 | 250 | 146 | 58.4 |
| ablation_251_300 | NO_POSITION_REINDEX_4000 | 50 | 250 | 144 | 57.6 |

| A | B | n | n_videos | accuracy_A_pct | accuracy_B_pct | difference_A_minus_B_pp | A_only_correct | B_only_correct | video_cluster_CI95_low_pp | video_cluster_CI95_high_pp | exact_discordance_p_exploratory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HERMES_4000 | NO_SMOOTHING_4000 | 250 | 50 | 59.2 | 59.2 | 0.0 | 3 | 3 | -2.0 | 2.0 | 1.0 |
| HERMES_4000 | NO_SUMMARY_4000 | 250 | 50 | 59.2 | 59.2 | 0.0 | 1 | 1 | -1.2 | 1.2 | 1.0 |
| HERMES_4000 | UNIFORM_LAYERS_4000 | 250 | 50 | 59.2 | 58.4 | 0.8 | 7 | 5 | -2.0 | 3.6 | 0.77441 |
| HERMES_4000 | NO_POSITION_REINDEX_4000 | 250 | 50 | 59.2 | 57.6 | 1.6 | 8 | 4 | -1.2 | 4.4 | 0.3877 |

The uniform-layer variant uses one mixed-query/recency policy at every depth; other mechanisms remain as documented in the saved transformed methods.

## 4. Interleaved efficiency

| method | video_runs | questions | correct | median_video_wall_s | p95_video_wall_s | total_wall_s | median_answer_s | total_encode_s | encoded_frames | total_compression_path_s | total_pseudo_forward_s | total_prune_s | max_cache_tokens | median_peak_allocated_gib | max_peak_allocated_gib | median_peak_reserved_gib | max_peak_reserved_gib | accuracy_pct | frames_per_encode_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fifo | 30 | 150 | 93 | 164.07862 | 332.49786 | 4883.32831 | 0.13919 | 268.90823 | 8892 | 2.22743 | 0.0 | 0.0 | 4013 | 4.68908 | 4.68908 | 4.90234 | 4.92578 | 62.0 | 33.06704 |
| hermes | 30 | 150 | 90 | 165.88339 | 342.11358 | 4968.77771 | 0.13854 | 271.15382 | 8892 | 54.87248 | 54.61374 | 16.10288 | 4013 | 4.68945 | 4.68945 | 4.91016 | 4.92969 | 60.0 | 32.79319 |

Wall time is end-to-end per-video analyzer time. Encoding, pseudo-query, pruning, and answer events are separately instrumented. TTFT from legacy logs is not used as end-to-end latency.

## 5. Full benchmark

| scope | method | videos | questions | correct | accuracy_pct |
| --- | --- | --- | --- | --- | --- |
| full_498 | HERMES_4000 | 498 | 2495 | 1555 | 62.3246 |
| full_498 | FIFO_4000 | 498 | 2495 | 1546 | 61.9639 |

| A | B | n | n_videos | accuracy_A_pct | accuracy_B_pct | difference_A_minus_B_pp | A_only_correct | B_only_correct | video_cluster_CI95_low_pp | video_cluster_CI95_high_pp | exact_discordance_p_exploratory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| HERMES_4000 | FIFO_4000 | 2495 | 498 | 62.32465 | 61.96393 | 0.36072 | 58 | 49 | -0.48096 | 1.16279 | 0.43944 |

No statistically clear accuracy difference was detected for the primary full-benchmark contrast; this is not proof of equivalence.

## Reuse and completion status

| name | status |
| --- | --- |
| HERMES_4000 | completed |
| HERMES_6000 | completed |
| FIFO_4000 | completed |
| FIFO_6000 | completed |
| HERMES_50000 | completed |

Full shard completion is in `full_benchmark_shard_status.csv`; per-video raw predictions and timing events are included in the archive. This is a shard/benchmark-specific result, not evidence that HERMES can never help under other models, budgets, or workloads.
