# Observed HERMES AutoDL research summary

## Execution status

```json
{
  "preserved_and_reanalyzed": [
    "HERMES_4000",
    "HERMES_6000",
    "HERMES_50000"
  ],
  "completed_or_reused_new": [
    "HERMES_4000_REPEAT1",
    "HERMES_6000_REPEAT1",
    "FIFO_4000",
    "FIFO_6000",
    "ATTENTION_DIFFERENTIAL"
  ],
  "failed": [],
  "skipped": {
    "native_full_prefix": "known OOM/context limitations; no truncated substitute",
    "component_ablations": "pending matched FIFO interpretation",
    "full_498_video_benchmark": "not selected",
    "other_benchmarks": "not selected"
  }
}
```

## Accuracy

```
             method         generation  kv_budget   n  correct  accuracy_pct
        HERMES_4000 preserved_original       4000 250      144          57.6
        HERMES_6000 preserved_original       6000 250      144          57.6
       HERMES_50000 preserved_original      50000 250      138          55.2
HERMES_4000_REPEAT1        new_repeat1       4000 250      143          57.2
HERMES_6000_REPEAT1        new_repeat1       6000 250      143          57.2
          FIFO_4000        new_control       4000 250      144          57.6
          FIFO_6000        new_control       6000 250      144          57.6
```

## Repeatability

```
 kv_budget   n  n_videos  original_correct  repeat_correct  original_accuracy_pct  repeat_accuracy_pct  net_accuracy_change_pp  changed_choices  changed_choice_pct  changed_correctness  changed_correctness_pct  became_correct  became_wrong
      4000 250        50               144             143                   57.6                 57.2                    -0.4                4                 1.6                    3                      1.2               1             2
      6000 250        50               144             143                   57.6                 57.2                    -0.4                5                 2.0                    3                      1.2               1             2
```

## FIFO paired comparisons

```
        A            B   n  n_videos  accuracy_A_pct  accuracy_B_pct  difference_A_minus_B_pp  A_only_correct  B_only_correct  video_cluster_CI95_low_pp  video_cluster_CI95_high_pp  exact_discordance_p_exploratory
FIFO_4000  HERMES_4000 250        50            57.6            57.6                      0.0               6               6                       -2.4                         2.4                           1.0000
FIFO_6000  HERMES_6000 250        50            57.6            57.6                      0.0               5               5                       -2.4                         2.4                           1.0000
FIFO_4000 HERMES_50000 250        50            57.6            55.2                      2.4              11               5                       -0.4                         5.2                           0.2101
FIFO_6000 HERMES_50000 250        50            57.6            55.2                      2.4              11               5                       -0.8                         5.6                           0.2101
```

## Efficiency

```
             method  ttft_n  median_ttft_s  mean_ttft_s  max_reported_allocated_gib  memory_log_count  max_answering_cache  compression_events compression_marker  encoded_chunks  encoded_frames  outer_elapsed_min_this_session
          FIFO_4000     250        0.05050      0.05290                     4.68903               829                 4013                 770               FIFO             829           11434                        29.39437
          FIFO_6000     250        0.04855      0.05112                     4.73470               829                 6013                 732               FIFO             829           11434                        29.30081
        HERMES_4000     250        0.04083      0.04050                     4.68946               829                 4013                 770             HERMES             829           11434                        28.16541
        HERMES_6000     250        0.04056      0.03916                     4.73532               829                 6013                 732             HERMES             829           11434                        28.25526
       HERMES_50000     250        0.05192      0.05882                    11.83590               829                50013                  79             HERMES             829           11434                        31.04613
HERMES_4000_REPEAT1     250        0.05064      0.05235                     4.68946               829                 4013                 770             HERMES             829           11434                        31.44019
HERMES_6000_REPEAT1     250        0.04934      0.05003                     4.73532               829                 6013                 732             HERMES             829           11434                        31.35321
```

## Attention differential

```
              comparison  max_abs_max  mean_abs_mean  cosine_min  cosine_mean  past_top13_overlap
 causal_frozen_vs_native     0.999992       0.024993    0.356427     0.724323                 1.0
current_vs_causal_frozen     0.999512       0.014561    0.762802     0.883062                 1.0
       current_vs_native     0.999992       0.031577    0.220430     0.636826                 1.0
```

## Limits

- Subset results do not establish full-benchmark reproduction.
- 50K is near-uncompressed HERMES, not native LLaVA.
- TTFT is not end-to-end latency.
- Reserved GPU memory was not logged.
- Runtime is specific to the AutoDL GPU and preserved/new session.
- Attention differential uses a text-cache/query test, not a full-video ablation.
- Exact discordance tests are exploratory; confidence intervals use video-cluster bootstrap.
