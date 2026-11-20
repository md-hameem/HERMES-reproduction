# Short query-cache results and confirmation decision

Physical removal of older K/V slots reproduced **all 40 saved masked-contextual answers**, including their answer text. The short cache preserved the original recent logical positions through prefill and decode and reduced median query K/V from **46.09 MiB to 9.34 MiB**, a **79.7% reduction**. The reduction applies to the retained K/V tensors passed to answering. Peak live memory, encoding cost and whole-system throughput require separate measurements.

| Arm | Correct | Median query K/V | Median QA time before extraction |
|---|---:|---:|---:|
| Full contextual | 21/40 | 46.09 MiB | 156.94 ms |
| Masked contextual | 27/40 | 46.09 MiB | 161.35 ms |
| Short contextual | 27/40 | 9.34 MiB | 156.23 ms |
| Compact recent-four | 25/40 | 9.34 MiB | 154.74 ms |

The per-question paired difference against masked contextual was **7.49 ms** in answering alone. All 40 pairs had equal numbers of decode calls. Median short-cache extraction took **3.03 ms**. Charging each question its own extraction cost left a **4.45 ms** median net saving, positive on **35/40** questions. These are medians of paired differences. They differ from subtracting the arm medians above.

Against compact recent-four, the short path had three wins and one loss, the same query bytes and a median **2.99 ms extra per question** after extraction. Against full contextual, it had seven wins and one loss, fewer query bytes and a median **2.19 ms extra** after extraction. These comparisons share preparation, so they cannot establish that the candidate speeds up a live application. As in the matched run, omitting video 208 removes the net accuracy gain over compact recent-four.

The audit verified 115 archive payloads, 40 checkpoints, 160 arm-level answers, 320 timed calls and 120 historical replays. It reconstructed the position and physical-slot traces, timing order, and all exported comparisons. All eight videos reused their verified feature caches. Recorded run time after model load was 73.49 seconds. The FP32 fixture passed. In FP16 all three tested top tokens agreed even with 66 full-logit entries outside the recorded elementwise bound. The result supports answer-level agreement on this sample. The eight development videos are the same ones used to choose the approach.

## Next decision

Freeze the short contextual path as a **candidate for confirmation**, with no fallback or new window sweep before that comparison. The primary accuracy baseline is compact recent-four because it uses the same query bytes and is simpler. The primary efficiency reference is masked contextual because it uses the same contextual representations and reproduced the same saved answers. Retain full contextual as an additional control. Report query KV bytes, extraction, prefill/decode and total query wall time separately from cache construction and total live memory.

Before renting GPU time, build a confirmation manifest from source groups outside the development set. Group near duplicate clips from the same source before assigning splits. Freeze the manifest, model, sampling, window, prompts and primary comparisons before examining candidate results. A new subset of previously analyzed StreamingBench is a within-benchmark confirmation. Use genuinely separate footage for an external generalization claim. Report paired video-level uncertainty, question and video denominators, and regression cases. Keep the current 40 questions out of the confirmation estimate.

For the thesis and Neurocomputing, the live research question is whether contextual encoding provides a reliable accuracy gain at the same query-cache budget. The compact implementation has established local feasibility. Novelty relative to the closest decoupled and compact cache methods still needs a direct comparison, and a Q1 submission remains contingent on independent results and a defensible original contribution.

[Audited tables](../../results/short_query_2026-11-21/) · [Audit and hashes](../../evidence/audits/short_query_2026-11-21_audit.json) · [Executed notebook](../../notebooks/executed/2026-11-21_hermes-short-query-cache.ipynb)
