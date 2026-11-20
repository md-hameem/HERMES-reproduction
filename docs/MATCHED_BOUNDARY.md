# AutoDL matched-boundary control

The GPU run completed all **200 answers**. The primary pair recorded two wins for C matched and zero for D, and all 200 historical choices matched. [Audit and next decision](<reports/2026-11-21 matched-boundary results and next decision.md>) · [Executed notebook](../notebooks/executed/2026-11-21_hermes-matched-boundary.ipynb).

The [short query-cache follow-up](SHORT_QUERY_CACHE.md) uses these answers as pinned replay references and tests physical history removal.

Open [hermes-matched-boundary.ipynb](../notebooks/autodl/hermes-matched-boundary.ipynb) in the same AutoDL container and choose **Restart Kernel → Run All**. Defaults reuse the assets under `/root/autodl-tmp/hermes_causal_pilot`. The new work directory is `/root/autodl-tmp/hermes_matched_boundary`. The notebook supports fresh setup through the same pinned dependency and model checks, with additional download time.

The primary comparison is **C matched versus D** on the same 40 development questions. Both arms hold older history physically in the cache and hide it during every question and answer call. Both start from the same older cache prefix and encode the same recent features in the same groups, positions and explicit causal mask shape. During recent encoding, C can attend to older history and D cannot. A matched and B give the paired comparison when history is visible during readout. Compact recent-four is the simple accuracy baseline.

The notebook reruns all five arms for **200 answers** and requires B, D and compact recent-four to match all 120 previous choices. Historical A and C are included as saved references. Those comparisons measure the combined effect of the boundary split and the explicit mask path. The prior long-prefix case stays outside this experiment. Video features, question timing and sampling match the previous screen. No extra imagery or annotation is needed for this control.

Preflight checks a physical reference in FP32 and same-shape hidden-history invariance in FP16. It records full FP16 logit differences and requires matching tested top tokens and zero blocked attention. Any replay mismatch preserves evidence and stops the run. These checks cover the preflight fixture and the specified historical choices.

Watch for `MATCHED_PREFLIGHT_PASS`, `QUESTION_BLOCK_DONE 1/40` and `MATCHED_BOUNDARY_COMPLETE`. Download **HERMES_matched_boundary_evidence.zip** from the last cell and save the executed notebook. The ZIP includes primary and secondary paired summaries, individual transitions, cache placement and mask traces, source hashes, and closed logs. If the run stops before QA, run the download cell manually when failure evidence exists.

Local validation exercises embedded source materialization and runner import, both precision checks on a small CPU Qwen decoder, and exact state replay for the existing independent and compact paths. The completed GPU run used the RTX 4090 and all eight existing feature caches. [Previous experiment and decision](<reports/2026-11-21 stage attribution analysis and next decision.md>).
