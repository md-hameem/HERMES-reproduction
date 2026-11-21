# Visual review of the confirmation disagreements

Run [hermes-confirmation-visual-review.ipynb](../notebooks/autodl/hermes-confirmation-visual-review.ipynb) in the same AutoDL workspace. It can run on CPU. Keep the existing confirmation videos in `/root/autodl-tmp/hermes_short_query_confirmation_v2/videos` and the shared Python environment in `/root/autodl-tmp/hermes_causal_pilot/venv`. It uses the existing Decord and Pillow packages to render frames.

The case list is the 12 primary correctness disagreements from the [audited confirmation](<reports/2026-11-21 short query confirmation analysis.md>). These cover 10 clips. The notebook checks the video hashes, uses the recorded source frame IDs, verifies decoder timestamps and rejects post-query frames. It renders the recent four separately from the older frames used during full-history construction.

Open each case with `show_case(case_id)`. The sheets and the extracted `index.html` display the question and options. The answer key and model outcomes are in `after_review/outcome_key.csv`. Read that file after saving visual judgments. Use `show_original(case_id, sampled_index)` to inspect small text and fine detail at original resolution.

Use `record_review` or edit `review_notes.csv` to record the support indices, an optional answer letter, confidence and any ambiguity. Mark whether the decisive evidence is recent only, older only, independently visible in both, requires both, not visible, or uncertain. A still sequence may not resolve motion, audio or ambiguity. Preserve uncertain cases instead of inferring support.

Select **Restart Kernel → Run All** to generate the first package. The last cell links to **HERMES_confirmation_visual_review.zip**. Rerun that cell after adding notes or inspecting original frames. Send the ZIP for analysis. Rerunning frame generation preserves existing notes with the matching case list. The package contains review images and metadata for the 12 cases. Raw videos and model weights remain in the workspace.

The review is diagnostic because it selects disagreements after observing answers. It can uncover missing or distracting evidence and inform a subsequent experiment. It does not estimate a benchmark-wide mechanism or establish a new method.
