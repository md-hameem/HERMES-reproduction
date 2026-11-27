# Research evidence

Each experiment ZIP is stored once under `archives/`. Full local extractions are under ignored `local/` subdirectories. Published summary tables and figures are available directly in the repository. [Five-stage results](../results/five_stage/) · [Project status](../docs/PROJECT_STATUS.md)

| Location | Contents |
|---|---|
| [Blinded support review audit](../results/target_binding_2026-11-22/blind_review_first_pass_audit.json) | Archive hashes, execution verification and first-pass counts. The original image ZIP is stored in the local evidence directory. |
| [TwinPast media checks](../results/twinpast_development_2026-11-23/media_check.json) | Pinned pair IDs, video and sampled frame hashes, timestamps, and the exclusion reason for one pair |
| [TwinPast visual audit](../results/twinpast_development_2026-11-23/visual_audit.json) | Nine development contrasts with a matching visual option in both branches |
| [TwinPast GPU archive](archives/HERMES_twinpast_diagnostic_evidence.zip) | Original ZIP containing 9 checkpoints and 72 score rows |
| [TwinPast GPU audit](audits/twinpast_diagnostic_2026-11-23_audit.json) | Recomputed paired outcomes and archive, notebook and source integrity checks |
| [TwinPast history-only archive](archives/HERMES_twinpast_history_only_evidence.zip) | Original 9-pair run, 52 score rows and historical-state contact sheets |
| [TwinPast history-only audit](audits/twinpast_history_only_2026-11-23_audit.json) | Independent archive, prior score, frame alignment and paired-outcome verification |
| [TwinPast 7B archive](archives/HERMES_twinpast_backbone_evidence.zip) | Saved 7B run with seven checkpoints and 28 scores |
| [TwinPast 7B audit](audits/twinpast_backbone_2026-11-23_audit.json) | Exact-frame comparison against the 0.5B archive and recomputed decision |
| [TwinPast fixed-recent archive](archives/HERMES_twinpast_fixed_recent_evidence.zip) | Seven saved 7B checkpoints and 70 scores across history-only, history-plus-recent and recent-only conditions |
| [TwinPast fixed-recent audit](audits/twinpast_fixed_recent_2026-11-23_audit.json) | Verified inputs, prior replay, outcomes and decision |
| [TwinPast automatic-selection archive](archives/HERMES_twinpast_automatic_evidence.zip) | Seven saved checkpoints, six method outcomes and frame coverage |
| [TwinPast automatic-selection audit](audits/twinpast_automatic_2026-11-23_audit.json) | Recomputed ranking, predictions, correction, costs and curated-frame overlap |
| [TwinPast candidate-pool archive](archives/HERMES_twinpast_candidate_pool_evidence.zip) | Four checkpoints and 48 scores at a fixed answer frame budget |
| [TwinPast candidate-pool audit](audits/twinpast_candidate_pool_2026-11-23_audit.json) | Recomputed selections, curated replay, paired outcomes and advancement rule |
| [TwinPast temporal-window archive](archives/HERMES_twinpast_temporal_windows_evidence.zip) | Eight fresh-pair checkpoints, source-screen logs and four method outcomes |
| [TwinPast temporal-window audit](audits/twinpast_temporal_windows_2026-11-23_audit.json) | Recomputed selections, paired outcomes, annotation overlap and forward calls |
| [TwinPast sub-second archive](archives/HERMES_twinpast_subsecond_evidence.zip) | Seven checkpoints, 56 scores and 14 contact sheets |
| [TwinPast sub-second audit](audits/twinpast_subsecond_2026-11-23_audit.json) | Exact uniform replay and two robust recoveries |
| [TwinPast matched refinement archive](archives/HERMES_twinpast_refinement_evidence.zip) | Eight checkpoints and 96 answer scores |
| [TwinPast matched refinement audit](audits/twinpast_refinement_2026-11-27_audit.json) | Verified paired outcomes, frame budgets and timestamp overlap |
| [TwinPast fixed-frame readout archive](archives/HERMES_twinpast_readout_evidence.zip) | Eight checkpoints, 64 exact direct replays and 128 total choice scores |
| [TwinPast fixed-frame readout audit](audits/twinpast_readout_2026-11-27_audit.json) | Paired outcomes, observation agreement and decoder cost |
| [TwinPast CPU boundary-coverage audit](audits/twinpast_boundary_cpu_2026-11-27.json) | Frozen candidates, frame selections and paired interval coverage for 16 branches |
| [TwinPast boundary GPU archive](archives/HERMES_twinpast_boundary_evidence.zip) | Eight checkpoints, frozen selections, 64 exact baseline replays and 96 total choice scores |
| [TwinPast boundary GPU audit](audits/twinpast_boundary_gpu_2026-11-27.json) | Paired outcomes and synchronized stage times |
| [DROID state-control archive](archives/HERMES_droid_state_control_evidence.zip) | Source receipts and 96 choice scores from six source-separated pairs |
| [DROID state-control audit](audits/droid_state_control_2026-11-27_audit.json) | Identical-input checks and recomputed robust outcomes |
| [DROID natural-history archive](archives/HERMES_droid_natural_history_evidence.zip) | Seven scene pairs, 56 answer scores and 28 exact uniform repeats |
| [DROID natural-history audit](audits/droid_natural_history_2026-11-27_audit.json) | Verified 114 input frames, one recovery and no regression |
| [DROID earlier-state masking archive](archives/HERMES_droid_state_interference_evidence.zip) | Equal-area masking with fixed terminal and recent frames |
| [DROID earlier-state masking audit](audits/droid_state_interference_2026-11-27_audit.json) | 28 replay checks, 56 verified mask hashes and 84 choice scores |
| [DROID terminal-context archive](archives/HERMES_droid_terminal_context_evidence.zip) | Three context lengths, 84 choice scores and 28 exact replays |
| [DROID terminal-context audit](audits/droid_terminal_context_2026-11-27_audit.json) | Verified terminal images and paired context transitions |
| [DROID earlier-state swap archive](archives/HERMES_droid_state_swap_evidence.zip) | Frozen symmetric substitutions, 56 scores and 28 exact replays |
| [DROID earlier-state swap audit](audits/droid_state_swap_2026-11-27_audit.json) | Robust pair exchange and gold-margin analysis |
| [DROID terminal attention archive](archives/HERMES_droid_terminal_attention_evidence.zip) | 112 score vectors and 56 replay comparisons across four conditions |
| [DROID terminal attention audit](audits/droid_terminal_attention_2026-11-27_audit.json) | Verification of frame receipts, replay checks and paired outcomes |
| [DROID automatic span selection archive](archives/HERMES_droid_span_selection_evidence.zip) | Six source-separated scene pairs, 240 scores and 48 parity checks |
| [DROID automatic span selection audit](audits/droid_span_selection_2026-11-28_audit.json) | Recomputed accuracy, support coverage, window choices and advancement decision |
| [DROID interior-evidence CPU audit](audits/droid_interior_cpu_2026-11-28.json) | 14 reviewed clips, seven accepted clips and exhaustive three-frame fixed-window coverage |
| [DROID interior-evidence GPU archive](archives/HERMES_droid_interior_evidence.zip) | Input manifest, 56 scores and 14 causal-mask parity checks |
| [DROID interior-evidence GPU audit](audits/droid_interior_gpu_2026-11-28_audit.json) | Per-clip outcomes, frame receipts and comparisons with fixed windows |
| [DROID matched event-relative CPU audit](audits/droid_matched_event_cpu_2026-11-28.json) | Verified source receipts and 14 review dispositions |
| [Target-binding archive](archives/HERMES_target_binding_evidence.zip) | Original 60-clip run, 300 checkpoints, 1,800 selector outcomes and blinded window-pair worksheet |
| [Target-binding audit](audits/target_binding_2026-11-22_audit.json) | Reconstructed integrity, candidate windows, selector outputs, paired estimates and component timing |
| [Context-diagnostic archive](archives/HERMES_context_diagnostic_evidence.zip) | Completed seven-case run: 140 payloads, 84 checkpoints, 74 answers, ten skips and original review images |
| [Context-diagnostic audit](audits/context_diagnostic_2026-11-20_audit.json) | Independent source-slot/position/partition, replay, status and paired-export verification |
| [Input-diagnostic archive](archives/HERMES_input_diagnostic_evidence.zip) | Completed input controls: 287 manifest payloads, 200 checkpoints, 134 answers, 66 explicit skips and 57 review pages |
| [Input-diagnostic audit](audits/input_diagnostic_2026-11-20_audit.json) | Independent identity, frame/status, paired-subset, export and provenance checks |
| [Causal-scorer pilot archive](archives/HERMES_causal_scorer_pilot_evidence.zip) | Supplied completed eight-video screen: 103 manifest payloads, 24 checkpoints, 120 predictions, source, logs, sampling metadata and contact sheets |
| [Causal-pilot audit](audits/causal_pilot_2026-11-20_audit.json) | Independent integrity, identity, table, timing and historical replay checks |
| [Original five-stage archive](archives/HERMES_AutoDL_five_stage_evidence.zip) | All 2,921 original manifest payloads, including predictions, video checkpoints, timing events, annotations, scripts, logs and embedded October evidence |
| [Historical supplement](archives/historical_evidence_supplement.zip) | Three recovered October files and their original manifest hashes |
| [Original AutoDL evidence](autodl_2026-10-25/) | Original three-budget results, logs, paired analyses and efficiency exports. Internal historical paths are preserved. |
| [October controls](autodl_2026-10-28/) | Hash-verified HERMES repeats, FIFO controls, original attention diagnostic and preserved earlier evidence |
| [Local five-stage audit](audits/five_stage_local_audit.json) | Independently recomputed integrity, prediction, checkpoint, statistics and timing checks |

To re-audit the DROID terminal-attention run, place the original `DROID_natural_history_frames.zip` in `evidence/local/droid-natural-history/` and run `python scripts/validate_droid_terminal_attention_evidence.py` from the project root. The frame pack stays local and is checked against the frozen SHA-256 and the 114 per-frame receipts. [Frame-pack preparation](<../docs/reports/2026-11-27 droid natural history protocol.md>)

The interior coverage and matched-event CPU audits can be recomputed from the committed intake receipts and visual review decisions with `python scripts/check_droid_interior_coverage.py` and `python scripts/check_droid_matched_event_screen.py`. The span-selection and interior GPU audits also check the local frame ZIPs under `evidence/local/`. Use the companion ZIPs created by the respective notebook builders. [Span-selection protocol](<../docs/reports/2026-11-27 droid span selection protocol.md>) · [Interior GPU results](<../docs/reports/2026-11-28 droid interior GPU results.md>)

The five-stage archive SHA-256 is `b8bc4cf8a3b28e6854b2e76b8c0239b12d3b5598ee1a07dafc03d0326a900518`. The supplement SHA-256 is `4be19b26c61237860c322c5431090dcafc900aae400b9c0ed8204a2c68aab66a`. The uploaded executed notebook is stored under `notebooks/executed/` and its SHA-256 is recorded in the local audit.

Run both audits from the project root with Python, NumPy, pandas and SciPy installed.

```bash
python scripts/validate_final_evidence.py
python scripts/validate_five_stage_evidence.py
```

The older audit reports the three omissions inside its original bundle. The new audit also verifies their recovered bytes in the separate supplement.

The latest validator reads directly from the ZIP. A full extraction is optional for local inspection.

```bash
python -m zipfile -e evidence/archives/HERMES_AutoDL_five_stage_evidence.zip evidence/local/five_stage
```

Historical evidence, the uploaded executed notebook and published summary copies retain the measured output bytes. Current workflow and validation code live outside the evidence packages. The full-benchmark aggregate reuses October samples 201–250 across sessions and hardware. Live recovery patches are documented in the current status report. [Execution record](../notebooks/executed/2026-11-18_hermes-five-stage-autodl.ipynb) · [Audit implementation](../scripts/validate_five_stage_evidence.py)

## Causal-scorer pilot received 20 November 2026

The archive SHA256 is `af4b7392e90bd3f433ad20182a9058d8c8027a8ede936ed5d759572be04e0361`. The [executed notebook](../notebooks/executed/2026-11-20_hermes-causal-scorer-pilot.ipynb) SHA256 is `24620321fc4b2783e68706928e8831456579aa8e8e03a4eccbf2c5a1345686a6`. They preserve the uploaded bytes. Full extraction under `local/causal_pilot_2026-11-20/` is ignored; [compact results](../results/causal_pilot_2026-11-20/README.md) and the [publication decision](<../docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) are readable without extraction.

Run `python scripts/validate_causal_pilot_evidence.py` using Python's standard library. The validator never executes archived code. It verifies all 103 payloads and recomputes prediction, pairwise and timing tables, while documenting that raw videos/features and the GPU computation cannot be independently rerun from the package alone. The original ZIP retains three benign editor-checkpoint logs; reader-facing exports omit those caches. Later workflow heartbeat changes are not rewritten into the archived execution.

## Input/capability diagnostic received 20 November 2026

The canonical archive SHA256 is `f184271bd8929f2ca5dead71a1ec60e4e1b555f33f2ed66d9909e3d7e0ffa1e8`. The [executed notebook](../notebooks/executed/2026-11-20_hermes-input-diagnostic.ipynb) SHA256 is `24df662b86955295cae4aa24a559806b72021ded6ce6a0815a0eb1ce321313bf`. Both preserve the uploaded bytes. Full extraction under `local/input_diagnostic_2026-11-20/` is ignored. [Compact results](../results/input_diagnostic_2026-11-20/) and [analysis](<../docs/reports/2026-11-20 input diagnostic analysis and next decision.md>) are published separately.

Run `python scripts/validate_input_diagnostic_evidence.py` with the standard library. It verifies all 287 payloads, 200 unique case checkpoints, 120 mandatory answers plus 14 native answers, explicit skips, all export tables, historical exact-subset comparisons and eight historical feature-reuse provenance records. It never executes archived code. Five audit tests include corrupted correctness/cache/input identities and skipped-case counting. Raw videos and tensors remain outside the package; oracle tests and original-method claims are not established.

## Context diagnostic received 20 November 2026

Archive SHA256: `fee3165e323c0f7106688a37cb185ec390d4f831e9debdfbf0352bd72881b774`. [Executed notebook](../notebooks/executed/2026-11-20_hermes-context-diagnostic.ipynb) SHA256: `26f2ec995b044b7d1595c73945eebc0285675882a7f93bd7719b53dfa6bf2055`. Both preserve uploaded bytes. Full extraction under `local/context_diagnostic_2026-11-20/` is ignored; [compact results](../results/context_diagnostic_2026-11-20/) and [analysis](<../docs/reports/2026-11-20 context diagnostic analysis and next decision.md>) are published separately.

Run `python scripts/validate_context_diagnostic_evidence.py` using the standard library. It verifies all 140 payloads, 84 checkpoints, 74 answers/ten explicit skips, exact source-slot maps (including declared duplicates), tail positions, ingestion groups, prior replay choices and all exported tables. Five new audit tests pass. Archived code is parsed/read rather than executed. Local auditing cannot rerun absent video/tensors or GPU inference; repeated filler and selected support are diagnostic conditions, not established original methods.

## Stage attribution received 21 November 2026

The [evidence ZIP](archives/HERMES_stage_attribution_evidence.zip) and [executed notebook](../notebooks/executed/2026-11-21_hermes-stage-attribution.ipynb) preserve the original upload. Run `python scripts/validate_stage_attribution_evidence.py` to verify 271 payloads, 204 checkpoints, mask/position metadata, 81 historical replays and reconstructed counts. It reads uploaded source and cells as data. [Audit and hashes](audits/stage_attribution_2026-11-21_audit.json) · [Tables](../results/stage_attribution_2026-11-21/).

## Matched-boundary control received 21 November 2026

The [evidence ZIP](archives/HERMES_matched_boundary_evidence.zip) and [executed notebook](../notebooks/executed/2026-11-21_hermes-matched-boundary.ipynb) preserve the original uploads. Run `python scripts/validate_matched_boundary_evidence.py` to verify 274 payloads, 200 checkpoints, construction and readout mask traces, 120 required replays and all paired counts. It reads embedded sources as literals and does not run uploaded code. [Audit and hashes](audits/matched_boundary_2026-11-21_audit.json) · [Tables](../results/matched_boundary_2026-11-21/).

## Short query-cache run received 21 November 2026

The [evidence ZIP](archives/HERMES_short_query_evidence.zip) and [executed notebook](../notebooks/executed/2026-11-21_hermes-short-query-cache.ipynb) preserve the original uploads. Run `python scripts/validate_short_query_evidence.py` to verify 115 payloads, 40 checkpoints, 160 answers, 320 timed calls and the paired cost accounting. The validator parses the recorded source and outputs without executing them. [Audit and hashes](audits/short_query_2026-11-21_audit.json) · [Tables](../results/short_query_2026-11-21/).

## Construction selection received 21 November 2026

The [evidence ZIP](archives/HERMES_construction_selection_evidence.zip) preserves the upload. The accompanying notebook was byte-identical to the [maintained source](../notebooks/autodl/hermes-construction-selection.ipynb) and had no execution outputs. Its local copy is under ignored `local/construction_selection_2026-11-21/`. Run `python scripts/validate_construction_selection_evidence.py` to verify 117 payloads, 240 answers, 80 replays, position traces and paired results from the ZIP. [Audit](audits/construction_selection_2026-11-21_audit.json) · [Tables](../results/construction_selection_2026-11-21/).

The subsequently saved [executed notebook](../notebooks/executed/2026-11-21_hermes-construction-selection.ipynb) is archived separately. The validator now also checks that its source and logged final ZIP hash match the audited run.

## Short query confirmation notebook received 21 November 2026

The [completed v2 archive](archives/HERMES_short_query_confirmation_evidence.zip) matches the final SHA256 in the [executed notebook](../notebooks/executed/2026-11-21_hermes-short-query-confirmation.ipynb). Run `python scripts/validate_short_query_confirmation_evidence.py` to verify all 360 payloads, 150 question checkpoints, 450 answers and the video-paired comparison. The separate v1 checksum-failure ZIP remains under ignored `local/short_query_confirmation_v1_failed/`. [Audit and hashes](audits/short_query_confirmation_2026-11-21_audit.json) · [Tables](../results/short_query_confirmation_2026-11-21/).

## Confirmation visual review received 21 November 2026

The [visual review ZIP](archives/HERMES_confirmation_visual_review.zip) matches the final hash in the [executed notebook](../notebooks/executed/2026-11-21_hermes-confirmation-visual-review.ipynb). The package contains all 35 contact sheets and 231 frame slots. The submitted notes were blank and remain intact. Run `python scripts/validate_confirmation_visual_review_evidence.py` to verify the saved material. [Analyst observations](../results/confirmation_visual_review_2026-11-21/) · [Audit](audits/confirmation_visual_review_2026-11-21_audit.json).

The follow-up ZIP (SHA256 `dbe820afbb90e5f60cc5e479250da07f0f33812bf6ce03443c5b3ca2fdb7d526`) adds two PNGs. Every original ZIP member matches byte for byte. The [verified PNGs](../results/confirmation_visual_review_2026-11-21/originals/) are tracked directly, and the full follow-up ZIP is preserved under ignored `local/confirmation_visual_review_2026-11-21/`.
