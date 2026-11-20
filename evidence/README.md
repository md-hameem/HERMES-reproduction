# Research evidence

The five-stage, causal-scorer, input, context and stage-attribution ZIPs are each stored once as canonical evidence payloads in Git. Full local extractions are under ignored `local/` subdirectories. Published summary tables and figures are available directly in the repository. [Five-stage results](../results/five_stage/) · [Project status](../docs/PROJECT_STATUS.md)

| Location | Contents |
|---|---|
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
