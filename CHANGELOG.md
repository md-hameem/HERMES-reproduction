# Changelog

All notable reproduction, evidence, control, and documentation changes for this project are recorded here. Dates use the evidence-update date; older sections are retained as historical milestones rather than rewritten.

## 2026-10-28 — Final AutoDL Controls and Evidence Audit

### Added

- Executed final AutoDL control workflow in `hermes-final-autodl.ipynb`.
- Extracted `HERMES_AutoDL_final_evidence/` with raw repeat/FIFO CSVs, logs, scripts, paired analyses, attention metrics, figures, provenance, and `exports/final/master_manifest_sha256.csv`.
- Added `scripts/validate_final_evidence.py`, an independent validator for manifest hashes, question identity, seven prediction files, recomputed accuracy/efficiency, paired statistics, attention artifacts, FIFO source isolation, final summaries, figures, and executed-notebook state.
- Added same-hardware HERMES KV=4000/6000 repeat predictions and budget-matched FIFO KV=4000/6000 predictions.
- Added pseudo-query attention differential outputs for current manual, causal-masked frozen-hidden, and native transformer attention.

### Findings

- Original AutoDL HERMES: 57.6% at 4K and 6K; 55.2% at 50K.
- AutoDL repeats: 57.2% at both 4K and 6K, with 4–5 changed choices and three correctness changes per budget. Same-environment nondeterminism is now demonstrated.
- FIFO matches HERMES aggregate accuracy at both budgets: exact paired p = 1.000 for 4K and 6K.
- HERMES/FIFO versus 50K remains an observed +2.4-point subset difference, not statistically conclusive (paired p ≈ 0.210).
- Manual pseudo-query attention is an approximation, not native-equivalent: mean cosine similarity is 0.637 current-versus-native and 0.724 causal-frozen-versus-native.
- Peak allocated memory and maximum cache reductions are stable; median TTFT varies between sessions and should not be overinterpreted as a single-process measurement.

### Known issues

- The uploaded final evidence directory verifies 53 payload files, but the original `HERMES_AutoDL_final_evidence.zip` byte stream is not present locally.
- Three files referenced by the embedded historical manifest were omitted from the final bundle: `exports/figures/accuracy_50.png`, `exports/figures/paired_differences_50.png`, and `exports/hermes_50_by_task.csv`.
- The executed final notebook differs from repository commit `af79580` through AutoDL network setup, direct PyTorch wheel installation, and a `DynamicCache` compatibility patch for the native attention reference. Outputs remain internally validated, but the run should not be described as an unmodified one-click execution of that commit.

## 2026-10-26 — AutoDL Efficiency Extraction

### Added

- `scripts/extract_efficiency.py` and hash-checked exports under `exports/efficiency/`.
- Question-level, video-level, and summary TTFT/memory/cache tables derived from the three preserved AutoDL logs.

### Findings

- Every original AutoDL run has 250 TTFT values, 829 chunks, and 11,434 encoded frames.
- 4K reduces peak allocated memory by 60.4% and maximum cache length by 92.0% relative to the 50K near-uncompressed HERMES control.
- TTFT is not end-to-end latency, reserved memory was not logged, and encoding/compression durations are not isolated.

## 2026-10-25 — AutoDL Preservation and Kaggle Comparison

### Added

- Fresh three-budget AutoDL generation preserved separately under `HERMES/results/`, `logs/`, and `exports/`.
- Positional Kaggle–AutoDL comparison using validated `(video_id, qa_index)` keys.

### Findings

- AutoDL accuracy: 57.6% at 4K and 6K; 55.2% at 50K.
- Kaggle-to-AutoDL changes: 4K has three choice changes and three correctness reversals; 6K has ten choice changes and nine correctness reversals; 50K is identical.
- The small cross-platform differences were plausible nondeterminism but remained unproven until the 28 October same-environment repeats.

## 2026-10-16 — Comprehensive Technical Research Report

### Added

- `docs/HERMES_Comprehensive_Technical_Research_Report.md` with literature taxonomy, paper-level technical reviews, HERMES mechanism analysis, cross-paper comparison, research gaps, and the EDMI research proposal.

### Changed

- The EDMI proposal was explicitly marked as an unvalidated research idea contingent on native controls, component ablations, and compressed-only failure analysis.

## 2026-10-13 — Artifact Intake and Paired-Analysis Verification

### Added

- Local intake and verification of 43 Kaggle artifact files, including six 10-video budget CSVs, the 50-video 4K CSV, aggregate result tables, figures, and inference logs.

### Findings

- Raw-row recomputation confirmed 57.2% at 4K and 6K and 55.2% at 50K for the preserved Kaggle generation.
- The 50-video paired table showed 4, 13, and 15 discordant questions for 4K–6K, 4K–50K, and 6K–50K respectively.
- Historical 6K and 50K Kaggle logs remained missing locally, so their memory/TTFT values could not be re-parsed.

## 2026-10-09 — Reproduction Audit and Next Research Plan

### Added

- `docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md` as the living technical audit.
- Evidence-grading framework, chronology, source review, dataset representativeness analysis, statistical guidance, risk register, phased controls, and next research plan.

### Findings

- The project was assessed as a successful subset reproduction and systems study, but not yet a complete component-level reproduction.
- Missing native non-HERMES controls, unresolved source-fidelity questions, single-shard generalization, and uninstrumented efficiency boundaries were recorded as major limitations.
