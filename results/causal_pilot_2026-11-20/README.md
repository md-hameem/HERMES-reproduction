# Audited causal-scorer pilot results

Completed AutoDL evidence received and audited 20 November 2026. Eight annotated videos, 40 distinct questions, three methods, 120 answers. All methods score 24/40; fresh causal versus current HERMES changes one wrong choice and zero correctness outcomes.

| Files | Role |
|---|---|
| `summary.csv`, `predictions.csv`, `paired_questions.csv`, `run_report.md` | Byte-identical copies of the original ZIP's exports; `run_report.md` was originally `exports/report.md` |
| `audited_method_metrics.csv`, `audited_video_metrics.csv` | Independently reconstructed QA, replay, component timing and memory values |
| `pairwise_outcomes.csv` | Choice changes and paired correctness wins/losses, without population significance claims |
| `question_casebook.csv` | All 40 question identities, outcomes, sampling boundary and prior-eviction status; support/cause still unclassified |
| `historical_replay_comparison.csv` | New FIFO/HERMES predictions match the historical predictions exactly for these 40 questions |
| `task_outcomes.csv` | Descriptive task-level counts for the selected screen |
| `next_review_queue.csv` | Generated 18-case review queue. Copy to a separate working file before annotating; regeneration replaces this file. |

Rebuild and validate with `python scripts/validate_causal_pilot_evidence.py`. The validator uses only the Python standard library and reads the preserved ZIP and executed notebook without executing their code. It also reads existing annotations/historical evidence and validates the source identities against the checkout/pinned runner hash.

This is a selected development screen. The 60% accuracy is not a new full-benchmark estimate, and replay timings exclude shared vision preparation. Raw videos and feature tensors are not included in the evidence ZIP. Contact sheets are sparse visual aids; do not treat missing thumbnails as proof that evidence was absent. See the [analysis and publication decision](<../../docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>), [audit JSON](../../evidence/audits/causal_pilot_2026-11-20_audit.json), [source archive](../../evidence/archives/HERMES_causal_scorer_pilot_evidence.zip) and [execution record](../../notebooks/executed/2026-11-20_hermes-causal-scorer-pilot.ipynb).
