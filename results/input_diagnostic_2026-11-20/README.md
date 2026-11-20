# Audited input/capability diagnostic results

Received 20 November 2026: eight videos, 40 questions, 200 case records. There are 134 answers and 66 explicit skips. Question-only scores 14/40; recent-4 25/40; recent-20 21/40. Native prefix scores 10/14, exactly matching all historical baselines on those identities. No support-oracle answers were requested.

| Files | Provenance |
|---|---|
| `summary.csv`, `predictions.csv`, `paired_historical.csv`, `run_report.md` | Byte-identical original ZIP exports; `run_report.md` was `exports/report.md` |
| `within_run_pairs.csv` | Independently reconstructed matched-input-control correctness/choice comparisons |
| `question_casebook.csv` | All 40 question identities, historical HERMES outcomes and new control outcomes/statuses |
| `per_video_metrics.csv`, `task_outcomes.csv` | Descriptive source-video/task counts; small selected groups |
| `figures/sample_204_q0_prefix.jpg` | Original eight-frame prefix sheet; yellow addition visible at sampled index 7, 14 s |
| `figures/sample_204_q3_recent_page.jpg` | Original final prefix page; requested purple text overlay visible at sampled index 69, 138 s |

Images are unchanged copies from the preserved ZIP. Their analyst-reviewed observations do not replace original-resolution support checks or create executed oracle results. Full 57-page review material and pending 40-row support sheet remain inside the canonical ZIP/ignored local extraction.

Rebuild and validate with `python scripts/validate_input_diagnostic_evidence.py`. The validator reads/parses evidence without executing archived sources. Videos/tensors are absent, so local auditing cannot rerun GPU inference. The 1.68-minute recorded phase excludes setup/download/model loading and final packaging and reused the prior feature cache. Historical comparisons are not same-session efficiency tests. Native's 14-case score cannot be compared directly to 40-case scores; no population effect, equivalence or original contribution is established.

[Detailed analysis and next decision](<../../docs/reports/2026-11-20 input diagnostic analysis and next decision.md>) · [Audit JSON](../../evidence/audits/input_diagnostic_2026-11-20_audit.json) · [Canonical ZIP](../../evidence/archives/HERMES_input_diagnostic_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-20_hermes-input-diagnostic.ipynb)
