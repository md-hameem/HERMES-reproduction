# Audited context/content diagnostic

Received 20 November 2026: seven selected case blocks, 84 checkpoints, 74 answers and ten explicit planned skips. All 26 comparable earlier choices replay exactly; 19 answered chunk-partition contrasts change zero choices. Full-prefix opening-frame replacement recovers three errors on the same six answered identities; local replacement recovers two and loses one. Two analyst support candidates both answer correctly.

| Files | Role |
|---|---|
| `summary.csv`, `predictions.csv`, `replay_checks.csv`, `contrasts.csv`, `run_report.md` | Byte-identical original ZIP exports; `run_report.md` was `exports/report.md` |
| `audited_pair_summary.csv` | Independently reconstructed paired recoveries/regressions and effective replacement counts |
| `question_casebook.csv` | All seven questions, gold and every arm's choice/correctness/status |
| `figures/` | Unchanged source-resolution candidate PNGs for 204/q0 and 204/q3; original last sampled page for 203/q2 |

Rebuild with `python scripts/validate_context_diagnostic_evidence.py`. The standard-library validator parses/reads evidence without executing archived source. It verifies all 140 manifest payloads, repeated source/position/group maps, row/config hashes and exact export tables. Raw videos/features and numerical inference cannot be independently rerun from the ZIP alone. Five rejection/control tests accompany the audit.

The controls establish intervention-sensitive **selected examples**, not population gains or a new method. Repeated filler is not certified neutral; the opening frame can contain meaningful objects. Multiple arms on one question are not independent samples. Candidate support is analyst-selected, not certified human oracle evaluation. The 75.330-second recorded phase reused five feature sets and excludes setup/download/model loading/final packaging.

[Detailed analysis and next decision](<../../docs/reports/2026-11-20 context diagnostic analysis and next decision.md>) · [Stage-attribution protocol](../../paper/neurocomputing/stage_attribution_protocol.md) · [Audit JSON](../../evidence/audits/context_diagnostic_2026-11-20_audit.json) · [Canonical ZIP](../../evidence/archives/HERMES_context_diagnostic_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-20_hermes-context-diagnostic.ipynb)
