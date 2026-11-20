# Fresh-state input diagnostic

Selected development cases; no population or original-method claim.

| Method | Answered | Skipped | Correct | Accuracy on answered |
|---|---:|---:|---:|---:|
| question_only | 40 | 0 | 14 | 35.0% |
| recent_4 | 40 | 0 | 25 | 62.5% |
| recent_20 | 40 | 0 | 21 | 52.5% |
| native_prefix | 14 | 26 | 10 | 71.4% |
| support_oracle | 0 | 40 | 0 | N/A |

Native-prefix/oracle rows may cover different subsets: use paired_historical.csv for exact-subset comparisons.
All controls reset per question; no eviction or past-answer history. Recent/oracle inputs compact selected frames chronologically.
Historical comparisons reuse old predictions; they do not establish same-session efficiency or isolate memory policy alone.
Oracle is privileged human support, not a deployable method. No oracle runs without verified annotations.
Question wall time includes reset/feature reading/ingestion/answer; excludes setup and shared feature preparation.
Review sheets show original sampled frames, not proof of support annotation. Source videos and feature tensors are excluded from ZIP.
