# Context/partition diagnostic

Seven selected development cases; not an algorithm evaluation or population accuracy estimate.

| Arm | Answered | Skipped | Correct |
|---|---:|---:|---:|
| question_only | 7 | 0 | 1 |
| recent_4_original | 7 | 0 | 3 |
| recent_20_original | 7 | 0 | 2 |
| full_prefix_original | 6 | 1 | 1 |
| full_prefix_framewise | 6 | 1 | 1 |
| recent_20_fixed4 | 7 | 0 | 2 |
| recent_20_opening_fill | 7 | 0 | 4 |
| recent_20_local_fill | 7 | 0 | 2 |
| full_prefix_fixed4 | 6 | 1 | 1 |
| full_prefix_opening_fill | 6 | 1 | 4 |
| full_prefix_local_fill | 6 | 1 | 2 |
| candidate_support | 2 | 5 | 2 |

Historical replay choice mismatches: 0. Inspect mismatches before interpreting interventions.
Inspect exact-content chunking controls before attributing differences to semantics.
Replacement arms keep last-four source frames and their token positions fixed at equal input length; they repeat pre-query fillers, which are not certified neutral.
When there are no earlier slots, replacement arms are identical-input controls, not independent evidence.
Candidate support at 204/q0 [6,7] and 204/q3 [69] comes from analyst-reviewed saved sheets; no human oracle certification is claimed.
Gold is used only for scoring. All answers reset state and use no compression/past answers. Resource-limited full prefixes are skipped without truncation.
Question wall times exclude model/setup/shared preparation. No publication-ready improvement is implied.
