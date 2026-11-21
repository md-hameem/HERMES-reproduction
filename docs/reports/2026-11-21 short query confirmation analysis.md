# Short query confirmation results

The frozen comparison completed **450 answers across 150 questions and 30 clips**. The audit verified 360 archive payloads, all checkpoint and annotation identities, the 30 media hash receipts, cache sizes, query/decode position traces and the paired bootstrap calculation. The saved media hashes are distinct. Original source overlap remains unverified, and the raw media are outside the evidence ZIP.

| Arm | Correct | Accuracy | Median answer K/V |
|---|---:|---:|---:|
| Compact recent-four | 74/150 | 49.33% | 9.34 MiB |
| Latest-two contextual short readout | 72/150 | 48.00% | 9.34 MiB |
| All-history contextual short readout | 70/150 | 46.67% | 9.34 MiB |

**Primary result:** all-history versus compact recent-four is **−2.67 percentage points** (95% video-bootstrap interval **−7.33 to +1.33**). All-history recovered four answers and lost eight, and changed 19 choices. The per-video balance was three positive, six negative and 21 tied. Omitting one video at a time yields −3.45 to −1.38 points. The point estimate did not depend on one clip, while the interval leaves the true effect uncertain. The gain seen on the original eight development videos did not carry over in this comparison.

Latest-two versus compact had three recoveries and five losses. All-history versus latest-two had two recoveries and four losses. The second comparison keeps the contextual position convention fixed. The primary comparison changes recent construction context and also uses the compact baseline's ordinary positions. It assesses the full candidate against a practical equal-budget baseline. The interval cannot establish equivalence.

The measured scoring-free construction, extraction and answer stages were faster for the contextual arms after the shared reference preparation. The median paired prepared-path differences versus compact were 9.72 ms for all-history and 12.08 ms for latest-two. The shared preparation already encoded the original history and recent frames so both contextual arms could reuse an exact historical prefix. The timing difference does not establish a whole-system speedup. The run used about 4.95 GB of range transfers and took 700.88 seconds after model load, including feature preparation and preflight.

## Research decision

Do not advance unconditional contextual encoding as an accuracy improvement. Retain compact recent-four as the working baseline. The supported thesis finding is narrower: withholding direct history can help on controlled examples, and physical removal matched masked answering on the development questions. The same-cache-size contextual policy did not improve the frozen 30-clip point estimate.

The next useful work is a **visual evidence review of the 12 primary discordant questions**. Compare the frames actually available before each query and document whether decisive evidence appears in the recent four, in older history, in both, or in neither. Keep visible evidence and annotation ambiguity separate. This is a diagnostic follow-up after seeing outcomes. Use it to identify a mechanism that predicts when history could help, then evaluate any resulting rule on footage kept separate from both development and confirmation. Avoid tuning the older-frame count on these 30 clips. A distinct validated method for the Q1 submission remains open.

[Audited tables](../../results/short_query_confirmation_2026-11-21/) · [Audit and hashes](../../evidence/audits/short_query_confirmation_2026-11-21_audit.json) · [Executed notebook](../../notebooks/executed/2026-11-21_hermes-short-query-confirmation.ipynb)
