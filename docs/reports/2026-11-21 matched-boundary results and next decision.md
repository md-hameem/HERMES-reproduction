# Matched-boundary results and next experiment

Matching the recent encoding groups preserved all 200 answer choices from the previous stage-attribution run. The construction effect survived on the same two questions, 208/q4 and 209/q0, with no losses in the primary comparison. The audit verified 274 payloads, 200 checkpoints and all 120 required replays. It also checked that the two construction traces differed only in the blocked history span. The experiment reused all eight feature caches and took 55.53 seconds after model load. The elapsed time describes this controlled replay.

| Matched condition | History visible during recent encoding | History visible during answering | Correct |
|---|---|---|---:|
| A matched | Yes | Yes | 21/40 |
| B | No | Yes | 24/40 |
| C matched | Yes | No | 27/40 |
| D | No | No | 25/40 |
| Compact recent-four | Absent | Absent | 25/40 |

- **Construction effect with history hidden at readout:** C matched versus D had two wins and zero losses. The recovery cases ask for the current number of green blocks (208/q4) and the next action after a green block is placed (209/q0). The two new matched encoding calls keep the older prefix, recent inputs, mask shape and positions fixed. The answer mask is also held fixed.
- **The effect depends on readout.** When older tokens remain visible during answering, A matched versus B had zero wins and three losses. Hiding those tokens in the same contextualized cache (A matched → C matched) had seven wins and one loss. The data support a distinction between using history to build recent representations and making history directly available to the decoder.
- **The practical accuracy margin is small.** C matched beat compact recent-four by three wins and one loss. Six choices changed, including two changes between incorrect choices. The net advantage disappears without video 208. C matched also retains the older physical cache slots. This run provides no measurement of the storage or timing effect of removing them.

The strict FP32 physical reference passed, as did the exact FP16 same-shape perturbation checks. All nine tested FP16 top tokens agreed. The maximum FP16 full-logit difference between cross-path references was 0.03125, with 44 entries above the earlier elementwise tolerance. Those differences remain in the evidence. The matched experiment resolves the grouping issue for this comparison and revisits the same questions. The two runs count as one development sample when assessing generalization.

## Next experiment

Implement one **physically short query cache** for the matched C policy. Let the recent frames attend to older history during encoding, then construct an answer cache that contains the prompt and recent slots only. Preserve the recent keys' logical positions and the prefill/decode position starts. Use the existing masked C run as the behavior reference. First verify the physical and masked paths numerically and record choice changes on the 40 development questions. Keep mismatches visible because finite-precision replay can differ when the physical shape changes.

Freeze one candidate before evaluating accuracy. Compare it against compact recent-four, full contextual readout and masked C. Use footage from source groups that did not inform development, with video-level pairing, point estimates and intervals. If using a new subset of the previously inspected StreamingBench benchmark, report it as within-benchmark confirmation and retain a source-separated external test for any generalization claim. Set the evaluation manifest using metadata and source overlap checks before reviewing candidate outcomes. Use actual GPU throughput to choose sample size, and distinguish query-cache bytes and query time from total live cache memory, construction work and whole-system timing.

Advance the publication candidate only if it achieves a credible accuracy advantage over compact recent-four or a measured efficiency advantage at comparable quality to a strong baseline. A physical query-cache reduction by itself does not establish novelty. Review the closest compact and decoupled cache methods before selecting the manuscript contribution. If the short query cache does not survive the controls, record that outcome before adding a history fallback or another policy.

[Audited tables](../../results/matched_boundary_2026-11-21/) · [Audit and hashes](../../evidence/audits/matched_boundary_2026-11-21_audit.json) · [Executed notebook](../../notebooks/executed/2026-11-21_hermes-matched-boundary.ipynb)
