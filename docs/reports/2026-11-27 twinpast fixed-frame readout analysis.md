# TwinPast fixed-frame readout audit

The AutoDL run reproduced all 64 archived direct score vectors with maximum absolute error 0.0 before testing the observation readout. The audit verified 27 payloads and recomputed 128 choice scores from eight development pairs. Each comparison reused the same six history frames, four recent frames, model and precision. The observation was generated once per branch and frame set without answer options, then supplied to both option orders.

| Frame set | Direct robust pairs | Observation robust pairs | Correct branch-order answers, direct → observation | Decoder forwards, direct → observation |
|---|---:|---:|---:|---:|
| Coarse uniform, primary | 1/8 | 1/8 | 16/32 → 17/32 | 32 → 120 |
| RGB change, diagnostic | 0/8 | 0/8 | 16/32 → 13/32 | 32 → 118 |

The primary comparison changed five answers from wrong to correct and four from correct to wrong. The diagnostic comparison regressed three answers and recovered none. Neither frame set gained or lost a robust pair. The development criterion set before the run required at least one robust recovery without losing the primary baseline pair.

On coarse uniform, the observation readout processed 98,846 decoder input tokens versus 65,512 for direct scoring. Summed forward time was 18.45 seconds versus 10.67 seconds. These are decoder measurements from this run. They exclude media I/O, vision encoding and shared retrieval.

The two branches have different correct answers. Despite that contrast, the generated observations were textually identical on all eight coarse pairs and six of eight RGB-change pairs. For example, both RGB-change branches of the publication pair produced “closed,” although the contact sheets show opposite terminal states. Both blue-lid branches produced “unclear.” The observation text is a diagnostic of this one prompt and model, and the second pass still sees the frames. Identical observations do not establish why every prediction changed.

## Research decision

Retire this free-text observation readout and keep its result as a fixed-input ablation. The earlier visual review showed that the selected windows retained annotated timestamps on 6/12 branches versus 10/12 when auditing all candidate windows. Use that gap to guide a CPU-only test of boundary-aware window ranking.

Keep the existing question-relevance scores and the budget of three windows. Compare their original ranking with a rule that preserves a candidate's last frame before a detected scene cut. Measure terminal-state and interval coverage on the reviewed clips without using labels to choose frames. This tests whether ranking and sampling can retain more relevant evidence before another GPU run. A favorable development result would justify freezing the rule and testing matched answer accuracy and cost on fresh, source-separated cases. It would not establish improvement on its own.

[Audit JSON](../../evidence/audits/twinpast_readout_2026-11-27_audit.json) · [Evidence ZIP](../../evidence/archives/HERMES_twinpast_readout_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-twinpast-fixed-frame-readout.ipynb) · [Visual review](<2026-11-27 twinpast refinement visual review.md>)
