# TwinPast visual review of matched refinement

Reviewed 16 branch contact sheets from the eight completed development pairs. The renderer verified each clip against its frozen size and SHA-256 and checked the RGB hashes of all displayed answer frames against the GPU checkpoints. It displayed the annotated reference frames separately. The review used the known answers and annotations and was conducted by one assistant. The judgments below are exploratory.

## Where the evidence was lost

| Stage | Branches with a timestamp inside an annotated interval |
|---|---:|
| All 12 candidate windows, sampled at 4 FPS for this audit | 10/12 |
| Three selected windows at 4 FPS | 6/12 |
| Six frames selected by RGB change | 6/12 |
| Six frames selected by local uniform | 4/12 |

Four branches lost interval coverage when the 12 candidates were reduced to three windows. These were both branches of the letter-order and guitar-case pairs. Both branches of the blue-lid pair lacked an interval hit even across all candidate windows. This is a timestamp check on logged windows. It does not establish that the endpoint frames used for scoring exposed the annotated state.

| Development pair | What the frames show |
|---|---|
| `twinpast_0255`, letter order | Both local policies omit the opening interval on both branches. Coarse uniform retains the opening frame and solves both branches in both orders. |
| `twinpast_0118`, guitar case | The selected windows omit the terminal guitar-case state at 5–5.625 seconds. On A, both local policies show the earlier open case while the reference shows the cover lowered. On B, local uniform includes an open-case view earlier in the scene. Its answers still fail both orders. |
| `twinpast_0338`, yellow box | RGB change picks the hand moving the cover. The stable post-action frames around 3.6–4.2 seconds lie outside the selected windows on both branches. B is answered correctly in only one order despite the box becoming visible in a selected transition frame. |
| `twinpast_0164`, publication | RGB change selects the terminal closed publication on A and the terminal open publication on B at 4.5 seconds. A is correct in only one order. B is correct in both. |
| `twinpast_0085`, pillow | A shows the exposed pillow at 4.0 seconds in the RGB-change row. The answer changes with option order. B is correct in both orders even though its RGB-change row shows the sheet moving toward the pillow before the final covered state. |
| `twinpast_0206`, blue lid | Annotated interval overlap is zero for both local policies. At 11.75 seconds, just outside the interval, the lid and supporting object are already visible on both branches in the RGB-change row. It answers both branches incorrectly in both orders. |
| `twinpast_0120`, box flaps | On A, the RGB-change row includes the box with its flaps down before the cut. The model answers correctly in only one order. On B, both local policies show the later backpack scene instead of the packing box. The pair has no scored evidence intervals. |
| `twinpast_0021`, eggs | On A, the local selections start when eggs are already being poured or cooked. On B, the bowl appears earlier but the decisive view of separate yolks is unclear at the contact-sheet scale. The pair has no scored evidence intervals. |

## Next controlled experiment

Prioritize a **fixed-frame readout test**. The publication and pillow examples contain the required state in the exact frames supplied to the model, yet the answer depends on option order. The blue-lid example shows why timestamp hits alone would miss a readout problem.

1. Replay the archived direct scores on the same six historical and four shared recent frames, with both option orders. Use coarse uniform as the primary frame set because its direct baseline solved one robust pair. Use RGB-change frames as a separate diagnostic frame set because they contain visible states the direct readout missed. Check that predictions reproduce before testing the intervention.
2. Compare direct choice scoring with one readout that extracts the visible state of the named target before scoring the same answer options. Use the same frames, chronological order, model, and precision. Account for any extra prompts, tokens and latency.
3. Score robust pair accuracy and count both recoveries and regressions on all eight reviewed pairs. The clear visual cases explain behavior. Keep them in the overall denominator and keep the annotated reference frames out of the readout comparison.
4. A development signal requires at least one robust recovery and no loss of the robust baseline pair on the primary frame set. Report the RGB-change comparison separately. Use fresh, source-separated cases for confirmation before making a publication claim.

The separate retrieval result suggests a follow-up that preserves segment boundaries during window ranking. Keep that change out of the fixed-frame experiment so an improvement can be attributed to the readout.

## Reproduce the contact sheets

From the repository root, run `python scripts/review_twinpast_refinement.py --media PATH_TO_CLIPS` in an environment with Decord, NumPy and Pillow. Supply clips as `pair_id/A.mp4` and `pair_id/B.mp4`. The default output under `evidence/local/twinpast-refinement-review` contains the 16 contact sheets and a JSON receipt of the timestamp checks. [Matched refinement audit](<2026-11-27 twinpast matched refinement analysis.md>)
