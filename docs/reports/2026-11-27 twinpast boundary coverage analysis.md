# TwinPast scene-boundary coverage on CPU

This retrospective development test used the 16 clips from the matched refinement experiment. The CPU script verified each clip against the frozen media manifest and reproduced the archived local-uniform frame indices on all 16 branches. Twelve branches across six pairs have evidence intervals.

The rule keeps the 12 archived candidate windows, their question-relevance scores and the three-window budget. It flags a possible cut when adjacent 4 FPS frames differ by at least 0.20 mean absolute RGB distance after resizing to 32 × 18 and normalizing by 255. The ranking takes the highest-relevance nonoverlapping triple containing a flagged window when one exists. The anchor variant keeps the frame immediately before the strongest flagged cut in the selected triple and retains the farther fixed-offset sample in that window. All methods select six earlier frames and reserve the four original recent frames. The rule uses neither interval labels nor gold answers to choose frames.

| Policy | Any annotated interval | Final annotated interval | Pairs with both final intervals covered |
|---|---:|---:|---:|
| Original ranking and local uniform sampling | 4/12 | 0/12 | 0/6 |
| Boundary-constrained ranking only | 5/12 | 2/12 | 1/6 |
| Pre-cut frame anchor only | 6/12 | 3/12 | 1/6 |
| Boundary-constrained ranking and pre-cut anchor | **8/12** | **8/12** | **4/6** |

The combined rule gained interval coverage on both branches of the guitar-case and publication pairs and retained all four existing hits. It moved the hits to the final intervals on both branches of the yellow-box and pillow pairs. The opening letter-order and blue-lid pairs were still missed by the combined rule. Across all 16 branches, the constraint changed seven window triples and the combined rule changed 11 six-frame histories.

The coverage result met the development criterion of two or more new final-interval hits, no final-interval losses and no reduction in overall interval coverage. The final interval is selected by its latest annotated end time. It is a timing proxy, and visual sufficiency still needs review. The rule and threshold were chosen after inspecting this development set, then held fixed during the CPU run. The threshold has no independent calibration. Camera or object motion may also trigger the cut detector.

The cut scan requested 1,996 distinct candidate frame indices summed across branches, compared with 624 selected-window sample positions in the original local refinement. These counts describe the retrieval and sampling work, which can overlap with other decoding. They do not measure end-to-end latency. The answer-frame budget stayed at ten, and the CPU run did not score answers.

## Next experiment

Freeze the combined rule and send its selected frames to the original direct readout. Compare it with coarse uniform and local uniform in the same session using identical budgets, model, prompts, label orders and shared recent frames. Use the eight reviewed pairs for this development decision and count robust recoveries and regressions against coarse uniform. Report the added CPU work and total latency before evaluating fresh, source-separated footage. This test will show whether the timestamp gain translates into answer accuracy.

The [GPU notebook](../../notebooks/autodl/hermes-twinpast-boundary.ipynb) recomputes the relevance scores and boundary selections and replays 64 archived answer scores. It records synchronized stage times and allocates shared work to each policy for a timing estimate. Independent end-to-end runs would be needed for a latency claim.

The GPU run later passed numerical replay and recovered zero robust pairs. [GPU audit and research decision](<2026-11-27 twinpast boundary GPU analysis.md>)

Run `python scripts/check_twinpast_boundary_coverage.py` from the repository root with Decord, NumPy and Pillow and the same verified media used for the contact sheets. [CPU audit and per-branch selections](../../evidence/audits/twinpast_boundary_cpu_2026-11-27.json) · [Fixed-frame readout result](<2026-11-27 twinpast fixed-frame readout analysis.md>)
