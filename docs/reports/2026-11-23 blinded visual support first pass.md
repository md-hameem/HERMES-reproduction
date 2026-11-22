# Blinded visual support review

The CPU run produced 147 sheets for 32 questions from 28 clips. Local verification decoded every JPEG, matched the blank worksheet and HTML index, and verified the notebook completion log. The executed notebook contains one added network setup cell. Its review cells match the maintained version. The original ZIP is preserved under the local evidence directory with its SHA-256 recorded. [Executed notebook](../../notebooks/executed/2026-11-23_hermes-blind-support-review.ipynb) · [Packet audit](../../results/target_binding_2026-11-22/blind_review_first_pass_audit.json)

## First reviewer pass

I inspected all 147 window sheets without opening the gold answers, model predictions, or sampling strata. This is a single assistant visual review of sampled stills. Motion, small text and inferred causes can require the original footage. The counts describe the purposive review set.

| Window label | Count |
|---|---:|
| Visible support at the query time | 17 |
| Wrong target, setting, or time | 63 |
| Insufficient visible information | 24 |
| Uncertain | 43 |

- **Q03, masking tape.** The contact sheet shows the tape being applied across the painted horizon at samples 72 and 73. Samples 70 and 71 show a similar horizontal mark made with a brush before tape is applied. The newest window still shows the tape.
- **Q17, workout.** Samples 264–266 label the left exercise panel **Low-to-High**. Samples 261–262 show an adjacent single-panel exercise view. The newest window also shows the labeled left panel.
- **Q14, animated rabbit.** The earlier shoe advertisement and later wall breach are visible. The stills do not settle the character's motive. This is a candidate for independent review using contiguous footage.
- **Q22, vehicle ahead.** An older window shows a white minivan. The nearest observed frame shows a yellow taxi directly ahead. This is a useful control for the risk of replacing an answer with stale history.

Two questions had clearly visible, time-valid support in an older window on this pass. Both also had visible support in the latest window. Eight questions across four clips were required to move to the GPU pilot. The first pass falls short even before assessing whether each distractor meets the matching rule. No second, independent review or adjudication has been completed. [First-pass labels](../../results/target_binding_2026-11-22/visual_support_first_pass.csv) · [Pilot decision rule](../../paper/neurocomputing/historical_correction_execution_plan.md)

## Next decision

The independent packet remains available for later adjudication. The assistant's full-size recheck confirmed both earlier-support cases and is part of the same reviewer assessment. The team proceeded with a separate TwinPast development screen, which found nine clear earlier-evidence contrasts with identical recent frames. The next experiment is a paired history-swap diagnostic on that set. [Structured source review](../../results/target_binding_2026-11-22/visual_support_assistant_review.csv) · [TwinPast CPU findings](<2026-11-23 twinpast development review.md>)
