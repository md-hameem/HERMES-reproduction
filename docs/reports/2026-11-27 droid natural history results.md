# DROID natural-history GPU audit

The run completed all seven pairs across four DROID collection sites. The audit verified 21 evidence payloads and 114 input-frame hashes and recomputed 56 choice scores. All 28 repeats of the uniform readout matched exactly. Both arms used six chronological history frames, four identical recent frames and both option orders.

| History input | Robust pairs | Correct branch-order answers | Predictions of open |
|---|---:|---:|---:|
| Uniform, with one or two final-state frames | 0/7 | 12/28 | 24/28 |
| Annotated, with three final-state frames | **1/7** | **14/28** | 22/28 |

The two corrected answers came from the closed branch of the same wide-white-drawer pair, `droid_natural_87abc314ec`. It changed from open to closed in both option orders. No other predicted answer changed. Both arms had two branches whose predictions differed between the two option orders. The annotated arm retained an initial-state frame and increased emphasis on the final state. It recovered one robust pair with no regression, below the two-recovery threshold fixed before GPU evaluation.

The previous neutral-filler DROID pilot solved 4/6 with a restored cue. The scene set, object views and intervening content differ between that experiment and this one, so their gap cannot isolate the effect of competing states. Here, uniform already received a final-state frame on every branch, and annotated received three. The remaining failures can arise even when the final state is represented in the input. The predominance of “Open” across both scoring orders also makes individual-answer accuracy an incomplete diagnostic.

## Research decision

Hold further automatic selector development. Run one fixed-input attribution test on these development scenes before renting time for a larger benchmark. Keep the three known terminal-state frames, their positions, the ten-frame budget and the four real recent frames identical. In the earlier frames, compare the original visible competing state with a condition that masks the competing object region. Add an equal-area mask on a nearby non-target region to control for the visual perturbation. Review the masks against the frames before scoring and use both answer orders. Any gained pairs support interference from earlier target evidence under that manipulation. If the matched manipulation does not recover pairs, compare cue-only recognition on the exact same natural clips before considering a different backbone or readout. Keep all reviewed scenes out of confirmation.

[Audit JSON](../../evidence/audits/droid_natural_history_2026-11-27_audit.json) · [Evidence ZIP](../../evidence/archives/HERMES_droid_natural_history_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-droid-natural-history.ipynb) · [Frozen protocol](<2026-11-27 droid natural history protocol.md>)

The masks and same-input GPU notebook are frozen in the [earlier-state masking protocol](<2026-11-27 droid earlier state masking protocol.md>).
