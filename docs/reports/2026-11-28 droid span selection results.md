# DROID automatic span selection results

The endpoint-contrast selector passed the technical checks and missed the research advancement criterion. It improved on recency on the six fresh scene pairs and tied the question-only first/last rule on every prediction.

| Condition | First-view robust pairs | Last-view robust pairs | Total robust | Correct branch-order answers |
|---|---:|---:|---:|---:|
| Unrestricted native | 3/6 | 2/6 | 5/12 | 39/48 |
| Last-three recency | 0/6 | 3/6 | 3/12 | 30/48 |
| Question-only first/last | 2/6 | 3/6 | 5/12 | 39/48 |
| Endpoint-contrast selector | 2/6 | 3/6 | 5/12 | 39/48 |
| Annotated reference | 2/6 | 3/6 | 5/12 | 39/48 |

Robust correctness requires the correct answer for both episodes and both option orders. The selector corrected nine recency answers with zero regressions. Compared with unrestricted decoding, it corrected four answers and regressed four. The paired gain versus recency was 16.7 percentage points with a descriptive 95% scene-bootstrap interval of 0 to 33.3 points. The difference versus the question-only rule was zero.

On 23 of 24 episode-question decisions, the selector chose the same window as the question-only rule. On the remaining last-view washing-machine question, it chose slots 5–7 instead of 7–9. This removed one annotated support frame and preserved both predictions. The fixed first/last rule attained the annotated reference's maximum possible support overlap within the tested windows on every episode-question. These questions offered little opportunity for visual selection to improve coverage. [Protocol](<2026-11-27 droid span selection protocol.md>) · [Audit](../../evidence/audits/droid_span_selection_2026-11-28_audit.json)

The RTX 4080 run logged 240 choice scores across five reported conditions. The audit verified 120 input frames and all 48 native versus explicit causal-mask prediction checks. The maximum option-logit difference was 0.03125 against a tolerance of 0.05. The first and last decoder layers received both mask types. The audit recomputed window choices from logged contrast scores. The run archived those scores and omitted the feature vectors used to produce them. [Executed notebook](../../notebooks/executed/2026-11-28_hermes-droid-span-selection.ipynb) · [Evidence ZIP](../../evidence/archives/HERMES_droid_span_selection_evidence.zip)

## Research decision

Retire endpoint contrast as the current method candidate. Keep first/last routing as a baseline. The next experiment should start with a CPU visual-support screen on new source groups. Use event-relative questions whose answer appears inside a clip, such as the state immediately before a second interaction. Freeze the frame grid and compare annotated window coverage with fixed first/last and recency windows before using model outputs. Proceed to a small GPU feasibility run only if the screen produces cases with clear interior evidence and a measurable coverage advantage over the simple controls. Treat that run as a test of whether the backbone can use the interior evidence before implementing a new automatic selector.
