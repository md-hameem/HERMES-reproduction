# DROID fixed-length earlier-state swap results

The AutoDL run completed all seven pairs. The audit verified 24 evidence payloads, 114 frame hashes, 56 choice scores and 28 exact original replays on an RTX 4080. Each branch retained the same ten positions, terminal frames, recent frames and prompt. Only the first two history images were swapped.

| Input | Robust pairs | Correct branch-order answers |
|---|---:|---:|
| Original ten frames | 1/7 | 14/28 |
| Two earlier frames replaced with the paired branch's frames | 1/7 | **18/28** |

The swap corrected six branch-order answers and regressed two. It recovered the wooden cabinet pair beside the range hood, `droid_natural_6d74030dc0`, and lost the wide white drawer pair, `droid_natural_87abc314ec`. The gold-versus-alternative logit margin rose on 23/28 branch-order observations, with a median paired change of +0.375. These observations share scenes and answer orders, so the margin count is descriptive.

The result shows that earlier image content can change scores and decisions while the frame count and terminal positions stay fixed. The donor frames come from a different episode in the same scene. The state change is coupled to visual differences such as robot pose, background details and the splice itself. The result therefore supports a content effect under this manipulation. A state-specific mechanism and a robust improvement remain to be tested.

## Research decision

Close the swap as a development diagnostic. The next method candidate should control which visual time span the answer readout uses while keeping the same sequence and token positions. First specify an annotation-guided feasibility comparison on the seven inspected scenes. Compare ordinary decoding with a readout that restricts answer-token attention to the verified terminal span. Include an equally sized control restriction and retain the existing terminal-only diagnostic as a capability reference. Audit which keys remain visible and which earlier information may already be represented in later tokens.

Only an improvement that preserves baseline successes would justify an annotation-blind version. Compare that version with simple recency and time-window baselines on fresh sources, and check the closest prior work before claiming originality.

The [fixed-position terminal-attention protocol](<2026-11-27 droid terminal attention protocol.md>) specifies the four conditions, replay checks and feasibility criterion.

[Audit JSON](../../evidence/audits/droid_state_swap_2026-11-27_audit.json) · [Evidence ZIP](../../evidence/archives/HERMES_droid_state_swap_evidence.zip) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-droid-state-swap.ipynb)
