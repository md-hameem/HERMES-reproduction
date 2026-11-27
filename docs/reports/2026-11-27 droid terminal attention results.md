# DROID fixed-position terminal attention results

The annotation-guided mask passed its feasibility criterion on seven DROID development pairs. It recovered three robust pairs that both explicit controls missed and preserved the causal control's only robust pair. All four conditions used the same ten frames, frame order and token positions. A robust pair requires correct answers for both branches in both option orders.

| Condition | Robust pairs | Correct branch-order answers |
|---|---:|---:|
| Native attention | 1/7 | 14/28 |
| Explicit causal control | 1/7 | 14/28 |
| Annotated terminal access | 4/7 | 24/28 |
| Matched earlier access | 0/7 | 2/28 |

Relative to the causal control, terminal access corrected 11 answers and flipped one correct answer to wrong. The median change in gold-versus-alternative logit margin was +0.1875. The three robust recoveries were `droid_natural_8b01107c59`, `droid_natural_c00374385e` and `droid_natural_19bf4a67ee`.

The RTX 4080 run yielded 112 score vectors. All 28 native vectors exactly matched the archived state-swap originals. All 28 causal-control predictions matched native attention, with a maximum per-choice logit difference of 0.015625 against the 0.05 tolerance. The runner verified delivery of each explicit mask to the first and last decoder attention layers. The local audit checked all 114 frame receipts and recomputed the score totals and replay differences. [Audit](../../evidence/audits/droid_terminal_attention_2026-11-27_audit.json) · [Executed notebook](../../notebooks/executed/2026-11-27_hermes-droid-terminal-attention.ipynb)

## What to test next

The result supports restricting direct visual access when the relevant span is known. Terminal representations can still carry earlier information. The earlier-span control has the same key budget but shows a different state and occupies different key positions. The three-frame physically isolated terminal input reached 6/7 on these pairs in a separate run. Keep that as a capability reference and treat the seven reused scenes as development data. [Protocol](<2026-11-27 droid terminal attention protocol.md>) · [Terminal-context results](<2026-11-27 droid terminal context results.md>)

The next experiment should freeze a question-conditioned span selector that sees only sampled frames, timestamps and the question. Give it no option text, answer labels or annotated spans. On source-disjoint pairs, compare it with unrestricted decoding, a three-frame recency window and an annotation-guided reference. Keep the input sequence and three-frame mask budget fixed for the masked arms. Report robust paired accuracy, annotated span coverage, failures on questions about earlier states and added inference cost. The selector needs to beat recency on fresh data to support a method contribution.
