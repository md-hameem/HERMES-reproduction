# DROID fixed-position terminal attention

The earlier terminal-context test scored six of seven robust pairs with three annotated terminal frames and one of seven with all ten frames. The next test asks whether limiting direct visual access during answer scoring can recover pairs while preserving the original ten input frames and their token positions.

## Comparison

Use the seven inspected DROID pairs, both branches and both option orders. Score the same inputs with the pinned LLaVA-OneVision Qwen2 7B model. Only the visual keys available to the question and answer scaffold vary across the explicit-mask runs.

| Condition | Directly visible visual frame slots, zero-indexed |
|---|---|
| Native attention | All ten |
| Explicit causal control | All ten |
| Terminal access | 3, 4, 5 |
| Earlier access | 0, 1, 2 |

All conditions preserve direct access to causal system and text keys. System and visual query rows retain their original causal attention. Terminal visual tokens can therefore still carry information about earlier frames. The comparison measures the effect of restricting direct attention during scoring.

## Replay and decision rules

The runner checks 28 native score vectors against the archived original state-swap scores and 28 explicit causal-control vectors against the current native scores. Each check requires matching predictions, prompt hashes and token counts, with an absolute difference of at most 0.05 per choice logit. The runner verifies that the explicit mask reached the first and last decoder attention layers for each mask condition.

A robust pair passes both branches in both option orders. Advance to an annotation-blind method only if terminal access recovers at least two pairs missed by both explicit-mask comparators and preserves every robust pair solved by the causal control. Report all four conditions, individual recoveries and regressions, and the paired median change in gold-versus-alternative margin against the causal control. The seven pairs are exploratory development data. An annotation-blind follow-up needs fresh sources and a simple recency baseline.

## AutoDL run

Open the [terminal-attention notebook](../../notebooks/autodl/hermes-droid-terminal-attention.ipynb) beside the existing `DROID_natural_history_frames.zip` and select **Restart Kernel → Run All** on the same GPU container used for the state-swap test. The notebook verifies the frame ZIP and can reuse the pinned environment and model. Save the executed notebook and download `HERMES_droid_terminal_attention_evidence.zip`. Run the last cell manually to collect the ZIP if the runner fails.

Local validation checked the notebook schema, frame receipts and mask and scoring flow on tiny CPU Qwen2 models with SDPA and eager attention. The subsequent 7B GPU run passed both replay gates. [Results](<2026-11-27 droid terminal attention results.md>)
