# DROID terminal-state context diagnostic

The earlier-state mask recovered no robust pairs. This diagnostic tests whether the pinned 7B model can read the exact terminal state from the same seven scenes and whether its answers change as context is added. [Masking audit](<2026-11-27 droid earlier state masking results.md>)

| Condition | Inputs |
|---|---|
| Terminal only, 3 frames | The three archived annotated terminal frames |
| Terminal plus recent, 7 frames | The same terminal frames followed by the four shared real recent frames |
| Full annotated, 10 frames | The original annotated history and shared recent frames |

Each condition reuses the same terminal-frame embeddings and original direct scorer. Both option orders are evaluated. The diagnostic reports robust pairs, individual answers, order flips and paired gains or losses when adding recent or earlier images. The 28 ten-frame scores must reproduce the archived predictions and token counts with at most 0.05 absolute logit drift. The preferred GPU is the RTX 4080 from the earlier-state masking run.

The comparison deliberately changes context length and some positions. If a pair fails with the terminal images alone, better retrieval would have limited benefit for those exact views under this readout. If a pair passes with terminal images and fails after context is added, the controlled observation is sensitivity to the added sequence. Length, positions and content may all contribute. All seven scenes remain development footage.

[GPU notebook](../../notebooks/autodl/hermes-droid-terminal-context.ipynb) · [Frozen frame manifest](../../data/twinpast/droid_natural_history_v1.json)

The completed experiment solved 6/7 robust pairs with terminal frames alone and 1/7 with full context. [GPU audit](<2026-11-27 droid terminal context results.md>)
