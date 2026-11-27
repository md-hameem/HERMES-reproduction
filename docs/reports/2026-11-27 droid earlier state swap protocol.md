# DROID fixed-length earlier-state swap

The terminal-context experiment lost four robust pairs when three earlier frames were prepended. This comparison holds length and terminal-frame positions fixed and changes the content in the first two history positions. [Terminal-context audit](<2026-11-27 droid terminal context results.md>)

For both directions of each pair, replace history positions one and two with the other branch's first two images. Those frames are inside its annotated initial-state interval and show the recipient branch's terminal state. The third early image, three terminal images and four shared recent images stay unchanged. Both arms use ten images, the same embeddings for shared frames, the same direct scorer and both option orders.

The notebook verifies the 114-frame pack and checks all 28 original scores against the terminal-context archive. Predictions, prompt hashes and token counts must match. The maximum absolute logit tolerance is 0.05. It reports 56 choice scores, robust pair recoveries and regressions, and how often the swap increases the gold logit minus the alternative logit.

The 14 substitutions were reviewed before GPU scoring. The cross-episode splice can also change the robot pose, local objects and lighting. The manipulation tests whether earlier input content affects the readout at fixed length. A result specific to state conflict would require a matched splice control and independent scenes. The seven pairs remain development footage.

The notebook uses the existing `DROID_natural_history_frames.zip`. Run `python scripts/prepare_droid_state_swap.py` with that frame pack and the terminal-context evidence archive to regenerate the manifest and review sheets. [Notebook](../../notebooks/autodl/hermes-droid-state-swap.ipynb) · [Frozen swap plan](../../data/twinpast/droid_state_swap_v1.json)

The completed run improved branch-order accuracy and exchanged one robust pair for another. [GPU audit](<2026-11-27 droid earlier state swap results.md>)
