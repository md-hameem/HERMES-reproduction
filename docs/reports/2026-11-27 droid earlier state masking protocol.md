# DROID earlier-state masking protocol

The natural-history pilot recovered one robust pair after emphasizing terminal frames. This diagnostic asks whether reducing evidence of the earlier, opposing state changes the answer while terminal evidence remains fixed. The seven scenes are development footage. [Natural-history audit](<2026-11-27 droid natural history results.md>)

| Condition | First two historical images | Remaining inputs |
|---|---|---|
| Original | Archived annotated images | One other history frame, three terminal frames and four real recent frames |
| Target mask | One fixed gray rectangle over the earlier target region | The same eight images in the same positions |
| Non-target mask | An equal-area rectangle of the same shape and gray value elsewhere in each image | The same eight images in the same positions |

Each pair uses the same target and control rectangles across both branches and both edited frames. The two edited images are inside the reviewed initial-state interval on both branches. I inspected all 28 original and masked image triplets before running the model. Some shapes and scene cues remain visible outside the target rectangles, and the untouched intermediate frame can still show an opposing state. The manipulation attenuates evidence in the targeted region. The equal-area comparator controls the amount of masking and leaves differences in the content covered by each placement.

The notebook checks all 114 original frame receipts and all 56 edited RGB hashes. It shares the unmodified embeddings across arms, holds the prompts and option orders fixed and compares the original arm with 28 archived annotated scores. The replay gate requires each prediction and token count to match, with at most 0.05 absolute difference per logit. Any replay failure blocks the comparison summary.

A robust pair requires both branches correct in both option orders. The exploratory criterion is at least two pairs correct with the target mask that fail in both comparison arms and zero robust losses against either comparator. The masks were specified after seeing the natural-history result and frozen before this GPU test. This diagnostic tests one masking manipulation on reviewed footage. Any follow-up claim needs different source scenes and independent review.

The notebook uses the same `DROID_natural_history_frames.zip` as the natural-history experiment. Run `python scripts/prepare_droid_interference.py` with the frozen pack and archived run to regenerate the rectangle manifest and review sheets. [GPU notebook](../../notebooks/autodl/hermes-droid-state-interference.ipynb) · [Frozen mask manifest](../../data/twinpast/droid_state_interference_v1.json)

The completed run recovered zero robust pairs with the target mask and lost the pair both comparators solved. [GPU audit](<2026-11-27 droid earlier state masking results.md>)
