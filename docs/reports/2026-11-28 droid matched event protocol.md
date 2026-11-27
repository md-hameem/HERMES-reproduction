# DROID matched event-relative screen

The previous annotated window raised coverage from 5/7 to 7/7 but answered fewer clips correctly than center-three. This screen asks whether fresh natural clips can support the same event-relative question with opposite answers and comparable fixed-window evidence. [Previous GPU results](<2026-11-28 droid interior GPU results.md>)

Nominate public DROID episodes recorded by the same collector with overlapping action and object descriptions. First look for opposite ordering of the target-state action and the event. Then allow both opening and closing annotations to nominate opposite event states. Language instructions are candidate hints. Ground truth and eligibility require visual review. [DROID dataset documentation](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md)

Skip all previously inspected DROID episode and site–scene IDs. For each nomination, require the same physical target, a clearly recognizable event that can anchor the same question, opposite visible answers at that event, and usable views. Compare first-three, center-three and last-three slots from a ten-frame uniform grid. A pair can advance only if center and last evidence offer no obvious cue to the answer difference when viewed without the event. Record mismatches and unresolved scene provenance as separate reasons.

The public raw release is the first source. For unresolved nominations, a third-party offset index can locate original public RLDS records. Verify record identity against both source paths before decoding frames. The original RLDS episode metadata provides recording and trajectory paths. A recovered record can be screened for visual eligibility even when its site–scene ID and camera serial could not be verified. Keep those provenance limits when choosing an evaluation cohort. [RLDS offset index](https://huggingface.co/datasets/EpicPinkPenguin/droid_dataset_segmentation_mask)

Use at least four eligible pairs from at least three verified new scenes before preparing a GPU comparison. The screen has no model inference. Keep the report and review dispositions if the source or visual checks block advancement.
