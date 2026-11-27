# Controlled event-relative video pairs

The natural DROID screen found zero eligible pairs in 14 nominations. The closest pair already exposed the queried state difference in the center window. This protocol tests whether the 7B backbone can use evidence at an event when fixed windows omit that evidence. [DROID matched-screen results](<2026-11-28 droid matched event results.md>)

## Film and sample

Start with two pilot pairs. If both pass review, build a cohort of eight pairs across at least three physical setups. Use ordinary drawers, hinged boxes or cupboard doors. Each pair has two separate continuous takes with a locked camera, the same object, background and lighting, and one shared question such as “When the yellow card touched the marked spot, was the drawer open or closed?” Put the marked spot beside the target so that touching it is possible in either state.

Within each pair, put the card down at the same point in time. Show opposite states when the card contacts the spot, then return both takes to the same visible state. Keep the card, hand position and surrounding action as consistent as possible. Across pairs, vary which take is filmed first and whether the common state is open or closed. Photographing hands and objects is sufficient.

Sample twelve frames on a fixed one-second grid starting at the first frame. Number the slots from 0 to 11. Use a stopwatch or cue outside the camera view. In half the pairs, bring the card into contact at about three seconds and restore the common state before slot 5. In the other half, bring it into contact at about eight seconds and restore the common state before slot 9. Leave time before and after the sampled interval for a steady camera. Verify the event timing in the footage. In particular, check that the state and any distinctive setup or reset motion stay out of the fixed windows.

| Condition | Visual slots |
|---|---|
| First three | 0–2 |
| Center three | 5–7 |
| Last three | 9–11 |
| Annotated window | Three contiguous slots containing the visibly verified event |

The two planned event positions are five slots apart. A single contiguous three-frame window cannot cover both exact positions. The review must still check for clues outside the event. The twelve-frame grid is specific to this experiment and changes the input budget relative to the ten-frame DROID screen.

## Review before GPU

1. View the full takes and record the sampled frame indices, the anchor slots and the states in both branches. Verify that the event is unambiguous and both frames show the event and target state together.
2. Review the fixed windows without viewing the full takes or seeing the answer labels. Flag any visible giveaway, including the target state, hand motion, persistent prop placement, or a lighting change. Record all dispositions. Ask a second reviewer to repeat the fixed-window check if available.
3. Freeze eight accepted pairs across at least three setups, with both event positions represented. Report support coverage for the annotation-guided and every static three-frame window before evaluating the model. Preserve the source clips, grid timestamps, frame hashes and rejection reasons.

If the cohort passes, compare annotated access with the strongest static window and unrestricted decoding on the pinned 7B model. Use the same twelve frames and token positions for all three arms, with both option orders. Treat each pair as robust only when both takes answer correctly in both orders. Report pair gains and losses, actual mask parity and an isolated annotated-frame reference to check whether the model can read the state from the relevant images. Keep any accuracy claim scoped to this controlled cohort until it is tested on natural video.
