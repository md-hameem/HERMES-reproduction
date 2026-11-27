# DROID matched event-relative screen results

The screen found **zero eligible natural pairs**. The first nomination pass produced 11 pairs and the broader pass produced 14, with overlap between passes. The raw bucket supplied no complete pairs. Using the published offset index, the fallback recovered all 19 original RLDS records needed for the broader set. Both source paths in each decoded record matched the index. The screen excluded 97 previously inspected episode IDs and 60 prior site–scene IDs. [DROID raw and RLDS documentation](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md) · [Offset index](https://huggingface.co/datasets/EpicPinkPenguin/droid_dataset_segmentation_mask) · [Acquisition receipts](../../data/twinpast/droid_matched_event_intake_v1.json)

| Main exclusion reason | Candidate pairs |
|---|---:|
| Different target | 6 |
| Different event | 3 |
| Visible state difference in a fixed window | 4 |
| Insufficient visibility at the event | 1 |

The closest pair shows the same red microwave and a pale green bowl, with the door closed at the end of both episodes. One episode removes the bowl from the microwave and the other initially picks it up from the table. Slot 4 already reveals an open door in one branch and a closed door in the other. The nightstand example has a similar failure, with the tray extended in one center window and already retracted in the other. These are visually useful near misses since the fixed baseline has a plausible shortcut. [Review dispositions](../../data/twinpast/droid_matched_event_review_v1.json)

The fallback provided episode-level source identity. Physical scene IDs and camera serials remained unverified for these fallback pairs. The review covered the ten-frame sample grids of all 14 pairs in a single pass. The nominations reuse episodes and come from two DROID sites, so the failure rate applies only to this shortlist. Recompute the counts with `python scripts/check_droid_matched_event_screen.py`. [Screen protocol](<2026-11-28 droid matched event protocol.md>) · [CPU audit](../../evidence/audits/droid_matched_event_cpu_2026-11-28.json)

## Research decision

Close this natural-pair screen and defer the GPU comparison. The next feasibility experiment uses filmed pairs with the same setup and neutral event. Vary the target state at that event while matching the first, center and last evidence. The follow-up protocol uses twelve sampled frames so that two event timings can sit outside the named baseline windows. A controlled cohort tests whether the model can use interior evidence. Broader claims would still require a separate evaluation on natural footage. [Controlled filming and review protocol](<2026-11-28 controlled event pair protocol.md>)
