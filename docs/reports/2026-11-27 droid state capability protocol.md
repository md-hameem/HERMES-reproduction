# Source-separated DROID state capability pilot

The public TwinPast release pins 350 pair records and names upstream datasets, while its pair records do not identify original recording IDs. It therefore cannot establish source-level disjointness between two TwinPast subsets using pair IDs alone. This pilot uses DROID because its public raw recordings carry episode, scene and collection-site identifiers, and DROID is absent from the sources named in TwinPast Appendix B. This establishes a different documented acquisition source. Exposure during model pretraining remains unknown. [TwinPast Appendix B](https://arxiv.org/html/2610.41983#A2) · [DROID raw release](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md)

Fifteen episodes were screened using the two external camera views before any model call. Six episodes from six distinct sites and scenes supplied clear open/closed contrasts. Each pair uses one external camera from one episode. Its two branches show opposite states of the same object, with three open-first and three closed-first episodes. The manifest records the official object generation, SHA-256, camera, scene, inclusive source-frame intervals and the selected RGB frame. Questions and answers were checked by one assistant and await independent review.

## Frozen comparison

The 20-second virtual timeline holds one resized camera frame for 0.5 seconds. All other frames are neutral gray, including the identical recent suffix. Only sample times matter for this run. The notebook embeds the 12 resized RGB frames and verifies them locally.

| Condition | Frames given to readout | Purpose |
|---|---|---|
| Aligned uniform | Six historical and four recent frames, including the real cue at 9.0 s | Check whether an ordinary sample can read the available state |
| Off-grid uniform | Same ten sampling positions, with the cue between 9.25 and 9.75 s | Check that an entirely missed cue cannot yield different answers across the paired branches |
| Off-grid annotated | Same ten-frame budget, with the sample at 9.0 s replaced by the cue at 9.5 s | Check whether presenting the missed frame restores robust state discrimination |
| Cue only | The same source frame alone | Diagnose state recognition when the ten-frame input fails |

Aligned uniform and off-grid annotated deliver identical pixel tensors in the same positions and must reproduce the choice logits. Off-grid uniform gives identical inputs to both branches and must also reproduce the logits. Robustness requires both correct states in both option orders. The pilot capability threshold is four of six robust pairs for the annotation-selected condition. That threshold was chosen before model evaluation and serves as a practical decision rule for this small pilot.

The hidden-cue result is fixed by the construction. A positive annotated result establishes that this model can read a clearly visible state under controlled inputs. The neutral filler and open/closed restriction make this a controlled diagnostic. A positive result does not validate a new selector, transfer to natural video, or readiness for journal submission. A natural-stream follow-up would need flowing history, distractor events and independently reviewed state annotations.

Derived images are 640 × 360 resized frames from DROID by Khazatsky et al. and the DROID Dataset Team, released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Raw source videos remain in the ignored local asset directory. [DROID project](https://droid-dataset.github.io/) · [DROID paper](https://www.roboticsproceedings.org/rss20/p120.pdf)

[Frozen manifest](../../data/twinpast/droid_state_control_v1.json) · [AutoDL notebook](../../notebooks/autodl/hermes-droid-state-control.ipynb) · [Prior boundary GPU result](<2026-11-27 twinpast boundary GPU analysis.md>)

The completed GPU run met the controlled capability threshold at 4/6 robust pairs. [GPU audit](<2026-11-27 droid state capability results.md>)
