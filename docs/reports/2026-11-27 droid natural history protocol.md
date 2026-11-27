# DROID natural-history capability pilot

The controlled DROID pilot reached 4/6 robust pairs with a restored state cue. This follow-up asks whether giving more frames from the final state helps when both the correct state and its opposite appear in real chronological history. [Controlled pilot](<2026-11-27 droid state capability results.md>)

## Frozen footage

Candidate open and close episodes were matched using success metadata, scene ID, camera serial and collection session in DROID Raw 1.0.1. The intake returned 19 candidate pairs. Visual inspection accepted seven pairs from distinct recorded scenes at four sites. Ambiguous or occluded transitions and visibly repeated physical settings were excluded. The 14 history episodes and their recorded scenes are disjoint from the earlier controlled-pilot intake. The pairs share collection sites and robot hardware, so the four sites are the conservative grouping for interpretation. The state annotations were made by one reviewer before GPU scoring. [DROID raw release](https://github.com/droid-dataset/droid/blob/main/docs/the-droid-dataset.md)

Each branch comes from a separate, chronological original recording of the same object in its pair. Each shows a stable initial state, a real transition and a stable opposite terminal state. Both branches and both arms then receive the same four sampled frames from a separate DROID tabletop activity. The continuation is an edit between recordings. Its four sampled frames show no target drawer, cabinet or oven. No neutral filler or reversed motion is used.

## Frozen GPU comparison

| Arm | Earlier frames | Recent frames |
|---|---|---|
| Uniform | Six approximately time-uniform samples over the trimmed original recording, including its first and last frames | Four identical frames from the shared continuation |
| Annotated | Six chronological samples with three in the verified final-state interval, preserving the first and last inputs | The same four continuation frames |

The uniform history includes at least one final-state frame and an initial-state frame in every branch. It has one final-state frame on 12 branches and two on the remaining two. The annotation-selected arm therefore tests stronger evidence for the final state in a history with an opposing state. Both arms use the same pinned 7B direct scorer, vision frame count, question and answer orders.

Report 56 paired choice scores, 28 exact repeats of the uniform score and each robust pair’s outcome. A robust pair requires the correct terminal state on both branches in both option orders. The exploratory decision rule requires two or more robust recoveries and zero regressions. High uniform accuracy would establish capability in these constructed streams and leave less headroom for this particular frame intervention. Any gain remains exploratory on reviewed development footage.

The annotated intervals and input pack were frozen before model evaluation. Run `python scripts/freeze_droid_natural_history.py` in the local media environment to rebuild the manifest and PNG archive. To repopulate a fresh cache, run `python scripts/prepare_droid_state_control.py` and then `python scripts/prepare_droid_natural_history.py --per-site 5` first. Raw videos remain in the ignored asset cache. The data pack consists of 114 resized frames from DROID, released by Khazatsky et al. and the DROID Dataset Team under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [DROID project](https://droid-dataset.github.io/)

[Frozen manifest](../../data/twinpast/droid_natural_history_v1.json) · [GPU notebook](../../notebooks/autodl/hermes-droid-natural-history.ipynb)

The completed GPU test recovered one robust pair, below the two-recovery criterion. [GPU audit](<2026-11-27 droid natural history results.md>)
