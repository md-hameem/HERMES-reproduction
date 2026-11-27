# DROID interior-evidence CPU screen

The endpoint selector tied a question-only first/last rule on every prediction. This screen tests whether there is enough visual evidence inside fresh clips to justify a new GPU diagnostic. [Previous run](<2026-11-28 droid span selection results.md>)

## Intake and review

Exclude prior DROID episode and site–scene IDs, including earlier previewed episodes. Use the DROID instruction index only to find candidate videos with multiple actions. The reviewer checks the videos against a ten-slot uniform frame grid. Record each candidate's disposition before any model inference. Keep a clip only if it shows a uniquely identifiable event, supports a question about the state at that event, and contains decisive interior evidence that can be identified visually. The question must identify the event without naming the answer or using the act of opening or closing the queried object as the event. Reviewers can inspect the original footage for the event order. The quantitative coverage check uses the fixed grid alone.

For accepted clips, record the source episode, scene, camera, frame grid, event timing, question, gold state, and all sampled slots that directly show the queried state at the anchor event. For the matched diagnostic, also record sampled slots at which the object clearly has the opposite state. Both labels must occur in the cohort. Hash the provenance files and compare under an equal three-frame budget.

## Coverage and gate

Use fixed first-three, center-three (slots 4–6), and last-three windows. Count direct support as covered if the window contains at least one verified support slot. Choose the annotation-guided contiguous three-frame window by maximum support count, breaking ties by earliest start. Report the best single fixed three-frame window across the entire observed cohort as an optimistic static comparator. Computing this ceiling on the reviewed cohort favors the static comparator. The comparison concerns coverage of sampled evidence rather than answer accuracy.

Schedule a small GPU feasibility run only if the reviewed screen contains at least six distinct scenes across at least three sites, at least two instances of each answer state, and at least two more covered clips than the best static window. These are engineering criteria for choosing the next experiment. A passing screen warrants a GPU comparison of annotation-guided access, matched baselines, and unrestricted decoding before any new selector is trained or tuned.
