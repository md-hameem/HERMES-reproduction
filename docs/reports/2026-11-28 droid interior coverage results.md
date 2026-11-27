# DROID event-relative evidence coverage

The CPU screen passed. Seven of 14 acquired clips qualified after visual review. The seven clips came from five DROID collection sites and contained four open-state and three closed-state answers. The intake excluded 83 prior episode IDs and 46 prior site–scene IDs. The retained videos have distinct hashes from the available earlier camera files. Physical-scene overlap across different dataset IDs remains unverified. [Review protocol](<2026-11-28 droid interior coverage protocol.md>) · [Review notes](../../data/twinpast/droid_interior_review_v1.json) · [Acquisition receipts](../../data/twinpast/droid_interior_intake_v1.json)

| Three-frame window | Clips with direct event-and-state support |
|---|---:|
| First three | 0/7 |
| Center three, slots 4–6 | 5/7 |
| Last three | 1/7 |
| Best fixed window across eight possible start slots | 5/7 |
| Annotation-guided window | 7/7 |

All seven accepted clips have a reviewed interior slot that shows the queried event and object state together. Four also have the opposite state visible at both endpoints. Several clips show the correct state away from the event, so a model may still infer answers from another window. The screen used one review pass and selected clips for visible support. Recompute the table with `python scripts/check_droid_interior_coverage.py`. [CPU audit](../../evidence/audits/droid_interior_cpu_2026-11-28.json)

## Small GPU comparison

The [AutoDL notebook](../../notebooks/autodl/hermes-droid-interior.ipynb) scores the seven clips in both option orders on the pinned 7B model. It compares annotated access with center three, last three and native decoding. The center window tied for best fixed coverage. All methods receive the same ten-frame visual sequence. Before interpreting accuracy, require matching native and explicit causal-mask predictions for all 14 scores and at most 0.05 absolute logit difference per answer option.

Use correctness in both answer orders as the primary descriptive outcome. Look for more clips correct with annotated attention than with center-three attention and inspect each gain and loss against unrestricted decoding. Both orders belong to the same clip. Group results by clip and separately inspect the four cases where both endpoints show the opposite state. The seven clips support a feasibility decision. Any method claim needs evaluation on fresh source groups after freezing an automatic policy.

Use the existing AutoDL GPU container. Place the notebook beside `DROID_interior_frames.zip` from `evidence/local/droid-interior/` and run all cells. Save the executed notebook and collect `HERMES_droid_interior_evidence.zip`.

Local validation checked the notebook schema, all 70 frame receipts, 56 scoring paths and 14 replay checks. The subsequent 7B run passed parity. The annotated window scored 5/7 in both option orders, compared with 6/7 for center-three. [GPU results](<2026-11-28 droid interior GPU results.md>)
