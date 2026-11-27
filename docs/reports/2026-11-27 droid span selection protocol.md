# DROID automatic span selection protocol

The [terminal-attention test](<2026-11-27 droid terminal attention results.md>) recovered three robust pairs using annotated spans. This screen tests whether an automatic selector improves on recency on fresh DROID source IDs, and whether visual features add value beyond a first/last rule.

## Cohort and inputs

The acquisition pass excluded 53 previously inspected episode IDs and 31 site–scene IDs. It matched candidates within site, scene and collector and required a shared exterior-camera serial. The initial pair search used DROID instruction metadata. Visual review then selected one view per accepted scene and required opposite beginning and end states for the same object in both branches. Six of 15 acquired scenes passed this single-pass visual review, across four sites. The other nine have recorded exclusion reasons. Some metadata lookups returned HTTP errors. The cohort is a targeted development screen, and physical-environment disjointness beyond the source identifiers remains unverified. [DROID project](https://droid-dataset.github.io/) · [Review dispositions](../../data/twinpast/droid_span_review_v1.json) · [Frozen manifest](../../data/twinpast/droid_span_cohort_v1.json)

Sample ten uniformly spaced frame indices from the beginning through the end of each episode and keep them chronological. There is no donor continuation. Each episode receives a first-view and a last-view question in both option orders. The labels and review marks are used for evaluation and the annotated reference. They are withheld from the selector.

## Fixed-budget comparison

| Condition | Direct visual attention during answer scoring |
|---|---|
| Unrestricted | All ten slots via native decoding |
| Recency | Last three slots for both question types |
| Question-only endpoint | First three or last three according to the question |
| Endpoint contrast | Three contiguous slots in the queried half, chosen using visual features |
| Annotated reference | Three contiguous slots in the queried half with the greatest overlap with visible support |

The selector recognizes the fixed phrases *first view* and *last view*. It compares frame-local, mean-pooled visual embeddings by cosine similarity to the queried endpoint minus similarity to the opposite endpoint. It averages this contrast over each eligible contiguous window and breaks ties toward the queried endpoint. The selector does not localize the object in the question. Full input length and visual positions stay the same, and masked visual information may be carried indirectly by contextualized prefix tokens. This experiment measures answer accuracy at a matched direct-attention budget. Each arm still encodes ten frames.

Reproduce the native predictions under an explicit full causal mask for all 48 branch-order tasks before interpreting any result. Require matching prompt hashes, token counts and predictions and at most 0.05 absolute logit difference per option. The runner checks delivery of full and restricted masks to the first and last decoder attention layers. Identical windows share a forward within each branch, question and order.

## Readout and decision

Robust correctness requires both episodes and both option orders to be correct for a question type. Report six first-view and six last-view endpoints, coverage counts, and the paired difference between the selector and each fixed-budget comparator. The descriptive bootstrap resamples scene pairs with both question types together. Six scenes support a prototype decision. A positive result against recency alone can come from the first/last wording, so require a positive observed difference against both comparators before spending on independent confirmation. Inspect all answer regressions and selector timing. Confirm any improvement on a larger source-grouped cohort before using it as a paper claim.

## AutoDL

Open the [notebook](../../notebooks/autodl/hermes-droid-span-selection.ipynb) alongside `DROID_span_cohort_frames.zip` on the existing GPU container. The local frame package is under `evidence/local/droid-span-cohort/`. Run all cells and save the executed notebook and `HERMES_droid_span_selection_evidence.zip`; the final cell can also retrieve runner failure evidence.

Local CPU validation checked all 120 frames, the notebook schema, mask dispatch and 48 parity comparisons with a tiny Qwen2 model. The pinned 7B GPU run subsequently passed parity and missed the research advancement criterion. [Results](<2026-11-28 droid span selection results.md>)
