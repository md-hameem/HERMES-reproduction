# Historical evidence correction execution plan

**Decision on 22 November 2026:** The next experiment is defined. The active milestone is blinded visual-support review. Recent-first correction remains a candidate method, with originality and performance still open.

**23 November update:** The review packet and one blinded pass are complete. The first pass found only two questions with clear earlier support. The GPU pilot stays pending an independent review and enough matched cases. [First-pass findings](<../../docs/reports/2026-11-23 blinded visual support first pass.md>)

The local independent reviewer packet is `evidence/local/target_binding_2026-11-22/HERMES_independent_reviewer_packet.zip`. It includes the same sheets and an empty `independent_review.csv` with explicit option, supporting frame and temporal validity fields. Validate the returned CSV with `python scripts/prepare_independent_support_review.py --validate PATH_TO_CSV` before reconciling the two reviews.

## Evidence and current stage

| Completed work | What it establishes |
|---|---|
| Full HERMES/FIFO reproduction | 2,495 matched questions across 498 clips. The observed +0.36 percentage-point HERMES advantage had a confidence interval crossing zero. |
| Causal scorer, stage and query-cache controls | The corrected scorer did not improve correctness on the selected development cases. History visibility and cache construction can change some answers. Physical cache shortening preserved the 40 tested masked answers. The later 150-question confirmation favored recent-only in the point estimate. |
| Frozen target-binding run | Across 60 clips and 300 questions, recent-only scored 156/269, averaging scored 152/269, and the minimum rule scored 147/269 on the eligible subset. The primary interval included zero. |
| Failure analysis and Q1 audit | The minimum rule collapsed to the binding-only ranking. The saved answer table has 33 older-only opportunities, 14 recent-only cases, 142 with correct answers in both recent and older windows, and 80 with no correct window. Of the 33 older-only cases, 12 are prospective, six causal, and four text-rich. Correctness by itself does not locate the visual evidence. |

[Project status](../../docs/PROJECT_STATUS.md) · [Target-binding results](<../../docs/reports/2026-11-22 target binding analysis.md>) · [Q1 experiment audit](<../../docs/reports/2026-11-22 q1 audit of target binding.md>)

## Milestones and decisions

| Stage | Deliverable and decision |
|---|---|
| **1. Review on CPU, ready now** | Render the 147 archived windows across 32 questions and 28 clips from the existing video cache. Two reviewers independently label the supported option, the frame slots supporting it, whether it applies at query time, and uncertainty. Hide model outputs, gold, and selection strata until labels are fixed. Report agreement and adjudicate disagreements. |
| **2. Test the mechanism on one GPU, pending labels** | Use question-only and recent-four controls. Hold the same four recent frames fixed in both primary arms and add either two verified, time-valid older frames or two same-clip wrong-target or wrong-setting frames. Choose matched distractors by proximity in age among eligible frames and freeze identities before inference. Keep the token budget, prompts, option-order permutations, and decoder fixed. Save the correct option's margin and paired correctness for each arm. |
| **3. Test a deployable method, conditional** | If the diagnostic passes, prototype an automatic check of whether earlier evidence supports replacing the recent answer with a specific option. Compare recent-four with two recent plus two chronological older frames, an equal-budget relevance selector, and the closest implementable comparator. Keep candidate access and scan budgets matched among selectors. Report every answer, recoveries and regressions, end-to-end latency and peak memory. Lock the method on development clips. |
| **4. Confirm and prepare the paper, conditional** | Use source-disjoint confirmation and a second backbone after the policy is frozen. TwinPast is a candidate historical-evidence check. Keep pairs together in splits and bootstrap samples. Show a positive predeclared primary contrast with a paired confidence interval and report the cost trade-off before making a method-improvement claim. Establish the difference from PACE and MACD before drafting novelty claims. |

**Development gate.** Proceed to the GPU diagnostic if review yields at least eight usable support-versus-distractor cases across at least four distinct clip IDs. A usable case needs two supporting older frames, two comparably aged distractor frames, and a supported option that applies at query time. If the review yields fewer, record why and reassess the evidence pool before scheduling inference. Advance to an automatic method only if support wins outnumber distractor wins, support-only wins occur on at least three clip IDs, and leaving out any one clip keeps the pooled difference positive. These are resource-allocation criteria for a small purposive pilot. They do not establish statistical significance or an external accuracy gain. Record any originals later found to share a source and account for them in confirmation.

## Prepare the reviewer packet

The [self-contained CPU notebook](../../notebooks/autodl/hermes-blind-support-review.ipynb) is the quickest route on the original AutoDL instance. Upload it and choose **Restart Kernel → Run All**. Download the ZIP, extract it, and open `review.html`.

The following terminal option assumes a current repository checkout on the machine holding the original clips and a Python environment with Decord and Pillow.

```bash
python scripts/render_target_binding_review.py \
  --worksheet results/target_binding_2026-11-22/visual_support_review.csv \
  --evidence evidence/archives/HERMES_target_binding_evidence.zip \
  --videos /root/autodl-tmp/hermes_target_binding/videos \
  --output /root/autodl-tmp/HERMES_blind_support_review.zip
```

The ZIP contains images and a worksheet, with no ground truth or model predictions. Use the original source frames to resolve small text or apparent gaps. Keep each reviewer's worksheet separate until adjudication. In the existing notes field, record the supporting sampled indices and temporal-validity judgment.

## Skills applied

- **Q1 Ablation and Error Analysis** yielded the task breakdown, the blinded error categories, and the matched support/distractor contrast.
- **Q1 Method and Planning** yielded the pilot acceptance criteria and the split between a diagnostic with verified frames and a four-frame deployment test.
- **Q1 Research Orchestrator** yielded the active milestone and dependencies. The literature screen is complete for planning, novelty is provisional, the mechanism test is pending, and a manuscript improvement claim remains blocked.
- **Q1 Experiment Auditor** supplied the earlier checks on accuracy, compute, and reproducibility.

[Prior-art decision](<../../docs/reports/2026-11-22 historical evidence correction decision.md>) · [TwinPast](https://arxiv.org/abs/2610.41983v2) · [PACE](https://arxiv.org/abs/2608.26355) · [MACD](https://proceedings.mlr.press/v306/xiao26w.html)

## TwinPast fallback prepared

The pinned public annotation file provides evidence intervals for 289 of 350 pairs. The development manifest fixes six Historical State Recall pairs and six Attribute and Identity Recall pairs. Eligible pairs provide at least two shared 0.5 FPS grid times inside both branches' annotated evidence, at least eight seconds before the query. Selection uses a fixed hash order without answer values or model results. Keep these 12 pairs out of confirmation and verify source overlap separately.

A CPU smoke check decoded both videos for `twinpast_0208`. The selected earlier samples at 0 and 2 seconds differ and show distinct book states. At 26, 28, 30 and 32 seconds, the compared 192×108 RGB samples match exactly. This verifies one case and the frame access path. If the fallback is activated, check temporal validity and recent-context agreement for each pair before using it. A cross-branch history swap would be a separate diagnostic from the same-clip distractor test. [Frozen development manifest](../../data/twinpast/historical_correction_development.json) · [Pinned dataset annotations](https://huggingface.co/datasets/younght/TwinPast/blob/c1f1cbab68bfda2982565c0491f1a97474b73558/data.jsonl)

The full CPU screen subsequently verified 11 pairs with disjoint sampled windows. All 44 sampled recent frame pairs matched at decoded RGB resolution. Nine pairs gave clear visual answer contrasts. Those nine are ready for a small cross-branch development diagnostic. The StreamingBench review remains a single-reviewer assessment. [CPU findings and next test](<../../docs/reports/2026-11-23 twinpast development review.md>)

For the next development run, use the nine clear pairs and preserve the method-free recent-only control. Report the two uncertain pairs separately if they are run. The StreamingBench eight-case threshold governs the same-clip test. The TwinPast diagnostic changes history by swapping branches of an existing pair. Keep the entire frozen 12-pair set out of confirmation.
