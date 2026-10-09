# HERMES Reproduction Project — Research Log

> **Streaming Video QA with Hierarchical KV-Cache Memory**

| | |
|---|---|
| **Project Status** | Ongoing reproduction, validation, and extension study |
| **Primary Environment** | Kaggle · Tesla T4 GPU |
| **Model** | LLaVA-OneVision-Qwen2-0.5B |
| **Framework** | HERMES VideoQA inference pipeline |
| **Current Benchmark Scale** | StreamingBench subset: 50 videos / 250 questions |
| **Last Updated** | 10 October 2026 |
| **Supporting Documents** | [Technical Research Report](docs/HERMES_Comprehensive_Technical_Research_Report.md) · [Reproduction Audit](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md) |

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Research Scope and Evaluation Principles](#2-research-scope-and-evaluation-principles)
3. [Timeline and Development Log](#3-timeline-and-development-log)
4. [Reproduction Environment](#4-reproduction-environment)
5. [KV-Cache Compression Validation](#5-kv-cache-compression-validation)
6. [StreamingBench Preparation](#6-streamingbench-preparation)
7. [10-Video KV-Budget Ablation](#7-10-video-kv-budget-ablation)
8. [50-Video Scaled Benchmark](#8-50-video-scaled-benchmark)
9. [Task-Level Analysis](#9-task-level-analysis)
10. [Error Analysis](#10-error-analysis)
11. [Efficiency Analysis](#11-efficiency-analysis)
12. [Statistical Interpretation](#12-statistical-interpretation)
13. [Reproducibility and Backup](#13-reproducibility-and-backup)
14. [Current Research Status](#14-current-research-status)
15. [Remaining Experiments and Future Work](#15-remaining-experiments-and-future-work)
16. [Planned Final Report Structure](#16-planned-final-report-structure)
17. [Research Notes and Interpretation Constraints](#17-research-notes-and-interpretation-constraints)
18. [Literature Context and Related Work](#18-literature-context-and-related-work)
19. [HERMES Deep Mechanistic Analysis](#19-hermes-deep-mechanistic-analysis)
20. [Cross-Paper Technical Comparison](#20-cross-paper-technical-comparison)
21. [Evolution of the Research Field](#21-evolution-of-the-research-field)
22. [Reproduction Audit and Evidence Assessment](#22-reproduction-audit-and-evidence-assessment)
23. [Risks, Inconsistencies, and Open Issues](#23-risks-inconsistencies-and-open-issues)
24. [Phased Next-Stage Research Plan](#24-phased-next-stage-research-plan)
25. [Experimental Design Specifications](#25-experimental-design-specifications)
26. [Candidate Research Extensions](#26-candidate-research-extensions)
27. [Deliverables and Completion Criteria](#27-deliverables-and-completion-criteria)

---

# 1. Project Objective

The goal of this project is to reproduce, validate, and analyze the HERMES approach for efficient streaming video understanding using hierarchical KV-cache memory.

The project began as an engineering reproduction exercise and has progressed into an experimental study of the relationship between:

- KV-cache budget,
- VideoQA accuracy,
- GPU memory consumption,
- time-to-first-token (TTFT),
- task-specific performance, and
- qualitative failure modes.

The current work focuses on LLaVA-OneVision-Qwen2-0.5B under the HERMES streaming VideoQA pipeline.

## Research Questions

1. Can the HERMES KV-cache compression mechanism be reproduced reliably on publicly available GPU infrastructure?
2. How does KV-cache budget influence StreamingBench accuracy?
3. How much GPU memory can HERMES save relative to a very large KV budget?
4. Does reducing the KV budget affect latency?
5. Which categories of video understanding are most vulnerable under constrained memory?
6. Are performance differences between cache budgets statistically meaningful?
7. How representative are conclusions drawn from a small benchmark subset?

## Extended Analysis Scope

The study currently includes:

- reproduction fidelity checks,
- fixed-budget KV-cache experiments,
- accuracy-memory trade-off analysis,
- latency analysis,
- confidence intervals,
- paired prediction analysis,
- task-level accuracy,
- qualitative error taxonomy,
- dataset-scale sensitivity, and
- reproducibility packaging.

---

# 2. Research Scope and Evaluation Principles

Several distinctions are important for interpreting the experiments correctly.

## 2.1 HERMES configuration vs. original model baseline

A KV budget of `50000` is used as a **large-budget / near-uncompressed HERMES control**.

It should **not** be described as the original LLaVA-OneVision baseline because it still executes through the HERMES inference pipeline.

On the 10-video subset, KV=50000 did not trigger cache compression. On the larger 50-video subset, however, compression was triggered 79 times and the maximum answering cache reached approximately `50013` tokens. Therefore, KV=50000 is best interpreted as a **near-uncompressed HERMES control**, not as a guaranteed no-compression baseline.

## 2.2 Small-subset results are exploratory

The first StreamingBench experiment contains only:

- 10 videos,
- 50 questions.

This subset is useful for validating the pipeline and exploring trends, but it is too small to support strong general conclusions.

A 50-video / 250-question evaluation was therefore performed as the next scale-up step.

## 2.3 Accuracy differences must be interpreted with uncertainty

Accuracy is reported together with 95% Wilson confidence intervals.

Because all configurations are evaluated on the same questions, paired comparisons are more informative than comparing only aggregate percentages.

---

# 3. Timeline and Development Log

## Day 1 — Environment Setup and Initial Reproduction

The initial objective was to reproduce the HERMES inference pipeline on Kaggle.

### Platform

- Kaggle Notebook
- CUDA-enabled Tesla T4 GPU
- Python 3.12
- PyTorch with CUDA support

The LLaVA-OneVision-Qwen2-0.5B model was successfully loaded and executed on GPU.

Initial verification:

```text
Model loaded successfully
Main device: cuda:0
```

This established that the core model and HERMES repository could run on the available hardware.

---

## Day 2 — Dependency and Compatibility Debugging

The original HERMES implementation depends on a specific Transformers development revision. Running against newer Transformers releases introduced several compatibility problems.

### Problem 1 — LLaVA-OneVision internal API differences

Some code paths expected model internals such as:

```python
base_model.language_model.config
```

while newer library releases exposed a different internal structure.

This initially required compatibility investigation.

### Problem 2 — Qwen2 rotary-embedding API mismatch

HERMES expects a Qwen2 RoPE interface compatible with:

```python
rotary_emb(x, seq_len=...)
```

while newer Transformers releases changed the rotary-embedding API.

Temporary compatibility patches were explored during debugging.

### Problem 3 — CUDA device mismatch

When Kaggle exposed two Tesla T4 GPUs, automatic model placement distributed components across devices. HERMES performs manual KV-cache and rotary-position operations, leading to tensor device mismatches.

The stable experimental configuration therefore constrains HERMES to one GPU:

```bash
CUDA_VISIBLE_DEVICES=0
```

This also makes memory and latency measurements easier to interpret.

### Final compatibility strategy

For reproducibility, the project returned to the HERMES-compatible Transformers Git revision rather than keeping ad-hoc inference patches.

The final environment records:

```text
Transformers: 4.45.0.dev0
Tokenizers:   0.19.1
```

The Transformers revision used by HERMES is:

```text
66bc4def9505fa7c7fe4aa7a248c34a026bb552b
```

---

## Day 3 — Custom Video Smoke Test

A custom sandwich-preparation video was used as a controlled smoke test.

The model was evaluated on timestamped questions covering:

- scene understanding,
- action recognition,
- event understanding.

Example predictions included:

| Question | Example Prediction |
|---|---|
| What is happening in the video right now? | The video is set in a kitchen, where the sandwich is being prepared. |
| What is the person doing at this point? | The person is preparing a sandwich, specifically making the filling for it. |

The experiment confirmed that:

- streaming frame ingestion worked,
- conversation history was preserved,
- HERMES question generation executed,
- cache compression could be triggered,
- predictions were produced successfully.

---

## Day 4 — Reproduction Fidelity Check and StreamingBench Pilot

Before benchmark evaluation, the official HERMES position re-indexing / RoPE correction path was restored and tested.

The faithful smoke test completed successfully without the temporary compatibility shortcut that had disabled key re-rotation during earlier debugging.

A 10-video StreamingBench subset was then constructed and evaluated.

Initial benchmark configurations:

- KV=4000,
- KV=6000,
- KV=50000 large-budget control.

The first benchmark established an end-to-end evaluation pipeline with saved CSV predictions, automatic multiple-choice accuracy, task labels, logs, and cache statistics.

---

## Day 5 — Full 10-Video Budget Ablation

The pilot was expanded into a six-budget ablation:

```text
KV = 500
KV = 1000
KV = 2000
KV = 4000
KV = 6000
KV = 50000
```

The evaluation included:

- accuracy,
- Wilson confidence intervals,
- maximum reported GPU memory,
- median TTFT,
- P95 TTFT,
- maximum answering-cache length,
- compression-event counts.

Paired prediction comparisons were also added.

---

## Day 6 — Scale-Up to 50 Videos

The StreamingBench shard was expanded from 10 to 50 videos.

The resulting benchmark contained:

```text
50 videos
250 questions
5 questions per video
```

All video paths were validated before inference:

```text
Missing video files: 0
All 50 video paths are valid.
```

Three primary memory settings were evaluated:

- KV=4000,
- KV=6000,
- KV=50000 near-uncompressed HERMES control.

This became the current primary experimental result.

---

## Day 7 — Reproducibility Packaging

The complete research state was backed up, including:

- experiment CSVs,
- generated figures,
- raw experiment logs,
- annotations,
- HERMES source snapshot,
- environment metadata,
- Git revision,
- Git working-tree status,
- source diff,
- serialized notebook variables,
- SHA-256 manifest,
- full Kaggle working-directory archive.

The final ZIP integrity test passed successfully.

---

# 4. Reproduction Environment

The final saved environment metadata is:

| Component | Version / Value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| Transformers | 4.45.0.dev0 |
| Tokenizers | 0.19.1 |
| CUDA available | Yes |
| Visible GPUs | 2 × Tesla T4 |
| Experimental GPU usage | Single T4 via `CUDA_VISIBLE_DEVICES=0` |
| HERMES Git commit | `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789` |
| HERMES branch | `main` |

The repository state was recorded together with:

- `git_commit.txt`
- `git_branch.txt`
- `git_status.txt`
- `source_changes.diff`
- `pip_freeze.txt`
- `nvidia_smi.txt`
- `environment.json`

This is important because the reproduction required a library revision compatible with the HERMES implementation.

---

# 5. KV-Cache Compression Validation

Before benchmarking, cache compression was validated using the custom video.

## KV = 500

| Metric | Value |
|---|---:|
| Compression triggered | Yes |
| Final compressed cache length | 513 |
| Static prefix (`n_init`) | 13 |
| Approx. GPU memory | 4.43–4.62 GB |

The final length corresponds to approximately:

```text
500 budgeted tokens + 13 initial/static tokens = 513
```

## KV = 4000

| Metric | Value |
|---|---:|
| Compression triggered | Yes |
| Final compressed cache length | 4013 |
| Approx. GPU memory | 4.52 GB |

## KV = 50000

On the short custom test, the sequence did not exceed the 50K budget.

| Metric | Value |
|---|---:|
| Compression triggered | No |
| Approx. GPU memory | 4.58 GB |

This confirmed that cache compression was activated only when required by the configured memory budget.

<p align="center">
  <img src="img/KV Cache After HERMES Compression.png" alt="Final KV-cache length after HERMES compression" width="520" />
</p>

<p align="center"><b>Figure 1.</b> Final KV-cache length after HERMES compression in the initial custom-video experiment.</p>

<p align="center">
  <img src="img/GPU Memory Usage vs KV Cache Budget.png" alt="GPU memory usage across KV-cache budgets" width="520" />
</p>

<p align="center"><b>Figure 2.</b> GPU memory usage across the initial KV-cache settings.</p>

<p align="center">
  <img src="img/Time To First Token vs KV Budget.png" alt="TTFT across KV budgets" width="520" />
</p>

<p align="center"><b>Figure 3.</b> Initial TTFT measurements across KV budgets.</p>

---

# 6. StreamingBench Preparation

The official StreamingBench real-time annotation file used by the repository is:

```text
data/streamingbench/streamingbench_realtime.json
```

The annotation file contains approximately 498 video records.

A downloadable shard covering 50 videos was prepared locally. The archive structure used folders such as:

```text
sample_217/video.mp4
```

rather than annotation filenames of the form:

```text
sample_217_real.mp4
```

Therefore, video paths were matched by sample identifier.

Apple metadata files such as:

```text
._video.mp4
```

were explicitly excluded.

## 6.1 Initial 10-video subset

The pilot subset contained:

- 10 videos,
- 50 questions,
- 5 questions per video.

This stage validated the evaluation pipeline before larger runs.

<p align="center">
  <img src="img/StreamingBench 10-Video Subset Durations.png" alt="Duration of each video in the 10-video subset" width="650" />
</p>

<p align="center"><b>Figure 4.</b> Duration distribution for the original 10-video pilot subset.</p>

<p align="center">
  <img src="img/StreamingBench Subset Task Distribution.png" alt="Task distribution in the 10-video subset" width="650" />
</p>

<p align="center"><b>Figure 5.</b> Task distribution in the initial 10-video subset.</p>

## 6.2 Scaled 50-video subset

The scaled benchmark contains:

```text
50 videos
250 questions
5 questions per video
```

Video-duration statistics:

| Statistic | Duration (s) |
|---|---:|
| Minimum | 200.17 |
| Mean | 543.54 |
| Median | 614.09 |
| Maximum | 847.61 |

The videos therefore span approximately 3.3 to 14.1 minutes.

### Question distribution

| Task | Questions |
|---|---:|
| Causal Reasoning | 101 |
| Prospective Reasoning | 100 |
| Event Understanding | 16 |
| Action Recognition | 9 |
| Attribute Recognition | 9 |
| Object Recognition | 5 |
| Counting | 5 |
| Spatial Understanding | 3 |
| Text-Rich Understanding | 2 |
| **Total** | **250** |

The scaled subset is strongly dominated by Causal Reasoning and Prospective Reasoning. Consequently, task-specific conclusions for categories with very small sample counts must be treated cautiously.

<p align="center">
  <img src="img/Duration Distribution of the 50-Video StreamingBench Subset.png" alt="Duration distribution of the 50-video StreamingBench evaluation subset" width="650" />
</p>

<p align="center"><b>Figure 6.</b> Duration distribution of the 50-video StreamingBench evaluation subset.</p>

<p align="center">
  <img src="img/Task Distribution of the 50-Video Evaluation Set.png" alt="Task distribution of the 50-video StreamingBench evaluation subset" width="700" />
</p>

<p align="center"><b>Figure 7.</b> Task distribution of the 50-video evaluation subset.</p>

---

# 7. 10-Video KV-Budget Ablation

A six-budget ablation was performed on the same 10 videos and 50 questions.

## 7.1 Experimental setup

| Parameter | Value |
|---|---|
| Model | LLaVA-OneVision-Qwen2-0.5B |
| Benchmark | StreamingBench |
| Videos | 10 |
| Questions | 50 |
| Sampling | 0.5 FPS |
| Streaming | Enabled |
| GPU | Single Tesla T4 |
| KV budgets | 500, 1000, 2000, 4000, 6000, 50000 |

## 7.2 Main results

| KV Budget | Accuracy | 95% Wilson CI | Max GPU Memory | Median TTFT | P95 TTFT | Max Cache Length | Compression Events |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 500 | **68.0%** | 54.19–79.24 | 4.624 GB | 0.060 s | 0.064 s | 513 | 89 |
| 1000 | **68.0%** | 54.19–79.24 | 4.641 GB | 0.059 s | 0.065 s | 1013 | 87 |
| 2000 | 60.0% | 46.18–72.39 | 4.665 GB | 0.060 s | 0.065 s | 2013 | 79 |
| 4000 | 56.0% | 42.31–68.84 | 4.709 GB | 0.063 s | 0.075 s | 4013 | 76 |
| 6000 | 56.0% | 42.31–68.84 | 4.779 GB | 0.063 s | 0.077 s | 6013 | 67 |
| 50000 | 52.0% | 38.51–65.20 | 6.650 GB | 0.074 s | 0.120 s | 30981 | 0 |

### Observation

Accuracy is **not monotonic with KV budget** on this small subset.

The most aggressive settings, KV=500 and KV=1000, achieved the highest observed accuracy (68%), while the large-budget control achieved 52%.

This should not be interpreted as proof that stronger compression is universally better. The sample contains only 50 questions, the confidence intervals are wide, and the result may reflect dataset composition or the filtering of redundant context.

The result is nevertheless important because it shows that, on this subset, aggressive cache reduction did not automatically destroy VideoQA performance.

<p align="center">
  <img src="img/StreamingBench Accuracy vs KV-Cache Budget.png" alt="10-video StreamingBench accuracy versus KV-cache budget, with 95 percent Wilson confidence intervals" width="650" />
</p>

<p align="center"><b>Figure 8.</b> Accuracy versus KV-cache budget with 95% Wilson confidence intervals.</p>

<p align="center">
  <img src="img/GPU Memory vs KV-Cache Budget.png" alt="Maximum reported GPU memory versus KV-cache budget on the 10-video StreamingBench ablation" width="650" />
</p>

<p align="center"><b>Figure 9.</b> Maximum reported GPU memory versus KV-cache budget.</p>

<p align="center">
  <img src="img/Time-to-First-Token vs KV-Cache Budget.png" alt="Median and 95th-percentile time to first token versus KV-cache budget" width="650" />
</p>

<p align="center"><b>Figure 10.</b> Median and P95 TTFT versus KV-cache budget.</p>

<p align="center">
  <img src="img/Accuracy-Memory Trade-off.png" alt="Accuracy-memory trade-off across KV-cache budgets on the 10-video StreamingBench ablation" width="650" />
</p>

<p align="center"><b>Figure 11.</b> Accuracy-memory trade-off on the 10-video ablation.</p>

<p align="center">
  <img src="img/HERMES KV=4000 Performance Across Video Understanding Tasks.png" alt="KV equals 4000 pilot performance across video understanding tasks" width="700" />
</p>

<p align="center"><b>Figure 12.</b> Descriptive KV=4000 task-level performance in the initial pilot. Categories have unequal and, in several cases, small sample sizes.</p>

---

# 8. 50-Video Scaled Benchmark

The main benchmark was scaled to:

```text
50 videos
250 questions
```

Primary configurations:

```text
KV=4000
KV=6000
KV=50000
```

## 8.1 Main results

| Method | KV Budget | Correct | Accuracy | 95% Wilson CI | Max GPU Memory | Median TTFT | P95 TTFT | Max Cache Length | Compression Events |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HERMES | 4000 | 143 / 250 | **57.2%** | 51.00–63.18 | **4.729 GB** | **0.058 s** | 0.068 s | 4013 | 770 |
| HERMES | 6000 | 143 / 250 | **57.2%** | 51.00–63.18 | 4.796 GB | 0.060 s | **0.066 s** | 6013 | 732 |
| Near-uncompressed HERMES control | 50000 | 138 / 250 | 55.2% | 49.00–61.24 | 11.872 GB | 0.112 s | 0.217 s | 50013 | 79 |

## 8.2 Main findings

### Accuracy

KV=4000 and KV=6000 both achieved:

```text
57.2%
```

The KV=50000 control achieved:

```text
55.2%
```

The absolute difference is:

```text
+2.0 percentage points
```

in favor of the compressed 4K/6K settings.

However, the Wilson confidence intervals overlap substantially. Therefore, the current experiment does not justify a strong claim that 4K/6K are intrinsically more accurate than the 50K control.

The more defensible conclusion is:

> On the 50-video subset, HERMES maintained comparable accuracy at KV budgets of 4000 and 6000 while using substantially less GPU memory than the large-budget control.

### Memory

Relative to KV=50000:

- KV=4000 reduced maximum reported GPU memory from 11.872 GB to 4.729 GB, a reduction of approximately **60.2%**.
- KV=6000 reduced maximum reported GPU memory to 4.796 GB, a reduction of approximately **59.6%**.

### Latency

Median TTFT decreased from:

```text
0.112 s at KV=50000
```

to:

```text
0.058 s at KV=4000
0.060 s at KV=6000
```

This corresponds to approximate median TTFT reductions of:

- **48.2%** for KV=4000,
- **46.4%** for KV=6000.

The P95 TTFT difference is even larger:

```text
KV=4000: 0.068 s
KV=6000: 0.066 s
KV=50000: 0.217 s
```

### Cache size

The maximum answering-cache lengths were:

```text
4013   at KV=4000
6013   at KV=6000
50013  at KV=50000
```

This directly verifies that the configured HERMES budgets effectively bound cache growth.

<p align="center">
  <img src="img/StreamingBench 50-Video Evaluation.png" alt="50-video StreamingBench accuracy across the three main KV-cache configurations, with 95 percent Wilson confidence intervals" width="650" />
</p>

<p align="center"><b>Figure 13.</b> StreamingBench accuracy across the main 50-video KV-budget configurations, with 95% Wilson confidence intervals.</p>

---

# 9. Task-Level Analysis

Task-level performance was calculated for all three scaled configurations.

| Task | Questions | KV=4000 | KV=6000 | KV=50000 |
|---|---:|---:|---:|---:|
| Action Recognition | 9 | **55.56%** | **55.56%** | 44.44% |
| Attribute Recognition | 9 | 33.33% | 33.33% | 33.33% |
| Causal Reasoning | 101 | **57.43%** | **57.43%** | 55.45% |
| Counting | 5 | 80.00% | 80.00% | 80.00% |
| Event Understanding | 16 | 50.00% | 50.00% | 50.00% |
| Object Recognition | 5 | **80.00%** | **80.00%** | 60.00% |
| Prospective Reasoning | 100 | **58.00%** | **58.00%** | 57.00% |
| Spatial Understanding | 3 | 66.67% | 66.67% | 66.67% |
| Text-Rich Understanding | 2 | 50.00% | 50.00% | 50.00% |

## Interpretation

The 4K and 6K configurations achieve identical aggregate task-level accuracy values in this evaluation.

Differences relative to the 50K control appear in:

- Action Recognition,
- Causal Reasoning,
- Object Recognition,
- Prospective Reasoning.

However, several categories have extremely small sample counts. For example:

```text
Text-Rich Understanding: 2
Spatial Understanding: 3
Counting: 5
Object Recognition: 5
```

These values are useful for descriptive analysis but should not be treated as reliable estimates of population-level task performance.

The large categories, Causal Reasoning and Prospective Reasoning, dominate the overall benchmark result.

<p align="center">
  <img src="img/Task-Level Accuracy Across KV-Cache Budgets.png" alt="Task-level accuracy across 4000, 6000, and 50000 KV-cache budgets" width="750" />
</p>

<p align="center"><b>Figure 14.</b> Task-level accuracy across the 4000, 6000, and 50000 KV-cache configurations.</p>

---

# 10. Error Analysis

The initial KV=4000 experiment was used to build a qualitative error taxonomy.

## 10.1 Temporal / Event Memory

Observed errors included:

- action ordering,
- event sequence reconstruction,
- remembering prior events,
- temporal-state confusion.

The model could often identify relevant objects while still selecting an incorrect event ordering or state.

## 10.2 Fine-Grained Attribute Recognition

Observed failures included:

- colors,
- small visual details,
- object attributes.

These errors suggest that fine-grained details are a challenging component of the benchmark.

## 10.3 Counting

Observed failures included:

- number of objects,
- repeated objects,
- number of colors or components.

## 10.4 Spatial Understanding

Observed failures included:

- left/right relations,
- relative object placement,
- front/back relationships.

## Causal interpretation constraint

These error categories should **not yet be described as compression-induced failures**.

A failure can only be attributed plausibly to compression when a paired comparison shows that:

```text
large-budget/control = correct
compressed HERMES = incorrect
```

If all configurations fail on the same example, the failure is more likely related to the underlying model or task difficulty rather than cache compression specifically.

<p align="center">
  <img src="img/HERMES KV=4000 Error Distribution by Task.png" alt="Error distribution by task at KV=4000" width="600" />
</p>

<p align="center"><b>Figure 15.</b> Initial error distribution by task in the KV=4000 pilot evaluation.</p>

---

# 11. Efficiency Analysis

The scaled experiment provides the clearest current evidence for the practical benefit of HERMES.

## 11.1 Memory-accuracy trade-off

At KV=4000:

```text
Accuracy:       57.2%
Max GPU memory: 4.729 GB
Median TTFT:    0.058 s
Max cache:      4013
```

At KV=50000:

```text
Accuracy:       55.2%
Max GPU memory: 11.872 GB
Median TTFT:    0.112 s
Max cache:      50013
```

Thus, in this 50-video evaluation, the 4K HERMES setting:

- retained comparable accuracy,
- reduced measured peak GPU memory by approximately 60.2%,
- reduced median TTFT by approximately 48.2%,
- reduced maximum cache length by approximately 92.0%.

## 11.2 KV=4000 vs KV=6000

The 4K and 6K settings produced the same overall accuracy:

```text
57.2%
```

Their measured memory use was also close:

```text
4.729 GB vs 4.796 GB
```

The 4K setting therefore provides the stronger efficiency point in the current experiment because it achieves the same measured accuracy with a smaller cache budget and slightly lower memory consumption.

This is consistent with the broader HERMES motivation: more retained tokens do not automatically guarantee better downstream accuracy if the additional context is redundant or less useful.

---

# 12. Statistical Interpretation

## 12.1 Confidence intervals

For the 50-video benchmark:

| KV Budget | Accuracy | 95% Wilson CI |
|---:|---:|---:|
| 4000 | 57.2% | 51.00–63.18 |
| 6000 | 57.2% | 51.00–63.18 |
| 50000 | 55.2% | 49.00–61.24 |

The intervals overlap substantially.

Therefore, the current results support an **efficiency-preservation** claim more strongly than an **accuracy-improvement** claim.

## 12.2 Paired analysis on the 10-video subset

Exact paired comparisons were calculated using discordant question outcomes.

Some uncorrected pairwise tests produced p-values below 0.05, including comparisons involving the large-budget control.

However:

- 15 pairwise tests were performed,
- the dataset contains only 50 questions,
- the analysis was exploratory,
- no multiple-comparison correction was used in the saved table.

Under a conservative Bonferroni correction:

```text
0.05 / 15 ≈ 0.0033
```

none of the observed p-values would meet the corrected threshold.

Therefore, these pairwise results should be treated as exploratory evidence rather than confirmatory statistical findings.

## 12.3 Scale sensitivity

Comparison between the 10-video and 50-video evaluations:

| KV Budget | 10-video Accuracy | 50-video Accuracy | Difference |
|---:|---:|---:|---:|
| 4000 | 56.0% | 57.2% | +1.2 pp |
| 6000 | 56.0% | 57.2% | +1.2 pp |
| 50000 | 52.0% | 55.2% | +3.2 pp |

The direction of the 4K/6K versus 50K relationship remains similar after scaling, but the larger benchmark is more informative because it includes five times as many questions.

---

# 13. Reproducibility and Backup

A dedicated research backup was created.

## Saved contents

The reproducibility package includes:

- `results/`
- `data/`
- raw experiment logs,
- source-code snapshot,
- environment metadata,
- Git revision,
- Git status,
- source diff,
- Python package list,
- GPU information,
- serialized analysis state,
- SHA-256 file manifest.

The backup inventory found the following completed benchmark result files:

```text
streamingbench_10_kv500
streamingbench_10_kv1000
streamingbench_10_kv2000
streamingbench_10_kv4000
streamingbench_10_kv6000
streamingbench_10_kv50000

streamingbench_50_kv4000
streamingbench_50_kv6000
streamingbench_50_kv50000
```

The research archive contained:

```text
144 files
ZIP integrity: PASS
```

A separate full Kaggle working-directory archive was also generated containing:

```text
445 files
ZIP integrity: PASS
```

The full archive included the repository, benchmark outputs, logs, downloaded shard, backups, and Kaggle notebook source artifact.

---

# 14. Current Research Status

| Component | Status |
|---|---|
| LLaVA-OneVision-0.5B loading | Complete |
| HERMES streaming inference | Complete |
| KV-cache compression validation | Complete |
| RoPE / Transformers compatibility investigation | Complete |
| Official/fidelity inference restoration | Complete |
| Custom-video smoke test | Complete |
| StreamingBench 10-video subset | Complete |
| KV=500 evaluation | Complete |
| KV=1000 evaluation | Complete |
| KV=2000 evaluation | Complete |
| KV=4000 evaluation | Complete |
| KV=6000 evaluation | Complete |
| KV=50000 large-budget control | Complete |
| 10-video six-budget ablation | Complete |
| Wilson confidence intervals | Complete |
| 10-video paired comparison | Complete |
| 50-video benchmark construction | Complete |
| 50-video KV=4000 evaluation | Complete |
| 50-video KV=6000 evaluation | Complete |
| 50-video KV=50000 evaluation | Complete |
| 50-video task analysis | Complete |
| Efficiency analysis | Complete |
| Reproducibility backup | Complete |
| Full 498-video StreamingBench | Pending |
| Exact original LLaVA baseline | Pending |
| Scaled paired significance analysis | Pending |
| Final paper-style report | In progress |

---

# 15. Remaining Experiments and Future Work

## 15.1 Full StreamingBench evaluation

The current primary benchmark contains 50 of approximately 498 videos.

The next major reproduction step is the full benchmark.

A practical strategy is to process StreamingBench in shards rather than storing all videos simultaneously.

Planned target:

```text
~498 videos
~2490 questions
```

## 15.2 Exact LLaVA-OneVision baseline

The current KV=50000 configuration is not equivalent to the original non-HERMES LLaVA-OneVision inference path.

A proper baseline experiment should evaluate the same questions with the base model without the HERMES cache-management mechanism.

This will allow a cleaner decomposition of:

- backbone/model errors,
- streaming-pipeline effects,
- HERMES compression effects.

## 15.3 Paired 50-video analysis

The next statistical comparison should merge predictions by:

```text
video_id + question
```

for:

```text
KV=4000
KV=6000
KV=50000
```

Useful categories include:

- all configurations correct,
- all configurations wrong,
- compressed HERMES correct / large-budget wrong,
- large-budget correct / compressed HERMES wrong.

This will identify candidate compression-sensitive examples.

## 15.4 Compression-specific error taxonomy

The existing error taxonomy is descriptive.

A stronger analysis will classify only examples where prediction correctness changes between memory settings.

This can answer questions such as:

- Are temporal questions more sensitive to compression?
- Are fine-grained visual attributes preferentially lost?
- Does compression remove redundancy without harming relevant history?
- Are some tasks improved when stale context is discarded?

## 15.5 Repeatability

Where computationally feasible, repeated runs or controlled seed experiments should be considered to test whether observed differences are stable.

## 15.6 Paper-level comparison

The final reproduction report should distinguish clearly between:

- the original HERMES paper's published full-benchmark results,
- the current Kaggle subset reproduction,
- hardware differences,
- implementation/environment differences,
- additional ablations introduced in this project.

---

# 16. Planned Final Report Structure

The final thesis/report chapter can be organized as follows:

1. Introduction
2. Streaming Video Understanding Background
3. HERMES Architecture
4. Hierarchical KV-Cache Memory
5. Reproduction Environment
6. Implementation and Compatibility Challenges
7. Reproduction Fidelity Verification
8. Experimental Methodology
9. KV-Budget Ablation
10. StreamingBench Evaluation
11. Accuracy-Memory-Latency Trade-off
12. Task-Level Analysis
13. Error Analysis
14. Statistical Analysis
15. Comparison with HERMES Paper
16. Limitations
17. Future Research Directions
18. Conclusion

---

# 17. Research Notes and Interpretation Constraints

This document is a **living research log** and should continue to be updated as experiments are completed.

The following interpretation rules should be maintained throughout the project:

1. **Do not call KV=50000 the original LLaVA baseline.**
   It is a large-budget HERMES control.

2. **Do not attribute an error directly to compression without a paired comparison.**
   Shared errors may originate from the base model.

3. **Do not compare the 10-video or 50-video accuracy directly with the paper's full-dataset accuracy as if they were equivalent experiments.**

4. **Report hardware when discussing latency or memory.**
   The current experiments use a Tesla T4, while published results may use different accelerators.

5. **Treat very small task categories cautiously.**
   Some task-level accuracies are based on only 2–5 questions.

6. **Preserve raw CSVs and logs as the authoritative experimental record.**
   Notebook variables and pickled objects are secondary conveniences.

7. **Keep the exact Git commit and package revisions with every reproducibility release.**

---

## Current Research Takeaway

The reproduction has progressed beyond a basic engineering validation.

The current 50-video experiment provides evidence that HERMES can constrain streaming KV-cache growth to approximately 4K–6K tokens while maintaining comparable VideoQA accuracy to a much larger 50K cache setting.

On the evaluated 250 questions:

```text
KV=4000: 57.2% accuracy, 4.729 GB max reported GPU memory
KV=6000: 57.2% accuracy, 4.796 GB max reported GPU memory
KV=50000: 55.2% accuracy, 11.872 GB max reported GPU memory
```

The strongest current result is therefore not that compression improves accuracy, but that **substantial memory and latency reductions were achieved without an observed accuracy penalty on the evaluated subset**.

The next stage is to determine whether this finding remains stable on the full benchmark and against the exact non-HERMES LLaVA-OneVision baseline.

---

# 18. Literature Context and Related Work

> *Sourced from the [Comprehensive Technical Research Report](docs/HERMES_Comprehensive_Technical_Research_Report.md). Full bibliographic details, including author lists and venue information, are available in that document.*

## 18.1 Streaming Video Understanding and KV-Cache Management

| Paper | Core Contribution |
|---|---|
| **HERMES** (Zhang et al., 2026) | 3-tier hierarchical KV-cache pruning with summary-token aggregation — the central reference for this reproduction |
| **InfiniPot** (Qualcomm / Hanyang, 2024–2025) | Context-Aware Pruning (CaP-G + CaP-Q) for multi-hop reasoning; demonstrates layer hit-rate dynamics |
| **StreamFlow** (2026) | Dynamics-aware GOP buffering with Visual Attention Score (VAS) injection to counteract attention drift |
| **StreamKV** (Chen et al., 2025) | Segment-level in-context KV retrieval for streaming QA |
| **Streaming Long Video** (Qian et al., NeurIPS 2024) | Long streaming video modeling via LLMs |
| **DispidER** (Qian et al., CVPR 2025) | Disentangled real-time perception, decision, and reaction |
| **ReKV** (Di et al., 2025) | Retrieval-augmented KV offloading — external CPU storage with query-time retrieval |
| **StreamingTOM** (Chen et al., CVPR 2026) | Dynamic visual token merging and pruning |
| **LM-Infinite** (Han et al., NAACL 2024) | Zero-shot extreme length generalization with static cache attention masks |

## 18.2 Foundational Multimodal Models

| Paper | Core Contribution |
|---|---|
| **Qwen2.5-VL** (Bai et al., 2025) | Dynamic-resolution ViT with 3D M-RoPE (temporal, height, width coordinates) |
| **Qwen3-VL** (Bai et al., 2025) | 235B-A22B MoE; 39-language OCR; 57.0% on MMLongBench-Doc |
| **LLaVA-NeXT / OneVision** (Li et al., 2024–2025) | Visual projection architecture, multi-image and video SFT — the backbone used in this reproduction |
| **InternVL 1.5** (Chen et al., 2024) | Open-source multimodal alignment with high-resolution ViT scaling |
| **Gemini 2.5** (Comanici et al., Google DeepMind, 2025) | Frontier agentic reasoning and safety evaluations |
| **MiniCPM** (Hu et al., 2024) | Small language model scaling with high-density tokenizers |

## 18.3 Datasets and Benchmarks

| Benchmark | Focus |
|---|---|
| **StreamingBench** | Streaming video QA — used in this reproduction |
| **EgoSchema** (Malik Group, UC Berkeley) | Long-horizon egocentric temporal QA with rigorous human verification |
| **Ego4D** (Grauman et al., 2022–2024) | 3,000-hour egocentric perception dataset |
| **OVO-Bench** (Li et al., 2025) | Online real-time streaming evaluation (SSR, CRR) |
| **MovieNet** (Huang et al., ECCV 2020) | Narrative long-form video structure |
| **Video-MME** (Fu et al., 2024–2025) | Multimodal video evaluation across short, medium, and long inputs |

---

# 19. HERMES Deep Mechanistic Analysis

> *Synthesized from the [Technical Research Report](docs/HERMES_Comprehensive_Technical_Research_Report.md), Report C.*

## 19.1 Layer-Wise Attention Dynamics

HERMES exploits the natural stratification of attention allocation across transformer layer depth:

- **Shallow Layers (Sensory Memory, Layers 0–5):** Over 80% of attention mass concentrates on the most recent chunk. These layers function as a sensory buffer capturing high-frequency spatial features. Older tokens can be pruned with minimal impact on semantic reasoning.

- **Middle Layers (Working Memory, Layers 6–20):** Recency bias moderates. Attention distributes across recent tokens and intermediate semantic representations, tracking ongoing state changes and multi-frame actions.

- **Deep Layers (Long-Term Memory, Layers 21–31):** Recency bias disappears entirely. Attention becomes sparse and rhythmic, forming periodic peaks at frame patch boundaries (e.g., every 196 tokens). These peaks identify "anchor tokens" that capture the core semantic content of entire frames.

## 19.2 Hierarchical KV-Cache Management Formulation

HERMES calculates a layer-dependent token retention score for each token $i$ at layer $l$:

$$S_i^{(l)} = \alpha_l \cdot \mathcal{R}_i + (1 - \alpha_l) \cdot \mathcal{A}_i^{(l)}$$

where:

- $\mathcal{R}_i = \exp(-\gamma \cdot (T - t_i))$ — exponential recency score
- $\mathcal{A}_i^{(l)} = \frac{1}{|Q|} \sum_{q \in Q} A_{q, i}^{(l)}$ — accumulated attention score
- $\alpha_l \to 1.0$ in shallow layers (recency priority), $\alpha_l \to 0.0$ in deep layers (attention-peak priority)

Tokens are ranked by $S_i^{(l)}$ within each layer and the lowest-scoring tokens are evicted once the cache exceeds the assigned memory budget.

## 19.3 Streaming Ingestion Pipeline

```
Streaming Video Ingestion Pipeline (HERMES)
========================================================================================
Input Video Stream: ... [Frame t-2] ---> [Frame t-1] ---> [Frame t] (Chunk Arrival)
                               |
                               v
Visual Patch Projection:  X_v = ViT_Patch_Embed(Frame_t)  [e.g., 196 tokens/frame]
                               |
                               v
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 0-5  [Sensory Memory]:                                                         │
│   - Attention profile: Extreme Recency Bias                                          │
│   - Eviction: Low S_i^(l) (Old frames purged, latest chunk fully preserved)          │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │
                                           v
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 6-20 [Working Memory]:                                                         │
│   - Attention profile: Hybrid Recency + Semantic Tracking                            │
│   - Eviction: Balanced S_i^(l) (Recent window maintained + salient history)          │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │
                                           v
┌──────────────────────────────────────────────────────────────────────────────────────┐
│ Layer 21-31 [Long-Term Memory]:                                                      │
│   - Attention profile: Sparse Rhythmic Peaks (Anchor Tokens)                         │
│   - Eviction: Keep top-K periodic anchor tokens across all historical frames         │
│   - Aggregation: Low-scoring tokens mean-pooled into Summary Token (k_sum, v_sum)    │
└──────────────────────────────────────────┬───────────────────────────────────────────┘
                                           │
                                           v
Cross-Layer Memory Smoothing → Positional Re-Indexing → Query Arrival (TTFT < 30 ms)
========================================================================================
```

## 19.4 Cross-Layer Memory Smoothing

Evicting tokens independently across layers can disrupt feature representations. If token $i$ is retained in Layer 26 as an anchor but pruned in Layer 8, the upper layer loses its lower-level feature pathway.

HERMES propagates importance scores backward through the network:

$$\tilde{S}_i^{(l)} = S_i^{(l)} + \sum_{k > l} \lambda_k \cdot \tilde{S}_i^{(k)}$$

Smoothing coefficients: $\lambda_{shallow}=0.1$, $\lambda_{middle}=0.3$, $\lambda_{deep}=0.4$.

## 19.5 M-RoPE Positional Re-Indexing

Pruning intermediate tokens creates non-contiguous gaps in the position index sequence, which introduces phase shifts in the relative attention distances for newly arriving tokens.

HERMES counteracts this with a bijective re-indexing map:

$$\tilde{m}_i = \phi(m_i) = \text{rank}(m_i \in \mathcal{M}_{retained})$$

This preserves contiguous index sequences, maintaining positional consistency during generation.

## 19.6 Summary Token Aggregation

Non-anchor tokens evicted in deep layers are not purged entirely. Their key and value representations are mean-pooled into a summary token:

$$k_{sum}^{(l)} = \frac{1}{|\mathcal{E}_l|} \sum_{j \in \mathcal{E}_l} k_j^{(l)}, \quad v_{sum}^{(l)} = \frac{1}{|\mathcal{E}_l|} \sum_{j \in \mathcal{E}_l} v_j^{(l)}$$

The summary token is appended to the layer's KV-cache, ensuring cumulative background context remains accessible.

---

# 20. Cross-Paper Technical Comparison

> *Complete comparison profiles from the [Technical Research Report](docs/HERMES_Comprehensive_Technical_Research_Report.md), Report D.*

| Dimension | HERMES | InfiniPot | StreamFlow |
|---|---|---|---|
| **Objective** | Unbounded streaming video QA under fixed memory | Multi-hop reasoning context retention in long documents | Visual attention drift prevention and encoding redundancy reduction |
| **Backbones** | LLaVA-OV-7B, Qwen2.5-VL-7B | LLaMA-3-8B, Mistral-7B | Qwen3.5-9B |
| **Training** | Training-free (plug-and-play) | Training-free pruning | Hybrid (frozen MLLM + trained compressor) |
| **Memory** | 3-tier internal KV hierarchy | Dual-scoring KV-cache (CaP-G + CaP-Q) | Mid-term GOP buffer + latent long-term store |
| **Selection** | Recency (shallow) / attention spikes (deep) | Global query hit-rate scoring | Pixel temporal residuals / cosine similarity |
| **Positional** | Contiguous M-RoPE re-indexing | Static RoPE preserved | Anchor I-frame chronological sorting |
| **Memory Footprint** | Bounded & constant (< 20 GB across 10⁴ frames) | Bounded (4K–8K budgets) | 21.1% peak GPU reduction |
| **TTFT** | < 30 ms (10× faster than retrieval) | Low prefill latency over 4K tokens | 50.4% end-to-end latency reduction |
| **Key Results** | +11.4% accuracy on streaming benchmarks | 49.75% on HotpotQA | 67.73% on StreamingBench |

---

# 21. Evolution of the Research Field

```
Evolutionary Roadmap of Long-Context Multimodal Systems
===================================================================================================
Phase 1: Offline Video Modeling (Pre-2023)
  - Uniform sampling (16-32 frames), full prefill
  - Quadratic attention complexity O(T²), OOM failures on long inputs
                                   │
                                   ▼
Phase 2: Context-Window Scaling (2024)
  - Attention window expansion via 3D M-RoPE (Qwen2-VL, Qwen2.5-VL to 32k-256k tokens)
  - High prefill latency and unbounded memory growth under continuous streaming
                                   │
                                   ▼
Phase 3: Query-Guided KV-Cache Pruning (2024-2025)
  - Selective pruning (SnapKV, InfiniPot CaP-Q) effective for static multi-hop QA
  - Fails in streaming: cannot predict future queries during ingestion
                                   │
                                   ▼
Phase 4: Retrieval-Augmented Video Offloading (2025)
  - External storage architectures (ReKV): offloads KV-cache to CPU/disk
  - Resolves GPU memory limits, but retrieval introduces high latency (TTFT > 0.60 s)
                                   │
                                   ▼
Phase 5: Internal Hierarchical & Dynamics-Aware Architectures (2026 — Frontier)
  - HERMES: Layer-wise attention dynamics (Sensory, Working, Long-Term anchors)
  - StreamFlow: Dynamics-aware GOPs and online VAS injection
  - Convergence: Sub-30ms latency with bounded memory and high semantic retention
===================================================================================================
```

---

# 22. Reproduction Audit and Evidence Assessment

> *Sourced from the [Reproduction Audit](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md), prepared 9 October 2026.*

## 22.1 Executive Assessment

**Main finding.** On the documented 50-video / 250-question subset, HERMES with 4,000 or 6,000 cache-budgeted tokens answered 143/250 questions correctly (57.2%), while a 50,000-token HERMES control answered 138/250 (55.2%). The 4K and 6K settings used much less reported GPU memory and had lower TTFT.

**Measured trade-off against the 50K control:**

- **60.2%** lower highest logged GPU memory
- **48.2%** lower median TTFT
- **92.0%** fewer cache positions at the logged maximum
- **+2.0 pp** observed accuracy difference (point estimate, not a demonstrated causal improvement)

**Verdict:** This is substantive engineering and empirical progress, but best described as a **successful subset reproduction and systems study**, not yet a fully audited reproduction of every component in the HERMES paper.

## 22.2 Evidence Grading System

| Grade | Meaning |
|---|---|
| **A** | Notebook directly corroborated — executable cell contents or explicit saved output |
| **B** | Research-log reported — present in project log, consistent with notebook summaries |
| **C** | Analytic interpretation — reasoned inference from A/B evidence |
| **D** | Unverified — requires further artifacts or testing |

## 22.3 Source Materials Inspected

| Asset | Establishes | Cannot Establish |
|---|---|---|
| Research project log | Researcher-reported workflow, configurations, interpretation | Independent proof of saved CSV values |
| Kaggle notebook (191 cells) | Actual procedures and reported outputs | Not a self-contained executable repo |
| HERMES paper (previously uploaded) | Paper methodology and baselines | Does not validate the Kaggle implementation |

## 22.4 Important Missing Artifacts

The project log records two successful ZIP integrity checks, but neither ZIP has been uploaded for independent audit. The notebook references Kaggle paths (`/kaggle/working/HERMES/`) for CSVs, logs, and figures that are not locally available. Specific required artifacts:

1. `source_changes.diff` and `video_qa/base.py` from the actual experiment commit
2. All three 50-video `1_0.csv` files and corresponding `.log` files
3. `streamingbench_50_kaggle.json`
4. `environment.json`, `pip_freeze.txt`, `git_commit.txt`, `git_status.txt`
5. SHA-256 manifest and research ZIP

## 22.5 Code-Path Review

### Correctly working subsystems (per saved outputs)

- Model loads on CUDA and generates responses
- Timestamped videos processed in chunks
- KV compression appears in logs when budgets exceeded
- Cache maxima conform to budget + 13-token prefix
- 50-video batches produce all 250 expected result rows
- The notebook extracts `qa_acc`, task classifications, runtime measurements, and figures

### Not yet proven

- Exact semantics of the modification to `video_qa/base.py`
- Equivalence of the final repository tree to official HERMES
- Unit tests of layer partitioning, pseudo-query attention scoring, cross-layer smoothing, summary pooling, and rotary delta correction in isolation
- Exact definition of memory and TTFT instrumentation within inference code
- Separation of conversation history from gold labels for all benchmark calls

---

# 23. Risks, Inconsistencies, and Open Issues

> *From the [Reproduction Audit](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md), Section 11.*

| Priority | Issue | Consequence | Resolution Required |
|---|---|---|---|
| **Critical** | `video_qa/base.py` remains modified per `git status` | Unverified change may affect evaluation, prompting, or inference | Inspect full `source_changes.diff`; explain/diff-test every functional change |
| **Critical** | No exact non-HERMES baseline | Cannot isolate benefit against base LLaVA | Run native baseline with same questions, times, and sampling |
| **High** | Single nonrandom 50-video shard | Limited generalisability; extreme task imbalance | Add stratified/random independent shards, then full set |
| **High** | Question-level uncertainty ignores video clustering | Intervals may be too optimistic | Paired video-cluster bootstrap or hierarchical analysis |
| **High** | No completed 50-video paired table | Equal totals may hide significant answer swaps | Join 250 exact records across three budgets |
| **High** | "50K = no compression" can be misleading | 79 logged compressions in 50-video control | Consistently label as "near-uncompressed HERMES control" |
| **High** | Metric definitions not independently audited | "max GPU memory" and TTFT may omit important costs | Inspect source instrumentation; add CUDA peak + end-to-end timers |
| **High** | Gold-answer history contamination not formally ruled out | Would compromise the meaning of accuracy if gold answers leak | Trace actual history-writing and prediction-input code path |
| **Medium** | Some saved-output cells lack execution counts | Ambiguity in chronological rebuild | Fresh-kernel run and exported execution manifest |
| **Medium** | Conditional skip checks file non-emptiness only | Partial/old CSV could be reused | Validate 250 unique keys, expected IDs, completion marker |
| **Medium** | Individual HERMES mechanisms not ablated locally | Cannot determine which component drives benefits | Ablate smoothing, summaries, position policy, importance scoring |
| **Medium** | Reported TTFT excludes unspecified components | Incomplete real-time claims | Measure ingestion cost, latency percentiles, total wall time |

**Integrity note:** No contradiction was found between the main accuracy rows reported in the log and the corresponding notebook CSV-summary outputs.

---

# 24. Phased Next-Stage Research Plan

> *From the [Reproduction Audit](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md), Section 12. The plan is deliberately gated: do not spend substantial GPU resources on the full dataset until cheap code and analysis checks pass.*

## Phase 0 — Recover and Freeze the Real Executable Experiment (No GPU)

1. Import the full backed-up HERMES repository/source snapshot and `source_changes.diff`
2. Diff the modified `video_qa/base.py` against recorded upstream HEAD
3. Inspect all modified source for leakage from reference answers into model inputs
4. Verify no temporary "disable key rotation" patch survived in cache code
5. Record SHA-256 for effective configuration, model identity, source files, annotation file, and CSVs
6. Convert the exploratory notebook into a canonical `setup → run → analyse → export` workflow
7. Create an experiment manifest with all provenance fields

**Gate 0 pass:** Every functional source deviation understood/documented; no gold-label leakage; exact executable environment recoverable.

## Phase 1 — Statistical Re-Analysis of Existing 50-Video CSVs (CPU-Only)

1. Load all three prediction CSVs; validate 250 rows, 50 videos, 5 QA/video, unique keys
2. Recompute accuracy directly from saved gold and predictions
3. Construct 4K ↔ 6K ↔ 50K paired correctness tables and discordance counts
4. Apply exact paired test as sensitivity analysis; bootstrap video-level paired differences
5. Report effect size and confidence intervals for each contrast
6. Build a 30–50 example qualitative casebook from discordances
7. Verify that the 10-video pilot is (or is not) nested in the 50-video set

**Gate 1 pass:** No missing/duplicated/misaligned questions; difference estimates and CIs reproducible from CSVs.

## Phase 2 — Establish Honest Controls and Component Fidelity (GPU)

1. Run **native LLaVA-OneVision-0.5B without HERMES** at matched sampling/questions
2. Retest HERMES 4K, 6K, 50K using the locked clean driver
3. Add **FIFO/sliding-window** cache retention at the same GPU/token budgets
4. Ablate **cross-layer smoothing**, **summary-token aggregation**, **attention-based retention**, and **position-re-indexing** separately
5. Unit-test RoPE/M-RoPE correction against controlled reference recomputation
6. Isolate streaming prefilling, compression overhead, query-time TTFT, and total QA response time

**Gate 2 pass:** At least one matched native baseline, one simple budget-matched compression baseline, and a documented source-fidelity audit.

## Phase 3 — Broader and More Representative Evaluation (GPU, Shard-Wise)

1. Source additional official StreamingBench video shards beyond the 201–250 block
2. Construct a reproducibly selected independent validation shard
3. Re-run strongest configurations from Phases 1–2
4. Evaluate full 498-video benchmark if access and compute allow
5. Report both micro accuracy and task-specific effects across video durations and compression events

**Gate 3 pass:** Results stable across non-overlapping evaluation shards; statistically grounded trade-off intervals.

## Phase 4 — Research Extension (Only After Baseline Evidence Is Credible)

Select an extension based on observed cases where the full model succeeds but compressed memory fails. Evaluate a minimal change first, use targeted ablation and independent holdout videos.

---

# 25. Experimental Design Specifications

> *From the [Reproduction Audit](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md), Section 13.*

## 25.1 Mandatory Controls

| ID | Configuration | Scientific Role |
|---|---|---|
| C0 | Native LLaVA-OV-0.5B on matched questions and frames | True backbone reference — **currently missing** |
| C1 | HERMES 50K with measured compression count | Large-budget HERMES control |
| C2 | HERMES 6K | Paper-style larger token budget |
| C3 | HERMES 4K | Current likely efficient operating point |
| C4 | Simple fixed 4K FIFO/sliding-window retention | Separates hierarchical selection from merely limiting budget |
| A1 | HERMES 4K without cross-layer smoothing | Attribution to memory alignment |
| A2 | HERMES 4K without summary-token aggregation | Attribution to long-term compressed memory |
| A3 | HERMES 4K with alternative importance policy | Attribution to depth-specific token selection |
| A4 | Lazy vs eager positional re-indexing | Position-fidelity/runtime trade-off |
| X1 | 500/1K/2K at sufficient sample scale | Test whether surprisingly high pilot accuracy generalises |

**Fairness requirements:** Same video set, annotation JSON, model checkpoint, seed, FPS, chunking, prompt, question-time cut-off, image preprocessing, and QA parser across all compared configurations.

## 25.2 Per-Question Output Schema

Save per-question results with at least:

```text
run_id, model_id, repo_sha, source_diff_sha, data_sha,
video_id, question_id, question, task, query_time_s,
gold_choice, prediction_text, parsed_choice, correct,
kv_budget, sampled_frames_seen, compression_events_so_far,
cache_length_at_query, ttft_s, end_to_end_query_s,
gpu_peak_allocated_gb, gpu_peak_reserved_gb
```

## 25.3 Evaluation Metrics

- **Accuracy:** micro overall, per task with $n$, macro only with caveats, per-duration band
- **Paired robustness:** win/loss/tie by question, video-cluster bootstrap CI, exact paired p (exploratory), 4K↔6K agreement rate
- **Efficiency:** peak allocated and reserved VRAM, per-question TTFT percentiles, total wall time/video, tokens/s, compression frequency
- **Memory quality:** cache length, retained-frame ages, summary-token counts
- **Reproducibility:** input hashes, package lock, source code diff, machine/GPU, run manifest

## 25.4 Suggested Acceptance Criteria

- **Completeness:** 100% expected video/QA keys, no duplicate rows, no silent dropped failures
- **Validity:** Same run manifest across compared methods except intended treatment variables
- **Control coverage:** True backbone and equal-budget FIFO available or honestly marked infeasible
- **Statistics:** Paired cluster-aware interval; predeclared non-inferiority margin (e.g., 3 pp accuracy loss)
- **Efficiency:** Lower memory and meaningful latency reduction with standardized instrumentation

---

# 26. Candidate Research Extensions

> *From the [Reproduction Audit](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md), Section 15 and the [Technical Research Report](docs/HERMES_Comprehensive_Technical_Research_Report.md), Reports F–H. These are hypotheses to consider only after code-fidelity and matched-baseline stages are complete.*

## 26.1 Critical Research Gaps Identified

| Gap | Status | Opportunity |
|---|---|---|
| **Retention of fine-grained micro-events** | Partially addressed by summary tokens, but individual micro-events cannot be recovered from pooled state | Dual-path memory combining motion anomaly detectors with semantic anchor extraction |
| **Autoregressive visual attention drift** | StreamFlow addresses via VAS + external GOP latents, but requires trained compressors | Training-free decoding hooks that dynamically adjust attention logits for cached visual anchors |
| **Audio-visual temporal synchronization** | Unresolved — cache eviction is purely vision-centric | Multi-modal cross-attention gating with audio event detectors modulating visual token retention |

## 26.2 Candidate Directions

### Direction 1 — Event-Sensitive Preservation of Short-Lived Evidence

Detect temporal change/novelty across sampled frames and reserve a small portion of the KV budget for rare event tokens. Compare against HERMES, StreamForest, TimeChat-Online, and StreamMem.

### Direction 2 — Per-Layer Adaptive Budget Allocation

Allocate KV slots across layers based on stable online signals (attention concentration, video novelty) rather than equal fixed budgets. Verify against closest methods before claiming novelty.

### Direction 3 — Confidence-Controlled Memory Degradation

Inexpensive online cues choose one of several predetermined compression levels without re-encoding historical video. Test quality-memory curve, switching overhead, and false trigger rates.

### Direction 4 — Decoding-Time Visual Focus Calibration (EDMI)

When Visual Attention Score drops below threshold $\tau$, apply dynamic logit adjustment to historical anchor tokens and summary tokens within the existing cache:

$$\tilde{S}_{t, i}^{(l, h)} = \begin{cases} S_{t, i}^{(l, h)} + \alpha \cdot (\tau - VAS_t) \cdot \Omega_i, & \text{if } i \in \mathcal{V} \text{ and } VAS_t < \tau \\ S_{t, i}^{(l, h)}, & \text{otherwise} \end{cases}$$

**Note:** This is a research idea (Entropy-Guided Dynamic Memory Injection), not a validated original contribution. Avoid adopting its novelty claims or performance assumptions without direct literature/code verification.

### Decision Rule

Choose the simplest candidate that (a) addresses reproducible compressed-only failures, (b) is not already implemented by close competitors, (c) has measurable outcomes on held-out videos, and (d) fits single-T4 or realistically accessible GPU resources. Document expected failure modes and falsifiable success criteria before coding.

---

# 27. Deliverables and Completion Criteria

> *From the [Reproduction Audit](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md), Section 16.*

| Priority | Deliverable | Completion Test |
|---|---|---|
| **P0** | `REPRODUCTION_PROVENANCE.md` | Every source change categorised; no leakage; git/runtime fingerprints archived |
| **P0** | Clean executable reproduction driver | One fresh restart reproduces a smoke test and verifies cache/key states |
| **P1** | `streamingbench_50_paired_correctness.csv` | Exactly 250 correctly aligned unique QA rows |
| **P1** | `streamingbench_50_pairwise_stats.md` | Exact discordances + clustered CIs + effect sizes |
| **P1** | `error_casebook.md` | Each analysed difference annotated and independently reviewable |
| **P2** | Non-HERMES LLaVA reference result | Identical matched protocol or clearly delimited feasibility constraints |
| **P2** | Equal-budget FIFO and component ablations | Budget/accuracy/runtime tables on same test cases |
| **P3** | New non-overlapping validation shard | Published sampling manifest + task distribution + results |
| **P3** | Full benchmark (if feasible) | Transparent split, input count, hardware, and official-protocol comparison |
| **P4** | Research novelty matrix and proposal | Closest-work analysis, falsifiable innovation, minimum viable experiment |
| **P4** | Final paper-style reproducibility report | Figures, tables, assumptions, negative results, and archive links |

### Suggested Session Order

1. **Session 1 (CPU-only):** Open Kaggle backup; review `video_qa/base.py` diff; verify CSV schema/row IDs; compute all 250-question paired contrasts; identify compression-sensitive videos.
2. **Session 2 (Minimal GPU):** Run short deterministic 4K/6K tests from clean environment; test RoPE correctness; run a small matched native-LLaVA baseline.
3. **Session 3 (Confirmation):** Finalise matched baseline plus FIFO and 1–2 mechanism ablations; decide whether 4K, 6K, and small-budget variants warrant full-dataset compute.
4. **Session 4+:** Broader shards, full evaluation, and research extension selection based on verified failure cases.

---

> *This document is a **living research log**. Future updates should append new experiments, additional baselines, analysis results, and conclusions with dates, experiment IDs, input hashes, and evidence status — without overwriting historical findings.*
