# HERMES Reproduction Project — Research Log

> **Streaming Video QA with KV-Cache Compression**

| | |
|---|---|
| **Project Status** | Ongoing reproduction and extension study |
| **Environment** | Kaggle Tesla T4 GPU |
| **Model** | LLaVA-OneVision-Qwen2-0.5B |
| **Framework** | HERMES VideoQA inference pipeline |

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Timeline and Development Log](#2-timeline-and-development-log)
3. [KV-Cache Compression Validation](#3-kv-cache-compression-validation)
4. [StreamingBench Preparation](#4-streamingbench-preparation)
5. [HERMES KV=4000 Evaluation](#5-hermes-kv4000-evaluation)
6. [Error Analysis](#6-error-analysis)
7. [Current Research Results](#7-current-research-results)
8. [Remaining Experiments](#8-remaining-experiments)
9. [Final Report Structure](#9-final-report-structure)
10. [Notes](#10-notes)

---

## 1. Project Objective

The goal of this project is to reproduce and analyze the HERMES paper's streaming video question answering system.

**Research Questions:**

1. Can the HERMES KV-cache compression mechanism be reproduced on publicly available infrastructure?
2. Does KV-cache compression preserve VideoQA accuracy while reducing memory requirements?
3. What types of video understanding abilities are affected by aggressive compression?
4. How does memory budget influence accuracy and latency?

**Extended Analysis Scope:**

- Memory / accuracy trade-offs
- KV-cache ablation experiments
- Task-level performance analysis
- Qualitative failure analysis

---

## 2. Timeline and Development Log

### Day 1 — Environment Setup and Initial Reproduction

**Platform:** Kaggle Notebook · Tesla T4 GPU · CUDA enabled

| Component | Version |
|---|---|
| Python | 3.12 |
| PyTorch | 2.10.0 + CUDA 12.8 |
| Transformers | 4.45.0.dev0 |
| GPU | Tesla T4 |

The LLaVA-OneVision-Qwen2-0.5B model was successfully loaded.

```
Model loaded successfully
Main device: cuda:0
```

---

### Day 2 — Dependency and Compatibility Debugging

Several compatibility issues were encountered between the original HERMES codebase and the current library versions:

#### Problem 1 — Missing `language_model` Attribute

Original HERMES code expected:

```python
base_model.language_model.config
```

However, the installed Transformers version exposed `LlavaOnevisionForConditionalGeneration` with a different internal structure. The loading code was modified to correctly access the language model components.

#### Problem 2 — Rotary Embedding API Mismatch

| Version | API Signature |
|---|---|
| Original HERMES | `rotary_emb(x, seq_len=...)` |
| Current Qwen2 | `rotary_emb(hidden_states, position_ids)` |

The rotary embedding helper functions were updated to match the new API.

#### Problem 3 — CUDA Device Mismatch

```
Error: Expected all tensors to be on the same device
```

**Cause:** RoPE tensors and KV-cache tensors were created on different CUDA devices.

**Solution:** All rotary tensors were explicitly moved to the active model device.

---

### Day 3 — Successful HERMES Smoke Test

A custom test video (`video_id: kaggle_test_001`) was created. The model successfully processed scene understanding, action recognition, and event summarization.

**Example Outputs:**

| Question | Prediction |
|---|---|
| *What is happening in the video right now?* | The video is set in a kitchen, where the sandwich is being prepared. |
| *What is the person doing at this point?* | The person is preparing a sandwich, specifically making the filling for it. |

The streaming pipeline successfully maintained history across video segments.

---

## 3. KV-Cache Compression Validation

Different KV budgets were tested to validate that the HERMES compression mechanism operates correctly.

### KV = 500

| Metric | Value |
|---|---|
| Compression triggered | ✅ Yes |
| Compressed length | 513 (`n_init=13` + `KV=500`) |
| GPU memory | ~4.43 GB |

### KV = 4000

| Metric | Value |
|---|---|
| Compression triggered | ✅ Yes |
| Compressed length | 4013 |
| GPU memory | ~4.52 GB |

### KV = 50000 (No-Compression Baseline)

| Metric | Value |
|---|---|
| Compression triggered | ❌ No |
| GPU memory | ~4.58 GB |

This configuration represents the **non-compressed baseline** — the cache never exceeds the budget, so no eviction occurs.

<p align="center">
  <img src="img/KV Cache After HERMES Compression.png" alt="Figure 1 — Final KV-cache length after HERMES compression at each budget setting" width="520" />
</p>

*<p align="center">Figure 1 — Final KV-cache length after HERMES compression. KV=500 and KV=4000 trigger compression, while KV=50000 never exceeds the budget.</p>*

<p align="center">
  <img src="img/GPU Memory Usage vs KV Cache Budget.png" alt="Figure 2 — GPU memory usage across KV cache budgets" width="520" />
</p>

*<p align="center">Figure 2 — GPU memory usage across KV cache budget settings. Memory increases modestly from 4.43 GB (KV=500) to 4.58 GB (KV=50000).</p>*

<p align="center">
  <img src="img/HERMES KV Budget vs GPU Memory.png" alt="Figure 3 — HERMES KV budget vs GPU memory" width="520" />
</p>

*<p align="center">Figure 3 — KV budget vs GPU memory comparison, confirming a modest linear relationship between cache size and VRAM consumption.</p>*

<p align="center">
  <img src="img/Memory Growth with KV Cache Size.png" alt="Figure 4 — Memory growth trend with KV cache size" width="520" />
</p>

*<p align="center">Figure 4 — Memory growth curve showing the continuous relationship between KV cache budget and GPU memory consumption.</p>*

<p align="center">
  <img src="img/Time To First Token vs KV Budget.png" alt="Figure 5 — Time to first token vs KV budget" width="520" />
</p>

*<p align="center">Figure 5 — Time To First Token (TTFT) latency across KV budgets. Compression provides a slight latency benefit at lower budgets.</p>*

<p align="center">
  <img src="img/HERMES KV Budget vs Answer Detail.png" alt="Figure 6 — Answer verbosity across KV budgets" width="520" />
</p>

*<p align="center">Figure 6 — Answer detail (word count) across KV budgets. Higher budgets produce more verbose answers, suggesting richer context retention.</p>*

---

## 4. StreamingBench Preparation

The official StreamingBench annotation file was used:

```
data/streamingbench/streamingbench_realtime.json
```

**Full dataset:** 498 videos

A **10-video evaluation subset** was created with **5 questions per video** (50 questions total).

<p align="center">
  <img src="img/StreamingBench 10-Video Subset Durations.png" alt="Figure 7 — Duration of each video in the 10-video subset" width="650" />
</p>

*<p align="center">Figure 7 — Duration of each video in the StreamingBench 10-video evaluation subset. Videos range from ~200 s to ~375 s.</p>*

<p align="center">
  <img src="img/StreamingBench Subset Task Distribution.png" alt="Figure 8 — Task distribution across the evaluation subset" width="650" />
</p>

*<p align="center">Figure 8 — Task distribution across the evaluation subset. Event Understanding dominates, followed by Action Recognition and Attribute Recognition.</p>*

---

## 5. HERMES KV=4000 Evaluation

| Parameter | Value |
|---|---|
| Model | LLaVA-OneVision-Qwen2-0.5B |
| Dataset | StreamingBench 10-video subset |
| Sampling rate | 0.5 FPS |
| Streaming | Enabled |
| KV budget | 4000 |

Evaluation completed successfully. Output schema:

```
video_id · question · choices · answer · correct_choice
pred_answer · pred_choice · qa_acc · task
```

The automatic evaluator was successfully integrated.

<p align="center">
  <img src="img/Prediction Correctness Distribution.png" alt="Figure 9 — Overall prediction correctness distribution" width="420" />
</p>

*<p align="center">Figure 9 — Overall prediction correctness distribution at KV=4000. The model correctly answers ~56% of the 50 evaluation questions.</p>*

<p align="center">
  <img src="img/HERMES KV=4000 StreamingBench Task Accuracy.png" alt="Figure 10 — Per-task accuracy breakdown at KV=4000" width="650" />
</p>

*<p align="center">Figure 10 — Per-task accuracy breakdown at KV=4000. Causal Reasoning achieves 100%, while Attribute Recognition drops to ~33%.</p>*

---

## 6. Error Analysis

Incorrect predictions were extracted and analyzed. The following failure taxonomy was identified:

### 6.1 Temporal / Event Memory

- Action ordering, sequence reconstruction, remembering previous events
- **Observed:** The model often recognized objects correctly but confused temporal order.

### 6.2 Fine-Grained Attribute Recognition

- Colors, small object details, visual attributes
- **Observed:** Compression can remove fine visual details.

### 6.3 Counting

- Number of blocks, colors, repeated objects
- **Observed:** Long-range object tracking is challenging after compression.

### 6.4 Spatial Understanding

- Front/back relations, relative position, object placement
- **Observed:** Spatial relationships are more fragile under memory reduction.

<p align="center">
  <img src="img/HERMES KV=4000 Error Distribution by Task.png" alt="Figure 11 — Error distribution by task type at KV=4000" width="600" />
</p>

*<p align="center">Figure 11 — Error distribution by task type at KV=4000. Event Understanding and Attribute Recognition account for the majority of errors.</p>*

---

## 7. Current Research Results

| Experiment | Status |
|---|---|
| Model loading | ✅ Complete |
| HERMES inference | ✅ Complete |
| KV compression | ✅ Complete |
| RoPE compatibility fixes | ✅ Complete |
| StreamingBench subset | ✅ Complete |
| KV=4000 evaluation | ✅ Complete |
| Error taxonomy | ✅ Complete |

---

## 8. Remaining Experiments

### Experiment 1 — KV=50000 Baseline

**Purpose:** Measure performance without compression.

**Comparison:** No compression vs HERMES compression.

### Experiment 2 — KV Budget Ablation

| KV Budget | Accuracy | Memory |
|---|---|---|
| 500 | TBD | TBD |
| 4000 | ✅ Complete | ✅ Complete |
| 6000 | ✅ Complete | TBD |
| 50000 | ⏳ Pending | ⏳ Pending |

### Experiment 3 — Larger Benchmark

| Scale | Videos | Questions |
|---|---|---|
| Current | 10 | 50 |
| Medium | 50 | 250 |
| Full | 498 | 2490 |

---

## 9. Final Report Structure

1. Introduction
2. HERMES architecture overview
3. Reproduction environment
4. Implementation challenges
5. Engineering fixes
6. Experimental setup
7. KV-cache ablation study
8. Accuracy–memory trade-off
9. Error analysis
10. Limitations
11. Future improvements

---

## 10. Notes

> This document is a **living research log**. Future updates should append new experiments, additional baselines, plots, tables, and conclusions.
