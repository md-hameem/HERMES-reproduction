# HERMES Reproduction Project Log

## Streaming Video QA with KV-Cache Compression

**Project Status:** Ongoing reproduction and extension study\
**Environment:** Kaggle Tesla T4 GPU\
**Model:** LLaVA-OneVision-Qwen2-0.5B\
**Framework:** HERMES VideoQA inference pipeline

------------------------------------------------------------------------

# 1. Project Objective

The goal of this project is to reproduce and analyze the HERMES paper's
streaming video question answering system.

The main research questions are:

1.  Can the HERMES KV-cache compression mechanism be reproduced on
    publicly available infrastructure?
2.  Does KV-cache compression preserve VideoQA accuracy while reducing
    memory requirements?
3.  What types of video understanding abilities are affected by
    aggressive compression?
4.  How does memory budget influence accuracy and latency?

The final goal is not only reproduction, but also an extended analysis
containing:

-   memory/accuracy trade-offs
-   KV-cache ablation experiments
-   task-level performance analysis
-   qualitative failure analysis

------------------------------------------------------------------------

# 2. Timeline and Development Log

## Day 1 --- Environment Setup and Initial Reproduction

### Environment

Platform:

-   Kaggle Notebook
-   Tesla T4 GPU
-   CUDA enabled

Initial environment:

    Python: 3.12
    PyTorch: 2.10.0 + CUDA 12.8
    Transformers: 4.45.0.dev0
    GPU: Tesla T4

The LLaVA-OneVision-Qwen2-0.5B model was successfully loaded.

Initial verification:

    Model loaded successfully
    Main device: cuda:0

------------------------------------------------------------------------

# Day 2 --- Dependency and Compatibility Debugging

Several compatibility issues were encountered:

-   Transformers version differences
-   Qwen2 rotary embedding API changes
-   LLaVA-OneVision internal model structure differences

## Problem 1 --- Missing language_model attribute

Original HERMES code expected:

``` python
base_model.language_model.config
```

However, the installed Transformers version exposed:

``` python
LlavaOnevisionForConditionalGeneration
```

with a different internal structure.

The loading code was modified to correctly access the language model
components.

------------------------------------------------------------------------

## Problem 2 --- Rotary Embedding API mismatch

Original HERMES implementation used:

``` python
rotary_emb(x, seq_len=...)
```

New Qwen2 implementation expected:

``` python
rotary_emb(hidden_states, position_ids)
```

The rotary embedding helper functions were updated.

------------------------------------------------------------------------

## Problem 3 --- CUDA device mismatch

Error:

    Expected all tensors to be on the same device

Cause:

RoPE tensors and KV-cache tensors were created on different CUDA
devices.

Solution:

All rotary tensors were explicitly moved to the active model device.

------------------------------------------------------------------------

# Day 3 --- Successful HERMES Smoke Test

A custom test video was created:

    video_id:
    kaggle_test_001

The model successfully processed:

-   scene understanding
-   action recognition
-   event summarization

Example outputs:

Question:

    What is happening in the video right now?

Prediction:

    The video is set in a kitchen, where the sandwich is being prepared.

Question:

    What is the person doing at this point?

Prediction:

    The person is preparing a sandwich, specifically making the filling for it.

The streaming pipeline successfully maintained history across video
segments.

------------------------------------------------------------------------

# 3. KV-Cache Compression Validation

Different KV budgets were tested.

## KV=500

Results:

    Compression triggered: Yes
    Compressed length: 513
    GPU memory: ~4.43 GB

Reason:

    n_init = 13
    KV budget = 500

    Final cache:
    500 + 13 = 513

------------------------------------------------------------------------

## KV=4000

Results:

    Compression triggered: Yes
    Compressed length: 4013
    GPU memory: ~4.52 GB

------------------------------------------------------------------------

## KV=50000

Results:

    Compression triggered: No
    GPU memory: ~4.58 GB

This configuration represents the non-compressed baseline.

------------------------------------------------------------------------

# 4. StreamingBench Preparation

The official StreamingBench annotation file was identified:

    data/streamingbench/streamingbench_realtime.json

Dataset statistics:

    Total videos: 498

The video files were not included with the repository, therefore a
StreamingBench shard was downloaded separately.

A subset was created:

    StreamingBench 10-video evaluation subset

Each video contains:

    5 questions/video

Total evaluation size:

    10 videos
    50 questions

------------------------------------------------------------------------

# 5. HERMES KV=4000 Evaluation

Configuration:

    Model:
    LLaVA-OneVision-Qwen2-0.5B

    Dataset:
    StreamingBench subset

    Sampling:
    0.5 FPS

    Streaming:
    Enabled

    KV budget:
    4000

Evaluation completed successfully.

Output schema:

    video_id
    question
    choices
    answer
    correct_choice
    pred_answer
    pred_choice
    qa_acc
    task

The automatic evaluator was successfully integrated.

------------------------------------------------------------------------

# 6. Error Analysis

Incorrect predictions were extracted and analyzed.

Observed failure patterns:

## 6.1 Temporal / Event Memory

Examples:

-   action ordering
-   sequence reconstruction
-   remembering previous events

Observed behavior:

The model often recognized objects correctly but confused temporal
order.

------------------------------------------------------------------------

## 6.2 Fine-Grained Attribute Recognition

Examples:

-   colors
-   small object details
-   visual attributes

Observed behavior:

Compression can remove fine visual details.

------------------------------------------------------------------------

## 6.3 Counting

Examples:

-   number of blocks
-   number of colors
-   repeated objects

Observed behavior:

Long-range object tracking is challenging after compression.

------------------------------------------------------------------------

## 6.4 Spatial Understanding

Examples:

-   front/back relations
-   relative position
-   object placement

Observed behavior:

Spatial relationships are more fragile under memory reduction.

------------------------------------------------------------------------

# 7. Current Research Results

Completed:

  Experiment                 Status
  -------------------------- ----------
  Model loading              Complete
  HERMES inference           Complete
  KV compression             Complete
  RoPE compatibility fixes   Complete
  StreamingBench subset      Complete
  KV=4000 evaluation         Complete
  Error taxonomy             Complete

------------------------------------------------------------------------

# 8. Remaining Experiments

## Experiment 1 --- KV=50000 Baseline

Purpose:

Measure performance without compression.

Comparison:

    No compression
    vs
    HERMES compression

------------------------------------------------------------------------

## Experiment 2 --- KV Budget Ablation

Planned table:

  KV Budget   Accuracy   Memory
  ----------- ---------- ----------
  500         TBD        TBD
  4000        Complete   Complete
  6000        Complete   TBD
  50000       Pending    Pending

------------------------------------------------------------------------

## Experiment 3 --- Larger Benchmark

Current:

    10 videos
    50 questions

Future:

    50 videos
    250 questions

or:

    498 videos
    2490 questions

------------------------------------------------------------------------

# 9. Final Report Structure

Planned final document:

1.  Introduction
2.  HERMES architecture overview
3.  Reproduction environment
4.  Implementation challenges
5.  Engineering fixes
6.  Experimental setup
7.  KV-cache ablation study
8.  Accuracy-memory trade-off
9.  Error analysis
10. Limitations
11. Future improvements

------------------------------------------------------------------------

# 10. Notes

This document is a living research log.

Future updates should append:

-   new experiments
-   additional baselines
-   plots
-   tables
-   conclusions
