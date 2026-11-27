# Direction B reset and Gate 0 specification

28 November 2026

## Research decisions

The DROID window/span selector, terminal-attention pilots, and controlled filming protocol are permanently retired. Their historical records remain archived as negative evidence. Direction A, pursuing a standalone StreamingBench accuracy/recall fix, is retired. Direction B investigates physical KV efficiency at measured whole-stream VRAM and synchronized latency. Gate 0 is a profiler and archive-parity check. Gate 1 is on hold pending Gate 0 attribution.

## Verified benchmark findings

[VERIFIED] The 498-video real-time visual benchmark has 2,495 paired questions. HERMES 4K answered 1,555 correctly (62.32%) and FIFO 4K answered 1,546 (61.96%). The difference was +0.36 pp with a video-cluster 95% interval of −0.48 to +1.16 pp. Both methods were correct on 1,497 questions and both were wrong on 891. Their binary correctness outcomes agreed on 2,388/2,495 questions (95.71%). HERMES alone answered 58 correctly and FIFO alone answered 49. Episodic memory was tied at 58.07% with 20 directional wins per arm. The audit found no robust specific recall failure attributable to one retention policy. These results prompted Direction B. [Paired audit script](../../scripts/analyze_streamingbench_pairs.py) · [Audited full comparison](../../results/five_stage/full_benchmark_paired.csv)

[VERIFIED] The raw archived exports and annotations join by video ID and zero-based question index. The annotations contain a task and query timestamp for all questions, five blank required-ability labels, and no question-level cue interval. Duration coverage is 50 videos and 250 questions from samples 201–250, from the saved frames/FPS table. The duration analysis below is secondary. Exact McNemar p-values use discordant question pairs and are exploratory because questions share videos and subgroup tests are unadjusted. [Annotations](../../data/streamingbench/streamingbench_realtime.json) · [Duration table](../../results/kaggle/streamingbench_50_video_statistics.csv)

## Verified cache, timing, and RoPE audit

- [VERIFIED] At 0.5 FPS, the ingestion cutoff is the ceiling of half the question timestamp. Twenty frames at 196 visual tokens per frame plus the 13-slot prompt give 3,933 slots at 40 s. The next frame takes the unpruned cache to 4,129 slots and triggers eviction after its chunk. All 1,986 questions past 40 s were answered after eviction by both arms. Of these, 1,966 had a new eviction since the previous answer and 20 reused the previous sampled cutoff. The remaining 509 questions had no prior eviction. [Ingestion](../../video_qa/hermes_vqa.py)
- [VERIFIED] The contextual short arm physically retained 797 slots (13 prompt plus 784 visual), preserving the last four frames’ contextual RoPE positions. Compact recent-four encoded those frames afresh at contiguous positions. The physical path reproduced all 40 masked answers and reduced median query KV from 46.09 to 9.34 MiB. Its median 4.45 ms paired saving includes extraction against the masked prepared path. It still constructed up to 3,933 slots first. In the frozen 150-question comparison, compact scored 74, latest-two contextual 72, and all-history contextual 70, at equal query budgets. The audits had no streaming FIFO-Small control. The contextual short-cache arm is retired. [Development report](<2026-11-21 short query-cache results and next decision.md>) · [Confirmation report](<2026-11-21 short query confirmation analysis.md>)
- [VERIFIED] On the archived RTX 4090, the interleaved per-arm allocated maxima were 4,801.62 MiB (FIFO) and 4,802.00 MiB (HERMES), with reserved maxima of 5,044 and 5,048 MiB. The 36.75 MiB prepared-path KV saving is about 0.77% of those allocated peaks. The streaming 4,013-to-797 slot delta is 37.6875 MiB. A full 16-frame chunk took 0.481–0.483 s median and the allocated high-water mark first appeared at a 16-frame chunk in all 60 runs. Vision and chunk prefill were timed together, so the peak cannot be attributed to either operation individually. Mean encoding was 8.96–9.04 s per clip, QA 0.74 s, and compression 0.07 s for FIFO versus 1.83 s for HERMES. The latter includes pseudo-forward and pruning. Around 153–154 s elapsed before the first question in the interleaved runs. The initial load path combines file I/O, video decode and frame materialization. Legacy TTFT logs used the next forward boundary. [Interleaved summary](../../results/five_stage/efficiency_interleaved_summary.csv) · [Archive](../../evidence/archives/HERMES_AutoDL_five_stage_evidence.zip)
- [VERIFIED] Both 4K arms compacted on exactly the same 536 chunks across 530 question intervals. Streaming eviction preserves positions until the maximum logical position exceeds 31,744 on the archived configuration. The first reset-bearing question was at 323 s. At end_time >322 s, raw reset-minus-no-reset accuracy was −7.55 pp (HERMES) and −6.86 pp (FIFO). One question lacked a previous timestamp, leaving 925 rows across 426 videos. Per-arm logistic models used categorical ability and task, query-time bins starting at 322, 400, 500, 600 and 700 s, and inter-question gap bins starting at 0, 30, 60, 90, 120 and 180 s. Standardized differences with video-cluster 95% intervals were −4.25 [−13.42, +4.92] and −3.75 [−12.43, +4.94] pp. The 359-question overlap model gave +0.64 and −0.14 pp. A quadratic time/gap sensitivity retained a HERMES association of −9.20 [−17.45, −0.95] pp. The association is model-sensitive and observational. The 4K compaction question is closed as a standalone claim. [Position policy](../../inference/llavaov_hermes.py) · [Paired inventory](../../results/failure_pilot/paired_inventory.csv)

## Frozen Gate 0

| Required media filename | Video ID | Last question (s) | Archived 4K resets per arm |
|---|---:|---:|---:|
| sample_290_real.mp4 | 438 | 575 | 1 |
| sample_278_real.mp4 | 69 | 583 | 1 |
| sample_480_real.mp4 | 471 | 675 | 2 |
| sample_200_real.mp4 | 27 | 755 | 2 |
| sample_453_real.mp4 | 81 | 804 | 2 |
| sample_449_real.mp4 | 132 | 958 | 3 |

Use the archived LLaVA-OneVision-Qwen2-0.5B snapshot 74dd0bf867a4cda7950c17663794267c60cf4b40 and archived AutoDL environment. Mount renamed copies of the six hash-matched originals. On AutoDL, run from the repository root. The model directory must contain the pinned snapshot and the output directory must be new.

```bash
python -B scripts/profile_streaming_gate0.py --manifest
python -B scripts/profile_streaming_gate0.py \
  --media-dir /mnt/streamingbench-gate0 \
  --model-path models/llava-onevision-qwen2-0.5b-ov-hf \
  --scratch-dir /tmp \
  --output-dir results/gate0_2026-11-28
```

[Gate 0 runner](../../scripts/profile_streaming_gate0.py) · [CPU contract tests](../../scripts/test_profile_streaming_gate0.py) · [Pinned AutoDL setup](../../notebooks/autodl/hermes-five-stage-autodl-cache-fix.ipynb)

Stage one clip at a time at the archived sample stride. Run FIFO 4K, HERMES 4K, FIFO-Small 784 and HERMES-Small 784 as separate processes against the shared CPU frame array. Remove the scratch directory after that clip. Each arm runs separate warmed profiling and lean latency passes. Preprocessing includes H2D transfer. Vision is the tower forward. Projection, pooling, and instrumentation gaps are recorded as other. The runner records chunk frame count and the exact LLM visual embedding sequence length alongside vision/chunk timing and memory. QA latency starts before catch-up and ends at the first selected token or complete answer after CUDA synchronization.

Each clip/arm directory records hardware, framework/CUDA versions, dtype, media and benchmark archive hashes, parity, stage measurements, TTFT and completion latency, chunk KV lengths, and per-layer pre/post position vectors for actual compaction. A compaction timestamp is the last source-frame timestamp of the chunk that caused it. Pass requires all 60 archived 4K choices, complete stage and latency records, and valid physical cache and compaction traces for all 24 clip–arm runs. There is no minimum Small-arm efficiency gain in this gate.

[VERIFIED] CPU contract tests exercised archived input/parity loading, artifact validation on synthetic records for all 24 clip–arm combinations, rejection of a parity mismatch, the first-token callback, and mocked subprocess staging and cleanup. No Gate 0 GPU measurements have been collected. Gate 1 remains on hold.

## Full CPU paired accuracy tables

The following is the output of the verified CPU audit command.
Integrity PASS - 2,495 joined questions, 498 videos, 1,555 HERMES correct, 1,546 FIFO correct.

### Overall paired outcomes

| | FIFO right | FIFO wrong |
|---|---:|---:|
| HERMES right | 1,497 | 58 |
| HERMES wrong | 49 | 891 |

### Overall accuracy

| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All questions | 2,495 | 498 | 62.32% | 61.96% | +0.36 | 58 | 49 | 0.4394 |

### Task category

| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Action Recognition | 353 | 146 | 65.16% | 62.89% | +2.27 | 13 | 5 | 0.0963 |
| Attribute Recognition | 303 | 190 | 72.94% | 73.60% | -0.66 | 8 | 10 | 0.8145 |
| Causal Reasoning | 128 | 47 | 61.72% | 62.50% | -0.78 | 0 | 1 | 1.0000 |
| Clips Summarize | 317 | 83 | 71.92% | 72.87% | -0.95 | 2 | 5 | 0.4531 |
| Counting | 193 | 55 | 19.17% | 19.69% | -0.52 | 3 | 4 | 1.0000 |
| Event Understanding | 159 | 102 | 70.44% | 67.92% | +2.52 | 6 | 2 | 0.2891 |
| Object Recognition | 367 | 199 | 71.93% | 72.21% | -0.27 | 7 | 8 | 1.0000 |
| Prospective Reasoning | 108 | 28 | 61.11% | 59.26% | +1.85 | 6 | 4 | 0.7539 |
| Spatial Understanding | 246 | 109 | 54.07% | 54.07% | +0.00 | 4 | 4 | 1.0000 |
| Text-Rich Understanding | 321 | 154 | 57.63% | 56.70% | +0.93 | 9 | 6 | 0.6072 |

### Required ability

| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unspecified | 5 | 1 | 100.00% | 100.00% | +0.00 | 0 | 0 | 1.0000 |
| episodic memory | 997 | 332 | 58.07% | 58.07% | +0.00 | 20 | 20 | 1.0000 |
| working memory | 1,493 | 369 | 65.04% | 64.43% | +0.60 | 38 | 29 | 0.3284 |

### Question index within video

| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Early (first half rounded up) | 1,494 | 498 | 63.72% | 62.92% | +0.80 | 30 | 18 | 0.1114 |
| Late | 1,001 | 498 | 60.24% | 60.54% | -0.30 | 28 | 31 | 0.7948 |

### Question timestamp tertiles

| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <= 141 s | 832 | 481 | 64.42% | 63.82% | +0.60 | 9 | 4 | 0.2668 |
| 141 < t <= 340 s | 839 | 479 | 59.95% | 59.48% | +0.48 | 23 | 19 | 0.6440 |
| > 340 s | 824 | 404 | 62.62% | 62.62% | +0.00 | 26 | 26 | 1.0000 |

### High question timestamp

| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Top decile incl. ties (>= 550 s) | 250 | 181 | 62.00% | 64.40% | -2.40 | 6 | 12 | 0.2379 |
| Remaining (< 550 s) | 2,245 | 498 | 62.36% | 61.69% | +0.67 | 52 | 37 | 0.1374 |

### Exploratory video duration tertiles (50 videos, 250 questions)

| Group | Questions | Videos | HERMES | FIFO | Delta (pp) | HERMES only | FIFO only | Exact p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| <= 459.426 s | 85 | 17 | 50.59% | 52.94% | -2.35 | 0 | 2 | 0.5000 |
| 459.426 < t <= 645.44 s | 85 | 17 | 52.94% | 49.41% | +3.53 | 4 | 1 | 0.3750 |
| > 645.44 s | 80 | 16 | 70.00% | 71.25% | -1.25 | 2 | 3 | 1.0000 |
