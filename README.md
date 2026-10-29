<h1 align="center">
  <img src="./asset/logo.png" width="40" alt="logo"> HERMES
</h1>
<p align="center">
  <b>[ACL 2026] KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding</b>
</p>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://hermes-streaming.github.io/)
[![Paper](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2601.14724)
[![HF Paper](https://img.shields.io/badge/Dataset-HuggingFace-yellow)](https://huggingface.co/papers/2601.14724)

</div>

## 🔥 News
- **[2026.04.06]** HERMES is accepted to ACL 2026 Main 🎉
- **[2026.03.23]** Full code released!
- **[2025.01.23]** HERMES reached **#3 Paper of the day** on [Hugging Face Daily Papers](https://huggingface.co/papers/2601.14724)!
- **[2025.01.21]** HERMES is available on [arXiv](https://arxiv.org/abs/2601.14724).

---

## 📋 Reproduction Study

This fork contains an independent **reproduction, validation, and control study** of HERMES on Kaggle Tesla T4 and AutoDL RTX 4080-family infrastructure.

### Overview

| | |
|---|---|
| **Environments** | Kaggle · single Tesla T4; AutoDL · single RTX 4080 SUPER or RTX 4080, depending on session |
| **Model** | LLaVA-OneVision-Qwen2-0.5B |
| **Benchmark** | StreamingBench samples 201–250 (50 videos / 250 questions) |
| **KV Budgets Tested** | 500, 1K, 2K, 4K, 6K, 50K |
| **Canonical original 50-video workflow** | [`hermes-v2-autodl.ipynb`](./hermes-v2-autodl.ipynb) |
| **Final control workflow** | [`hermes-final-autodl.ipynb`](./hermes-final-autodl.ipynb) |
| **Status** | Two preserved HERMES generations plus same-hardware repeats, attention differential, and FIFO controls; component ablations and broader shards pending |

### Original Validated AutoDL Run — 25 October 2026

| Configuration | Correct | Accuracy | Total runtime |
|---|---:|---:|---:|
| **HERMES KV=4000** | **144/250** | **57.6%** | **28.17 min** |
| **HERMES KV=6000** | **144/250** | **57.6%** | **28.26 min** |
| HERMES KV=50000 (control) | 138/250 | 55.2% | 31.05 min |

Each run completed all 50 videos and 250 questions. The executed notebook validated question order, choices, gold labels, source revision, and output structure. Runtime is total session time per configuration on an RTX 4080 SUPER; it is not directly comparable with Kaggle runtime.

### AutoDL Efficiency Evidence

The preserved logs were re-parsed with [`scripts/extract_efficiency.py`](./scripts/extract_efficiency.py). All budgets processed the same 829 chunks / 11,434 frames and produced 250 TTFT measurements. Allocated memory is in GiB; outer elapsed time is coarse whole-second video-loop progress, while total runtime includes startup and other notebook subprocess overhead.

| KV budget | Accuracy | Peak allocated GiB | Median TTFT (s) | P95 TTFT (s) | Max cache | Compression events | Outer elapsed | Total runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 57.6% | **4.689** | **0.0408** | 0.0489 | **4,013** | 770 | 28.00 min | 28.17 min |
| 6000 | 57.6% | 4.735 | 0.0406 | 0.0459 | 6,013 | 732 | 28.17 min | 28.26 min |
| 50000 (control) | 55.2% | 11.836 | 0.0519 | 0.0882 | 50,013 | 79 | 30.95 min | 31.05 min |

Relative to the 50K control, 4K has **60.4% lower peak allocated memory**, **21.4% lower median logged TTFT**, **9.5% lower outer elapsed time**, and **92.0% lower maximum answering-cache length**. These are derived from existing AutoDL logs, not new GPU measurements. Reserved memory and isolated encoding/compression durations were not logged; TTFT excludes ingestion and pseudo-query compression and is not end-to-end latency. The process-lifetime allocated-memory counter was not reset per video. The nearly identical 4K/6K TTFT values do not support a monotonic latency claim.

Reproduce the tables with `python scripts/extract_efficiency.py`. Outputs are in [`exports/efficiency/`](./exports/efficiency/): the [summary](./exports/efficiency/autodl_50_efficiency_summary.csv), [750 question rows](./exports/efficiency/autodl_50_question_metrics.csv), [150 video rows](./exports/efficiency/autodl_50_video_metrics.csv), and [hash-checked provenance](./exports/efficiency/autodl_50_efficiency_provenance.json).

### Final AutoDL Controls — 28 October 2026

`hermes-final-autodl.ipynb` reused the hash-verified original predictions and added same-hardware HERMES repeats, a differential test of pseudo-query attention, and budget-matched FIFO eviction controls. All seven raw prediction files independently validated for row count, question identity, gold alignment, and parsed-choice agreement.

| Method | Correct | Accuracy | Peak allocated GiB | Median TTFT (s) | Max cache | Elapsed |
|---|---:|---:|---:|---:|---:|---:|
| HERMES 4K, original | 144/250 | 57.6% | 4.689 | 0.0408 | 4,013 | 28.17 min |
| HERMES 6K, original | 144/250 | 57.6% | 4.735 | 0.0406 | 6,013 | 28.26 min |
| HERMES 50K control | 138/250 | 55.2% | 11.836 | 0.0519 | 50,013 | 31.05 min |
| HERMES 4K repeat | 143/250 | 57.2% | 4.689 | 0.0506 | 4,013 | 31.44 min |
| HERMES 6K repeat | 143/250 | 57.2% | 4.735 | 0.0493 | 6,013 | 31.35 min |
| FIFO 4K | 144/250 | 57.6% | 4.689 | 0.0505 | 4,013 | 29.39 min |
| FIFO 6K | 144/250 | 57.6% | 4.734 | 0.0486 | 6,013 | 29.30 min |

Key control findings:

- **Repeatability:** 4K changed four choices and three correctness outcomes; 6K changed five choices and three correctness outcomes. Same-environment run-to-run nondeterminism is therefore demonstrated, with both repeats scoring 143/250.
- **FIFO parity:** FIFO 4K versus HERMES 4K had six wins each (exact paired p = 1.000); FIFO 6K versus HERMES 6K had five wins each (p = 1.000). HERMES is not shown to outperform simple newest-token FIFO retention on this shard.
- **Versus 50K:** the +2.4-point HERMES/FIFO advantage is not statistically conclusive (exact paired p ≈ 0.210; video-cluster intervals include negative differences).
- **Attention differential:** the manual pseudo-query path remains an approximation. Mean cosine similarity is 0.637 against native attention and 0.724 for the causal-masked frozen-hidden variant. The reported top-13 ranking overlap is uninformative because the test had only 13 past tokens.
- **Efficiency stability:** budget-related peak allocated memory and maximum cache reductions are consistent. Session-to-session TTFT varies, so latency should not be overinterpreted from one process run.

The extracted evidence directory is [`HERMES_AutoDL_final_evidence/`](./HERMES_AutoDL_final_evidence/); its machine-readable summary is [`observed_research_summary.json`](./HERMES_AutoDL_final_evidence/exports/final/observed_research_summary.json). Run the independent audit with:

```bash
python scripts/validate_final_evidence.py
```

The validator verifies 53 payload files against the final SHA-256 manifest, recomputes all seven accuracy and efficiency tables, reproduces paired statistics, checks attention artifacts, and inspects the executed notebook. It reports three archival warnings: the final bundle omits `exports/figures/accuracy_50.png`, `exports/figures/paired_differences_50.png`, and `exports/hermes_50_by_task.csv` from the embedded historical manifest. These omissions do not affect the new predictions, but the archive should be supplemented before being treated as fully self-contained.

### Earlier Kaggle Accuracy and Efficiency Reference

These are the previously reported Tesla T4 measurements. The separate AutoDL logs permit a fresh AutoDL efficiency reanalysis, but they do not independently revalidate these Kaggle metrics. Treat this table as historical logged results, not AutoDL measurements.

| Configuration | Accuracy | 95% Wilson CI | Max GPU Memory | Median TTFT | Max Cache Length |
|---|---:|---:|---:|---:|---:|
| **HERMES KV=4000** | **57.2%** | 51.00–63.18% | **4.73 GB** | **0.058 s** | 4,013 |
| **HERMES KV=6000** | **57.2%** | 51.00–63.18% | 4.80 GB | 0.060 s | 6,013 |
| HERMES KV=50000 (control) | 55.2% | 49.00–61.24% | 11.87 GB | 0.112 s | 50,013 |

**Historical Kaggle result:** HERMES at KV=4000 achieved comparable accuracy to the near-uncompressed 50K control while reducing reported peak GPU memory by **~60%** and median TTFT by **~48%**. These metrics exclude important end-to-end costs and are not hardware-independent claims.

### Kaggle–AutoDL Prediction Comparison

All 750 AutoDL rows were aligned to Kaggle rows by `(video_id, qa_index)`, with question, options, answer, correct choice, and task checked for equality.

| KV budget | Kaggle → AutoDL accuracy | Choice changes | Correctness changes | AutoDL-only correct | Kaggle-only correct |
|---|---:|---:|---:|---:|---:|
| 4000 | 57.2% → 57.6% | 3 | 3 | 2 | 1 |
| 6000 | 57.2% → 57.6% | 10 | 9 | 5 | 4 |
| 50000 | 55.2% → 55.2% | 0 | 0 | 0 | 0 |

The net one-question gain at 4K and 6K is **not a single changed question**. The detailed 13-row choice-discordance export is [`exports/kaggle_autodl_50_discordant_predictions.csv`](./exports/kaggle_autodl_50_discordant_predictions.csv), and aggregate counts are in [`exports/kaggle_autodl_50_comparison_summary.csv`](./exports/kaggle_autodl_50_comparison_summary.csv). Cross-GPU numerical nondeterminism is plausible, but it is not proven without a repeated identical-environment run and a fully matched Kaggle model snapshot.

### AutoDL Provenance and Evidence

| Component | Preserved identity |
|---|---|
| GPU | NVIDIA GeForce RTX 4080 SUPER, 32,760 MiB; driver 595.71.05 |
| Inference runtime | Python 3.12 · PyTorch 2.10.0+cu128 · Transformers 4.45.0.dev0 |
| HERMES revision | `8d699b16a6bedb9086c1b39ec4253c6a1d1ce789` |
| Transformers revision | `66bc4def9505fa7c7fe4aa7a248c34a026bb552b` |
| Model snapshot | `74dd0bf867a4cda7950c17663794267c60cf4b40` |
| Executed notebook SHA-256 | `8a2bd96900d2ecc6ab55a1674f69d9b1fd29fe7d0bfb23a8dd1646ac2e64d739` |
| Original AutoDL archive SHA-256 | `ae13e9f66f2d9e8a7202d0ebb64bf761c173aa1d667cdce626dd2e72cfd01120` |
| Locally repacked archive SHA-256 | `23ffbd2612c8424fcf2c295bbd3018f2a0b9560e102042235f9896c78f12e183` |

The [`HERMES_AutoDL_50_evidence.zip`](./HERMES_AutoDL_50_evidence.zip) archive was repacked locally from files that all passed the original [`exports/manifest_sha256.csv`](./exports/manifest_sha256.csv) checks. It has the same payload and byte size as the original archive but a different ZIP byte stream. [`exports/kaggle_autodl_50_comparison_provenance.json`](./exports/kaggle_autodl_50_comparison_provenance.json) records source-CSV hashes, comparison hashes, notebook identity, and both archive hashes.

### Completed Work

- Full HERMES inference pipeline on Kaggle T4 and AutoDL RTX 4080-family GPUs
- Transformers/Qwen2 RoPE compatibility fixes and pinned source checks
- Custom-video smoke test with KV compression validation
- 10-video six-budget ablation (KV=500 to KV=50K)
- Two preserved 50-video result generations (KV=4K, 6K, 50K)
- Same-hardware AutoDL 4K/6K repeat runs measuring prediction instability
- Differential test of manual pseudo-query attention against native transformer attention
- Budget-matched FIFO eviction controls at 4K and 6K
- Independent final-evidence validator with hash, prediction, statistics, efficiency, control, and notebook checks
- Task-level accuracy analysis and exploratory error taxonomy
- Paired prediction exports, exact discordance sensitivity tests, and video-cluster bootstrap intervals
- Kaggle–AutoDL positional prediction comparison and discordance case export
- Reproducibility packaging with raw CSVs, logs, source/model revisions, and SHA-256 manifests
- Reproducible AutoDL efficiency extraction with question/video tables and provenance hashes

### Next Experiments

1. Run one-at-a-time HERMES component ablations for cross-layer smoothing, summary tokens, differentiated shallow/middle/deep retention, and positional re-indexing.
2. Supplement or repackage the final evidence archive with the three historical files missing from its embedded preserved manifest.
3. Evaluate a non-overlapping 50-video shard before resumable shard-wise evaluation of all 498 videos.
4. Develop research extensions only after the component-ablation and generalization evidence is established.

The native full-prefix baseline remains incomplete and excluded where memory or context limits are exceeded. Inputs must not be silently truncated to force a comparison.

### Quick Start

For the validated 50-video workflow, use [`hermes-v2-autodl.ipynb`](./hermes-v2-autodl.ipynb) and run its cells in order on AutoDL.

The legacy Kaggle pilot is:

```bash
git clone https://github.com/md-hameem/HERMES-reproduction.git
cd HERMES-reproduction
bash scripts/run_kaggle.sh
```

The legacy script requires local video-path and annotation-subset setup; it is not the canonical 50-video reproduction path.

### Documentation

| Document | Description |
|---|---|
| [`research_log.md`](./research_log.md) | Complete research log (31 sections) — development timeline, all experimental results, mechanistic analysis, audit findings, and phased research plan |
| [`CHANGELOG.md`](./CHANGELOG.md) | Dated project changelog for major reproduction, evidence, control, and documentation milestones |
| [`docs/HERMES_Comprehensive_Technical_Research_Report.md`](./docs/HERMES_Comprehensive_Technical_Research_Report.md) | Literature review, deep HERMES mechanistic analysis, cross-paper comparisons, field evolution roadmap, research gaps, EDMI proposal, and 28 October control update |
| [`docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md`](./docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md) | Independent reproduction audit with evidence grading (22 sections), risk assessment, AutoDL verification, efficiency evidence, final controls, and next experiments |

### Interpretation Guidelines

1. KV=50000 is a **near-uncompressed HERMES control**, not the original LLaVA baseline — it still executes through the HERMES pipeline and triggered 79 compression events on the Kaggle 50-video run.
2. The 50-video subset is a **convenience shard** (samples 201–250), not a random or stratified sample of the full benchmark.
3. Task distribution is heavily imbalanced (Causal + Prospective Reasoning = 80.4% of questions).
4. Accuracy differences should be interpreted alongside their confidence intervals and the clustered nature of 5 questions per video.
5. Kaggle used a Tesla T4; AutoDL used an RTX 4080 SUPER for the original run and an RTX 4080 for the final control run. Do not directly compare their runtimes as hardware-independent results.
6. Small cross-platform and same-platform prediction differences are now a quantified reproducibility limitation, not evidence of a semantic improvement.
7. The 28 October FIFO controls match HERMES aggregate accuracy on this shard; do not claim HERMES-specific selection benefits without component-ablation support.
8. The pseudo-query attention differential quantifies approximation error; it does not establish numerical equivalence to native transformer attention.
9. TTFT and elapsed time are session-specific. Peak allocated memory and maximum cache size are the more stable cross-run efficiency measurements.

### Experimental Plots

All experimental figures are stored in the [`img/`](./img/) directory and embedded in the research log. AutoDL accuracy and paired-difference figures are in [`exports/figures/`](./exports/figures/).

---

## 🛠️ Installation


For **LLaVA** model inference:
```bash
conda create -n hermes-llava python=3.12 -y
conda activate hermes-llava
pip install -r requirements_llava.txt
pip install flash-attn --no-build-isolation
```

For **Qwen2.5-VL** model inference:
```bash
conda create -n hermes-qwen python=3.12 -y
conda activate hermes-qwen
pip install -r requirements_qwen.txt
pip install flash-attn --no-build-isolation
```


## 📦 Preparation

### Model Preparation

Create a `models` directory and download the model weights from HuggingFace:

```bash
mkdir models
```

We support the following models (choose one or more):

| Model Family | Model | HuggingFace Link |
|:---:|:---:|:---:|
| LLaVA-OneVision | llava-onevision-qwen2-0.5b-ov-hf | [llava-hf/llava-onevision-qwen2-0.5b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-0.5b-ov-hf) |
| LLaVA-OneVision | llava-onevision-qwen2-7b-ov-hf | [llava-hf/llava-onevision-qwen2-7b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-7b-ov-hf) |
| LLaVA-OneVision | llava-onevision-qwen2-72b-ov-hf | [llava-hf/llava-onevision-qwen2-72b-ov-hf](https://huggingface.co/llava-hf/llava-onevision-qwen2-72b-ov-hf) |
| Qwen2.5-VL | Qwen2.5-VL-3B-Instruct | [Qwen/Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) |
| Qwen2.5-VL | Qwen2.5-VL-7B-Instruct | [Qwen/Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| Qwen2.5-VL | Qwen2.5-VL-32B-Instruct | [Qwen/Qwen2.5-VL-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct) |


### Data Preparation

Download the benchmark videos from their official sources and place them according to the paths specified in the annotation files:

**Streaming Benchmarks:**

| Benchmark | Video Path | Official Source |
|:---:|:---:|:---:|
| StreamingBench | `/data/streamingbench/videos/` | 🤗 [StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench) |
| OVO-Bench | `/data/ovobench/videos/` | 🤗 [OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench) |
| RVS-Ego | `/data/rvs/ego/videos/` | 🤗 [RVS](https://huggingface.co/datasets/Becomebright/RVS) |
| RVS-Movie | `/data/rvs/movie/videos/` | 🤗 [RVS](https://huggingface.co/datasets/Becomebright/RVS) |

**Offline Benchmarks:**

| Benchmark | Video Path | Official Source |
|:---:|:---:|:---:|
| VideoMME | `/data/videomme/videos/` | 🤗 [VideoMME](https://huggingface.co/datasets/lmms-lab/Video-MME) |
| MVBench | `/data/mvbench/videos/` | 🤗 [MVBench](https://huggingface.co/datasets/OpenGVLab/MVBench) |
| EgoSchema | `/data/egoschema/videos/` | 🤗 [EgoSchema](https://huggingface.co/datasets/lmms-lab/egoschema) |

The annotation JSON files contain the same information as officially provided, with formatting adjustments to adapt to our codebase.

After preparation, the project structure should look like this:

```
HERMES/
├── asset/
│   └── logo.png
├── data/
│   ├── egoschema/
│   │   ├── videos/
│   │   └── egoschema.json
│   ├── mvbench/
│   │   ├── videos/
│   │   └── mvbench.json
│   ├── ovobench/
│   │   ├── videos/
│   │   └── ovobench_realtime_backeward.json
│   ├── rvs/
│   │   ├── ego/
│   │   │   ├── videos/
│   │   │   └── ego4d_oe.json
│   │   └── movie/
│   │       ├── videos/
│   │       └── movienet_oe.json
│   ├── streamingbench/
│   │   ├── videos/
│   │   └── streamingbench_realtime.json
│   └── videomme/
│       ├── videos/
│       └── videomme.json
├── docs/                          # Reproduction study documents
│   ├── HERMES_Comprehensive_Technical_Research_Report.md
│   └── HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md
├── eval/
│   ├── eval_multiple_choice.py
│   └── eval_open_ended.py
├── img/                           # Experimental plots and figures
├── inference/
│   ├── abstract_hermes.py
│   ├── llavaov_hermes.py
│   ├── qwenvl_hermes.py
│   ├── reindex_1d.py
│   └── reindex_3d.py
├── models/
│   └── ...
├── scripts/
│   ├── download_subset.py
│   ├── run_infer.sh
│   └── run_kaggle.sh
├── video_qa/
│   ├── base.py
│   ├── hermes_vqa.py
│   └── run_infer.py
├── LICENSE
├── README.md
├── research_log.md                # Full reproduction research log
├── requirements_cpu.txt
├── requirements_kaggle.txt
├── requirements_llava.txt
└── requirements_qwen.txt
```


## 🚀 Inference

Simply run the inference script:

```bash
bash scripts/run_infer.sh
```

Here is the content of `scripts/run_infer.sh`:

```bash
export PYTHONPATH=$(cd "$(dirname "$0")/.." && pwd):$PYTHONPATH

num_chunks=8
model=llava_ov_7b
dataset=streamingbench

python video_qa/run_infer.py \
    --num_chunks $num_chunks \
    --model ${model} \
    --dataset ${dataset} \
    --sample_fps 0.5 \
    --kv_size 6000
```

**Arguments:**

| Argument | Description |
|:---|:---|
| `model` | Model to use. Options: `llava_ov_0.5b`, `llava_ov_7b`, `llava_ov_72b`, `qwen2.5_vl_3b`, `qwen2.5_vl_7b`, `qwen2.5_vl_32b` |
| `dataset` | Benchmark dataset. Options: `videomme`, `mvbench`, `egoschema`, `rvs_ego`, `rvs_movie`, `ovobench`, `streamingbench` |
| `num_chunks` | Number of parallel processes for evaluation, typically set to the number of GPUs |
| `sample_fps` | Frame sampling rate (frames per second) from the video |
| `kv_size` | Maximum KV cache size for HERMES hierarchical memory management |
| `only_eval` | If set, skip inference and only run evaluation on existing results |


## 📊 Evaluation

The evaluation scripts compute metrics on the inference results:

- **Multiple-choice benchmarks** (VideoMME, MVBench, EgoSchema, OVBench, StreamingBench) are evaluated by `eval/eval_multiple_choice.py`, which takes a subcommand as its first argument:

| Subcommand | Description | Used by |
|:---|:---|:---|
| `general` | Compute overall accuracy, task-specific breakdown (auto-detects OVBench / StreamingBench), and prediction error analysis | MVBench, OVBench, StreamingBench, VideoMME |
| `videomme` | Report accuracy broken down by video duration (short / medium / long) | VideoMME |
| `egoschema` | Generate EgoSchema submission CSV file | EgoSchema |

```bash
python eval/eval_multiple_choice.py general --results_path results/llava_ov_7b/streamingbench/fps0.5-kv6000/results.csv
```

- **Open-ended benchmarks** (RVS-Ego, RVS-Movie) are evaluated by `eval/eval_open_ended.py`, which uses GPT for answer scoring:

```bash
python eval/eval_open_ended.py \
    --pred_path results/llava_ov_7b/rvs_ego/fps0.5-kv6000/results.csv \
    --output_dir results/llava_ov_7b/rvs_ego/fps0.5-kv6000/tmp \
    --output_json results/llava_ov_7b/rvs_ego/fps0.5-kv6000/results.json
```


## 📧 Contact

For any questions regarding the paper or the technical implementation, please feel free to contact haowei.zhang123@gmail.com


## 🙏 Acknowledgements

Our codebase is built upon [ReKV](https://github.com/Becomebright/ReKV). We gratefully acknowledge their contributions to the community.


## 📝 Citation

If you find our work useful for research, please cite our paper and give us a precious star 😄:

```bibtex
@misc{zhang2026hermeskvcachehierarchical,
      title={HERMES: KV Cache as Hierarchical Memory for Efficient Streaming Video Understanding}, 
      author={Haowei Zhang and Shudong Yang and Jinlan Fu and See-Kiong Ng and Xipeng Qiu},
      year={2026},
      eprint={2601.14724},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.14724}, 
}
```
