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

This fork contains an independent **reproduction, validation, and extension study** of HERMES, conducted on resource-constrained Kaggle infrastructure.

### Overview

| | |
|---|---|
| **Environment** | Kaggle · Tesla T4 GPU (single) |
| **Model** | LLaVA-OneVision-Qwen2-0.5B |
| **Benchmark** | StreamingBench (50-video / 250-question subset) |
| **KV Budgets Tested** | 500, 1K, 2K, 4K, 6K, 50K |
| **Status** | Subset reproduction complete · Full benchmark pending |

### Key Findings (50-Video Evaluation)

| Configuration | Accuracy | 95% Wilson CI | Max GPU Memory | Median TTFT | Max Cache Length |
|---|---:|---:|---:|---:|---:|
| **HERMES KV=4000** | **57.2%** | 51.00–63.18% | **4.73 GB** | **0.058 s** | 4,013 |
| **HERMES KV=6000** | **57.2%** | 51.00–63.18% | 4.80 GB | 0.060 s | 6,013 |
| HERMES KV=50000 (control) | 55.2% | 49.00–61.24% | 11.87 GB | 0.112 s | 50,013 |

**Main result:** HERMES at KV=4000 achieved comparable accuracy to the near-uncompressed 50K control while reducing peak GPU memory by **~60%** and median TTFT by **~48%**.

### Completed Work

- ✅ Full HERMES inference pipeline on Kaggle T4
- ✅ Transformers/Qwen2 RoPE compatibility fixes
- ✅ Custom-video smoke test with KV compression validation
- ✅ 10-video six-budget ablation (KV=500 to KV=50K)
- ✅ 50-video scaled benchmark (KV=4K, 6K, 50K)
- ✅ Task-level accuracy analysis (9 task categories)
- ✅ Error taxonomy (temporal, attribute, counting, spatial failures)
- ✅ Statistical analysis with Wilson CIs and paired comparisons
- ✅ Efficiency analysis (memory, TTFT, cache size)
- ✅ Comprehensive literature review and cross-paper comparison
- ✅ Reproduction audit with evidence grading
- ✅ Reproducibility packaging (CSVs, logs, source snapshot, SHA-256 manifest)

### Remaining

- ⏳ Non-HERMES LLaVA-OneVision native baseline
- ⏳ 50-video paired prediction analysis
- ⏳ HERMES component ablations (smoothing, summary tokens, RoPE re-indexing)
- ⏳ Full 498-video StreamingBench evaluation
- ⏳ Research extension development and evaluation

### Quick Start (Kaggle)

```bash
git clone https://github.com/md-hameem/HERMES-reproduction.git
cd HERMES-reproduction
bash scripts/run_kaggle.sh
```

This script automatically installs dependencies, downloads the model and a 10-video StreamingBench subset, runs HERMES inference with KV=4000, and evaluates accuracy.

### Documentation

| Document | Description |
|---|---|
| [`research_log.md`](./research_log.md) | Complete research log (27 sections) — development timeline, all experimental results, mechanistic analysis, audit findings, and phased research plan |
| [`docs/HERMES_Comprehensive_Technical_Research_Report.md`](./docs/HERMES_Comprehensive_Technical_Research_Report.md) | Literature review, deep HERMES mechanistic analysis, cross-paper comparisons, field evolution roadmap, research gaps, and EDMI proposal |
| [`docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md`](./docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md) | Independent reproduction audit with evidence grading, risk assessment, phased research plan, experimental design specifications, and deliverables |

### Interpretation Guidelines

1. KV=50000 is a **near-uncompressed HERMES control**, not the original LLaVA baseline — it still executes through the HERMES pipeline and triggered 79 compression events on the 50-video run
2. The 50-video subset is a **convenience shard** (samples 201–250), not a random or stratified sample of the full benchmark
3. Task distribution is heavily imbalanced (Causal + Prospective Reasoning = 80.4% of questions)
4. Accuracy differences should be interpreted alongside their confidence intervals and the clustered nature of 5 questions per video
5. The current experiments run on a Tesla T4, while published HERMES results use an A800 — hardware differences affect memory and latency comparisons

### Experimental Plots

All experimental figures are stored in the [`img/`](./img/) directory and embedded in the research log. These include KV-cache compression validation, GPU memory curves, accuracy vs. budget plots, TTFT analysis, task-level breakdowns, and error distributions.

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
