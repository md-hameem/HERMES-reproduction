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

This fork contains an independent **reproduction, validation, and control study** of HERMES on Kaggle Tesla T4 and AutoDL RTX 4080-family and RTX 4090 infrastructure.

### Repository layout

```text
HERMES/
├── inference/             # HERMES model and KV-cache implementation
├── video_qa/              # Video ingestion and question-answering entry points
├── eval/                  # Benchmark evaluators
├── scripts/               # Inference launchers and evidence validators
├── data/                  # Annotations and smoke-test assets
├── notebooks/             # AutoDL, Kaggle, executed and historical notebooks
├── evidence/              # Dated evidence, original archives and local audits
├── results/               # Published five-stage summaries and historical Kaggle results
├── logs/                  # Historical Kaggle logs
├── docs/                  # Research writing, figures and related papers
├── paper/                 # Manuscript planning and claim/evidence tracking
├── asset/                 # Project logo
├── models/                # Local model snapshots, excluded from Git
├── requirements_*.txt
├── CHANGELOG.md
├── LICENSE
└── README.md
```

The original five-stage ZIP is stored once under `evidence/archives/`. Its full local extraction lives under `evidence/local/` and is excluded from Git. Reader-facing summaries and figures are published under `results/five_stage/`. Notebook workflows and evidence commands are indexed in their directories. [Notebooks](notebooks/README.md) · [Evidence](evidence/README.md)

### Overview

| | |
|---|---|
| **Environments** | Kaggle Tesla T4; AutoDL RTX 4080 SUPER / RTX 4080 for October and RTX 4090 for the five-stage study |
| **Model** | LLaVA-OneVision-Qwen2-0.5B |
| **Benchmark** | Full StreamingBench real-time visual annotations, 498 videos / 2,495 questions per method |
| **KV Budgets Tested** | 500, 1K, 2K, 4K, 6K, 50K |
| **Canonical original 50-video workflow** | [`notebooks/autodl/hermes-v2-autodl.ipynb`](notebooks/autodl/hermes-v2-autodl.ipynb) |
| **Final control workflow** | [`notebooks/autodl/hermes-final-autodl.ipynb`](notebooks/autodl/hermes-final-autodl.ipynb) |
| **Five-stage workflow** | [`notebooks/autodl/hermes-five-stage-autodl-cache-fix.ipynb`](notebooks/autodl/hermes-five-stage-autodl-cache-fix.ipynb) |
| **Executed work file** | [`2026-11-18_hermes-five-stage-autodl.ipynb`](notebooks/executed/2026-11-18_hermes-five-stage-autodl.ipynb) |
| **Status** | All five stages completed and independently audited on 19 November 2026 |

### Five-stage evidence verified on 19 November 2026

The completed AutoDL study includes real-video attention validation, generalization on samples 251–300, four component ablations, 60 interleaved timing runs and the full 498-video real-time visual benchmark.

| Primary full-benchmark comparison | Correct / questions | Accuracy |
|---|---:|---:|
| HERMES 4K | 1,555 / 2,495 | 62.3246% |
| FIFO 4K | 1,546 / 2,495 | 61.9639% |

The observed difference is +0.3607 percentage points with a 95% video-cluster interval of −0.4810 to +1.1628 points and an exploratory exact paired p = 0.43944. An accuracy advantage remains unestablished for this configuration. The generalization shard gives 59.2% for HERMES 4K and 58.0% for FIFO 4K. All four component-ablation intervals include zero.

The real-video diagnostic changes retained-token sets, with mean current-versus-native Jaccard 0.6503 across 24 layers. The later selected 40-question corrected-scorer treatment below found no correctness gain over HERMES. In the interleaved study, median video wall time is 165.883 s for HERMES and 164.079 s for FIFO, and peak allocated memory is approximately 4.689 GiB for both. Whole-video and compression-path timings use different scopes.

The archive audit checked 2,921 payload files, 25 new runs, 1,246 video checkpoints, raw timing events, full aggregation, paired statistics and attention selection sets. Full-benchmark accuracy reuses the October 201–250 evidence across sessions and hardware. The executed notebook required live validator and shard fixes. A fresh Run All replay of the maintained workflow remains unverified. The three older archival omissions have been recovered in a separate supplement with their original hashes.

```bash
python scripts/validate_five_stage_evidence.py
```

The current status page describes completed work, remaining research and the evidence layout. The original uploaded ZIP, published summaries and executed notebook are indexed there. [Project status](docs/PROJECT_STATUS.md) · [Local audit](evidence/audits/five_stage_local_audit.json) · [Archive report](results/five_stage/final_research_report.md)

### Research extension in progress

**Neurocomputing is the working publication target.** The [author-guide assessment](<docs/reports/Neurocomputing submission requirements and HERMES readiness.md>) records submission requirements; an original method with validated benefit remains pending. The [research gap map](<docs/reports/HERMES research gap map.md>) accounts for 51 local PDF files, with a [39-entry register](<docs/reports/HERMES gap register.csv>). Q1 is mandatory according to the user; the applicable institutional ranking/category/year and recognition rules remain to be checked before submission.

**Input-control diagnostic:** The audit covers 200 case records. Question-only scored 14/40, recent-four 25/40, recent-20 21/40, and native prefix 10/14. All historical baselines also scored 10/14 on those native identities. [Analysis](<docs/reports/2026-11-20 input diagnostic analysis and next decision.md>) · [Results](results/input_diagnostic_2026-11-20/) · [Maintained notebook](notebooks/autodl/hermes-input-diagnostic.ipynb) · [Instructions](docs/INPUT_DIAGNOSTIC.md)

**Context diagnostic completed:** [analysis](<docs/reports/2026-11-20 context diagnostic analysis and next decision.md>) verifies 84 case records/74 answers, all 26 comparable historical choices, and zero choice changes across 19 answered chunk-partition contrasts. Fixed-position opening-frame replacement recovers three full-prefix failures; another filler also causes a regression. Two analyst support probes answer correctly. Next: [stage attribution and an existing-method comparator](paper/neurocomputing/stage_attribution_protocol.md), with no original method or population gain established. [Results](results/context_diagnostic_2026-11-20/) · [Maintained notebook](notebooks/autodl/hermes-context-diagnostic.ipynb)

**Stage attribution complete:** On 40 development questions, hiding direct history during readout with the same contextual cache improved 21/40 to 27/40 (seven recoveries, one loss). Compact recent-4 scored 25/40. [Analysis](<docs/reports/2026-11-21 stage attribution analysis and next decision.md>) · [Results](results/stage_attribution_2026-11-21/) · [Executed notebook](notebooks/executed/2026-11-21_hermes-stage-attribution.ipynb).

**Matched-boundary control complete:** The construction effect survives after fixing the recent encoding groups, positions and older prefix. C matched scores 27/40 versus 25/40 for D, with two wins and zero losses. All 200 choices reproduce the stage-attribution run on the same questions. Next, test a physically short query cache against compact recent-four on source-separated footage. [Analysis and next decision](<docs/reports/2026-11-21 matched-boundary results and next decision.md>) · [Audited tables](results/matched_boundary_2026-11-21/).

**Short query-cache run complete:** Physical history removal preserved all 40 masked answers and reduced median query K/V from 46.09 to 9.34 MiB. Net median saving versus masked readout was 4.45 ms after extraction. Compact recent-four used the same cache bytes, scored 25/40 versus 27/40, and was 2.99 ms faster after extraction. The next step is source-separated confirmation. [Analysis](<docs/reports/2026-11-21 short query-cache results and next decision.md>) · [Audited tables](results/short_query_2026-11-21/).

**Construction history selection complete:** The greedy selector tied latest-two and attention mass at 26/40 and incurred extra cost versus latest-two. All-history short readout remains at 27/40. The selector is retired. [Analysis](<docs/reports/2026-11-21 construction selection results and next decision.md>) · [Audited tables](results/construction_selection_2026-11-21/).

**Frozen confirmation complete:** Compact recent-four scored 74/150, latest-two 72/150, and all-history 70/150 at the same answer-cache size. The primary paired difference was −2.67 percentage points (95% video-bootstrap interval −7.33 to +1.33). Keep compact recent-four as the baseline and review the 12 discordant questions before proposing another intervention. [Analysis](<docs/reports/2026-11-21 short query confirmation analysis.md>) · [Tables](results/short_query_confirmation_2026-11-21/).

**Visual review complete:** On the 12 selected disagreements, the contact sheets show recent evidence in nine cases. Original frames verify one older-only chart recovery and a benchmark score discrepancy. Excluding the discrepant case leaves all-history 2.01 points below compact recent-four. [Analysis](<docs/reports/2026-11-21 confirmation visual review analysis.md>).

The causal-scorer pilot completed on 20 November 2026: **8 annotated videos, 40 distinct questions, 120 answers**. FIFO, HERMES and fresh causal HERMES each score **24/40**. Fresh causal scoring passes the native-attention preflight but changes one wrong choice and zero correctness outcomes. FIFO and HERMES match their historical predictions exactly on these 40 questions. This is selected development evidence, with no general accuracy-advantage or equivalence claim. [Audited analysis and publication decision](<docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) · [Results](results/causal_pilot_2026-11-20/README.md)

**Target-binding experiment complete:** On 269 eligible questions, recent-only scored 156, averaging verifier scores scored 152, and using the weaker score scored 147. The primary paired estimate was −1.86 points (95% video-bootstrap interval −4.40 to +0.37). The method missed its advancement gate. [Analysis](<docs/reports/2026-11-22 target binding analysis.md>) · [Executed notebook](notebooks/executed/2026-11-22_hermes-target-binding.ipynb).

**Current research direction.** On TwinPast, automatic uniform and relevance selection each passed 3/7 pairs, while the history-control correction rule passed 1/7. The next test checks whether denser causal sampling puts the evidence needed for historical-state questions into the candidate pool. [Automatic-selection audit](<docs/reports/2026-11-23 twinpast automatic selection analysis.md>) · [Publication plan](paper/neurocomputing/submission_phase_plan.md)

Run the standard-library evidence audit with `python scripts/validate_causal_pilot_evidence.py`. The original [ZIP](evidence/archives/HERMES_causal_scorer_pilot_evidence.zip) and [executed notebook](notebooks/executed/2026-11-20_hermes-causal-scorer-pilot.ipynb) retain their uploaded bytes. Full extraction is ignored under `evidence/local/causal_pilot_2026-11-20/`. The [maintained notebook](notebooks/autodl/hermes-causal-scorer-pilot.ipynb) includes later heartbeat/logging changes; the completed evidence comes from its pre-heartbeat runner version. Shared feature work and replay timing are separate; the measured post-model-load phase was 327.028 seconds, not total container rental time.

### Original Validated AutoDL Run — 25 October 2026

| Configuration | Correct | Accuracy | Total runtime |
|---|---:|---:|---:|
| **HERMES KV=4000** | **144/250** | **57.6%** | **28.17 min** |
| **HERMES KV=6000** | **144/250** | **57.6%** | **28.26 min** |
| HERMES KV=50000 (control) | 138/250 | 55.2% | 31.05 min |

Each run completed all 50 videos and 250 questions. The executed notebook validated question order, choices, gold labels, source revision, and output structure. Runtime is total session time per configuration on an RTX 4080 SUPER; it is not directly comparable with Kaggle runtime.

### AutoDL Efficiency Evidence

The preserved logs were re-parsed with [`scripts/extract_efficiency.py`](scripts/extract_efficiency.py). All budgets processed the same 829 chunks / 11,434 frames and produced 250 TTFT measurements. Allocated memory is in GiB; outer elapsed time is coarse whole-second video-loop progress, while total runtime includes startup and other notebook subprocess overhead.

| KV budget | Accuracy | Peak allocated GiB | Median TTFT (s) | P95 TTFT (s) | Max cache | Compression events | Outer elapsed | Total runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4000 | 57.6% | **4.689** | **0.0408** | 0.0489 | **4,013** | 770 | 28.00 min | 28.17 min |
| 6000 | 57.6% | 4.735 | 0.0406 | 0.0459 | 6,013 | 732 | 28.17 min | 28.26 min |
| 50000 (control) | 55.2% | 11.836 | 0.0519 | 0.0882 | 50,013 | 79 | 30.95 min | 31.05 min |

Relative to the 50K control, 4K has **60.4% lower peak allocated memory**, **21.4% lower median logged TTFT**, **9.5% lower outer elapsed time**, and **92.0% lower maximum answering-cache length**. These are derived from existing AutoDL logs, not new GPU measurements. Reserved memory and isolated encoding/compression durations were not logged; TTFT excludes ingestion and pseudo-query compression and is not end-to-end latency. The process-lifetime allocated-memory counter was not reset per video. The nearly identical 4K/6K TTFT values do not support a monotonic latency claim.

Reproduce the tables with `python scripts/extract_efficiency.py`. Outputs are in [`evidence/autodl_2026-10-25/exports/efficiency/`](evidence/autodl_2026-10-25/exports/efficiency/): the [summary](evidence/autodl_2026-10-25/exports/efficiency/autodl_50_efficiency_summary.csv), [750 question rows](evidence/autodl_2026-10-25/exports/efficiency/autodl_50_question_metrics.csv), [150 video rows](evidence/autodl_2026-10-25/exports/efficiency/autodl_50_video_metrics.csv), and [hash-checked provenance](evidence/autodl_2026-10-25/exports/efficiency/autodl_50_efficiency_provenance.json).

### Final AutoDL Controls — 28 October 2026

`notebooks/autodl/hermes-final-autodl.ipynb` reused the hash-verified original predictions and added same-hardware HERMES repeats, a differential test of pseudo-query attention, and budget-matched FIFO eviction controls. All seven raw prediction files independently validated for row count, question identity, gold alignment, and parsed-choice agreement.

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

The extracted evidence directory is [`evidence/autodl_2026-10-28/`](evidence/autodl_2026-10-28/); its machine-readable summary is [`observed_research_summary.json`](evidence/autodl_2026-10-28/exports/final/observed_research_summary.json). Run the independent audit with:

```bash
python scripts/validate_final_evidence.py
```

The validator verifies 53 payload files against the final SHA-256 manifest, recomputes all seven accuracy and efficiency tables, reproduces paired statistics, checks attention artifacts, and inspects the executed notebook. It reports three archival warnings: the final bundle omits `evidence/autodl_2026-10-25/exports/figures/accuracy_50.png`, `evidence/autodl_2026-10-25/exports/figures/paired_differences_50.png`, and `evidence/autodl_2026-10-25/exports/hermes_50_by_task.csv` from the embedded historical manifest. The three omitted files are recovered with their original hashes in the supplemental archive. [Historical supplement](evidence/archives/historical_evidence_supplement.zip)

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

The net one-question gain at 4K and 6K is **not a single changed question**. The detailed 13-row choice-discordance export is [`evidence/autodl_2026-10-25/exports/kaggle_autodl_50_discordant_predictions.csv`](evidence/autodl_2026-10-25/exports/kaggle_autodl_50_discordant_predictions.csv), and aggregate counts are in [`evidence/autodl_2026-10-25/exports/kaggle_autodl_50_comparison_summary.csv`](evidence/autodl_2026-10-25/exports/kaggle_autodl_50_comparison_summary.csv). Cross-GPU numerical nondeterminism is plausible, but it is not proven without a repeated identical-environment run and a fully matched Kaggle model snapshot.

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

The historical `HERMES_AutoDL_50_evidence.zip` was repacked from files that passed the original manifest checks. That ZIP is absent from the current checkout. Its payload is preserved within the later evidence packages and the historical supplement. The recorded original and repacked hashes describe different ZIP byte streams. [Historical manifest](evidence/autodl_2026-10-25/exports/manifest_sha256.csv) [`evidence/autodl_2026-10-25/exports/kaggle_autodl_50_comparison_provenance.json`](evidence/autodl_2026-10-25/exports/kaggle_autodl_50_comparison_provenance.json) records source-CSV hashes, comparison hashes, notebook identity, and both archive hashes.

### Completed Work

- Five-stage AutoDL study with full 498-video HERMES/FIFO 4K comparison
- Real-video attention selection diagnostic, second-shard generalization, four component ablations and interleaved timing
- Independent five-stage evidence audit and recovery of the three historical archival omissions
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

### Remaining research and writing

1. Complete the thesis analysis around the audited full-benchmark and component results.
2. Evaluate a corrected-attention QA variant if attributing answer-quality effects to the pseudo-query scorer.
3. Establish a matched feasible native LLaVA control if making claims against the base model.
4. Verify clean replay of the maintained workflow and decide the scope of any original extension with the advisor.

The detailed status page records the current evidence limits and completed stages. [Project status](docs/PROJECT_STATUS.md)

### Quick Start

For the current five-stage workflow and its audit, start with [Project status](docs/PROJECT_STATUS.md). For the historical 50-video workflow, use [`notebooks/autodl/hermes-v2-autodl.ipynb`](notebooks/autodl/hermes-v2-autodl.ipynb) and run its cells in order on AutoDL.

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
| [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) | Current results, five completed stages, evidence index, limitations and remaining work |
| [`docs/research_log.md`](docs/research_log.md) | Complete research log (32 sections) — development timeline, all experimental results, mechanistic analysis, audit findings, and phased research plan |
| [`CHANGELOG.md`](CHANGELOG.md) | Dated project changelog for major reproduction, evidence, control, and documentation milestones |
| [`docs/HERMES_Comprehensive_Technical_Research_Report.md`](docs/HERMES_Comprehensive_Technical_Research_Report.md) | Literature review, deep HERMES mechanistic analysis, cross-paper comparisons, field evolution roadmap, research gaps, EDMI proposal, and November five-stage update |
| [`docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md`](docs/HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md) | Independent reproduction audit with evidence grading (23 sections), risk assessment, AutoDL verification, efficiency evidence, final controls, and next experiments |

### Interpretation Guidelines

1. KV=50000 is a **near-uncompressed HERMES control**, not the original LLaVA baseline — it still executes through the HERMES pipeline and triggered 79 compression events on the Kaggle 50-video run.
2. The 50-video subset is a **convenience shard** (samples 201–250), not a random or stratified sample of the full benchmark.
3. Task distribution is heavily imbalanced (Causal + Prospective Reasoning = 80.4% of questions).
4. Accuracy differences should be interpreted alongside their confidence intervals and clustering by video. Question counts vary across the full benchmark.
5. Kaggle used a Tesla T4; AutoDL used an RTX 4080 SUPER for the original run, an RTX 4080 for October controls and an RTX 4090 for the new five-stage study. Do not directly compare their runtimes as hardware-independent results.
6. Small cross-platform and same-platform prediction differences are now a quantified reproducibility limitation, not evidence of a semantic improvement.
7. The 28 October FIFO controls match HERMES aggregate accuracy on this shard; do not claim HERMES-specific selection benefits without component-ablation support.
8. The pseudo-query attention differential quantifies approximation error; it does not establish numerical equivalence to native transformer attention.
9. TTFT and elapsed time are session-specific. Peak allocated memory and maximum cache size are the more stable cross-run efficiency measurements.

### Experimental Plots

All experimental figures are stored in the [`docs/figures/`](docs/figures/) directory and embedded in the research log. AutoDL accuracy and paired-difference figures are in [`evidence/autodl_2026-10-25/exports/figures/`](evidence/autodl_2026-10-25/exports/figures/).

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

The annotation paths describe the runtime video locations. Downloaded video directories and model snapshots are excluded from Git. The repository layout is listed above.

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
