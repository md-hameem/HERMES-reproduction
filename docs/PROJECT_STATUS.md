# HERMES project status

Updated 19 November 2026 after independent local verification of the five-stage AutoDL evidence. This page is the current project index. Dated entries in the research log and audit retain their historical context.

All five planned execution stages have completed. The project has progressed from a 50-video reproduction to an audited full StreamingBench real-time visual benchmark comparison on 498 videos and 2,495 questions per method. The latest GPU recorded in the evidence is an RTX 4090 with 24,564 MiB. Earlier runs used the hardware recorded in the October evidence.

## Main result

| Full real-time visual benchmark | Correct | Questions | Accuracy |
|---|---:|---:|---:|
| HERMES 4K | 1,555 | 2,495 | 62.3246% |
| FIFO 4K | 1,546 | 2,495 | 61.9639% |

The observed HERMES difference is +0.3607 percentage points. The 95% video-cluster bootstrap interval is −0.4810 to +1.1628 points using 10,000 resamples. HERMES alone answers 58 questions correctly and FIFO alone answers 49 correctly. The exploratory exact discordance test gives p = 0.43944. These measurements leave an accuracy advantage and equivalence unestablished for this model and budget. The evidence covers the benchmark's real-time visual annotations. Other StreamingBench tracks and other models require separate evaluation.

## Five completed stages

| Stage | Verified outcome |
|---|---|
| Real-video attention | One 16-frame chunk, 3,136 visual tokens, 24 layers and three pseudo-queries. Current versus native attention has mean cosine 0.5593. Mean retained-token Jaccard is 0.6503 at a constraining 256-token budget. The scorer changes which tokens are retained. QA effects of a corrected scorer remain unmeasured. |
| Generalization, samples 251–300 | HERMES 4K 59.2%, FIFO 4K 58.0%, HERMES 6K 57.6%, FIFO 6K 57.2%, HERMES 50K 56.4%. All five runs contain 250 questions. The 4K paired difference is +1.2 points with a clustered interval of −1.2 to +3.6. |
| Component ablations, samples 251–300 | No smoothing 59.2%, no summary 59.2%, uniform layers 58.4%, no position reindexing 57.6%. Full HERMES scores 59.2%. All four paired intervals include zero. Equal aggregate accuracy can still include changed predictions. |
| Interleaved efficiency | Ten videos, three repetitions and two methods produce 60 video runs. Median video wall time is 165.883 s for HERMES and 164.079 s for FIFO. Total compression-path time is 54.872 s versus 2.227 s over 30 runs each. Peak allocated memory is approximately 4.689 GiB for both. |
| Full real-time visual benchmark | Ten shards per method cover all 498 videos and 2,495 questions. Samples 201–250 reuse the audited October originals and 251–300 reuse the generalization stage. The other eight shards were run for both methods. |

Timing scopes are nested. Pruning is included within pseudo-forward time, which is included within the HERMES compression path. Adding those times would double count work. Whole-video wall time includes loading, decoding, CPU work and checkpoint validation within the analyzer measurement. The answer-event table measures the complete answering function. Its median is 0.13885 s for HERMES and 0.13693 s for FIFO. The efficiency summary's median-of-video-medians uses a different aggregation. Three repetitions reuse the same ten videos.

## Evidence organization

The GitHub copy stores one canonical five-stage ZIP alongside the executed work file and compact published results. The local extraction is ignored by Git. Notebook workflows are organized by environment and the research log, figures and literature are under `docs/`. [Repository README](../README.md) · [Notebook index](../notebooks/README.md) · [Evidence index](../evidence/README.md)

| Location | Role |
|---|---|
| `evidence/archives/HERMES_AutoDL_five_stage_evidence.zip` | Original uploaded five-stage archive, 2,921 manifest payloads |
| `evidence/local/five_stage/` | Local extraction of the archive, excluded from Git to avoid a duplicate payload |
| `results/five_stage/` | Published summary tables, report, figures and provenance for GitHub readers |
| `evidence/autodl_2026-10-25/` | Original AutoDL results, logs and derived analyses with historical internal paths preserved |
| `notebooks/executed/2026-11-18_hermes-five-stage-autodl.ipynb` | Uploaded executed work file, renamed from `hermes-five-stage-autodl-cache-fix (1).ipynb` with identical bytes |
| `notebooks/autodl/hermes-five-stage-autodl-cache-fix.ipynb` | Maintained workflow with permanent shard-count, CSV text and duplicate-option validator fixes |
| `evidence/autodl_2026-10-28/` | Earlier October evidence package |
| `evidence/archives/historical_evidence_supplement.zip` | Three recovered October payloads with hashes matching the historical manifest |
| `scripts/validate_five_stage_evidence.py` | CPU audit of archive integrity, annotations, predictions, checkpoints, statistics, timing events and retained-token sets |
| `evidence/audits/five_stage_local_audit.json` | Machine-readable outcome of the local audit |

The uploaded ZIP contains Python experiment code, JSON annotations, CSV predictions and Markdown analyses. It contains no R scripts.

The five-stage archive SHA-256 is `b8bc4cf8a3b28e6854b2e76b8c0239b12d3b5598ee1a07dafc03d0326a900518`. The executed notebook SHA-256 is `90174597f4132f3b116f3ab1f0d9410b92f655d9017d39b579c741ca1c3e0745`. Both match the supplied files and the archive hash printed in the notebook. The original archive's embedded notebook is an earlier source snapshot. The uploaded executed notebook is the execution record.

Run the current local audit from the project directory with Python, NumPy, pandas and SciPy installed.

```bash
python scripts/validate_five_stage_evidence.py
```

The audit verified every archived payload, all 53 earlier manifest payloads against the existing October evidence, 25 new complete runs, 1,246 video checkpoints and their timing totals, 18 master accuracy rows, task tables, all paired comparisons, 60 interleaved timing runs, and the attention/selection summaries. Run-completion logs agree with their saved completion records. Saved source hashes match the checkout after newline normalization. This is an audit of saved measurements.

## Recovery history and interpretation limits

The executed run required live notebook and runner fixes. Shard question counts are 250, 252, 253, 250, 250, 250, 240, 250, 250 and 250. Samples 301–350 contain 48 annotated videos, with samples 316 and 332 absent. Video IDs differ from sample IDs, so shard membership follows the sample ID in the source video path.

Numeric-looking gold answers require reading prediction CSVs as text. Sample 463, question index 4, repeats the gold text in options A and C. The validator now follows inference's existing first-match gold convention, A. The three runner hashes recorded across sessions reconstruct exactly from the original embedded runner, the CSV text fix, and the duplicate-option fix. They change validation rather than inference. The working notebook sources omit some patches applied through notebook history, so the saved execution record does not demonstrate a fresh Run All replay.

The full benchmark combines October results on samples 201–250 with November measurements for the other shards. Its primary accuracy comparison therefore uses multiple sessions and GPU types. The dedicated interleaved efficiency comparison uses its own matched session. The historical 60.4% memory reduction compares the October 4K and 50K HERMES runs. Both small-budget methods use approximately the same memory in the new matched comparison.

## Remaining work

1. Complete thesis synthesis around the audited reproduction, component diagnostics and full-benchmark result, including the null accuracy finding.
2. Measure a corrected-attention QA treatment if pursuing a claim about the scorer's effect on answer quality. Current attention results establish numerical and selection differences on one video chunk.
3. Establish a matched native LLaVA baseline on a feasible, explicitly defined context range if making claims against the base model. Current 50K results use the HERMES pipeline.
4. Verify a clean replay of the maintained notebook. The existing evidence establishes a recovered completed run.
5. Decide with the advisor whether the thesis needs an original extension or further models and workloads. EDMI remains a proposal with no measured implementation in this evidence.

All three historical archival omissions have been recovered in the supplemental ZIP with their original hashes. The old bundles still document the omissions. Raw model weights, videos and full attention tensors are outside this evidence package. Attention tables and selection sets can be checked locally, while rerunning the GPU computation requires the pinned assets and environment.

Related documents are the detailed research log, the dated technical audit and the comprehensive research report. [Research log](research_log.md) · [Technical audit](HERMES_Reproduction_Audit_and_Next_Research_Plan_2026-10-09.md) · [Technical report](HERMES_Comprehensive_Technical_Research_Report.md)
