# AutoDL input/capability diagnostic

Prepared 20 November 2026. [Notebook](../notebooks/autodl/hermes-input-diagnostic.ipynb) · [Scientific protocol](../paper/neurocomputing/next_diagnostic_protocol.md) · [Research plan](../paper/neurocomputing/research_execution_plan.md)

**Execution update:** the supplied reused-asset GPU run has completed and passed the [independent audit](../evidence/audits/input_diagnostic_2026-11-20_audit.json). It produced 134 answers and 66 explicit skips, including all 120 mandatory answers. The [analysis](<reports/2026-11-20 input diagnostic analysis and next decision.md>) now governs the next step: support review and controlled context/position tests. These instructions describe the maintained input-control workflow, not an implemented context-replacement treatment.

## Run now

Upload `hermes-input-diagnostic.ipynb` alone to AutoDL. Recommended: RTX 4090 / 24 GB, CUDA 12.8-compatible driver and at least 18 GiB free on `/root/autodl-tmp`. Choose **Restart Kernel and Run All**. Defaults are ready for the initial screen; no annotation or old ZIP upload is needed.

The default output directory is `/root/autodl-tmp/hermes_input_diagnostic`. The separate reuse candidate is `/root/autodl-tmp/hermes_causal_pilot`. If available, its environment must pass the exact pinned-version/CUDA probe; videos and old features must match the trusted historical hashes/metadata before reuse. Missing videos are downloaded, and missing/rejected features are regenerated in the new directory. The old experimental files are not overwritten. The notebook also works in a fresh container, with additional setup/download time.

The notebook embeds both runner sources and an audited data reference. It initializes a sparse Git checkout at `2a1385e0abc94debb64761b0465713c799a81c1c`, retries transient Git fetch/checkout failures and verifies source hashes. It uses the prior model revision, Python 3.12, PyTorch 2.10.0+cu128 and pinned Transformers source. Domestic package-index traffic bypasses AutoDL turbo proxies; GitHub/Hugging Face use the private inherited network environment. Proxy values are not printed.

## Planned answers

| Control | Scope | State/input |
|---|---|---|
| Question-only | 40 mandatory answers | Initial prompt and current question/options, no images |
| Recent 4 | 40 mandatory answers | Last up to four original sampled frames before query |
| Recent 20 | 40 mandatory answers | Last up to twenty original sampled frames before query |
| Native prefix | Up to 40 additional answers | Original sampled prefix, no eviction/summary; default cap 8,192 visual tokens plus separate model-context/VRAM checks |
| Verified support | Zero by default; optional reviewed cases | 1–20 human-verified pre-query sampled frames per annotated question |

Every question resets the model to its initial prompt. No previous generated answers, pseudo-query compression or retention policy are used. Native-prefix ingestion remains chronological with continuous positions; it never silently truncates a prefix. Recent/oracle selections receive fresh continuous positions in chronological selected-frame order. They change available input and history, so improvements do not isolate eviction causation.

Only the initial three controls constitute the mandatory **120-answer** matrix. All five arms get explicit case records: unused oracle cases and infeasible/disabled native cases are skipped with null correctness, not counted as wrong. A completed run contains 200 case records and at least 120 answered cases. Partial runs retain their coverage and checkpoints. The eight videos/40 questions are the already inspected development sample, not independent confirmation.

## Progress and resumption

Setup subprocesses print their stage and liveness every 30 seconds while otherwise silent. The GPU runner prints its stage, every question result, per-video elapsed/rough remaining estimates and a 30-second heartbeat. Liveness is distinct from successful work. From a separate AutoDL terminal:

```bash
tail -n 40 /root/autodl-tmp/hermes_input_diagnostic/logs/diagnostic.log
cat /root/autodl-tmp/hermes_input_diagnostic/progress.json
nvidia-smi
```

Before the runner starts, examine `logs/clone.log`, `logs/setup.log` or `logs/model_download.log` instead. Do not run a second concurrent job in the same WORK directory.

The default `MAX_HOURS=1.0` is a soft post-model experiment limit checked between whole video blocks after the first block. It excludes setup, downloads and model loading and can overrun by one block. It is a guard, not an estimate. Native prefixes may cost considerably more than the previous compressed replays. No fresh-container runtime has been measured for this notebook.

Restart Kernel and Run All with the same settings/WORK to resume. Every per-question checkpoint verifies configuration, feature identity, source question/gold/frame identities and result hash. Corrupt or incompatible checkpoints stop reuse. Change WORK when changing native settings, support annotations, embedded code or other experimental configuration. A rejected configuration does not replace the existing scientific completion record.

## Review and optional oracle

The evidence contains `review/support_review.csv` for all 40 questions and `review/index.html` linking complete original sampled-prefix pages for seven priority questions: 204/q0, 202/q0, 208/q0, 212/q3, 218/q3, 204/q3 and 219/q3. These pages contain all sampled frames available to those queries, rather than a sparse overview. Full source-video inspection may still be needed to determine whether sampling missed the cue. Review fields remain pending; contact sheets do not automatically annotate sufficient support.

For the initial run leave `SUPPORT_ITEMS=[]`. After reviewing evidence, create a new WORK and add reviewed objects with `sample_id`, `qa_index`, sorted unique `sampled_indices`, `verified=True` and nonempty `notes`. Use sampled-sequence indices, not source-frame numbers or seconds. The runner rejects unknown questions, duplicate cases, unreviewed/empty support, more than twenty frames, and frames outside the original prefix or after query time. This diagnostic uses privileged support knowledge and is not a deployable research method.

## What to return

Download `HERMES_input_diagnostic_evidence.zip` using the final notebook link and save the executed notebook. The ZIP includes predictions/statuses, summary, exact-subset paired historical comparisons, per-question checkpoints, run/environment/source/feature provenance, trusted original predictions, GPU preflight, review pages and SHA256 manifest. It excludes raw videos, weights and feature tensors. Keep those caches if further experiments are likely.

The runner packages diagnostic failures; after a failure stops Run All, execute the final download cell manually if the archive exists. Package-installation failures can precede archive creation and remain in setup logs. The notebook repackages after the runner's live log closes to include its final traceback/status.

`paired_historical.csv` compares each answered diagnostic subset against exactly those historical FIFO/HERMES/causal question identities. Historical and new timings come from different sessions and different input/history treatments; they are not a matched efficiency experiment. The report makes no population significance, original contribution or publication-readiness claim.

## Local validation

```bash
python scripts/make_input_diagnostic_notebook.py
python -m unittest discover -s scripts -p test_input_diagnostic.py -v
```

Eighteen implementation tests passed locally, including native small-decoder numerical checks, exact frame slicing, question execution/resume, context skips, damaged metadata/checkpoint rejection, workspace protection, partial asset reuse, environment reuse, package integrity and execution of every notebook code cell with network/GPU actions disabled. Eleven reused pilot utility tests and three setup-routing tests also passed. Numerical tests used the local CPU environment with pinned Transformers. The supplied reused-asset AutoDL execution subsequently passed mandatory GPU preflight and completed; fresh-container installation was not independently rerun. Five new saved-evidence audit tests also pass.
