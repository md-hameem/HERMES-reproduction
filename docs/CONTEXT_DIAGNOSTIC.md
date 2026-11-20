# AutoDL context/content diagnostic

Prepared 20 November 2026 after the [completed input diagnostic](<reports/2026-11-20 input diagnostic analysis and next decision.md>). This is a targeted causal diagnostic, not an implemented original method or independent confirmation.

**Completed-run update:** the [audited context result](<reports/2026-11-20 context diagnostic analysis and next decision.md>) verifies all seven blocks, 84 records/74 answers, 26 matching historical replays and 19 unchanged answered partition contrasts. Three full-prefix failures recover with fixed-position opening replacement; local replacement also causes a regression. The next [stage-attribution protocol](../paper/neurocomputing/stage_attribution_protocol.md) is a plan, not an implemented follow-up notebook. The instructions below describe the completed maintained workflow.

## Run

Upload [hermes-context-diagnostic.ipynb](../notebooks/autodl/hermes-context-diagnostic.ipynb) alone, preferably to the same RTX 4090 / 24 GB container, then **Restart Kernel and Run All**. Defaults require no edits. Output goes to `/root/autodl-tmp/hermes_context_diagnostic`; the separate reuse candidate remains `/root/autodl-tmp/hermes_causal_pilot`. The old input diagnostic directory is not modified. Pinned environment/videos/features are verified before reuse; a fresh container is supported with extra setup/download/encoding work. Fresh setup checks 18 GiB free and a CUDA 12.8-compatible GPU environment.

The notebook is self-contained: three current source modules and SHA-verified historical comparison data are embedded. It checks out only pinned inference/annotation paths, handles package routing/retries as before, and runs the mandatory native/feature-ingestion GPU preflight. No archived notebook or Python source from the evidence ZIP is executed.

## Seven selected questions and twelve arms

Cases are 204/q0, 203/q2, 204/q3 (recent-window recoveries), 209/q0 and 208/q4 (regressions), and 202/q0/208/q0 (unrecovered controls). Five original source videos are needed. These are inspected development cases; seven-case correct counts are not new benchmark scores.

| Arms | Purpose |
|---|---|
| Question-only; recent-4/recent-20/full prefix with original feature-chunk boundaries | Replay the earlier fresh-state controls on exact question identities |
| Full prefix framewise; recent-20/full prefix in fixed batches of four frames | Check whether ingestion partition/numerical effects change answers without changing input content |
| Recent-20/full prefix with opening-frame filler; with local-frame filler | Replace only earlier input slots, keeping input length, final four source frames and their token positions unchanged |
| Candidate support | Probe two saved-sheet candidates: 204/q0 sampled indices [6,7] and 204/q3 [69] |

Replacement fillers repeat either original sampled index 0 or the earliest of the preserved final four frames. They are all pre-query inputs, but neither filler is certified neutral. Repetition, salience and modified temporal content remain confounds. These are declared counterfactual sequences, not natural stream replay or deployable policies. If a prefix has only four frames, replacement leaves the input unchanged; that is an identical-input null control.

Candidate support is **analyst-selected from saved review sheets, not a certified human oracle**. Original-resolution PNGs for those three candidate frames are exported for checking. No support is invented for the other five questions. Complete original sampled-prefix review pages for all seven cases are also exported and contain no filler interventions.

All arms reset to the same initial prompt and use no compression, previous answers or gold-dependent deployment policy. They use the same model, source features, answer prompt/parser and greedy 16-token policy. The slot-to-source map, timestamps, input identity, chunk groups, cache lengths and protected token positions are saved in checkpoints.

## Coverage, progress and resumption

There are **84 case records**. At the default `MAX_VISUAL_TOKENS=16384`, up to **74 answers** are expected before any additional context/VRAM limitation: five unavailable support candidates and five too-large full-prefix arms for 208/q4 are explicit skips. Those full prefixes are never silently truncated. Complete status requires all seven case blocks and 21 successful core question-only/recent-original answers; inspect full arm statuses and skip reasons as well as completion.

Setup has liveness messages while silent; the runner has per-arm output, case completion estimates and a 30-second heartbeat. From another AutoDL terminal:

```bash
tail -n 40 /root/autodl-tmp/hermes_context_diagnostic/logs/context.log
cat /root/autodl-tmp/hermes_context_diagnostic/progress.json
nvidia-smi
```

`MAX_HOURS=0.75` is a soft post-model guard checked between complete case blocks, not a measured runtime estimate. Setup/downloads/model loading/final packaging are excluded; a block can overrun. Keep WORK/settings unchanged and rerun All to resume validated checkpoints. Use a new WORK when changing code, reference or visual-token cap. A configuration conflict preserves the older scientific completion record. Avoid concurrent runs in one WORK directory.

## Return and interpretation

Download **`HERMES_context_diagnostic_evidence.zip`** from the final link and save the executed notebook. It contains predictions, explicit statuses, per-arm checkpoints, summary, `replay_checks.csv`, `contrasts.csv`, source/config/feature provenance, source-only review images, GPU preflight, logs and SHA256 manifest. Videos, feature tensors and weights are excluded. Keep them on AutoDL for reuse.

The notebook repackages after closing the live log. If a runner failure stops Run All, execute the final download cell manually when failure evidence exists. Setup failures may precede ZIP creation and are recorded in WORK/logs.

First inspect `replay_checks.csv` for comparable historical choice mismatches. Then inspect identical-content original/fixed/framewise contrasts. If either is unstable, audit numerical/implementation behavior before attributing changes to earlier visual content. Only after those checks should equal-length, fixed-tail replacement results inform the context hypothesis. Compare both gains and regressions, avoid gold-based selection of a winning arm, and keep candidate-support assumptions explicit. A repeatable effect would motivate one mechanism-level prior-art comparison and possible repair; it does not establish originality by itself.

## Local checks

```bash
python scripts/make_context_diagnostic_notebook.py
python -m unittest discover -s scripts -p test_context_diagnostic.py -v
```

Fourteen implementation tests pass: source/time bounds, fixed-length/tail invariants, declared duplicates and ineffective replacements, candidate bounds, exact default denominators, chunk-group content identity, checkpoint/config protection, archive integrity, headless execution of every notebook cell, feature-bank mapping, a small native-decoder numerical partition check, mocked arm execution/resume, and exact-content contrast reporting. CPU tests use pinned Transformers. The supplied reused-asset AutoDL execution subsequently completed and passed GPU preflight; five independent saved-evidence audit tests also pass. This does not establish a fresh-container installation test or an original method.
