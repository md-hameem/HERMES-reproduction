# Experiment notebooks

The causal-scorer pilot has a completed eight-video execution record; use the maintained workflow for any justified rerun, and the follow-up diagnostic protocol for the next research decision. The five-stage execution record contains the completed earlier GPU run and its outputs. Paths in archived execution records describe their original environment. Run repository CLI commands from the project root.

| Notebook | Purpose |
|---|---|
| [Matched-boundary workflow](autodl/hermes-matched-boundary.ipynb) | Completed control on the same 40 questions with matched recent encoding groups. [Run notes](../docs/MATCHED_BOUNDARY.md). |
| [Stage attribution workflow](autodl/hermes-stage-attribution.ipynb) | Base workflow for the 200-answer main matrix. The executed record below contains the final GPU preflight adjustments. [Run notes](../docs/STAGE_ATTRIBUTION.md). |
| [Context/content diagnostic](autodl/hermes-context-diagnostic.ipynb) | Maintained seven-case content/position and partition workflow; completed execution now audited. [Instructions](../docs/CONTEXT_DIAGNOSTIC.md) · [Analysis](<../docs/reports/2026-11-20 context diagnostic analysis and next decision.md>) |
| [Input/capability diagnostic — next run](autodl/hermes-input-diagnostic.ipynb) | Upload alone and Run All; 120 mandatory fresh-state answers, feasible uncompressed prefixes, optional reviewed-support inputs, verified reuse, progress/checkpoints and evidence export. [Instructions](../docs/INPUT_DIAGNOSTIC.md) |
| [Fresh-container causal-scorer pilot](autodl/hermes-causal-scorer-pilot.ipynb) | Upload this notebook alone and Run All. Default eight-video, three-method screen with pinned setup, selective downloads, shared video features, mandatory numerical preflight, checkpoints and an evidence ZIP |
| [Five-stage AutoDL workflow](autodl/hermes-five-stage-autodl-cache-fix.ipynb) | Maintained workflow with shard-count, CSV text and duplicate-option fixes and the current evidence paths |
| [Original AutoDL reproduction](autodl/hermes-v2-autodl.ipynb) | Historical 50-video, three-budget workflow and execution record |
| [October AutoDL controls](autodl/hermes-final-autodl.ipynb) | Executed repeatability, attention differential and FIFO control notebook |
| [Kaggle analysis workflow](kaggle/hermes-v2.ipynb) | Kaggle reproduction and thesis analyses |
| [Executed causal-scorer pilot](executed/2026-11-20_hermes-causal-scorer-pilot.ipynb) | Supplied completed eight-video, 120-answer execution record; preserved unchanged, with pre-heartbeat runner source |
| [Executed input diagnostic](executed/2026-11-20_hermes-input-diagnostic.ipynb) | Supplied successful reused-asset run: 120 mandatory answers, 14 native answers, 66 explicit skips, no oracle answers; original bytes preserved |
| [Executed context diagnostic](executed/2026-11-20_hermes-context-diagnostic.ipynb) | Supplied completed seven-case run: 84 records, 74 answers, ten planned skips; original bytes preserved |
| [Executed stage attribution](executed/2026-11-21_hermes-stage-attribution.ipynb) | Complete 200-answer main matrix and four separate long-case answers with the same-shape FP16 preflight. Original bytes and exploratory failure outputs preserved. |
| [Executed matched-boundary control](executed/2026-11-21_hermes-matched-boundary.ipynb) | Completed 200 answers, 120 required replays and zero choice changes relative to the preceding stage run. Original bytes preserved. |
| [Executed five-stage work file](executed/2026-11-18_hermes-five-stage-autodl.ipynb) | Uploaded completed November run, preserving its exact bytes and outputs |
| [Historical notebooks](archive/) | Earlier Kaggle development notebooks and the superseded five-stage AutoDL draft |

The maintained workflow has permanent fixes for the issues recovered during the executed run. A fresh complete GPU replay remains unverified. The evidence validator audits saved measurements independently. [Current project status](../docs/PROJECT_STATUS.md) · [Evidence index](../evidence/README.md)

The maintained pilot has eleven local safety/numerical/monitor tests, separate setup tests and headless notebook validation. The supplied pre-heartbeat execution passed GPU preflight and completed all QA; fresh causal scoring changed zero correctness outcomes versus HERMES on these 40 questions. The default is a development screen on eight videos from the inspected 201–250 shard, distinct from the broader 30-case review set. [Pilot protocol](../docs/FAILURE_PILOT.md)

For an RTX 4090 fresh-container session, provision roughly 2–3 hours for the default pilot. This is an unmeasured planning estimate: setup/downloads may take 20–45 minutes and experiments 45–120 minutes. `PER_OUTCOME=1` reduces the screen to four videos and 60 answers, with approximately 30–60 minutes of experiments plus setup. The default two-hour soft experiment budget excludes setup/downloads, checks after complete video blocks and can overrun by a block. Keep the same `WORK` directory for resumption and download the evidence ZIP before deleting the container.

The [audited scorer result](<../docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) records 327.028 seconds for the post-model-load phase, excluding setup/downloads/model load/packaging. The separate [input diagnostic has now completed](<../docs/reports/2026-11-20 input diagnostic analysis and next decision.md>): 100.961 seconds after model loading with all eight prior feature caches reused. Question-only/recent-4/recent-20 score 14/25/21 out of 40; native scores 10/14, exactly matching historical baselines on that subset. Neither timing establishes total rental time or a fresh-container startup result. Preserve existing assets for the next context/support controls. Eighteen implementation tests and five new saved-evidence audit tests pass locally.
