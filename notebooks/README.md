# Experiment notebooks

The causal-scorer pilot has a completed eight-video execution record; use the maintained workflow for any justified rerun, and the follow-up diagnostic protocol for the next research decision. The five-stage execution record contains the completed earlier GPU run and its outputs. Paths in archived execution records describe their original environment. Run repository CLI commands from the project root.

| Notebook | Purpose |
|---|---|
| [Fresh-container causal-scorer pilot](autodl/hermes-causal-scorer-pilot.ipynb) | Upload this notebook alone and Run All. Default eight-video, three-method screen with pinned setup, selective downloads, shared video features, mandatory numerical preflight, checkpoints and an evidence ZIP |
| [Five-stage AutoDL workflow](autodl/hermes-five-stage-autodl-cache-fix.ipynb) | Maintained workflow with shard-count, CSV text and duplicate-option fixes and the current evidence paths |
| [Original AutoDL reproduction](autodl/hermes-v2-autodl.ipynb) | Historical 50-video, three-budget workflow and execution record |
| [October AutoDL controls](autodl/hermes-final-autodl.ipynb) | Executed repeatability, attention differential and FIFO control notebook |
| [Kaggle analysis workflow](kaggle/hermes-v2.ipynb) | Kaggle reproduction and thesis analyses |
| [Executed causal-scorer pilot](executed/2026-11-20_hermes-causal-scorer-pilot.ipynb) | Supplied completed eight-video, 120-answer execution record; preserved unchanged, with pre-heartbeat runner source |
| [Executed five-stage work file](executed/2026-11-18_hermes-five-stage-autodl.ipynb) | Uploaded completed November run, preserving its exact bytes and outputs |
| [Historical notebooks](archive/) | Earlier Kaggle development notebooks and the superseded five-stage AutoDL draft |

The maintained workflow has permanent fixes for the issues recovered during the executed run. A fresh complete GPU replay remains unverified. The evidence validator audits saved measurements independently. [Current project status](../docs/PROJECT_STATUS.md) · [Evidence index](../evidence/README.md)

The maintained pilot has eleven local safety/numerical/monitor tests, separate setup tests and headless notebook validation. The supplied pre-heartbeat execution passed GPU preflight and completed all QA; fresh causal scoring changed zero correctness outcomes versus HERMES on these 40 questions. The default is a development screen on eight videos from the inspected 201–250 shard, distinct from the broader 30-case review set. [Pilot protocol](../docs/FAILURE_PILOT.md)

For an RTX 4090 fresh-container session, provision roughly 2–3 hours for the default pilot. This is an unmeasured planning estimate: setup/downloads may take 20–45 minutes and experiments 45–120 minutes. `PER_OUTCOME=1` reduces the screen to four videos and 60 answers, with approximately 30–60 minutes of experiments plus setup. The default two-hour soft experiment budget excludes setup/downloads, checks after complete video blocks and can overrun by a block. Keep the same `WORK` directory for resumption and download the evidence ZIP before deleting the container.

The [audited result](<../docs/reports/2026-11-20 causal scorer pilot analysis and publication decision.md>) records 327.028 seconds for the post-model-load phase, much shorter than the conservative earlier planning estimate above. This excludes setup/downloads/model load/packaging and does not establish total rental time. The [next control protocol](../paper/neurocomputing/next_diagnostic_protocol.md) is a plan rather than a new executable notebook. Preserve the existing videos/features if available.
