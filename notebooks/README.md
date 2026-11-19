# Experiment notebooks

Use the new causal-scorer pilot for the next AutoDL session. The five-stage execution record contains the completed earlier GPU run and its outputs. Paths in archived execution records describe their original environment. Run repository CLI commands from the project root.

| Notebook | Purpose |
|---|---|
| [Fresh-container causal-scorer pilot](autodl/hermes-causal-scorer-pilot.ipynb) | Upload this notebook alone and Run All. Default eight-video, three-method screen with pinned setup, selective downloads, shared video features, mandatory numerical preflight, checkpoints and an evidence ZIP |
| [Five-stage AutoDL workflow](autodl/hermes-five-stage-autodl-cache-fix.ipynb) | Maintained workflow with shard-count, CSV text and duplicate-option fixes and the current evidence paths |
| [Original AutoDL reproduction](autodl/hermes-v2-autodl.ipynb) | Historical 50-video, three-budget workflow and execution record |
| [October AutoDL controls](autodl/hermes-final-autodl.ipynb) | Executed repeatability, attention differential and FIFO control notebook |
| [Kaggle analysis workflow](kaggle/hermes-v2.ipynb) | Kaggle reproduction and thesis analyses |
| [Executed five-stage work file](executed/2026-11-18_hermes-five-stage-autodl.ipynb) | Uploaded completed November run, preserving its exact bytes and outputs |
| [Historical notebooks](archive/) | Earlier Kaggle development notebooks and the superseded five-stage AutoDL draft |

The maintained workflow has permanent fixes for the issues recovered during the executed run. A fresh complete GPU replay remains unverified. The evidence validator audits saved measurements independently. [Current project status](../docs/PROJECT_STATUS.md) · [Evidence index](../evidence/README.md)

The new pilot has nine local safety/numerical tests and a headless notebook Run All validation that skips installation, network and GPU execution. Its GPU preflight must pass in AutoDL before QA runs. The default is a development screen on eight videos from the inspected 201–250 shard, distinct from the broader 30-case review set. [Pilot protocol](../docs/FAILURE_PILOT.md)
