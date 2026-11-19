# Research evidence

The original five-stage ZIP is the canonical new evidence payload in Git. Its full local extraction is kept under `local/five_stage/` and excluded from Git. Published summary tables and figures are available directly in the repository. [Five-stage results](../results/five_stage/) · [Project status](../docs/PROJECT_STATUS.md)

| Location | Contents |
|---|---|
| [Original five-stage archive](archives/HERMES_AutoDL_five_stage_evidence.zip) | All 2,921 original manifest payloads, including predictions, video checkpoints, timing events, annotations, scripts, logs and embedded October evidence |
| [Historical supplement](archives/historical_evidence_supplement.zip) | Three recovered October files and their original manifest hashes |
| [Original AutoDL evidence](autodl_2026-10-25/) | Original three-budget results, logs, paired analyses and efficiency exports. Internal historical paths are preserved. |
| [October controls](autodl_2026-10-28/) | Hash-verified HERMES repeats, FIFO controls, original attention diagnostic and preserved earlier evidence |
| [Local five-stage audit](audits/five_stage_local_audit.json) | Independently recomputed integrity, prediction, checkpoint, statistics and timing checks |

The five-stage archive SHA-256 is `b8bc4cf8a3b28e6854b2e76b8c0239b12d3b5598ee1a07dafc03d0326a900518`. The supplement SHA-256 is `4be19b26c61237860c322c5431090dcafc900aae400b9c0ed8204a2c68aab66a`. The uploaded executed notebook is stored under `notebooks/executed/` and its SHA-256 is recorded in the local audit.

Run both audits from the project root with Python, NumPy, pandas and SciPy installed.

```bash
python scripts/validate_final_evidence.py
python scripts/validate_five_stage_evidence.py
```

The older audit reports the three omissions inside its original bundle. The new audit also verifies their recovered bytes in the separate supplement.

The latest validator reads directly from the ZIP. A full extraction is optional for local inspection.

```bash
python -m zipfile -e evidence/archives/HERMES_AutoDL_five_stage_evidence.zip evidence/local/five_stage
```

Historical evidence, the uploaded executed notebook and published summary copies retain the measured output bytes. Current workflow and validation code live outside the evidence packages. The full-benchmark aggregate reuses October samples 201–250 across sessions and hardware. Live recovery patches are documented in the current status report. [Execution record](../notebooks/executed/2026-11-18_hermes-five-stage-autodl.ipynb) · [Audit implementation](../scripts/validate_five_stage_evidence.py)
