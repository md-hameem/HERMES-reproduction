# AutoDL stage attribution

Prepared 20 November 2026. [Notebook](../notebooks/autodl/hermes-stage-attribution.ipynb) · [Scientific protocol](../paper/neurocomputing/stage_attribution_protocol.md) · [Previous evidence](<reports/2026-11-20 context diagnostic analysis and next decision.md>)

## Run now

Upload **hermes-stage-attribution.ipynb alone**, preferably into the same AutoDL container, and choose **Restart Kernel and Run All**. Defaults require no edits. Output is `/root/autodl-tmp/hermes_stage_attribution_v2`, which preserves evidence from the first attempt. The verified reuse candidate is `/root/autodl-tmp/hermes_causal_pilot`. A fresh container is supported with additional setup/download/feature preparation. Recommended: RTX 4090 / 24 GB, CUDA 12.8-compatible driver and 18 GiB free for fresh setup.

The notebook embeds four current source modules and immutable historical comparison data. It uses the previously tested network/package routing, sparse pinned inference checkout, exact environment/model probe and video/feature hash checks. Notebook kernel imports remain CPU/standard-library; the isolated subprocess owns the GPU. No archived Python/notebook cells are executed from evidence ZIPs.

## Conditions and coverage

For each of the same 40 development questions, use the last up to 20 original sampled frames. Older H is up to 16 frames; R is the last up to four. All inputs remain within the original pre-query boundary.

| Arm | H visible during R construction | H directly visible to question/decode |
|---|---|---|
| A | Yes | Yes |
| B | No | Yes |
| C | Yes | No |
| D | No | No |
| Compact recent-4 | No H present | No H present; compact positions |

The main matrix is **200 answers**. By default, **four additional A–D answers** evaluate the longer 204/q3 prefix, reported in a separate scope. Set `RUN_LONG_CASE=False` before starting a new WORK if only the main matrix is desired. The soft time budget can produce an explicitly partial matrix. No fixed runtime has been measured for this notebook.

A uses the exact earlier recent-20 input and original feature-chunk partition. Its H prefix is shared identically with B; causal attention prevents H from observing later R tokens. B hides H during R construction using an instance-local causal attention mask, retaining H's physical cache slots and R's original logical positions. Mandatory preflight compares this against physically independent construction with the same position gap. C/D hide H at every layer for the question prefill **and every generated token**. The mask preserves prompt/sink access and causality. C can still carry H indirectly through R; it is not complete historical erasure.

A–D have the same answer-time physical cache lengths and positions. Compact recent-4 has fewer physical tokens and compact logical positions. Its factor flags are not applicable to A–D's binary design; exported `factor_cell`, `history_physically_present`, actual visual indices and position fields make that distinction explicit. Questions with empty H are null controls, not extra independent interventions.

The implementation uses per-question A/B/compact control snapshots. Shared preparation time, peak allocation, state byte sizes and construction mask/group traces are saved under `preparation/`. Readout timing and peak allocation include held shared snapshots and transient restore buffers. They are **not standalone-method efficiency measurements**, even though A–D answer-cache slots match. No new eviction, past-answer guidance, adaptive selection rule or original algorithm is introduced. B is an independently constructed-recent comparator inspired by DSCache, not its full method reproduction.

## Mandatory gates

Before QA, the runner checks the original feature/native-attention preflight plus a new two-frame stage preflight:

- An explicit all-visible causal mask agrees with the native logits/argmax.
- H-masked R construction agrees with a physically H-free reference at identical RoPE positions.
- Masked historical readout agrees with physically removed H while preserving all remaining logical positions, through question prefill and two forced decoding steps.
- Every layer assigns zero attention to hidden H and future tokens on the checked fixture.
- Source cache snapshots remain unchanged.

The first AutoDL attempt stopped before QA when one of 25,088 entries in a K/V comparison exceeded the original FP16 bound. Its absolute difference was approximately 0.005615. The screenshot alone does not establish the cause.

The revised preflight first checks the two-frame fixture with FP32 math attention (K/V absolute and relative tolerances of 0.0001 each, logit tolerances of 0.001 each). It then restores the original language-model weights and buffers and repeats the comparisons in FP16 (K/V tolerances of 0.01 each, logit absolute and relative tolerances of 0.02 and 0.005). Both passes require identical tested argmaxes and zero attention to blocked history and future positions. Per-layer K/V differences and exceedance counts for both the original and active bounds are saved to `exports/stage_preflight_diagnostics.json` before the K/V assertions. The successful summary is saved to `stage_preflight.json`.

During QA, the runner traces masking on every LM call and restores and checks the source states after answering. **A and compact recent-4 must reproduce their earlier choices on every main question.** The longer A must reproduce the previous full-prefix choice. Any mismatch stops attribution and saves evidence. A matching rerun cannot bypass a previously stored mismatch.

## Progress, resume and return

Setup prints stage/liveness while silent. The runner prints preflight, per-arm results, question-block completion and 30-second heartbeat. From another AutoDL terminal:

```bash
tail -n 40 /root/autodl-tmp/hermes_stage_attribution_v2/logs/stage.log
cat /root/autodl-tmp/hermes_stage_attribution_v2/progress.json
nvidia-smi
```

`MAX_HOURS=1.0` is a soft post-model budget checked between question blocks, not a runtime estimate. It excludes setup/download/model load/final packaging and can overrun by one block. Rerun All with unchanged settings/WORK to resume validated checkpoints. Changing code, references or long-case settings requires a new WORK directory. Do not run concurrent jobs in one WORK.

Download **HERMES_stage_attribution_evidence.zip** using the final link and save the executed notebook. The archive includes checkpoints, source/config/feature provenance, shared preparation records, all readout mask traces, logical/physical position metadata, preflight metrics, summaries/pairs and final closed logs. Weights, videos, feature tensors and GPU cache tensors are excluded. Keep those assets for reuse.

If a runner gate/error stops Run All, run the final download cell manually when an archive exists. Setup failures may precede archive creation; inspect WORK/logs. No GPU run or scientific outcome for this matrix is established by creating the notebook.

## Local validation

```bash
python scripts/make_stage_attribution_notebook.py
python -m unittest discover -s scripts -p test_stage_attribution.py -v
```

Local tests exercise both precision passes on a small Qwen decoder, exact restoration of parameters and mixed-precision buffers, and failure with saved diagnostics when an injected error exposes hidden history. They also check the real answer function's decode-mask coverage, checkpoint and archive integrity, and headless execution of every notebook cell with network and GPU actions disabled. Complete AutoDL GPU execution remains pending.
