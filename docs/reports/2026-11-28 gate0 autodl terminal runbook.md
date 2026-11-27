# Gate 0 AutoDL terminal runbook

28 November 2026

Use the pushed Gate 0 commit on `main`. The tracked annotation JSON and 7,938,475-byte benchmark evidence ZIP supply all reference inputs. The ignored local evidence extraction is optional. This runbook assumes the fresh AutoDL image has `git`, `curl`, `python3`, `nvidia-smi`, and `tmux`.

## Clone or update

```bash
export WORK=/root/autodl-tmp
cd "$WORK"
# Fresh checkout
git clone --branch main --single-branch --depth 1 https://github.com/md-hameem/HERMES-reproduction.git HERMES
# For an existing checkout, instead use
# git -C "$WORK/HERMES" pull --ff-only origin main
cd "$WORK/HERMES"
git rev-parse HEAD
```

Verify the commit against the handoff before continuing. All dependency and media commands below run under the data disk.

## Install the archived environment

```bash
set -euo pipefail
export WORK=/root/autodl-tmp
export CACHE="$WORK/cache"
export PIP_CACHE_DIR="$CACHE/pip" HF_HOME="$CACHE/hf" XDG_CACHE_HOME="$CACHE/xdg"
export UV_CACHE_DIR="$CACHE/uv" UV_PYTHON_INSTALL_DIR="$CACHE/python"
export TMPDIR="$WORK/tmp" UV_HTTP_TIMEOUT=180 UV_CONCURRENT_DOWNLOADS=3
export UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" \
  "$UV_PYTHON_INSTALL_DIR" "$TMPDIR" "$WORK/bin" "$WORK/gate0_scratch"
cd "$WORK/HERMES"
df -h "$WORK"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

if [ ! -x "$WORK/bin/uv" ]; then
  curl -fsSL https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="$WORK/bin" sh
fi
UV="$WORK/bin/uv"
VENV="$WORK/gate0_venv"
PY="$VENV/bin/python"
"$UV" venv "$VENV" --python 3.12.3 --seed

python3 - <<'PY'
import os
from pathlib import Path
from zipfile import ZipFile
with ZipFile("evidence/archives/HERMES_AutoDL_five_stage_evidence.zip") as archive:
    freeze = archive.read("exports/environment_freeze.txt").decode()
Path(os.environ["CACHE"], "gate0-freeze.txt").write_text(
    "".join(line + "\n" for line in freeze.splitlines()
            if line and not line.startswith(("torch @ ", "transformers @ ")))
)
PY

TORCH='torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl'
"$UV" pip install --python "$PY" -r "$CACHE/gate0-freeze.txt" "$TORCH" ||
  UV_DEFAULT_INDEX=https://pypi.org/simple "$UV" pip install --python "$PY" -r "$CACHE/gate0-freeze.txt" "$TORCH"

TF=66bc4def9505fa7c7fe4aa7a248c34a026bb552b
curl -fL --retry 3 "https://codeload.github.com/huggingface/transformers/zip/$TF" -o "$CACHE/transformers-$TF.zip"
"$UV" pip install --python "$PY" --no-deps "$CACHE/transformers-$TF.zip"
"$UV" pip check --python "$PY"
"$PY" -B - <<'PY'
import inspect, sys, torch, transformers, tokenizers, huggingface_hub, decord
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
assert sys.version_info[:3] == (3, 12, 3)
assert torch.__version__ == "2.10.0+cu128" and torch.version.cuda == "12.8"
assert transformers.__version__ == "4.45.0.dev0"
assert tokenizers.__version__ == "0.19.1" and huggingface_hub.__version__ == "0.34.4"
assert decord.__version__ == "0.6.0" and torch.cuda.is_available()
assert "seq_len" in inspect.signature(Qwen2RotaryEmbedding.forward).parameters
print(torch.cuda.get_device_name(0))
PY
"$PY" -B scripts/test_profile_streaming_gate0.py
```

The version list is read from the tracked archive. Its original notebook installation used the same direct CUDA wheel and Transformers revision. uv and Hugging Face caching stay on the data disk. [Archived setup](../../notebooks/autodl/hermes-five-stage-autodl-cache-fix.ipynb)

## Download the pinned model and six videos

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
"$PY" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

revision = "74dd0bf867a4cda7950c17663794267c60cf4b40"
snapshot = Path(snapshot_download(
    repo_id="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", revision=revision,
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.vocab", "tokenizer.*", "merges.txt"],
    ignore_patterns=["onnx/*", "*.onnx", "*.bin", "*.msgpack"],
    max_workers=4,
))
assert snapshot.name == revision and (snapshot / "config.json").is_file()
assert list(snapshot.glob("*.safetensors"))
Path(os.environ["WORK"], "gate0_model_path").write_text(str(snapshot) + "\n")
print(snapshot)
PY

"$PY" -B scripts/profile_streaming_gate0.py --manifest
"$PY" -B scripts/fetch_streaming_gate0_media.py --media-dir "$WORK/gate0_media"
"$PY" -B scripts/fetch_streaming_gate0_media.py --media-dir "$WORK/gate0_media" --check
```

The source is `mjuicem/StreamingBench` on Hugging Face. The helper requests ZIP central directories and byte ranges for the six selected members, checks ZIP CRC during extraction, and verifies each extracted file against its archived size and SHA-256. It retries via the official endpoint if the mirror fails and rejects an endpoint that returns the full shard to a range request. The same `--check` command accepts already-mounted copies in that directory. [Media helper](../../scripts/fetch_streaming_gate0_media.py)

## Launch only Gate 0

```bash
cat > "$WORK/run-gate0.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
WORK=/root/autodl-tmp
export PIP_CACHE_DIR="$WORK/cache/pip" HF_HOME="$WORK/cache/hf"
export XDG_CACHE_HOME="$WORK/cache/xdg" TMPDIR="$WORK/tmp"
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "$WORK/HERMES"
test ! -e "$WORK/gate0_results"
"$WORK/gate0_venv/bin/python" -u -B scripts/profile_streaming_gate0.py \
  --media-dir "$WORK/gate0_media" --model-path "$(cat "$WORK/gate0_model_path")" \
  --scratch-dir "$WORK/gate0_scratch" --output-dir "$WORK/gate0_results" \
  2>&1 | tee "$WORK/gate0_run.log"
SH

tmux new-session -d -s hermes-gate0 "bash '$WORK/run-gate0.sh'"
tail -F "$WORK/gate0_run.log"
```

The runner requires a new output directory. Keep a failed run's log and output directory for diagnosis. The successful final line is `Gate 0 PASS: all 24 isolated clip-arm runs validated`. Gate 1 remains on hold.
