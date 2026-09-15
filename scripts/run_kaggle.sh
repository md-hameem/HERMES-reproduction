#!/bin/bash
set -e

echo "=== Setting up HERMES Kaggle Environment ==="
# Install minimal requirements (assuming Kaggle environment with PyTorch/Transformers)
pip install -r requirements_kaggle.txt --no-deps

echo "=== Downloading LLaVA-OV-0.5B Model ==="
# Download model to models/ directory
mkdir -p models/llava-onevision-qwen2-0.5b-ov-hf
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
    local_dir='models/llava-onevision-qwen2-0.5b-ov-hf'
)"

echo "=== Downloading StreamingBench 10-Video Subset ==="
python scripts/download_subset.py

echo "=== Running HERMES VQA on Subset (KV=4000) ==="
export PYTHONPATH=$(pwd):$PYTHONPATH

# Ensure results directory exists
mkdir -p results/kaggle_pilot

# We run inference. 
# We don't use the standard scripts/run_infer.sh because we want to test specifically on our subset with KV=4000
python -m video_qa.hermes_vqa \
    --model llava_ov_0.5b \
    --sample_fps 0.5 \
    --save_dir results/kaggle_pilot \
    --anno_path data/streamingbench/streamingbench_realtime.json \
    --kv_size 4000 \
    --streaming true

echo "=== Running Multiple-Choice Evaluation ==="
# Find the exact CSV file created by hermes_vqa
# Assuming it saves to results/kaggle_pilot/llava_ov_0.5b/streamingbench/fps0.5-kv4000/results.csv
RESULT_CSV="results/kaggle_pilot/1_0.csv"
if [ -f "$RESULT_CSV" ]; then
    python eval/eval_multiple_choice.py general --results_path "$RESULT_CSV"
else
    echo "Warning: Expected result CSV not found at $RESULT_CSV. Inference may have failed or used a different path structure."
fi

echo "=== Kaggle Pilot Test Complete! ==="
