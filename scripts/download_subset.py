import json
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

def download_subset(json_path, num_videos=10):
    print(f"Loading annotation from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    # Get unique video paths
    unique_videos = list(dict.fromkeys(sample["video_path"] for sample in data))
    subset_videos = unique_videos[:num_videos]
    
    print(f"Found {len(unique_videos)} unique videos. Downloading {len(subset_videos)} for the pilot test...")
    
    # Target directory structure
    target_dir = Path("data/streamingbench/videos")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Download from HuggingFace
    repo_id = "mjuicem/StreamingBench"
    
    for i, video_path in enumerate(subset_videos):
        # The JSON uses paths like "/data/streamingbench/videos/sample_100_real.mp4"
        # We need the basename to download from HF
        basename = os.path.basename(video_path)
        
        print(f"[{i+1}/{len(subset_videos)}] Downloading {basename}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=repo_id,
                filename=basename,
                repo_type="dataset",
                local_dir=str(target_dir)
            )
            print(f"  -> Saved to {downloaded_path}")
        except Exception as e:
            print(f"  -> Failed to download {basename}: {e}")
            
    print("Download complete!")

if __name__ == "__main__":
    download_subset("data/streamingbench/streamingbench_realtime.json", num_videos=10)
