"""
Download the full BatteryLife_Processed dataset from HuggingFace.

Uses snapshot_download which:
- Resumes interrupted downloads automatically
- Stores files under ~/.cache/huggingface/hub/ and symlinks into local_dir
- Validates checksums on download

Usage:
    conda activate Eco-Research
    python scripts/download_batterylife.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGET = PROJECT_ROOT / "data" / "raw" / "batterylife" / "batterylife_processed"


def main() -> int:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set in environment", file=sys.stderr)
        return 1

    TARGET.mkdir(parents=True, exist_ok=True)
    print(f"target: {TARGET}")
    print("starting snapshot_download (~86 GB; will resume on interruption)")

    t0 = time.time()
    path = snapshot_download(
        repo_id="Battery-Life/BatteryLife_Processed",
        repo_type="dataset",
        local_dir=str(TARGET),
        token=token,
        max_workers=8,
        # Skip nothing -- full dataset
    )
    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
