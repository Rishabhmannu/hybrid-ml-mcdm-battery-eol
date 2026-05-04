"""
Dataset download scripts for all 4 battery cycling datasets.
Run this script first to acquire all raw data.

Usage:
    conda activate Eco-Research
    python src/data/download.py
"""
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import RAW_DIR, BATTERYLIFE_DIR, STANFORD_DIR, NASA_DIR, CALCE_DIR


def download_batterylife():
    """
    Download BatteryLife_Processed from HuggingFace.
    837 Li-ion batteries, multiple chemistries.
    Source: https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed
    """
    print("=" * 60)
    print("Downloading BatteryLife_Processed from HuggingFace...")
    print("=" * 60)

    try:
        from datasets import load_dataset

        dataset = load_dataset("Battery-Life/BatteryLife_Processed")
        print(f"Dataset loaded. Keys: {list(dataset.keys())}")

        # Save to local directory
        BATTERYLIFE_DIR.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(BATTERYLIFE_DIR / "batterylife_processed"))
        print(f"Saved to {BATTERYLIFE_DIR / 'batterylife_processed'}")

    except Exception as e:
        print(f"Error downloading BatteryLife: {e}")
        print("Manual download: https://huggingface.co/datasets/Battery-Life/BatteryLife_Processed")


def download_stanford():
    """
    Download Stanford Second-Life Battery Dataset.
    8 retired Nissan Leaf cells, 15 months cycling.
    Source: https://osf.io/fns57/
    Paper: Cui et al. (2024), Cell Reports Physical Science, DOI: 10.1016/j.xcrp.2024.101941
    """
    print("\n" + "=" * 60)
    print("Stanford Second-Life Battery Dataset")
    print("=" * 60)
    print("This dataset must be downloaded manually from:")
    print("  Primary: https://osf.io/fns57/")
    print("  Alt:     https://purl.stanford.edu/td676xr4322")
    print(f"\nPlease download and extract to: {STANFORD_DIR}")
    STANFORD_DIR.mkdir(parents=True, exist_ok=True)


def download_nasa():
    """
    Download NASA Randomized & Recommissioned Battery Dataset.
    26 battery packs with constant, random, and second-life cycling.
    Source: https://data.nasa.gov/dataset/randomized-and-recommissioned-battery-dataset
    Kaggle mirror: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset
    """
    print("\n" + "=" * 60)
    print("NASA Randomized & Recommissioned Battery Dataset")
    print("=" * 60)
    print("Download options:")
    print("  1. NASA Open Data Portal: https://data.nasa.gov/dataset/randomized-and-recommissioned-battery-dataset")
    print("  2. Kaggle mirror: https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset")
    print("  3. Contact: christopher.a.teubert@nasa.gov")
    print(f"\nPlease download and extract to: {NASA_DIR}")
    NASA_DIR.mkdir(parents=True, exist_ok=True)


def download_calce():
    """
    Download CALCE Battery Data from University of Maryland.
    Multi-chemistry (LCO, LFP, NMC), multi-format benchmark.
    Source: https://calce.umd.edu/battery-data
    """
    print("\n" + "=" * 60)
    print("CALCE Battery Data (University of Maryland)")
    print("=" * 60)
    print("Download from:")
    print("  https://calce.umd.edu/battery-data")
    print("  https://web.calce.umd.edu/batteries/data/")
    print(f"\nPlease download and extract to: {CALCE_DIR}")
    CALCE_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("EV Battery EoL Routing Framework — Data Acquisition")
    print("=" * 60)
    print(f"Raw data directory: {RAW_DIR}\n")

    # Create all directories
    for d in [BATTERYLIFE_DIR, STANFORD_DIR, NASA_DIR, CALCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Download BatteryLife (automated)
    download_batterylife()

    # Instructions for manual downloads
    download_stanford()
    download_nasa()
    download_calce()

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"  BatteryLife: {'Check' if (BATTERYLIFE_DIR / 'batterylife_processed').exists() else 'Pending'}")
    print(f"  Stanford:    {'Check' if any(STANFORD_DIR.iterdir()) else 'Manual download needed'}")
    print(f"  NASA:        {'Check' if any(NASA_DIR.iterdir()) else 'Manual download needed'}")
    print(f"  CALCE:       {'Check' if any(CALCE_DIR.iterdir()) else 'Manual download needed'}")


if __name__ == "__main__":
    main()
