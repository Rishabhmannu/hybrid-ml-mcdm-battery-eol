"""
Data preprocessing: load raw datasets, standardize formats, handle missing values.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import (
    BATTERYLIFE_DIR, STANFORD_DIR, NASA_DIR, CALCE_DIR,
    PROCESSED_DIR, MISSING_THRESHOLD, OUTLIER_ZSCORE,
    VOLTAGE_MIN, VOLTAGE_MAX, TEMPERATURE_MIN, TEMPERATURE_MAX,
    RANDOM_SEED,
)

# Standard column names across all datasets
STANDARD_COLUMNS = [
    "battery_id", "cycle", "voltage", "current", "temperature",
    "capacity", "internal_resistance", "charge_energy", "discharge_energy", "time"
]


def load_batterylife():
    """Load BatteryLife_Processed from local HuggingFace cache."""
    print("Loading BatteryLife_Processed...")
    try:
        from datasets import load_from_disk
        ds = load_from_disk(str(BATTERYLIFE_DIR / "batterylife_processed"))
        df = ds["train"].to_pandas() if "train" in ds else pd.DataFrame(ds)
        print(f"  Loaded {len(df)} rows")
        return df
    except Exception as e:
        print(f"  Error: {e}")
        return pd.DataFrame()


def load_stanford():
    """Load Stanford Second-Life dataset from local files."""
    print("Loading Stanford Second-Life dataset...")
    # TODO: Implement based on actual file format after download
    # Expected: CSV or MAT files with aging cycle data
    print("  Placeholder — implement after manual download")
    return pd.DataFrame()


def load_nasa():
    """Load NASA Randomized & Recommissioned dataset from local files."""
    print("Loading NASA Battery dataset...")
    # TODO: Implement based on actual file format after download
    # Expected: MAT files in three subdirectories
    print("  Placeholder — implement after manual download")
    return pd.DataFrame()


def load_calce():
    """Load CALCE benchmark data from local files."""
    print("Loading CALCE Battery data...")
    # TODO: Implement based on actual file format after download
    print("  Placeholder — implement after manual download")
    return pd.DataFrame()


def validate_data(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Apply data quality checks."""
    initial_count = len(df)

    # Voltage range check
    if "voltage" in df.columns:
        mask = (df["voltage"] >= VOLTAGE_MIN) & (df["voltage"] <= VOLTAGE_MAX)
        df = df[mask]

    # Temperature range check
    if "temperature" in df.columns:
        mask = (df["temperature"] >= TEMPERATURE_MIN) & (df["temperature"] <= TEMPERATURE_MAX)
        df = df[mask]

    # Missing value check per battery
    if "battery_id" in df.columns:
        missing_pct = df.groupby("battery_id").apply(
            lambda x: x.isnull().mean().mean()
        )
        valid_batteries = missing_pct[missing_pct < MISSING_THRESHOLD].index
        df = df[df["battery_id"].isin(valid_batteries)]

    removed = initial_count - len(df)
    if removed > 0:
        print(f"  [{source}] Removed {removed} rows during validation")

    return df


def merge_datasets(dfs: dict) -> pd.DataFrame:
    """
    Merge all datasets into a single DataFrame.
    Adds source prefix to battery_id for traceability.
    """
    frames = []
    prefixes = {"batterylife": "BL", "stanford": "STAN", "nasa": "NASA", "calce": "CALCE"}

    for source, df in dfs.items():
        if df.empty:
            continue
        prefix = prefixes.get(source, source.upper())
        if "battery_id" in df.columns:
            df["battery_id"] = prefix + "_" + df["battery_id"].astype(str)
        df["source"] = source
        frames.append(df)

    if not frames:
        print("Warning: No datasets loaded!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nMerged dataset: {len(combined)} rows, {combined['battery_id'].nunique()} unique batteries")
    return combined


def normalize_features(df: pd.DataFrame, feature_cols: list) -> tuple:
    """StandardScaler normalization for ML features."""
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])
    return df, scaler


def main():
    """Full preprocessing pipeline."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load all datasets
    datasets = {
        "batterylife": load_batterylife(),
        "stanford": load_stanford(),
        "nasa": load_nasa(),
        "calce": load_calce(),
    }

    # Step 2: Validate each dataset
    for source, df in datasets.items():
        if not df.empty:
            datasets[source] = validate_data(df, source)

    # Step 3: Merge
    combined = merge_datasets(datasets)

    if not combined.empty:
        # Step 4: Save processed data
        combined.to_csv(PROCESSED_DIR / "combined_raw.csv", index=False)
        print(f"Saved combined raw data to {PROCESSED_DIR / 'combined_raw.csv'}")

    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
