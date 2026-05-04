"""
Feature engineering: extract 16+ features per cycle from raw battery data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DIR, GRADE_THRESHOLDS


def compute_soh(capacity_current: float, capacity_nominal: float) -> float:
    """State of Health = (current capacity / nominal capacity) * 100."""
    if capacity_nominal <= 0:
        return np.nan
    return (capacity_current / capacity_nominal) * 100


def classify_grade(soh: float) -> str:
    """Classify battery into grade based on SoH thresholds."""
    if soh > GRADE_THRESHOLDS["A"]:
        return "A"
    elif soh > GRADE_THRESHOLDS["B"]:
        return "B"
    elif soh > GRADE_THRESHOLDS["C"]:
        return "C"
    else:
        return "D"


def extract_cycle_features(cycle_df: pd.DataFrame) -> dict:
    """
    Extract engineered features from a single cycle's data.

    Parameters:
        cycle_df: DataFrame containing time-series data for one cycle
                  Expected columns: voltage, current, temperature, capacity, time

    Returns:
        Dictionary of engineered features for this cycle
    """
    features = {}

    # Basic statistics
    if "voltage" in cycle_df.columns:
        features["voltage_mean"] = cycle_df["voltage"].mean()
        features["voltage_std"] = cycle_df["voltage"].std()
        features["voltage_max"] = cycle_df["voltage"].max()
        features["voltage_min"] = cycle_df["voltage"].min()

    if "current" in cycle_df.columns:
        features["current_mean"] = cycle_df["current"].mean()
        features["current_std"] = cycle_df["current"].std()

    # Temperature features
    if "temperature" in cycle_df.columns:
        features["temp_mean"] = cycle_df["temperature"].mean()
        features["temp_std"] = cycle_df["temperature"].std()
        features["temp_range"] = cycle_df["temperature"].max() - cycle_df["temperature"].min()
        features["temp_max"] = cycle_df["temperature"].max()

    # Capacity
    if "capacity" in cycle_df.columns:
        features["capacity"] = cycle_df["capacity"].iloc[-1] if len(cycle_df) > 0 else np.nan

    # Internal resistance
    if "internal_resistance" in cycle_df.columns:
        features["ir_mean"] = cycle_df["internal_resistance"].mean()

    # Energy features
    if "charge_energy" in cycle_df.columns:
        features["charge_energy"] = cycle_df["charge_energy"].sum()
    if "discharge_energy" in cycle_df.columns:
        features["discharge_energy"] = cycle_df["discharge_energy"].sum()

    # Derived features
    if "charge_energy" in features and "discharge_energy" in features:
        if features["charge_energy"] > 0:
            features["energy_efficiency"] = features["discharge_energy"] / features["charge_energy"]
        else:
            features["energy_efficiency"] = np.nan

    # Charge time (if time column available)
    if "time" in cycle_df.columns:
        features["charge_time"] = cycle_df["time"].max() - cycle_df["time"].min()

    return features


def compute_degradation_features(battery_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute degradation-rate features across cycles for one battery.

    Parameters:
        battery_df: DataFrame with per-cycle features, sorted by cycle number

    Returns:
        DataFrame with additional degradation features added
    """
    df = battery_df.copy().sort_values("cycle")

    # Capacity fade rate (dQ/dN)
    if "capacity" in df.columns:
        df["capacity_fade_rate"] = df["capacity"].diff() / df["cycle"].diff()

    # Internal resistance growth (normalized to first cycle)
    if "ir_mean" in df.columns:
        ir_initial = df["ir_mean"].iloc[0]
        if ir_initial > 0:
            df["ir_growth"] = df["ir_mean"] / ir_initial
        else:
            df["ir_growth"] = np.nan

    # Coulombic efficiency approximation
    if "capacity" in df.columns:
        nominal = df["capacity"].iloc[0]
        df["soh"] = df["capacity"].apply(lambda x: compute_soh(x, nominal))

    # Cumulative energy throughput
    if "discharge_energy" in df.columns:
        df["cumulative_energy"] = df["discharge_energy"].cumsum()

    # Knee-point indicator (second derivative of capacity curve)
    if "capacity" in df.columns:
        capacity_diff = df["capacity"].diff()
        df["capacity_acceleration"] = capacity_diff.diff()
        # Knee point = where acceleration becomes strongly negative
        threshold = df["capacity_acceleration"].quantile(0.05)
        df["knee_point"] = (df["capacity_acceleration"] < threshold).astype(int)

    return df


def build_feature_matrix(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full feature matrix from combined raw data.

    Parameters:
        combined_df: Merged dataset from preprocess.py

    Returns:
        Feature matrix with one row per battery-cycle
    """
    print("Building feature matrix...")

    all_features = []

    for battery_id, battery_data in combined_df.groupby("battery_id"):
        # Extract per-cycle features
        for cycle_num, cycle_data in battery_data.groupby("cycle"):
            features = extract_cycle_features(cycle_data)
            features["battery_id"] = battery_id
            features["cycle"] = cycle_num
            all_features.append(features)

    feature_df = pd.DataFrame(all_features)

    # Compute degradation features per battery
    enriched_dfs = []
    for battery_id, battery_features in feature_df.groupby("battery_id"):
        enriched = compute_degradation_features(battery_features)
        enriched_dfs.append(enriched)

    if enriched_dfs:
        result = pd.concat(enriched_dfs, ignore_index=True)

        # Add grade labels
        if "soh" in result.columns:
            result["grade"] = result["soh"].apply(classify_grade)

        print(f"Feature matrix: {result.shape[0]} rows x {result.shape[1]} columns")
        return result

    return pd.DataFrame()


def main():
    """Build feature matrix from preprocessed data."""
    input_path = PROCESSED_DIR / "combined_raw.csv"

    if not input_path.exists():
        print(f"Error: {input_path} not found. Run preprocess.py first.")
        return

    combined = pd.read_csv(input_path)
    features = build_feature_matrix(combined)

    if not features.empty:
        output_path = PROCESSED_DIR / "features.csv"
        features.to_csv(output_path, index=False)
        print(f"Saved feature matrix to {output_path}")


if __name__ == "__main__":
    main()
