"""
Stage 13c — Export the IsoForest/VAE training-time scaler + featurize metadata.

Both anomaly models were trained via `load_feature_bundle()` with default args
(no `exclude_capacity_features`), producing a 38-dim feature space:
  25 numeric (full set minus the 5 high-missing columns)
  + 7 chemistry one-hots
  + 6 form_factor one-hots

Neither trainer persisted its scaler, so we reproduce it deterministically:
load unified.parquet + splits.json, build the same feature frame, concat
numeric + one-hots, fit StandardScaler on the train split.

Outputs (saved under models/anomaly_shared/ so both IsoForest and VAE can
load from a single canonical location — they share the same featurization):

  models/anomaly_shared/feature_scaler.pkl   ← 38-dim scaler
  models/anomaly_shared/feature_meta.json    ← numeric list, categories, medians

Run once. Idempotent.

Usage
-----
    python scripts/export_anomaly_scalers.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import (
    HIGH_MISSING_FEATURES,
    NUMERIC_FEATURES_FULL,
    SPLITS_JSON,
    UNIFIED_PARQUET,
    _build_feature_frame,
    _onehot_with_categories,
)

OUT_DIR = PROJECT_ROOT / "models" / "anomaly_shared"
SCALER_PATH = OUT_DIR / "feature_scaler.pkl"
META_PATH = OUT_DIR / "feature_meta.json"


def main() -> None:
    print("Stage 13c — Export anomaly (IsoForest + VAE) inference assets")
    print("=" * 70)
    numeric_features = [c for c in NUMERIC_FEATURES_FULL if c not in HIGH_MISSING_FEATURES]
    print(f"  full numeric features ({len(numeric_features)}): {numeric_features}")

    print(f"\n[Load] {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)}")
    df = pd.read_parquet(UNIFIED_PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())

    pre_medians = df[numeric_features].median(numeric_only=True).to_dict()
    print(f"  computed feature medians on full corpus ({len(pre_medians)} features)")

    df = _build_feature_frame(df, numeric_features)
    train_df = df[df["battery_id"].isin(splits["train"])]
    print(f"  train rows: {len(train_df):,} (from {len(splits['train'])} batteries)")

    categories = {
        "chemistry": sorted(train_df["chemistry"].dropna().unique().tolist()),
        "form_factor": sorted(train_df["form_factor"].dropna().astype(str).unique().tolist()),
    }
    print(f"  chemistry categories ({len(categories['chemistry'])}): {categories['chemistry']}")
    print(f"  form_factor categories ({len(categories['form_factor'])}): {categories['form_factor']}")

    x_num = train_df[numeric_features].to_numpy(dtype=np.float32)
    x_cat = _onehot_with_categories(train_df, categories).to_numpy(dtype=np.float32)
    x = np.concatenate([x_num, x_cat], axis=1)
    print(f"  concatenated feature matrix: {x.shape}")

    scaler = StandardScaler()
    scaler.fit(x)
    print(f"  fit StandardScaler on {scaler.n_features_in_} features")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    META_PATH.write_text(json.dumps({
        "numeric_features": numeric_features,
        "categories": categories,
        "feature_medians": {k: float(v) for k, v in pre_medians.items()},
        "feature_dim": int(x.shape[1]),
        "fit_source": "load_feature_bundle()-equivalent on full (non-audited) train split",
        "consumed_by": ["isolation_forest", "vae_anomaly"],
        "n_train_rows": int(len(train_df)),
        "n_train_batteries": int(len(splits["train"])),
    }, indent=2))
    print(f"\n[Save] scaler → {SCALER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[Save] meta   → {META_PATH.relative_to(PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
