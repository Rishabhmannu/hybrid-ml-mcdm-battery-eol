"""
Stage 13b — Export the TCN's training-time scaler + shared featurize metadata.

The TCN was trained via `load_battery_sequences` (src/data/training_data.py),
which fits a StandardScaler internally on `train_df[numeric_features]` and
discards it after training — so `models/tcn_rul/best.pt` has no companion
`feature_scaler.pkl` (unlike XGBoost / per-chemistry which do persist theirs).

This script reproduces the scaler fit deterministically, and ALSO dumps the
shared featurize metadata that every demo inference path needs:
  - numeric feature list (audited)
  - chemistry / form_factor category lists (exact order used at training time)
  - per-feature medians (matches `_build_feature_frame` NaN imputation)

Outputs
-------
  models/tcn_rul/feature_scaler.pkl   ← TCN scaler (19 numeric features only)
  models/tcn_rul/feature_meta.json    ← shared meta used by the demo featurizer

Run once after TCN training. Idempotent.

Usage
-----
    python scripts/export_tcn_scaler.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import (
    CAPACITY_LEAK_FEATURES,
    HIGH_MISSING_FEATURES,
    NUMERIC_FEATURES_FULL,
    SPLITS_JSON,
    UNIFIED_PARQUET,
    _build_feature_frame,
)

OUT_DIR = PROJECT_ROOT / "models" / "tcn_rul"
SCALER_PATH = OUT_DIR / "feature_scaler.pkl"
META_PATH = OUT_DIR / "feature_meta.json"


def main() -> None:
    print("Stage 13b — Export TCN training scaler (audited config)")
    print("=" * 70)
    numeric_features = [
        c for c in NUMERIC_FEATURES_FULL
        if c not in HIGH_MISSING_FEATURES and c not in CAPACITY_LEAK_FEATURES
    ]
    print(f"  audited numeric features ({len(numeric_features)}): {numeric_features}")

    print(f"\n[Load] {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)}")
    df = pd.read_parquet(UNIFIED_PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())

    # Capture pre-imputation medians (matches `_build_feature_frame` semantics:
    # medians computed on the full corpus, before split, then applied to fill NaNs).
    pre_medians = df[numeric_features].median(numeric_only=True).to_dict()
    print(f"  computed feature medians on full corpus ({len(pre_medians)} features)")

    df = _build_feature_frame(df, numeric_features)
    train_df = df[df["battery_id"].isin(splits["train"])]
    print(f"  train rows: {len(train_df):,} (from {len(splits['train'])} batteries)")

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_features].to_numpy(dtype="float32"))
    print(f"  fit StandardScaler on {scaler.n_features_in_} features")

    categories = {
        "chemistry": sorted(train_df["chemistry"].dropna().unique().tolist()),
        "form_factor": sorted(
            train_df["form_factor"].dropna().astype(str).unique().tolist()
        ),
    }
    print(f"  chemistry categories ({len(categories['chemistry'])}): "
          f"{categories['chemistry']}")
    print(f"  form_factor categories ({len(categories['form_factor'])}): "
          f"{categories['form_factor']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    META_PATH.write_text(json.dumps({
        "numeric_features": numeric_features,
        "categories": categories,
        "feature_medians": {k: float(v) for k, v in pre_medians.items()},
        "fit_source": "load_battery_sequences-equivalent on audited train split",
        "n_train_rows": int(len(train_df)),
        "n_train_batteries": int(len(splits["train"])),
    }, indent=2))
    print(f"\n[Save] scaler → {SCALER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"[Save] meta   → {META_PATH.relative_to(PROJECT_ROOT)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
