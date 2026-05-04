"""
Stage 9 training-dataset builder.

Reads `data/processed/cycling/unified.parquet` and `splits.json`, produces:
- numerical feature matrix X (cycle-level)
- targets y_soh (SoH percent) and y_rul (cycles to EoL @ SoH=0.8)
- one-hot-encoded categorical context (chemistry, source family, form_factor)
- per-battery sequences for LSTM
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DIR

UNIFIED_PARQUET = PROCESSED_DIR / "cycling" / "unified.parquet"
SPLITS_JSON = PROCESSED_DIR / "cycling" / "splits.json"
EOL_THRESHOLD = 0.80

# Per Stage-EDA finding: ir_ohm is 93 % missing and the 4 temperature columns
# are 87 % missing — median-imputing them would manufacture a fake signal in
# the majority of rows, so they are excluded from the default feature set.
# Set `include_high_missing=True` in load_feature_bundle() to override (e.g.,
# for the IR-availability ablation study).
HIGH_MISSING_FEATURES = ["ir_ohm", "t_min", "t_max", "t_mean", "t_range"]

# Iteration-2 dQ/dV peak features (Severson 2019 / Greenbank & Howey 2022).
# Available for ~85 % of the corpus (BatteryLife + NASA Rand & Recomm + the
# discharge-side of synthetic), NaN for the rest. Median-imputed at training
# time which is acceptable since the imputation is per-source.
DQDV_FEATURES = [
    "v_peak_dqdv_charge",
    "dqdv_peak_height_charge",
    "dqdv_peak_width_charge",
    "v_peak_dqdv_discharge",
    "dqdv_peak_height_discharge",
    "dqdv_peak_width_discharge",
    "q_at_v_lo",
    "q_at_v_hi",
]

NUMERIC_FEATURES_FULL = [
    "nominal_Ah",
    "capacity_Ah",
    "v_min", "v_max", "v_mean",
    "i_min", "i_max", "i_mean",
    "t_min", "t_max", "t_mean", "t_range",
    "charge_time_s", "discharge_time_s",
    "ir_ohm",
    "coulombic_eff",
    "capacity_delta",
    "capacity_roll5_mean", "capacity_roll5_std",
    "capacity_roll20_mean", "capacity_roll20_std",
    "cycle_count_so_far",
] + DQDV_FEATURES

NUMERIC_FEATURES = [c for c in NUMERIC_FEATURES_FULL if c not in HIGH_MISSING_FEATURES]


@dataclass
class FeatureBundle:
    X_train: np.ndarray
    y_train_soh: np.ndarray
    y_train_rul: np.ndarray
    X_val: np.ndarray
    y_val_soh: np.ndarray
    y_val_rul: np.ndarray
    X_test: np.ndarray
    y_test_soh: np.ndarray
    y_test_rul: np.ndarray
    feature_names: list
    train_battery_ids: pd.Series
    val_battery_ids: pd.Series
    test_battery_ids: pd.Series
    train_cycles: pd.Series
    val_cycles: pd.Series
    test_cycles: pd.Series
    train_sources: pd.Series
    val_sources: pd.Series
    test_sources: pd.Series
    train_chemistries: pd.Series
    val_chemistries: pd.Series
    test_chemistries: pd.Series
    scaler: StandardScaler

    def shapes(self) -> dict:
        return {
            "train": tuple(self.X_train.shape),
            "val": tuple(self.X_val.shape),
            "test": tuple(self.X_test.shape),
            "n_features": len(self.feature_names),
        }


def _compute_rul_per_battery(group: pd.DataFrame, eol_threshold: float) -> np.ndarray:
    """RUL = (cycle of first SoH<EoL) - current cycle. Right-censored otherwise."""
    cycles = group["cycle"].to_numpy()
    soh = group["soh"].to_numpy()
    below = np.where(soh < eol_threshold)[0]
    if len(below) > 0:
        eol_cycle = cycles[below[0]]
    else:
        eol_cycle = cycles.max()
    return np.maximum(0, eol_cycle - cycles).astype(float)


def _build_feature_frame(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    df = df.copy()
    rul_pieces = []
    for _, group in df.groupby("battery_id", sort=False):
        rul_pieces.append(pd.Series(_compute_rul_per_battery(group, EOL_THRESHOLD),
                                    index=group.index))
    df["rul"] = pd.concat(rul_pieces).reindex(df.index)

    df["soh_pct"] = df["soh"].clip(lower=0.0, upper=1.5) * 100.0
    df["source_family"] = df["source"].str.split("_").str[0]

    df[numeric_features] = df[numeric_features].fillna(
        df[numeric_features].median(numeric_only=True)
    )

    n_before = len(df)
    df = df[df["soh_pct"].notna() & np.isfinite(df["soh_pct"])]
    df = df[df["rul"].notna() & np.isfinite(df["rul"])]
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"  dropped {n_dropped:,} rows with NaN/inf labels "
              f"({n_dropped/max(n_before,1)*100:.2f}% of total)")
    return df


def _split_frame(df: pd.DataFrame, splits: dict) -> dict:
    return {
        "train": df[df["battery_id"].isin(splits["train"])],
        "val":   df[df["battery_id"].isin(splits["val"])],
        "test":  df[df["battery_id"].isin(splits["test"])],
    }


def _onehot_with_categories(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """One-hot encode using fixed category lists for cross-split consistency."""
    pieces = []
    for col, cats in categories.items():
        s = df[col].astype("category").cat.set_categories(cats)
        pieces.append(pd.get_dummies(s, prefix=col, dtype="float32"))
    return pd.concat(pieces, axis=1)


def load_feature_bundle(
    *,
    sample_frac: float | None = None,
    smoke: bool = False,
    use_categoricals: bool = True,
    include_source_onehot: bool = False,
    include_high_missing: bool = False,
    verbose: bool = True,
) -> FeatureBundle:
    """
    Load a train/val/test feature bundle from unified.parquet + splits.json.

    Parameters
    ----------
    sample_frac : optional fraction to subsample within each split for speed
    smoke       : if True, hard-cap to ~50k rows total for fast iteration
    use_categoricals : whether to one-hot encode chemistry + form_factor
    include_source_onehot : whether to also one-hot encode source_family.
        DEFAULT FALSE per Roman et al. 2021 (Nature Machine Intelligence):
        source-ID features encourage protocol-fingerprint memorization and
        hurt cross-source generalization. Set True only for the ablation
        study that quantifies the within-source-memorization gap.
    include_high_missing : include `ir_ohm` and the 4 temperature columns
        (default: False; per Stage-EDA finding they are 87–93 % missing and
        median-imputing them manufactures a fake signal in most rows).
    """
    numeric_features = (NUMERIC_FEATURES_FULL if include_high_missing
                        else NUMERIC_FEATURES)
    if verbose:
        print(f"Loading {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} ...")
        if not include_high_missing:
            print(f"  feature set: {len(numeric_features)} numeric features "
                  f"(excluded high-missing: {HIGH_MISSING_FEATURES})")
    df = pd.read_parquet(UNIFIED_PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    df = _build_feature_frame(df, numeric_features)

    splits_dict = _split_frame(df, splits)

    if smoke:
        for k, sub in splits_dict.items():
            cap = 20000 if k == "train" else 5000
            if len(sub) > cap:
                splits_dict[k] = sub.sample(n=cap, random_state=42)
    elif sample_frac is not None:
        splits_dict = {
            k: sub.sample(frac=sample_frac, random_state=42) for k, sub in splits_dict.items()
        }

    train_df = splits_dict["train"]
    categories = {
        "chemistry": sorted(train_df["chemistry"].dropna().unique().tolist()),
        "form_factor": sorted(train_df["form_factor"].dropna().astype(str).unique().tolist()),
    }
    if include_source_onehot:
        # Ablation-only: source-ID one-hots encourage memorization (Roman 2021).
        categories["source_family"] = sorted(
            train_df["source_family"].dropna().unique().tolist()
        )

    def _build_xy(sub: pd.DataFrame):
        x_num = sub[numeric_features].to_numpy(dtype=np.float32)
        if use_categoricals:
            x_cat = _onehot_with_categories(sub, categories).to_numpy(dtype=np.float32)
            x = np.concatenate([x_num, x_cat], axis=1)
        else:
            x = x_num
        y_soh = sub["soh_pct"].to_numpy(dtype=np.float32)
        y_rul = sub["rul"].to_numpy(dtype=np.float32)
        return x, y_soh, y_rul

    X_train, y_train_soh, y_train_rul = _build_xy(splits_dict["train"])
    X_val,   y_val_soh,   y_val_rul   = _build_xy(splits_dict["val"])
    X_test,  y_test_soh,  y_test_rul  = _build_xy(splits_dict["test"])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    feature_names = list(numeric_features)
    if use_categoricals:
        for col, cats in categories.items():
            feature_names.extend([f"{col}_{c}" for c in cats])

    bundle = FeatureBundle(
        X_train=X_train, y_train_soh=y_train_soh, y_train_rul=y_train_rul,
        X_val=X_val,     y_val_soh=y_val_soh,     y_val_rul=y_val_rul,
        X_test=X_test,   y_test_soh=y_test_soh,   y_test_rul=y_test_rul,
        feature_names=feature_names,
        train_battery_ids=splits_dict["train"]["battery_id"].reset_index(drop=True),
        val_battery_ids=splits_dict["val"]["battery_id"].reset_index(drop=True),
        test_battery_ids=splits_dict["test"]["battery_id"].reset_index(drop=True),
        train_cycles=splits_dict["train"]["cycle"].reset_index(drop=True),
        val_cycles=splits_dict["val"]["cycle"].reset_index(drop=True),
        test_cycles=splits_dict["test"]["cycle"].reset_index(drop=True),
        train_sources=splits_dict["train"]["source"].reset_index(drop=True),
        val_sources=splits_dict["val"]["source"].reset_index(drop=True),
        test_sources=splits_dict["test"]["source"].reset_index(drop=True),
        train_chemistries=splits_dict["train"]["chemistry"].reset_index(drop=True),
        val_chemistries=splits_dict["val"]["chemistry"].reset_index(drop=True),
        test_chemistries=splits_dict["test"]["chemistry"].reset_index(drop=True),
        scaler=scaler,
    )

    if verbose:
        print(f"  shapes: {bundle.shapes()}")
        print(f"  SoH train mean: {bundle.y_train_soh.mean():.2f}%  "
              f"val: {bundle.y_val_soh.mean():.2f}%  test: {bundle.y_test_soh.mean():.2f}%")

    return bundle


def soh_to_grade(soh_pct: np.ndarray) -> np.ndarray:
    """Vectorized SoH (0-110%) → A/B/C/D grade."""
    return np.where(soh_pct > 80, "A",
            np.where(soh_pct > 60, "B",
              np.where(soh_pct > 40, "C", "D")))


def load_battery_sequences(
    *,
    sequence_length: int = 30,
    smoke: bool = False,
    max_batteries_per_split: int | None = None,
    include_source_onehot: bool = False,
    include_high_missing: bool = False,
    verbose: bool = True,
):
    """
    Build per-battery sequences for LSTM RUL training.

    Returns a dict with keys 'train', 'val', 'test', each a tuple
    (X_sequences, y_rul_targets, battery_ids) where X has shape
    (n_windows, sequence_length, n_features).

    See `load_feature_bundle` for `include_source_onehot` and
    `include_high_missing` semantics.
    """
    numeric_features = (NUMERIC_FEATURES_FULL if include_high_missing
                        else NUMERIC_FEATURES)
    if verbose:
        print(f"Loading sequences from {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_parquet(UNIFIED_PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    df = _build_feature_frame(df, numeric_features)
    splits_dict = _split_frame(df, splits)

    train_df = splits_dict["train"]
    cats = {
        "chemistry": sorted(train_df["chemistry"].dropna().unique().tolist()),
        "form_factor": sorted(train_df["form_factor"].dropna().astype(str).unique().tolist()),
    }
    if include_source_onehot:
        cats["source_family"] = sorted(
            train_df["source_family"].dropna().unique().tolist()
        )

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_features].to_numpy(dtype=np.float32))

    def _build_split(name: str, sub: pd.DataFrame):
        battery_groups = list(sub.groupby("battery_id", sort=False))
        if smoke:
            cap = 30 if name == "train" else 10
            battery_groups = battery_groups[:cap]
        elif max_batteries_per_split:
            battery_groups = battery_groups[:max_batteries_per_split]

        X_seqs = []
        y_rul = []
        bids = []
        for bid, group in battery_groups:
            group = group.sort_values("cycle")
            x_num = scaler.transform(group[numeric_features].to_numpy(dtype=np.float32))
            x_cat = _onehot_with_categories(group, cats).to_numpy(dtype=np.float32)
            x = np.concatenate([x_num, x_cat], axis=1)
            r = group["rul"].to_numpy(dtype=np.float32)
            if len(group) <= sequence_length:
                continue
            for i in range(len(group) - sequence_length):
                X_seqs.append(x[i : i + sequence_length])
                y_rul.append(r[i + sequence_length])
                bids.append(bid)
        if not X_seqs:
            return np.empty((0, sequence_length, x.shape[1])), np.empty((0,)), pd.Series(dtype=str)
        return (
            np.stack(X_seqs).astype(np.float32),
            np.array(y_rul, dtype=np.float32),
            pd.Series(bids, name="battery_id"),
        )

    out = {k: _build_split(k, v) for k, v in splits_dict.items()}
    n_features = out["train"][0].shape[2] if out["train"][0].size else 0

    if verbose:
        print(f"  sequence shapes: train={out['train'][0].shape}  "
              f"val={out['val'][0].shape}  test={out['test'][0].shape}")
    return out, n_features


if __name__ == "__main__":
    bundle = load_feature_bundle(smoke=True)
    print("OK — feature bundle loaded:", bundle.shapes())
