"""
Single-cell featurizer for the live demo.

Three entry points:

  featurize_single_cell(cell_df, cycle_n, scaler, mode="audited")
      → (1, 32) array for XGBoost SoH/RUL + ChemistryRouter
        (audited: drops capacity-leak features).

  featurize_single_cell(cell_df, cycle_n, scaler, mode="full")
      → (1, 38) array for IsoForest + VAE
        (full feature set, includes capacity-derived features).

  prepare_tcn_sequence(cell_df, cycle_n, scaler, seq_len=30)
      → (1, 30, 32) for TCN
        (numerics scaled, one-hots appended raw — matches load_battery_sequences).

All three honour the exact category lists + numeric column order from training,
read from feature_meta.json files produced by `scripts/export_*_scaler*.py`.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from frontend.lib.hf_resolve import hf_path

META_FILES = {
    "audited": "tcn_rul/feature_meta.json",
    "full":    "anomaly_shared/feature_meta.json",
}


@lru_cache(maxsize=4)
def _load_meta(mode: str) -> dict:
    meta = json.loads(Path(hf_path(META_FILES[mode])).read_text())
    cats = meta["categories"]
    meta["onehot_columns"] = (
        [f"chemistry_{c}" for c in cats["chemistry"]]
        + [f"form_factor_{c}" for c in cats["form_factor"]]
    )
    meta["feature_dim_total"] = (
        len(meta["numeric_features"]) + len(meta["onehot_columns"])
    )
    return meta


def _onehot_row(chemistry: str, form_factor, meta: dict) -> np.ndarray:
    cats = meta["categories"]
    chem_vec = [1.0 if c == chemistry else 0.0 for c in cats["chemistry"]]
    ff_vec = [1.0 if c == str(form_factor) else 0.0 for c in cats["form_factor"]]
    return np.asarray(chem_vec + ff_vec, dtype=np.float32)


def _impute_numeric_block(block: pd.DataFrame, meta: dict) -> np.ndarray:
    cols = meta["numeric_features"]
    medians = meta["feature_medians"]
    filled = block[cols].copy()
    for c in cols:
        filled[c] = filled[c].fillna(medians[c])
    return filled.to_numpy(dtype=np.float32)


def _pick_row(cell_df: pd.DataFrame, cycle_n: int) -> pd.DataFrame:
    """Return a single-row dataframe for the cycle closest to `cycle_n`."""
    idx = (cell_df["cycle"] - cycle_n).abs().idxmin()
    return cell_df.loc[[idx]]


def featurize_single_cell(
    cell_df: pd.DataFrame,
    cycle_n: int,
    scaler,
    mode: str = "audited",
) -> np.ndarray:
    """Build a (1, N) scaled feature vector at the given cycle.

    mode='audited' → (1, 32) for XGBoost SoH/RUL/ChemistryRouter
    mode='full'    → (1, 38) for IsoForest/VAE
    """
    if mode not in META_FILES:
        raise ValueError(f"mode must be one of {list(META_FILES)}, got {mode!r}")
    meta = _load_meta(mode)
    row = _pick_row(cell_df, cycle_n)
    x_num = _impute_numeric_block(row, meta)
    x_cat = _onehot_row(
        str(row["chemistry"].iloc[0]), row["form_factor"].iloc[0], meta
    )[None, :]
    x = np.concatenate([x_num, x_cat], axis=1)
    if x.shape != (1, meta["feature_dim_total"]):
        raise AssertionError(
            f"feature dim mismatch: built {x.shape}, expected "
            f"(1, {meta['feature_dim_total']}) for mode={mode!r}"
        )
    return scaler.transform(x).astype(np.float32)


def prepare_tcn_sequence(
    cell_df: pd.DataFrame,
    cycle_n: int,
    scaler,
    seq_len: int = 30,
) -> np.ndarray:
    """Build a (1, seq_len, 32) TCN input ending at cycle_n (audited mode).

    Numerics scaled with the TCN scaler; one-hots appended raw (matches the
    training-time `load_battery_sequences` layout).
    """
    meta = _load_meta("audited")
    sub = cell_df[cell_df["cycle"] <= cycle_n].sort_values("cycle")
    if sub.empty:
        raise ValueError(f"no cycles ≤ {cycle_n} in cell_df")
    window = sub.tail(seq_len)
    if len(window) < seq_len:
        pad = window.iloc[[0]].copy()
        window = pd.concat([pad] * (seq_len - len(window)) + [window], ignore_index=True)

    x_num = _impute_numeric_block(window, meta)
    x_num_scaled = scaler.transform(x_num).astype(np.float32)

    chemistry = str(window["chemistry"].iloc[-1])
    form_factor = window["form_factor"].iloc[-1]
    cat_row = _onehot_row(chemistry, form_factor, meta)
    x_cat = np.tile(cat_row, (seq_len, 1))

    x = np.concatenate([x_num_scaled, x_cat], axis=1)
    return x[None, :, :].astype(np.float32)
