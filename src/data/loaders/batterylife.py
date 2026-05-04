"""BatteryLife loader — handles all 18 sub-sources (MATR, CALCE, HUST, Stanford,
Stanford_2, ISU_ILCC, Tongji, RWTH, SDU, ZN-coin, NA-ion, CALB, MICH, MICH_EXP,
HNEI, SNL, UL_PUR, XJTU).

All 1,382 cells share an identical .pkl schema:
  top-level: cell_id, form_factor, cathode_material, anode_material,
             nominal_capacity_in_Ah, depth_of_charge, depth_of_discharge,
             already_spent_cycles, max/min_voltage_limit_in_V, ...
  cycle_data[i]: {cycle_number, current_in_A, voltage_in_V,
                  charge_capacity_in_Ah, discharge_capacity_in_Ah,
                  time_in_s, temperature_in_C, internal_resistance_in_ohm}

Per-cycle aggregation: this loader summarizes each cycle's time-series arrays
into a single row of stats (min/max/mean/range), discarding the raw waveforms
to keep memory footprint manageable. The raw waveforms remain available in the
.pkl files if downstream needs them (e.g. dQ/dV feature engineering).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.data.dqdv_features import compute_dqdv_features
from src.data.loaders.schema import (
    UNIFIED_COLUMNS,
    normalize_chemistry,
    normalize_form_factor,
)


BL_ROOT = Path("data/raw/batterylife/batterylife_processed")

# Subdirectories that contain .pkl cell files (excludes labels and READMEs)
SUBSETS = [
    "CALB", "CALCE", "HNEI", "HUST", "ISU_ILCC", "MATR", "MICH", "MICH_EXP",
    "NA-ion", "RWTH", "SDU", "SNL", "Stanford", "Stanford_2", "Tongji",
    "UL_PUR", "XJTU", "ZN-coin",
]


def _arr(x) -> np.ndarray:
    """Coerce list/None to a 1-D float ndarray (NaN for None)."""
    if x is None:
        return np.array([], dtype=float)
    return np.asarray(x, dtype=float)


def _safe_stat(arr: np.ndarray, fn) -> float:
    if arr.size == 0:
        return float("nan")
    try:
        return float(fn(arr))
    except Exception:
        return float("nan")


def _summarize_cycle(cell_meta: dict, cyc: dict) -> dict:
    v = _arr(cyc.get("voltage_in_V"))
    i = _arr(cyc.get("current_in_A"))
    t = _arr(cyc.get("temperature_in_C"))
    qc = _arr(cyc.get("charge_capacity_in_Ah"))
    qd = _arr(cyc.get("discharge_capacity_in_Ah"))
    ts = _arr(cyc.get("time_in_s"))

    # Discharge capacity = max(discharge_capacity_in_Ah). Many BatteryLife
    # cycles also reset charge/discharge capacity at the start of each phase.
    capacity_Ah = _safe_stat(qd, np.max)
    if not np.isfinite(capacity_Ah) and qc.size > 0:
        capacity_Ah = _safe_stat(qc, np.max)

    nominal = float(cell_meta.get("nominal_capacity_in_Ah") or float("nan"))
    soh = capacity_Ah / nominal if nominal and np.isfinite(capacity_Ah) else float("nan")

    # Charge/discharge times: rough heuristic from current sign (positive in BL = discharge by convention?
    # Actually BatteryLife's sign convention is implicit; we use absolute current for stats and split
    # phases by current sign for time accounting.
    charge_time_s = float("nan")
    discharge_time_s = float("nan")
    if i.size > 1 and ts.size == i.size:
        dts = np.diff(ts)
        # mask middle: include both endpoints by using sign of i[:-1]
        charge_mask = i[:-1] < 0      # convention: charge = negative current
        discharge_mask = i[:-1] > 0
        if dts.size:
            charge_time_s = float(dts[charge_mask].sum())
            discharge_time_s = float(dts[discharge_mask].sum())

    coulombic_eff = float("nan")
    qc_max = _safe_stat(qc, np.max)
    qd_max = _safe_stat(qd, np.max)
    if np.isfinite(qc_max) and qc_max > 0 and np.isfinite(qd_max):
        coulombic_eff = qd_max / qc_max

    cycle_num = int(cyc.get("cycle_number") or -1)

    ir = cyc.get("internal_resistance_in_ohm")
    ir_ohm = float(ir) if ir is not None and np.isfinite(float(ir)) else float("nan")

    # Per-cycle dQ/dV peak features (NaN if waveform is too sparse).
    dqdv = compute_dqdv_features(v, i, qc, qd)

    return {
        "battery_id": cell_meta["battery_id"],
        "cycle": cycle_num,
        "source": cell_meta["source"],
        "chemistry": cell_meta["chemistry"],
        "form_factor": cell_meta["form_factor"],
        "nominal_Ah": nominal,
        "second_life": cell_meta["second_life"],
        "capacity_Ah": capacity_Ah,
        "soh": soh,
        "v_min": _safe_stat(v, np.min),
        "v_max": _safe_stat(v, np.max),
        "v_mean": _safe_stat(v, np.mean),
        "i_min": _safe_stat(i, np.min),
        "i_max": _safe_stat(i, np.max),
        "i_mean": _safe_stat(i, np.mean),
        "t_min": _safe_stat(t, np.min),
        "t_max": _safe_stat(t, np.max),
        "t_mean": _safe_stat(t, np.mean),
        "t_range": _safe_stat(t, np.max) - _safe_stat(t, np.min) if t.size else float("nan"),
        "charge_time_s": charge_time_s,
        "discharge_time_s": discharge_time_s,
        "ir_ohm": ir_ohm,
        "coulombic_eff": coulombic_eff,
        **dqdv,
    }


def _load_cell(pkl_path: Path, subset: str) -> list[dict]:
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict) or "cycle_data" not in obj:
        return []
    cell_id_raw = str(obj.get("cell_id") or pkl_path.stem)
    # Source-prefix the battery id for global uniqueness.
    battery_id = f"BL_{subset}_{cell_id_raw}".replace(" ", "_")
    cell_meta = {
        "battery_id": battery_id,
        "source": f"BL_{subset}",
        "chemistry": normalize_chemistry(obj.get("cathode_material"), source_hint=subset),
        "form_factor": normalize_form_factor(obj.get("form_factor")),
        "second_life": (obj.get("already_spent_cycles") or 0) > 0,
        "nominal_capacity_in_Ah": obj.get("nominal_capacity_in_Ah"),
    }
    rows = []
    for cyc in obj["cycle_data"]:
        if not isinstance(cyc, dict):
            continue
        rows.append(_summarize_cycle(cell_meta, cyc))
    return rows


def iter_batterylife() -> Iterator[pd.DataFrame]:
    """Yield one DataFrame per .pkl file. Use this for streaming/incremental write."""
    for subset in SUBSETS:
        sub = BL_ROOT / subset
        if not sub.is_dir():
            continue
        pkls = sorted(sub.glob("*.pkl"))
        for pkl in pkls:
            try:
                rows = _load_cell(pkl, subset)
            except Exception as exc:
                print(f"  [BL/{subset}] FAILED {pkl.name}: {exc}")
                continue
            if rows:
                df = pd.DataFrame(rows, columns=UNIFIED_COLUMNS)
                yield df


def load_all(limit_per_subset: int | None = None) -> pd.DataFrame:
    """Load all BatteryLife cells into a single DataFrame. Memory-bounded with limit_per_subset."""
    parts = []
    for subset in SUBSETS:
        sub = BL_ROOT / subset
        if not sub.is_dir():
            continue
        pkls = sorted(sub.glob("*.pkl"))
        if limit_per_subset:
            pkls = pkls[:limit_per_subset]
        n_loaded = 0
        for pkl in pkls:
            try:
                rows = _load_cell(pkl, subset)
            except Exception as exc:
                print(f"  [BL/{subset}] FAILED {pkl.name}: {exc}")
                continue
            if rows:
                parts.append(pd.DataFrame(rows, columns=UNIFIED_COLUMNS))
                n_loaded += 1
        print(f"  [BL/{subset}] {n_loaded}/{len(pkls)} cells loaded")
    if not parts:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return pd.concat(parts, ignore_index=True)
