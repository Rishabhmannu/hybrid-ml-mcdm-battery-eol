"""CALCE (Kaggle mirror) loader.

The CALCE dataset has many xlsx files per cell, each representing one test
session with a "Statistics" sheet of per-cycle aggregates. We concatenate
all sessions for a cell, sort by Date_Time, and assign global cycle indices.

Source: data/raw/calce/CS2_data/CS2_data/{type_1,type_2}/CS2_NN/CS2_NN/*.xlsx
        data/raw/calce/CX2_data/CX2_data/{type_1,type_2}/CX2_NN/CX2_NN/*.xlsx

Cells are CS2_8, CS2_21, CS2_33–38 (CS2 series) and CX2_16, CX2_31, CX2_33–38 (CX2 series).
All are LCO chemistry, nominal capacity ~1.1 Ah.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loaders.schema import UNIFIED_COLUMNS

CALCE_ROOT = Path("data/raw/calce")
NOMINAL_AH = 1.1
CHEMISTRY = "LCO"
FORM_FACTOR = "prismatic"  # CS2/CX2 are prismatic pouch cells

STAT_COLS_OF_INTEREST = [
    "Cycle_Index", "Date_Time", "Voltage(V)", "Current(A)",
    "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)",
    "Internal_Resistance(Ohm)", "Charge_Time(s)", "DisCharge_Time(s)",
    "Vmax_On_Cycle(V)",
]


def _find_statistics_sheet(xl: pd.ExcelFile) -> str | None:
    for s in xl.sheet_names:
        if s.lower().startswith("statistics"):
            return s
    return None


def _load_one_cell(cell_dir: Path) -> pd.DataFrame:
    """Concatenate all xlsx Statistics sheets in a cell folder, ordered by Date_Time.

    The Statistics sheet stores Discharge_Capacity(Ah) and Charge_Capacity(Ah) as
    *cumulative* values within each test session (xlsx file). We compute per-cycle
    capacity as the diff within each session, taking the FIRST cycle of each
    session as the absolute value (which represents that single-cycle's capacity
    delivered, since cumulative starts from 0 each session).
    """
    xlsxs = sorted(cell_dir.glob("*.xlsx"))
    if not xlsxs:
        return pd.DataFrame()
    parts = []
    for xlsx in xlsxs:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xl = pd.ExcelFile(xlsx, engine="openpyxl")
            sheet = _find_statistics_sheet(xl)
            if not sheet:
                continue
            df = pd.read_excel(xlsx, sheet_name=sheet, engine="openpyxl")
        except Exception as exc:
            print(f"    [CALCE] failed {xlsx.name}: {exc}")
            continue
        keep = [c for c in STAT_COLS_OF_INTEREST if c in df.columns]
        if not keep:
            continue
        df = df[keep].copy()
        df["__file"] = xlsx.name

        # Convert cumulative discharge/charge capacity to per-cycle within this session
        df = df.sort_values("Cycle_Index" if "Cycle_Index" in df.columns else df.columns[0]).reset_index(drop=True)
        for col in ("Discharge_Capacity(Ah)", "Charge_Capacity(Ah)"):
            if col in df.columns:
                cum = pd.to_numeric(df[col], errors="coerce")
                # Per-cycle = current cumulative - previous cumulative (first cycle keeps its value)
                df[col] = cum.diff().fillna(cum)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    full = pd.concat(parts, ignore_index=True)
    # Sort by datetime to recover chronological order across sessions, then renumber globally
    full["Date_Time"] = pd.to_datetime(full["Date_Time"], errors="coerce")
    full = full.dropna(subset=["Date_Time"]).sort_values("Date_Time").reset_index(drop=True)
    full["__global_cycle"] = np.arange(1, len(full) + 1)
    return full


def _emit_cell(cell_dir: Path, series: str, type_label: str) -> pd.DataFrame:
    raw = _load_one_cell(cell_dir)
    if raw.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    cell_name = cell_dir.name
    battery_id = f"CALCE_{cell_name}"

    cap = raw["Discharge_Capacity(Ah)"].astype(float) if "Discharge_Capacity(Ah)" in raw else pd.Series([float("nan")] * len(raw))
    cap_c = raw["Charge_Capacity(Ah)"].astype(float) if "Charge_Capacity(Ah)" in raw else pd.Series([float("nan")] * len(raw))
    v_max_cycle = raw["Vmax_On_Cycle(V)"].astype(float) if "Vmax_On_Cycle(V)" in raw else pd.Series([float("nan")] * len(raw))
    ir = raw["Internal_Resistance(Ohm)"].astype(float) if "Internal_Resistance(Ohm)" in raw else pd.Series([float("nan")] * len(raw))
    cht = raw["Charge_Time(s)"].astype(float) if "Charge_Time(s)" in raw else pd.Series([float("nan")] * len(raw))
    dct = raw["DisCharge_Time(s)"].astype(float) if "DisCharge_Time(s)" in raw else pd.Series([float("nan")] * len(raw))

    out = pd.DataFrame({
        "battery_id": battery_id,
        "cycle": raw["__global_cycle"].astype(int),
        "source": f"CALCE_{series}_{type_label}",
        "chemistry": CHEMISTRY,
        "form_factor": FORM_FACTOR,
        "nominal_Ah": NOMINAL_AH,
        "second_life": False,
        "capacity_Ah": cap,
        "soh": cap / NOMINAL_AH,
        "v_min": float("nan"),
        "v_max": v_max_cycle,
        "v_mean": float("nan"),
        "i_min": float("nan"),
        "i_max": float("nan"),
        "i_mean": float("nan"),
        "t_min": float("nan"),
        "t_max": float("nan"),
        "t_mean": float("nan"),
        "t_range": float("nan"),
        "charge_time_s": cht,
        "discharge_time_s": dct,
        "ir_ohm": ir,
        "coulombic_eff": np.where((cap_c > 0) & cap_c.notna(), cap / cap_c, float("nan")),
    })
    # CALCE only retains pre-aggregated Statistics-sheet records, not within-cycle
    # waveforms — dQ/dV peak features therefore stay NaN for this source. reindex
    # fills any schema columns the dict doesn't carry with NaN.
    return out.reindex(columns=UNIFIED_COLUMNS)


def load() -> pd.DataFrame:
    parts = []
    for series_root_name, series_label in [("CS2_data", "CS2"), ("CX2_data", "CX2")]:
        series_root = CALCE_ROOT / series_root_name / series_root_name
        if not series_root.is_dir():
            continue
        for type_dir in sorted(series_root.glob("type_*")):
            type_label = type_dir.name
            for cell_outer in sorted(type_dir.iterdir()):
                if not cell_outer.is_dir():
                    continue
                # The xlsx files are nested one level deeper (CS2_35/CS2_35/*.xlsx)
                inner = cell_outer / cell_outer.name
                cell_dir = inner if inner.is_dir() else cell_outer
                df = _emit_cell(cell_dir, series_label, type_label)
                if not df.empty:
                    parts.append(df)
                    print(f"  [CALCE/{series_label}/{type_label}/{cell_outer.name}] {len(df)} cycles")
    if not parts:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return pd.concat(parts, ignore_index=True)
