"""NASA Randomized & Recommissioned Battery Dataset loader.

Source: data/raw/nasa/extracted/battery_alt_dataset/{regular_alt,recommissioned,second_life}_batteries/

CSV schema (per battery pack):
  start_time    — datetime; identifies one daily test session (~24 h ≈ 1 cycle)
  time          — relative seconds since start of life
  mode          — -1=discharge, 0=rest, 1=charge
  voltage_charger, temperature_battery — measured at charger
  voltage_load, current_load, temperature_mosfet, temperature_resistor — measured at load board (only during discharge)
  mission_type  — 0=reference (constant 2.5 A) discharge, 1=regular mission

Per-cycle aggregation: we group by (battery, start_time) and compute one
summary row per session. SoH is derived from the per-session integrated
discharge capacity (Coulomb counting on current_load during mode=-1), normalized
by the maximum capacity observed in the first 5 sessions of that battery
(used as effective nominal — these cells lack a published nominal and many are
already retired/recommissioned).
"""
from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from src.data.dqdv_features import compute_dqdv_features
from src.data.loaders.schema import UNIFIED_COLUMNS

NA_ROOT = Path("data/raw/nasa/extracted/battery_alt_dataset")

# 18650 Li-ion cells per the NASA paper; chemistry not specified in detail,
# but typical NASA prognostics datasets use NCM/LCO. Default to NMC for
# project-relevance (modern second-life context) — flag for review.
CHEMISTRY = "NMC"
FORM_FACTOR = "18650"


def _summarize_ref_discharge(grp: pd.DataFrame) -> dict:
    """One row per reference-discharge event (mission_type=0, constant 2.5A)."""
    t = grp["time"].values
    i_load = grp["current_load"].fillna(0).values
    v_load_full = grp["voltage_load"].values
    v_load = grp["voltage_load"].dropna()
    v_charger = grp["voltage_charger"].dropna()
    t_batt = grp["temperature_battery"].dropna()

    capacity_Ah = float("nan")
    discharge_time_s = float("nan")
    q_discharge_cumulative = np.full_like(t, np.nan, dtype=float)
    if len(t) > 1 and np.isfinite(t).all():
        dt = np.diff(t, prepend=t[0])
        # Per-sample incremental discharge capacity (Ah). Negative dt would
        # signal a non-monotonic timestamp; clip to 0.
        dt = np.clip(dt, 0, None)
        q_discharge_cumulative = np.cumsum(np.abs(i_load) * dt) / 3600.0
        capacity_Ah = float(q_discharge_cumulative[-1])
        discharge_time_s = float(t[-1] - t[0])

    # Use voltage_load for discharge V stats (more direct than charger reading)
    v = v_load if len(v_load) else v_charger

    # dQ/dV peak features on the discharge phase. We pass cumulative discharge
    # capacity as `discharge_capacity` and an all-NaN array for charge so
    # compute_dqdv_features only emits the discharge-side fields.
    dqdv = compute_dqdv_features(
        voltage=v_load_full,
        current=i_load,
        charge_capacity=np.full_like(v_load_full, np.nan, dtype=float),
        discharge_capacity=q_discharge_cumulative,
    )

    out = {
        "v_min": float(v.min()) if len(v) else float("nan"),
        "v_max": float(v.max()) if len(v) else float("nan"),
        "v_mean": float(v.mean()) if len(v) else float("nan"),
        "i_min": float(i_load.min()) if len(i_load) else float("nan"),
        "i_max": float(i_load.max()) if len(i_load) else float("nan"),
        "i_mean": float(i_load.mean()) if len(i_load) else float("nan"),
        "t_min": float(t_batt.min()) if len(t_batt) else float("nan"),
        "t_max": float(t_batt.max()) if len(t_batt) else float("nan"),
        "t_mean": float(t_batt.mean()) if len(t_batt) else float("nan"),
        "t_range": float(t_batt.max() - t_batt.min()) if len(t_batt) else float("nan"),
        "capacity_Ah": capacity_Ah,
        "charge_time_s": float("nan"),
        "discharge_time_s": discharge_time_s,
        "ir_ohm": float("nan"),
        "coulombic_eff": float("nan"),
    }
    out.update(dqdv)
    return out


def _process_csv(csv_path: Path, sub_folder: str, second_life: bool) -> pd.DataFrame:
    """Use only mission_type=0 (reference discharge) events as SoH checkpoints.

    This is the clean signal — each is a constant 2.5A discharge that gives an
    unambiguous capacity reading. Regular missions (mission_type=1) have variable
    load and are skipped to avoid integration noise.
    """
    cols = ["start_time", "time", "mode", "voltage_charger", "voltage_load",
            "temperature_battery", "current_load", "mission_type"]
    df = pd.read_csv(csv_path, usecols=cols)
    # Filter to reference discharges only
    ref = df[df["mission_type"] == 0].copy()
    if ref.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    sessions = ref.groupby("start_time", sort=True)
    cycle_rows = []
    for cycle_idx, (_, grp) in enumerate(sessions, start=1):
        s = _summarize_ref_discharge(grp)
        s["cycle"] = cycle_idx
        cycle_rows.append(s)
    if not cycle_rows:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    cyc_df = pd.DataFrame(cycle_rows)

    # Derive effective nominal from the first 5 sessions
    head_caps = cyc_df["capacity_Ah"].dropna().head(5)
    nominal_Ah = float(head_caps.max()) if len(head_caps) else float("nan")
    if not (nominal_Ah and np.isfinite(nominal_Ah) and nominal_Ah > 0):
        nominal_Ah = float("nan")
    cyc_df["nominal_Ah"] = nominal_Ah
    cyc_df["soh"] = cyc_df["capacity_Ah"] / nominal_Ah if nominal_Ah else float("nan")

    battery_id = f"NA_RAND_{sub_folder}_{csv_path.stem}"
    cyc_df["battery_id"] = battery_id
    cyc_df["source"] = f"NA_RAND_{sub_folder}"
    cyc_df["chemistry"] = CHEMISTRY
    cyc_df["form_factor"] = FORM_FACTOR
    cyc_df["second_life"] = second_life
    return cyc_df.reindex(columns=UNIFIED_COLUMNS)


def load() -> pd.DataFrame:
    parts = []
    sub_specs = [
        ("regular_alt_batteries", "regular_alt", False),
        ("recommissioned_batteries", "recommissioned", True),
        ("second_life_batteries", "second_life", True),
    ]
    for folder_name, label, second_life in sub_specs:
        folder = NA_ROOT / folder_name
        if not folder.is_dir():
            print(f"  [NA_RAND/{label}] folder missing")
            continue
        csvs = sorted(folder.glob("battery*.csv"))
        for csv in csvs:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    df = _process_csv(csv, label, second_life)
            except Exception as exc:
                print(f"  [NA_RAND/{label}] FAILED {csv.name}: {exc}")
                continue
            if not df.empty:
                parts.append(df)
                print(f"  [NA_RAND/{label}] {csv.name}: {len(df)} cycles")
    if not parts:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return pd.concat(parts, ignore_index=True)
