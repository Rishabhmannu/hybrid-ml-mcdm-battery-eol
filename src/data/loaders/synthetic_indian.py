"""Loader for PyBaMM-generated synthetic Indian-context cycling data.

Reads `data/processed/synthetic_indian/synthetic_indian_combined.csv` and maps
the columns into the unified per-cycle schema used by the rest of the pipeline.

The synthetic.py output uses different column names than the unified schema:
  voltage_* → v_*
  temperature_* → t_*
  capacity → capacity_Ah
  ambient_profile, ambient_K, driving_cycle → encoded into source for
    downstream stratification

Each synthetic cell becomes a single battery_id with the source pattern
`SYN_IN_<chemistry>_<thermal_profile>` so train/val/test stratification can
distribute thermal regimes evenly.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.schema import UNIFIED_COLUMNS

SYN_DIRS = [
    Path("data/processed/synthetic_indian"),       # NMC sweep (DFN + degradation)
    Path("data/processed/synthetic_indian_lfp"),   # LFP sweep (isothermal SPMe, no degradation)
]

# Mohtat2020 NMC and Prada2013 LFP both target 18650 form factor in their
# bundled parameter sets. Nominal capacity comes from the simulation itself
# (~5 Ah for Mohtat2020, ~2.3 Ah for Prada2013) — we record it as the
# cycle-1 capacity per cell.
FORM_FACTOR = "18650"


def load() -> pd.DataFrame:
    parts = []
    for d in SYN_DIRS:
        combined = d / "synthetic_indian_combined.csv"
        if combined.exists():
            df = pd.read_csv(combined)
            if not df.empty:
                parts.append(df)
    if not parts:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    raw = pd.concat(parts, ignore_index=True)
    if raw.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # Determine effective nominal per battery (cycle-1 capacity)
    nominal_per_batt = raw.groupby("battery_id")["capacity"].first().to_dict()
    raw["nominal_Ah"] = raw["battery_id"].map(nominal_per_batt)

    # Per-cycle SoH — the synthetic.py file already stores soh as % (0-100),
    # but our unified schema uses 0-1 fraction. Convert.
    soh_frac = (raw["soh"].astype(float) / 100.0).clip(lower=0, upper=1.5)

    out = pd.DataFrame({
        "battery_id": raw["battery_id"].astype(str),
        "cycle": raw["cycle"].astype(int),
        # Encode thermal profile into source for downstream stratification
        "source": "SYN_IN_" + raw["chemistry"].astype(str) + "_" + raw["ambient_profile"].astype(str),
        "chemistry": raw["chemistry"].astype(str),
        "form_factor": FORM_FACTOR,
        "nominal_Ah": raw["nominal_Ah"].astype(float),
        "second_life": False,
        "capacity_Ah": raw["capacity"].astype(float),
        "soh": soh_frac,
        "v_min": raw.get("voltage_min", float("nan")),
        "v_max": raw.get("voltage_max", float("nan")),
        "v_mean": raw.get("voltage_mean", float("nan")),
        "i_min": float("nan"),  # synthetic.py records only mean current
        "i_max": float("nan"),
        "i_mean": raw.get("current_mean", float("nan")),
        "t_min": float("nan"),
        "t_max": raw.get("temperature_max", float("nan")),
        "t_mean": raw.get("temperature_mean", float("nan")),
        "t_range": raw.get("temperature_range", float("nan")),
        "charge_time_s": float("nan"),
        "discharge_time_s": raw.get("time_s", float("nan")),
        "ir_ohm": float("nan"),
        "coulombic_eff": float("nan"),
    })
    # The PyBaMM combined CSVs store per-cycle aggregates only, not within-cycle
    # waveforms — re-running the simulator with full V/I/Q traces would dwarf the
    # gain (synthetic cells are <1 % of the corpus and pinned to train). dQ/dV
    # peak features therefore stay NaN for synthetic rows.
    return out.reindex(columns=UNIFIED_COLUMNS)
