"""NASA PCoE (Kaggle mirror, patrickfleith/nasa-battery-dataset) loader.

The Kaggle mirror provides a clean metadata.csv with one row per *test event*
(charge / discharge / impedance). We aggregate to per-discharge-cycle records
because that's what corresponds to the SoH trajectory for the project.

Source: data/raw/nasa_kaggle/pcoe/cleaned_dataset/
metadata.csv columns: type, start_time, ambient_temperature, battery_id,
                       test_id, uid, filename, Capacity, Re, Rct
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.schema import UNIFIED_COLUMNS

PCOE_ROOT = Path("data/raw/nasa_kaggle/pcoe/cleaned_dataset")
META_CSV = PCOE_ROOT / "metadata.csv"

# NASA PCoE 18650 cells: Sanyo (LiCoO2-graphite), 2 Ah nominal capacity
NOMINAL_AH = 2.0
CHEMISTRY = "LCO"
FORM_FACTOR = "18650"


def load() -> pd.DataFrame:
    if not META_CSV.exists():
        print(f"  [NA_PCOE] metadata.csv not found at {META_CSV}")
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    meta = pd.read_csv(META_CSV)

    # Filter to discharge events only — those have the Capacity field populated
    disc = meta[meta["type"] == "discharge"].copy()
    if disc.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # Per-battery cycle counter (sorted by test_id within battery, since test_id is sequential)
    disc = disc.sort_values(["battery_id", "test_id"]).reset_index(drop=True)
    disc["cycle"] = disc.groupby("battery_id").cumcount() + 1

    capacity = pd.to_numeric(disc["Capacity"], errors="coerce")
    t_amb = pd.to_numeric(disc["ambient_temperature"], errors="coerce")

    out = pd.DataFrame({
        "battery_id": "NA_PCOE_" + disc["battery_id"].astype(str),
        "cycle": disc["cycle"].astype(int),
        "source": "NA_PCOE",
        "chemistry": CHEMISTRY,
        "form_factor": FORM_FACTOR,
        "nominal_Ah": NOMINAL_AH,
        "second_life": False,  # PCoE original cells were never retired/recommissioned
        "capacity_Ah": capacity,
        "soh": capacity / NOMINAL_AH,
        "v_min": float("nan"), "v_max": float("nan"), "v_mean": float("nan"),
        "i_min": float("nan"), "i_max": float("nan"), "i_mean": float("nan"),
        "t_min": float("nan"), "t_max": float("nan"),
        "t_mean": t_amb,
        "t_range": float("nan"),
        "charge_time_s": float("nan"),
        "discharge_time_s": float("nan"),
        "ir_ohm": float("nan"),
        "coulombic_eff": float("nan"),
    })
    # NASA PCoE only exposes scalar `Capacity` per cycle in metadata.csv, no
    # within-cycle waveforms — dQ/dV peak features stay NaN for this source.
    return out.reindex(columns=UNIFIED_COLUMNS)
