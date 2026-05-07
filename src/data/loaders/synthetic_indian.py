"""Loader for synthetic Indian-context cycling data.

Iter-3 (May 2026) ships THREE distinct synthetic cohorts that this loader
folds into the unified per-cycle schema with **cohort-distinct source codes**
so downstream stratification can analyze each cohort separately:

  Source dir                                        Cohort                                Source code prefix
  -----------------------------------------------   -----------------------------------   --------------------------
  data/processed/synthetic_indian_iter3/            PyBaMM NMC, partially_reversible      SYN_IN_PYBAMM_NMC
                                                    plating (60 cells: 40 canonical
                                                    deterministic + 20 randomized)
  data/processed/synthetic_indian_iter3_irreversible/  PyBaMM NMC, irreversible plating   SYN_IN_PYBAMM_NMC_IRREV
                                                    ablation cohort (30 cells, all
                                                    randomized, deep Grade D)
  data/processed/synthetic_indian_iter3_lfp/        NREL BLAST-Lite LFP-Gr 250Ah          SYN_IN_BLAST_LFP
                                                    (40 cells, full Grade A→D, semi-
                                                    empirical aging from Smith/Gasper
                                                    et al. NREL J. Energy Storage 2024)

The Iter-1/Iter-2 paths (`data/processed/synthetic_indian/` and
`data/processed/synthetic_indian_lfp/`) are NOT loaded — those 70 cells stayed
at SoH ≈ 100 % and were superseded by the Iter-3 corpus (see Iteration 3 in
DATASET_IMPLEMENTATION_PLAN.md §12).

Schema mapping into the unified per-cycle schema:
  voltage_*       → v_*
  temperature_*   → t_*
  capacity        → capacity_Ah
  ambient_profile → encoded into source as `<prefix>_<climate>`
  soh (% scale)   → soh (0–1 fraction)

Per-cycle dQ/dV peak features stay NaN for synthetic rows: PyBaMM combined
CSVs store per-cycle aggregates only, and BLAST-Lite is a 0-D semi-empirical
model that doesn't produce within-cycle V/Q traces. NaN dQ/dV is handled
downstream by median imputation in src/data/training_data.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.loaders.schema import UNIFIED_COLUMNS

# Per-cohort directory + source-code prefix. Order matters only for log
# readability; loader concatenates results.
SYN_COHORTS = [
    {
        "dir": Path("data/processed/synthetic_indian_iter3"),
        "combined_csv": "synthetic_indian_combined.csv",
        "source_prefix": "SYN_IN_PYBAMM_NMC",   # canonical + randomized PyBaMM NMC
    },
    {
        "dir": Path("data/processed/synthetic_indian_iter3_irreversible"),
        "combined_csv": "synthetic_indian_combined.csv",
        "source_prefix": "SYN_IN_PYBAMM_NMC_IRREV",  # irreversible-plating ablation
    },
    {
        "dir": Path("data/processed/synthetic_indian_iter3_lfp"),
        "combined_csv": "synthetic_indian_lfp_combined.csv",
        "source_prefix": "SYN_IN_BLAST_LFP",   # NREL BLAST-Lite LFP
    },
]

# Reasonable form-factor mapping per cohort. PyBaMM cells use the OKane2022
# parameter set (LG M50T, 21700 cylindrical). BLAST-Lite cells are the
# Lfp_Gr_250AhPrismatic (large prismatic ~250 Ah, typical Indian-EV form
# factor for Tata/MG packs). Different form factors are useful for
# downstream chemistry-router features.
FORM_FACTOR_BY_PREFIX = {
    "SYN_IN_PYBAMM_NMC":       "21700",
    "SYN_IN_PYBAMM_NMC_IRREV": "21700",
    "SYN_IN_BLAST_LFP":        "prismatic",
}


def _load_one_cohort(cohort: dict) -> pd.DataFrame:
    combined = cohort["dir"] / cohort["combined_csv"]
    if not combined.exists():
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    raw = pd.read_csv(combined)
    if raw.empty:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)

    # Nominal capacity per battery = cycle-1 capacity (matches what the
    # synthetic generator records as the baseline).
    nominal_per_batt = raw.groupby("battery_id")["capacity"].first().to_dict()
    raw = raw.copy()
    raw["nominal_Ah"] = raw["battery_id"].map(nominal_per_batt)

    # SoH: synthetic stores percent (0-100); unified expects 0-1 fraction.
    soh_frac = (raw["soh"].astype(float) / 100.0).clip(lower=0, upper=1.5)

    prefix = cohort["source_prefix"]
    source = prefix + "_" + raw["ambient_profile"].astype(str)

    out = pd.DataFrame({
        "battery_id": raw["battery_id"].astype(str),
        "cycle": raw["cycle"].astype(int),
        "source": source,
        "chemistry": raw["chemistry"].astype(str),
        "form_factor": FORM_FACTOR_BY_PREFIX.get(prefix, "21700"),
        "nominal_Ah": raw["nominal_Ah"].astype(float),
        "second_life": False,
        "capacity_Ah": raw["capacity"].astype(float),
        "soh": soh_frac,
        "v_min": raw.get("voltage_min", float("nan")),
        "v_max": raw.get("voltage_max", float("nan")),
        "v_mean": raw.get("voltage_mean", float("nan")),
        "i_min": float("nan"),
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
    return out.reindex(columns=UNIFIED_COLUMNS)


def load() -> pd.DataFrame:
    parts = [_load_one_cohort(c) for c in SYN_COHORTS]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame(columns=UNIFIED_COLUMNS)
    return pd.concat(parts, ignore_index=True).reindex(columns=UNIFIED_COLUMNS)
