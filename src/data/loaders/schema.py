"""Unified per-cycle schema and normalization helpers.

Every per-source loader emits a DataFrame with these columns. Add only here.
"""
from __future__ import annotations

import re

# Canonical column order. Loaders must produce exactly these columns.
UNIFIED_COLUMNS = [
    # identity
    "battery_id",       # str — source-prefixed (BL_MATR_b1c0, NA_PCOE_B0005, ...)
    "cycle",            # int — 1-indexed
    "source",           # str — BL_<subset> | ST_OSF | ST_8JNR5 | NA_PCOE | NA_RAND_RECOMM | NA_KAGGLE_RW | CALCE | SYN_IN
    # cell metadata (constant per battery_id, repeated per cycle for joinability)
    "chemistry",        # str — NMC | LFP | LCO | NCA | LMO | Zn-ion | Na-ion | other
    "form_factor",      # str — 18650 | 21700 | pouch | prismatic | other | unknown
    "nominal_Ah",       # float — nominal capacity
    "second_life",      # bool — True if cell already cycled before reaching this dataset
    # per-cycle measurements
    "capacity_Ah",      # float — discharge capacity at this cycle
    "soh",              # float — capacity_Ah / nominal_Ah
    "v_min", "v_max", "v_mean",          # floats — voltage stats
    "i_min", "i_max", "i_mean",          # floats — current stats (pos = discharge)
    "t_min", "t_max", "t_mean", "t_range",  # floats — temperature °C (NaN if not measured)
    "charge_time_s",    # float — duration of charge phase
    "discharge_time_s", # float — duration of discharge phase
    "ir_ohm",           # float — internal resistance (NaN if not measured)
    "coulombic_eff",    # float — discharge_Ah / charge_Ah (NaN if charge_Ah missing)
    # per-cycle dQ/dV peak features (Severson 2019 / Greenbank & Howey 2022)
    # — NaN when within-cycle waveforms are not retained by the loader
    "v_peak_dqdv_charge",          # float V — voltage at max dQ/dV during charge
    "dqdv_peak_height_charge",     # float Ah/V — height of charge peak
    "dqdv_peak_width_charge",      # float V — FWHM of charge peak
    "v_peak_dqdv_discharge",       # float V — voltage at max dQ/dV during discharge
    "dqdv_peak_height_discharge",  # float Ah/V — height of discharge peak
    "dqdv_peak_width_discharge",   # float V — FWHM of discharge peak
    "q_at_v_lo",                   # float Ah — charge capacity at V = 3.5 V (Li-ion default)
    "q_at_v_hi",                   # float Ah — charge capacity at V = 3.9 V (Li-ion default)
]


# Canonical chemistry strings + lookup map for raw → canonical.
CHEMISTRY_CANONICAL = {"NMC", "LFP", "LCO", "NCA", "LMO", "Zn-ion", "Na-ion", "other"}

CHEMISTRY_RULES = [
    # (regex_pattern_lowercase, canonical)
    (r"^lfp$|lifepo4|li-fe-po|lithium iron phosphate", "LFP"),
    (r"lini[\d.]*(mn[\d.]*co[\d.]*|co[\d.]*mn[\d.]*)o2|nmc|lico_xnimn|lithium nickel manganese cobalt", "NMC"),
    (r"lico_?2|licoo2|lithium cobalt", "LCO"),
    (r"lin[ic][\d.]*co[\d.]*al[\d.]*o2|nca|lithium nickel cobalt aluminum", "NCA"),
    (r"limn_?2o_?4|lmo|lithium manganese", "LMO"),
    (r"^zinc$|zn[\W_]?ion|zinc[\W_]?ion", "Zn-ion"),
    (r"^sodium$|na[\W_]?ion|sodium[\W_]?ion", "Na-ion"),
]

# Source-name → chemistry fallback (used when cathode_material is Unknown/missing).
# Built from observed BatteryLife subset folder names.
SOURCE_CHEMISTRY_FALLBACK = {
    "NA-ion": "Na-ion",
    "ZN-coin": "Zn-ion",
}


def normalize_chemistry(raw: str | None, source_hint: str | None = None) -> str:
    """Map a raw chemistry string to one of the canonical labels.

    Falls back to source-name lookup when cathode is missing/unknown.
    """
    if raw and str(raw).strip().lower() not in {"unknown", "none", "n/a", ""}:
        s = str(raw).strip().lower()
        for pattern, canonical in CHEMISTRY_RULES:
            if re.search(pattern, s):
                return canonical
    # Fallback: derive from source name (e.g. "NA-ion" subset → Na-ion)
    if source_hint:
        for needle, chem in SOURCE_CHEMISTRY_FALLBACK.items():
            if needle.lower() in source_hint.lower():
                return chem
    return "other"


FORM_FACTOR_RULES = [
    (r"18650|cylindrical_18650", "18650"),
    (r"21700|cylindrical_21700", "21700"),
    (r"pouch", "pouch"),
    (r"prismatic", "prismatic"),
    (r"cylindrical", "cylindrical_other"),
]


def normalize_form_factor(raw: str | None) -> str:
    if not raw:
        return "unknown"
    s = str(raw).strip().lower()
    for pattern, canonical in FORM_FACTOR_RULES:
        if re.search(pattern, s):
            return canonical
    return "other"
