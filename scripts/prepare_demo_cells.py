"""
Stage 13 — Curate the Streamlit demo cell set.

Reads `data/processed/cycling/unified.parquet` (1,581 batteries) and stratifies
a small, representative subset across (chemistry × grade) plus a couple of
Indian-context synthetic cells. The output is written to
`frontend/data/demo_cells.parquet` (per-cycle rows for every picked cell) and
`frontend/data/demo_cells_manifest.json` (per-battery summary the Streamlit
sidebar can render without loading the parquet).

Target matrix (per FRONTEND_DESIGN.md §6):
    NMC      : A, B, C, D       (4 real cells)
    LFP      : A, B, C, D       (4 real cells)
    NCA      : A, B             (2 real cells)
    LCO      : B, D             (2 real cells)
    Zn-ion   : A, C             (2 real cells)
    Na-ion   : A, B             (2 real cells)
    other    : A                (1 real cell)
    synthetic: PyBaMM-NMC + BLAST-LFP   (2 Indian-context cells)
    ─────────────────────────────────────
    Total    : 19 cells

Grading uses src.data.training_data.soh_to_grade (A>80%, B>60%, C>40%, D≤40%)
applied to the last observed SoH.

Usage
-----
    python scripts/prepare_demo_cells.py
    python scripts/prepare_demo_cells.py --min-cycles 50 --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import soh_to_grade

UNIFIED_PARQUET = PROJECT_ROOT / "data" / "processed" / "cycling" / "unified.parquet"
OUT_DIR = PROJECT_ROOT / "frontend" / "data"
OUT_PARQUET = OUT_DIR / "demo_cells.parquet"
OUT_MANIFEST = OUT_DIR / "demo_cells_manifest.json"

EOL_THRESHOLD = 0.80  # matches training_data.EOL_THRESHOLD

# (chemistry, grade): how many real cells to pick. Synthetic handled separately.
REAL_TARGETS: dict[tuple[str, str], int] = {
    ("NMC",    "A"): 1, ("NMC",    "B"): 1, ("NMC",    "C"): 1, ("NMC",    "D"): 1,
    ("LFP",    "A"): 1, ("LFP",    "B"): 1, ("LFP",    "C"): 1, ("LFP",    "D"): 1,
    ("NCA",    "A"): 1, ("NCA",    "B"): 1,
    ("LCO",    "B"): 1, ("LCO",    "D"): 1,
    ("Zn-ion", "A"): 1, ("Zn-ion", "C"): 1,
    ("Na-ion", "A"): 1, ("Na-ion", "B"): 1,
    ("other",  "A"): 1,
}

# Synthetic Indian-context picks — one per simulator. Climate regions chosen
# for narrative variety (harsh summer vs monsoon).
SYNTHETIC_SOURCE_PATTERNS = [
    "SYN_IN_PYBAMM_NMC_Delhi_summer",     # PyBaMM electrochemical — harsh summer
    "SYN_IN_BLAST_LFP_Mumbai_monsoon",    # BLAST-Lite semi-empirical — humid stress
]


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-battery roll-up: source, chemistry, grade, censoring, cycle count."""
    g = df.groupby("battery_id", sort=False)
    last = g.tail(1).set_index("battery_id")
    first = g.head(1).set_index("battery_id")
    summary = pd.DataFrame({
        "source":        last["source"],
        "chemistry":     last["chemistry"],
        "form_factor":   last["form_factor"],
        "nominal_Ah":    last["nominal_Ah"],
        "second_life":   last["second_life"],
        "n_cycles":      g["cycle"].nunique(),
        "cycle_max":     g["cycle"].max(),
        "soh_first":     first["soh"],
        "soh_last":      last["soh"],
        "soh_min":       g["soh"].min(),
    }).reset_index()
    summary["censored"] = summary["soh_min"] >= EOL_THRESHOLD
    soh_pct_last = (summary["soh_last"].clip(0, 1.5) * 100.0).to_numpy()
    summary["grade_at_end"] = soh_to_grade(soh_pct_last)
    return summary


def _quality_filter(summary: pd.DataFrame, min_cycles: int) -> pd.DataFrame:
    """Drop trajectories that are too short or have a NaN final SoH."""
    keep = (
        summary["n_cycles"].ge(min_cycles)
        & summary["soh_last"].notna()
        & summary["chemistry"].notna()
    )
    return summary[keep].copy()


def _pick_representative(candidates: pd.DataFrame, rng: np.random.Generator) -> pd.Series | None:
    """Pick the cell closest to the candidate-pool median cycle count.

    Median is more demo-friendly than min/max — short cells underwhelm,
    huge cells (12k+ cycles) make the plot axis silly. Ties broken randomly
    for stability across runs with the same seed.
    """
    if candidates.empty:
        return None
    cands = candidates.copy()
    target = cands["n_cycles"].median()
    cands["dist"] = (cands["n_cycles"] - target).abs()
    min_dist = cands["dist"].min()
    tied = cands[cands["dist"] == min_dist]
    if len(tied) == 1:
        return tied.iloc[0]
    return tied.iloc[int(rng.integers(0, len(tied)))]


def stratify_picks(summary: pd.DataFrame, rng: np.random.Generator) -> list[dict]:
    """Run the (chemistry × grade) + synthetic stratification and return
    a list of pick records (dicts with battery_id + metadata + reason)."""
    is_synthetic = summary["source"].str.startswith("SYN_IN_", na=False)
    real_summary = summary[~is_synthetic]
    syn_summary = summary[is_synthetic]

    picks: list[dict] = []
    seen_ids: set[str] = set()

    for (chem, grade), n in REAL_TARGETS.items():
        pool = real_summary[
            (real_summary["chemistry"] == chem)
            & (real_summary["grade_at_end"] == grade)
            & (~real_summary["battery_id"].isin(seen_ids))
        ]
        for _ in range(n):
            row = _pick_representative(pool, rng)
            if row is None:
                print(f"  [skip] {chem}/grade {grade}: no candidates after filtering")
                break
            picks.append({
                "battery_id": row["battery_id"],
                "reason": f"real · {chem} · grade {grade}",
                **{k: row[k] for k in (
                    "source", "chemistry", "form_factor", "nominal_Ah",
                    "second_life", "n_cycles", "cycle_max",
                    "soh_first", "soh_last", "soh_min",
                    "censored", "grade_at_end",
                )},
            })
            seen_ids.add(row["battery_id"])
            pool = pool[pool["battery_id"] != row["battery_id"]]

    for pattern in SYNTHETIC_SOURCE_PATTERNS:
        pool = syn_summary[
            (syn_summary["source"] == pattern)
            & (~syn_summary["battery_id"].isin(seen_ids))
        ]
        row = _pick_representative(pool, rng)
        if row is None:
            print(f"  [skip] synthetic '{pattern}': no candidates")
            continue
        picks.append({
            "battery_id": row["battery_id"],
            "reason": f"synthetic · Indian-context · {pattern}",
            **{k: row[k] for k in (
                "source", "chemistry", "form_factor", "nominal_Ah",
                "second_life", "n_cycles", "cycle_max",
                "soh_first", "soh_last", "soh_min",
                "censored", "grade_at_end",
            )},
        })
        seen_ids.add(row["battery_id"])
    return picks


def _manifest_entry(pick: dict) -> dict:
    """Convert a pick record (with numpy types) into JSON-safe dict for the
    UI sidebar. Drops the heavy fields and rounds floats."""
    def f(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        if isinstance(v, (np.floating,)):
            return round(float(v), 4)
        if isinstance(v, (np.integer, np.bool_)):
            return v.item()
        return v
    return {
        "battery_id": str(pick["battery_id"]),
        "source": str(pick["source"]),
        "chemistry": str(pick["chemistry"]),
        "form_factor": (str(pick["form_factor"])
                        if pd.notna(pick["form_factor"]) else None),
        "nominal_Ah": f(pick["nominal_Ah"]),
        "second_life": bool(pick["second_life"]) if pd.notna(pick["second_life"]) else False,
        "grade_at_end": str(pick["grade_at_end"]),
        "censored": bool(pick["censored"]),
        "n_cycles": int(pick["n_cycles"]),
        "cycle_max": int(pick["cycle_max"]),
        "soh_first": f(pick["soh_first"]),
        "soh_last": f(pick["soh_last"]),
        "soh_min": f(pick["soh_min"]),
        "reason": pick["reason"],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Curate demo cell set for Streamlit frontend")
    p.add_argument("--min-cycles", type=int, default=50,
                   help="Drop cells with fewer than this many cycles (default %(default)s)")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed for tie-breaking among equally-good candidates")
    args = p.parse_args()

    print("Stage 13 — Demo-cell curation (frontend prep)")
    print("=" * 70)
    print(f"[Load] {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_parquet(UNIFIED_PARQUET)
    print(f"  {len(df):,} rows · {df['battery_id'].nunique():,} batteries")

    print("\n[1/4] Building per-battery summary")
    summary = _build_summary(df)
    print(f"  {len(summary)} batteries summarised")

    print(f"\n[2/4] Quality filter (min_cycles ≥ {args.min_cycles})")
    summary = _quality_filter(summary, args.min_cycles)
    print(f"  {len(summary)} batteries after filtering")

    print("\n[3/4] Stratified picks across (chemistry × grade) + synthetic")
    rng = np.random.default_rng(args.seed)
    picks = stratify_picks(summary, rng)
    if not picks:
        raise SystemExit("No cells selected — aborting.")
    pick_df = pd.DataFrame(picks)
    print(f"  picked {len(pick_df)} cells:")
    cols = ["battery_id", "chemistry", "grade_at_end", "source",
            "n_cycles", "soh_last", "censored"]
    print(pick_df[cols].to_string(index=False, float_format="%.3f"))

    print(f"\n[4/4] Writing demo set")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picked_ids = pick_df["battery_id"].tolist()
    demo_df = df[df["battery_id"].isin(picked_ids)].sort_values(
        ["battery_id", "cycle"]
    ).reset_index(drop=True)
    demo_df.to_parquet(OUT_PARQUET, index=False)
    size_mb = OUT_PARQUET.stat().st_size / (1024 * 1024)
    print(f"  parquet  → {OUT_PARQUET.relative_to(PROJECT_ROOT)}  "
          f"({len(demo_df):,} rows · {size_mb:.2f} MB)")

    manifest = {
        "schema_version": "1.0",
        "n_cells": len(pick_df),
        "n_real": int((~pick_df["source"].str.startswith("SYN_IN_")).sum()),
        "n_synthetic": int(pick_df["source"].str.startswith("SYN_IN_").sum()),
        "cells": [_manifest_entry(p) for p in picks],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"  manifest → {OUT_MANIFEST.relative_to(PROJECT_ROOT)}  "
          f"({len(manifest['cells'])} entries)")

    print(f"\n[Summary]")
    print(f"  chemistries covered: {sorted(pick_df['chemistry'].unique().tolist())}")
    print(f"  grades covered:      {sorted(pick_df['grade_at_end'].unique().tolist())}")
    print(f"  real / synthetic:    {manifest['n_real']} / {manifest['n_synthetic']}")
    print(f"  censored cells:      {int(pick_df['censored'].sum())} of {len(pick_df)}")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
