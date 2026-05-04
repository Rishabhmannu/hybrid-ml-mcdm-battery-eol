"""Unifier: orchestrates per-source loaders, applies quality filters, and writes
a single parquet at data/processed/cycling/unified.parquet.

Quality filters (per DATASET_IMPLEMENTATION_PLAN.md §5):
  - Voltage 2.0 ≤ v_mean ≤ 4.5 V (drop rows where v_mean is out of range)
  - Temperature −10 ≤ t_mean ≤ 60 °C (drop only if t_mean is present and OOR)
  - SoH plausibility: 0.0 ≤ soh ≤ 1.5 (allow some headroom for early-life > 1)
  - Drop battery if > 20% of its rows are missing capacity_Ah / soh
  - Drop duplicate (battery_id, cycle) keeping the row with max capacity_Ah
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import batterylife as ld_bl    # noqa: E402
from src.data.loaders import nasa_pcoe as ld_pcoe    # noqa: E402
from src.data.loaders import nasa_rand_recomm as ld_nrr  # noqa: E402
from src.data.loaders import calce as ld_calce       # noqa: E402
from src.data.loaders import stanford_osf as ld_osf  # noqa: E402
from src.data.loaders import stanford_8jnr5 as ld_8j # noqa: E402
from src.data.loaders import nasa_kaggle_random as ld_kr  # noqa: E402
from src.data.loaders import synthetic_indian as ld_syn  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "cycling"
OUT_PARQUET = OUT_DIR / "unified.parquet"


def apply_quality_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Quality filters with chemistry- and source-aware voltage bounds.

    Voltage windows (per cell):
      - Li-ion (NMC, LFP, LCO, NCA, LMO):  2.0 – 4.5 V
      - Zn-ion:                             0.8 – 2.0 V
      - Na-ion:                             1.5 – 4.5 V
      - other / unknown:                    skip filter (informational only)

    Sources that report multi-cell *pack* voltage (e.g. NASA Rand&Recomm has
    2-cell packs at ~7 V) are exempt from the per-cell window — the column is
    documented as pack-level for those rows.
    """
    stats: dict = {"rows_in": int(len(df))}

    # Build per-row voltage bounds based on chemistry
    chem = df["chemistry"].fillna("other")
    v_lower = chem.map({"NMC": 2.0, "LFP": 2.0, "LCO": 2.0, "NCA": 2.0, "LMO": 2.0,
                         "Zn-ion": 0.8, "Na-ion": 1.5}).fillna(0.0)
    v_upper = chem.map({"NMC": 4.5, "LFP": 4.5, "LCO": 4.5, "NCA": 4.5, "LMO": 4.5,
                         "Zn-ion": 2.0, "Na-ion": 4.5}).fillna(99.0)

    # Sources reporting pack-level voltage are exempt
    PACK_SOURCES = {"NA_RAND_regular_alt", "NA_RAND_recommissioned", "NA_RAND_second_life"}
    pack_exempt = df["source"].isin(PACK_SOURCES)

    v_ok = df["v_mean"].isna() | pack_exempt | (df["v_mean"].between(v_lower.values, v_upper.values))
    # The vectorized between with array bounds does element-wise comparison
    v_ok = df["v_mean"].isna() | pack_exempt | ((df["v_mean"] >= v_lower) & (df["v_mean"] <= v_upper))
    n_v_drop = int((~v_ok).sum())
    df = df[v_ok].copy()
    stats["dropped_voltage_oor"] = n_v_drop

    # --- Temperature range filter (only when t_mean is present)
    t_present = df["t_mean"].notna()
    t_in_range = (~t_present) | df["t_mean"].between(-10.0, 60.0)
    n_t_drop = int((~t_in_range).sum())
    df = df[t_in_range].copy()
    stats["dropped_temp_oor"] = n_t_drop

    # --- SoH plausibility filter
    soh_ok = df["soh"].isna() | df["soh"].between(0.0, 1.5)
    n_soh_drop = int((~soh_ok).sum())
    df = df[soh_ok].copy()
    stats["dropped_soh_implausible"] = n_soh_drop

    # --- Drop duplicate (battery_id, cycle) keeping max capacity_Ah
    before_dedup = len(df)
    df = (df.sort_values(["battery_id", "cycle", "capacity_Ah"], ascending=[True, True, False])
            .drop_duplicates(subset=["battery_id", "cycle"], keep="first"))
    stats["dropped_duplicate_cycles"] = before_dedup - len(df)

    # --- Drop batteries with > 20% missing capacity_Ah
    miss_per_battery = df.groupby("battery_id")["capacity_Ah"].apply(lambda s: s.isna().mean())
    drop_ids = miss_per_battery[miss_per_battery > 0.20].index.tolist()
    n_dropped_batts = len(drop_ids)
    if drop_ids:
        df = df[~df.battery_id.isin(drop_ids)].copy()
    stats["dropped_batteries_missing_capacity"] = n_dropped_batts
    stats["dropped_batteries_ids"] = drop_ids[:10]

    # --- Z-score outlier flags (informational; do NOT drop)
    for col in ("capacity_Ah", "v_mean", "t_mean"):
        if col not in df.columns:
            continue
        s = df[col]
        if s.notna().sum() < 30:
            continue
        z = (s - s.mean()) / s.std()
        df[f"{col}_z_outlier"] = (z.abs() > 3).fillna(False)

    stats["rows_out"] = int(len(df))
    return df, stats


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Backward-only derived features (no future leakage)."""
    df = df.sort_values(["battery_id", "cycle"]).reset_index(drop=True)
    g = df.groupby("battery_id", group_keys=False)

    # Capacity drop rate (cycle-to-cycle delta)
    df["capacity_delta"] = g["capacity_Ah"].diff()
    df["soh_delta"] = g["soh"].diff()

    # Rolling-window stats (5-cycle, 20-cycle) — backward-only
    for win in (5, 20):
        df[f"capacity_roll{win}_mean"] = g["capacity_Ah"].transform(
            lambda s: s.shift(0).rolling(win, min_periods=2).mean()
        )
        df[f"capacity_roll{win}_std"] = g["capacity_Ah"].transform(
            lambda s: s.shift(0).rolling(win, min_periods=2).std()
        )

    # Cumulative cycle count
    df["cycle_count_so_far"] = g.cumcount() + 1
    return df


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BUILDING UNIFIED CYCLING PARQUET")
    print("=" * 70)

    sources = [
        ("BatteryLife (18 subsets)", ld_bl.load_all),
        ("NASA PCoE",                ld_pcoe.load),
        ("NASA Randomized & Recommissioned", ld_nrr.load),
        ("CALCE",                    ld_calce.load),
        ("Stanford OSF (stub)",      ld_osf.load),
        ("Stanford 8jnr5 (stub)",    ld_8j.load),
        ("NASA Random v1 Kaggle (stub)", ld_kr.load),
        ("Synthetic Indian (PyBaMM)", ld_syn.load),
    ]

    parts: list[pd.DataFrame] = []
    per_source_counts: dict = {}
    for name, fn in sources:
        print(f"\n[loading] {name}")
        t0 = time.time()
        try:
            df = fn()
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        elapsed = time.time() - t0
        n = len(df)
        n_batts = df["battery_id"].nunique() if n else 0
        per_source_counts[name] = {"rows": n, "batteries": n_batts, "seconds": round(elapsed, 1)}
        print(f"  -> {n:,} rows, {n_batts} batteries ({elapsed:.1f}s)")
        if n:
            parts.append(df)

    if not parts:
        print("\nNothing loaded. Aborting.")
        return 1

    full = pd.concat(parts, ignore_index=True)
    print(f"\n[combine] total before filters: {len(full):,} rows, {full['battery_id'].nunique()} batteries")

    print("\n[filtering]")
    cleaned, qstats = apply_quality_filters(full)
    for k, v in qstats.items():
        print(f"  {k}: {v}")

    print("\n[derived features]")
    cleaned = add_derived_features(cleaned)
    print(f"  added: capacity_delta, soh_delta, capacity_roll5/20_*, cycle_count_so_far")

    # Write parquet
    cleaned.to_parquet(OUT_PARQUET, index=False)
    sz_mb = OUT_PARQUET.stat().st_size / 1e6
    print(f"\n[write] {OUT_PARQUET.relative_to(PROJECT_ROOT)} ({sz_mb:.1f} MB)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Final rows:       {len(cleaned):,}")
    print(f"Unique batteries: {cleaned['battery_id'].nunique():,}")
    print(f"Unique sources:   {cleaned['source'].nunique()}")
    print(f"\nPer-source counts (after filtering):")
    print(cleaned.groupby("source").agg(
        rows=("battery_id", "size"),
        batteries=("battery_id", "nunique"),
    ).to_string())
    print(f"\nChemistry distribution:")
    print(cleaned["chemistry"].value_counts().to_string())
    print(f"\nForm-factor distribution:")
    print(cleaned["form_factor"].value_counts().to_string())
    print(f"\nSecond-life cells: {cleaned[cleaned.second_life]['battery_id'].nunique()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
