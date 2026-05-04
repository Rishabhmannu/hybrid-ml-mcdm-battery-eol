"""
Stage 12 — End-to-end smoke test.

One battery -> proxy SoH -> grade -> Fuzzy BWM-TOPSIS routing -> unified DPP JSON.

Picks a real cell from unified.parquet, infers SoH from the latest cycle's
capacity-fade trend, classifies it into A-D, runs canonical-criteria TOPSIS
using literature weights from fuzzy_bwm_input.csv, and emits a DPP that
validates against the unified schema.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dpp.schema_mapper import (
    build_dpp,
    grade_from_soh,
    save_dpp,
    validate_against_schema,
)
from src.mcdm.topsis import run_canonical_topsis

UNIFIED_PARQUET = PROJECT_ROOT / "data" / "processed" / "cycling" / "unified.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "dpp_output"


def pick_battery(df: pd.DataFrame, battery_id: str | None) -> pd.DataFrame:
    if battery_id:
        sub = df[df["battery_id"] == battery_id]
        if sub.empty:
            raise SystemExit(f"battery_id {battery_id!r} not found in unified.parquet")
        return sub.sort_values("cycle")

    cycle_counts = df.groupby("battery_id")["cycle"].max()
    candidates = cycle_counts[(cycle_counts >= 100) & (cycle_counts <= 800)]
    if candidates.empty:
        raise SystemExit("No suitable smoke-test battery found")
    chosen = candidates.sort_values().index[len(candidates) // 2]
    return df[df["battery_id"] == chosen].sort_values("cycle")


def _safe_float(v, default=None):
    if v is None or pd.isna(v):
        return default
    return float(v)


def derive_battery_summary(cell_df: pd.DataFrame) -> dict:
    """Roll a per-cycle frame into a single-cell DPP-input dict."""
    cell_df = cell_df.copy()
    last = cell_df.iloc[-1]
    first = cell_df.iloc[0]

    nominal = _safe_float(last["nominal_Ah"]) or _safe_float(first["nominal_Ah"]) or 1.0
    last_cap = _safe_float(last["capacity_Ah"])

    if pd.notna(last["soh"]):
        soh_pct = float(last["soh"]) * 100.0 if last["soh"] <= 1.5 else float(last["soh"])
    elif last_cap is not None:
        soh_pct = max(0.0, min(110.0, (last_cap / nominal) * 100.0))
    else:
        soh_pct = float("nan")

    rolling = cell_df.tail(20)
    if len(rolling) >= 5 and rolling["soh"].notna().sum() >= 5:
        soh_series = rolling["soh"].dropna()
        if soh_series.iloc[-1] <= 1.5:
            soh_series = soh_series * 100.0
        delta_per_cycle = (soh_series.iloc[-1] - soh_series.iloc[0]) / max(1, len(soh_series) - 1)
        if delta_per_cycle < 0:
            cycles_to_eol = max(0, (soh_pct - 60.0) / abs(delta_per_cycle))
            rul = float(cycles_to_eol)
        else:
            rul = None
    else:
        rul = None

    return {
        "battery_id": str(last["battery_id"]),
        "chemistry": str(last["chemistry"]),
        "form_factor": (str(last["form_factor"]) if pd.notna(last["form_factor"]) else None),
        "nominal_Ah": nominal,
        "voltage_min_V": _safe_float(cell_df["v_min"].min(), default=2.0),
        "voltage_max_V": _safe_float(cell_df["v_max"].max(), default=4.2),
        "cycles_completed": int(cell_df["cycle"].max()),
        "soh_percent": soh_pct,
        "rul_remaining_cycles": rul,
        "second_life": bool(last["second_life"]) if pd.notna(last["second_life"]) else False,
        "source": str(last["source"]),
    }


def run(battery_id: str | None) -> Path:
    print("Stage 12 — End-to-end smoke test")
    print("=" * 70)
    print(f"Loading {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_parquet(UNIFIED_PARQUET)
    print(f"  {len(df):,} rows · {df['battery_id'].nunique():,} batteries")

    cell_df = pick_battery(df, battery_id)
    summary = derive_battery_summary(cell_df)
    print(f"\n[1/5] Selected: {summary['battery_id']}")
    print(f"      source={summary['source']}  chemistry={summary['chemistry']}  "
          f"cycles={summary['cycles_completed']}  nominal={summary['nominal_Ah']:.2f} Ah")

    print(f"\n[2/5] SoH inference (capacity-fade proxy)")
    print(f"      SoH = {summary['soh_percent']:.2f}%   "
          f"RUL ≈ {summary['rul_remaining_cycles']!s} cycles")

    grade = grade_from_soh(summary["soh_percent"])
    print(f"\n[3/5] Grade classification: {grade}")

    print(f"\n[4/5] Fuzzy BWM-TOPSIS routing (canonical 6 criteria)")
    topsis_out = run_canonical_topsis(grade)
    weights_str = ", ".join(f"{k}={v:.3f}" for k, v in topsis_out["weights"].items())
    print(f"      weights: {weights_str}")
    print(f"      ranking:")
    for r in topsis_out["ranked"]:
        marker = " <- recommended" if r["rank"] == 1 else ""
        print(f"        rank {r['rank']}  {r['alternative']:25s}  "
              f"closeness={r['closeness']:.4f}{marker}")

    rec = next(r for r in topsis_out["ranked"] if r["rank"] == 1)

    print(f"\n[5/5] Building unified DPP JSON")
    dpp = build_dpp(
        battery_id=summary["battery_id"],
        chemistry=summary["chemistry"],
        form_factor=summary["form_factor"],
        nominal_Ah=summary["nominal_Ah"],
        voltage_min_V=summary["voltage_min_V"],
        voltage_max_V=summary["voltage_max_V"],
        cycles_completed=summary["cycles_completed"],
        soh_percent=summary["soh_percent"],
        rul_remaining_cycles=summary["rul_remaining_cycles"],
        estimation_method="capacity-fade proxy (smoke test); will be replaced by XGBoost SoH (Stage 9)",
        estimation_confidence={
            "metric": "proxy",
            "value": 0.0,
            "validation_set": "none — Stage 12 placeholder",
        },
        data_source="field" if "synthetic" not in summary["source"].lower() else "simulated",
        grade=grade,
        recommended_route=rec["alternative"],
        route_score=rec["closeness"],
        mcdm_criteria=topsis_out["criteria"],
        mcdm_weights=topsis_out["weights"],
        all_route_scores=topsis_out["ranked"],
        second_life=summary["second_life"],
        data_sources=[
            f"unified.parquet ({summary['source']})",
            "data/processed/mcdm_weights/fuzzy_bwm_input.csv",
        ],
        model_artifacts=[
            "src.mcdm.topsis.run_canonical_topsis (literature weights)",
            "capacity-fade proxy (no trained ML model yet)",
        ],
    )

    ok, errors = validate_against_schema(dpp)
    dpp["provenance"]["schema_validated"] = ok
    if not ok:
        print(f"\n[FAIL] DPP failed schema validation:")
        for e in errors[:10]:
            print(f"   - {e}")
        raise SystemExit(1)

    out_path = save_dpp(dpp)
    print(f"\n[OK] DPP written to {out_path.relative_to(PROJECT_ROOT)}")
    print(f"     schema_validated=True   coverage={dpp['provenance']['coverage_pct']*100:.1f}%")

    summary_path = RESULTS_DIR / f"smoke_summary_{summary['battery_id'].replace('/', '_')}.json"
    with open(summary_path, "w") as f:
        json.dump({
            "battery": summary,
            "topsis": topsis_out,
            "grade": grade,
            "recommended_route": rec["alternative"],
            "dpp_coverage_pct": dpp["provenance"]["coverage_pct"],
            "schema_validated": ok,
        }, f, indent=2)
    print(f"     summary: {summary_path.relative_to(PROJECT_ROOT)}")
    return out_path


def main():
    p = argparse.ArgumentParser(description="Stage 12 end-to-end smoke test")
    p.add_argument("--battery-id", default=None,
                   help="Battery ID to use; default = a mid-cycled cell from unified.parquet")
    args = p.parse_args()
    run(args.battery_id)


if __name__ == "__main__":
    main()
