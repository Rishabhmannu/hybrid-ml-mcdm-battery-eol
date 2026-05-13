"""
Stage 12 — End-to-end smoke test.

One battery -> XGBoost SoH (audited) -> XGBoost RUL (audited+uncensored) ->
grade -> Fuzzy BWM-TOPSIS routing -> unified DPP JSON.

Picks a real cell from unified.parquet, runs the cell's last observed cycle
through the deployable ML models (audited XGBoost SoH for current SoH, audited
XGBoost RUL for remaining-life), classifies into A-D, runs canonical-criteria
TOPSIS using literature weights from fuzzy_bwm_input.csv, and emits a DPP that
validates against the unified schema.

Iter-3 §3.12 update (2026-05-08): replaced capacity-fade SoH proxy and linear
RUL extrapolation with actual ML predictions. RUL uses the Exp 2 winner —
audited XGBoost trained on uncensored cells only, the only RUL model to pass
the 2 % test-RMSE-of-range gate (1.92 % on uncensored test).
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import load_feature_bundle, soh_to_grade
from src.dpp.schema_mapper import (
    build_dpp,
    grade_from_soh,
    save_dpp,
    validate_against_schema,
)
from src.mcdm.topsis import run_canonical_topsis

UNIFIED_PARQUET = PROJECT_ROOT / "data" / "processed" / "cycling" / "unified.parquet"
RESULTS_DIR = PROJECT_ROOT / "results" / "dpp_output"

SOH_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_soh" / "xgboost_soh_audited.json"
RUL_MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_rul" / "xgboost_rul_audited_uncensored.json"


def _load_ml_models():
    """Return (soh_model, rul_model) — both audited XGBoost regressors."""
    if not SOH_MODEL_PATH.exists():
        raise SystemExit(f"Missing SoH model: {SOH_MODEL_PATH.relative_to(PROJECT_ROOT)} "
                         "— train via `scripts/train_xgboost_soh.py --exclude-capacity`")
    if not RUL_MODEL_PATH.exists():
        raise SystemExit(f"Missing RUL model: {RUL_MODEL_PATH.relative_to(PROJECT_ROOT)} "
                         "— train via `scripts/train_xgboost_rul.py --exclude-capacity --exclude-censored`")
    soh_model = xgb.XGBRegressor()
    soh_model.load_model(str(SOH_MODEL_PATH))
    rul_model = xgb.XGBRegressor()
    rul_model.load_model(str(RUL_MODEL_PATH))
    return soh_model, rul_model


def _find_battery_in_bundle(bundle, battery_id: str):
    """Return (X_last_cycle, last_cycle_num, split_name) for the chosen battery,
    or raise if not found in any split."""
    for split in ("train", "val", "test"):
        bids = getattr(bundle, f"{split}_battery_ids")
        mask = (bids == battery_id).to_numpy()
        if mask.any():
            X = getattr(bundle, f"X_{split}")
            cycles = getattr(bundle, f"{split}_cycles").to_numpy()
            cell_idx = np.where(mask)[0]
            local_last = cell_idx[np.argmax(cycles[cell_idx])]
            return X[local_last:local_last + 1], int(cycles[local_last]), split
    raise SystemExit(f"battery_id {battery_id!r} not found in any split of unified.parquet")


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


def derive_battery_summary(cell_df: pd.DataFrame, soh_pred_pct: float, rul_pred_cycles: float) -> dict:
    """Roll a per-cycle frame into a single-cell DPP-input dict using ML predictions
    for SoH and RUL (both from audited XGBoost models per Iter-3 §3.12)."""
    cell_df = cell_df.copy()
    last = cell_df.iloc[-1]
    first = cell_df.iloc[0]

    nominal = _safe_float(last["nominal_Ah"]) or _safe_float(first["nominal_Ah"]) or 1.0

    return {
        "battery_id": str(last["battery_id"]),
        "chemistry": str(last["chemistry"]),
        "form_factor": (str(last["form_factor"]) if pd.notna(last["form_factor"]) else None),
        "nominal_Ah": nominal,
        "voltage_min_V": _safe_float(cell_df["v_min"].min(), default=2.0),
        "voltage_max_V": _safe_float(cell_df["v_max"].max(), default=4.2),
        "cycles_completed": int(cell_df["cycle"].max()),
        "soh_percent": float(soh_pred_pct),
        "rul_remaining_cycles": float(max(0.0, rul_pred_cycles)) if not np.isnan(rul_pred_cycles) else None,
        "soh_observed_pct": (float(last["soh"]) * 100.0 if pd.notna(last["soh"]) and last["soh"] <= 1.5
                             else (float(last["soh"]) if pd.notna(last["soh"]) else float("nan"))),
        "second_life": bool(last["second_life"]) if pd.notna(last["second_life"]) else False,
        "source": str(last["source"]),
    }


def run(battery_id: str | None) -> Path:
    print("Stage 12 — End-to-end smoke test (Iter-3 §3.12 ML-deployable refresh)")
    print("=" * 70)
    print(f"[Load] {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)} ...")
    df = pd.read_parquet(UNIFIED_PARQUET)
    print(f"  {len(df):,} rows · {df['battery_id'].nunique():,} batteries")

    cell_df = pick_battery(df, battery_id)
    chosen_id = str(cell_df.iloc[-1]["battery_id"])
    chosen_source = str(cell_df.iloc[-1]["source"])
    chosen_chem = str(cell_df.iloc[-1]["chemistry"])
    print(f"\n[1/6] Selected battery: {chosen_id}")
    print(f"      source={chosen_source}  chemistry={chosen_chem}  "
          f"cycles_observed={int(cell_df['cycle'].max())}")

    print(f"\n[2/6] Loading audited ML models")
    soh_model, rul_model = _load_ml_models()
    print(f"      SoH: {SOH_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      RUL: {RUL_MODEL_PATH.relative_to(PROJECT_ROOT)}")

    print(f"\n[3/6] Building feature bundle (audited mode) and locating chosen cell ...")
    bundle = load_feature_bundle(exclude_capacity_features=True, verbose=False)
    X_last, cycle_last, split_name = _find_battery_in_bundle(bundle, chosen_id)
    print(f"      cell located in {split_name} split  ·  last observed cycle = {cycle_last}  "
          f"·  feature dim = {X_last.shape[1]}")

    print(f"\n[4/6] ML inference (XGBoost SoH audited + XGBoost RUL audited+uncensored)")
    soh_pred_pct = float(soh_model.predict(X_last)[0])
    rul_pred_cyc = float(rul_model.predict(X_last)[0])
    summary = derive_battery_summary(cell_df, soh_pred_pct, rul_pred_cyc)
    obs_str = (f" (observed {summary['soh_observed_pct']:.2f}%)"
               if not np.isnan(summary["soh_observed_pct"]) else "")
    print(f"      SoH (predicted) = {summary['soh_percent']:.2f}%{obs_str}")
    print(f"      RUL (predicted) ≈ {summary['rul_remaining_cycles']:.0f} cycles")

    grade = grade_from_soh(summary["soh_percent"])
    print(f"      Grade classification: {grade}")

    print(f"\n[5/6] Fuzzy BWM-TOPSIS routing (canonical 6 criteria)")
    topsis_out = run_canonical_topsis(grade)
    weights_str = ", ".join(f"{k}={v:.3f}" for k, v in topsis_out["weights"].items())
    print(f"      weights: {weights_str}")
    print(f"      ranking:")
    for r in topsis_out["ranked"]:
        marker = " <- recommended" if r["rank"] == 1 else ""
        print(f"        rank {r['rank']}  {r['alternative']:25s}  "
              f"closeness={r['closeness']:.4f}{marker}")

    rec = next(r for r in topsis_out["ranked"] if r["rank"] == 1)

    print(f"\n[6/6] Building unified DPP JSON")
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
        estimation_method=("XGBoost SoH (audited, R²=0.996, RMSE=2.43% on test) + "
                           "XGBoost RUL (audited+uncensored, 1.92% RMSE-of-range on uncensored test). "
                           "Iter-3 §3.12 deployable."),
        estimation_confidence={
            "metric": "audited test RMSE",
            "value": 0.0243,  # SoH RMSE fraction
            "validation_set": "uncensored test partition (254,107 rows) — Iter-3 audited gate metric",
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
            f"models/xgboost_soh/xgboost_soh_audited.json",
            f"models/xgboost_rul/xgboost_rul_audited_uncensored.json",
            "src.mcdm.topsis.run_canonical_topsis (literature weights)",
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
