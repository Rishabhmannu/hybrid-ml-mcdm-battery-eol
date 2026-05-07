"""End-to-end evaluation of the deployable ChemistryRouter.

Sanity-checks that:
  1. The router loads cleanly from `models/per_chemistry/router_manifest.json`.
  2. Per-row dispatch (group by chemistry, predict in batch, reassemble) gives
     identical predictions to running each per-chemistry model on its slice
     individually — confirms vectorized dispatch is correct.
  3. Routed predictions outperform the global audited model on aggregated
     test grade-routing accuracy (the architectural payoff).

Outputs results to `results/tables/chemistry_router/eval.json` so the
manuscript can cite a single deployable-router headline number.

Usage
-----
    python scripts/eval_chemistry_router.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import load_feature_bundle, soh_to_grade
from src.models.chemistry_router import ChemistryRouter
from src.utils.config import MODELS_DIR, RESULTS_DIR

OUT_TBL = RESULTS_DIR / "tables" / "chemistry_router"
MANIFEST = MODELS_DIR / "per_chemistry" / "router_manifest.json"


def main():
    OUT_TBL.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ChemistryRouter end-to-end eval")
    print("=" * 70)

    if not MANIFEST.exists():
        print(f"Manifest not found at {MANIFEST.relative_to(PROJECT_ROOT)}.")
        print("Run `python scripts/train_xgboost_per_chemistry.py` first.")
        sys.exit(1)

    print(f"Loading router from {MANIFEST.relative_to(PROJECT_ROOT)} ...")
    router = ChemistryRouter.load(MANIFEST)
    print(f"  {router}")
    print()

    print("Loading audited test bundle ...")
    bundle = load_feature_bundle(exclude_capacity_features=True)
    X_te = bundle.X_test
    y_te = bundle.y_test_soh
    chems_te = bundle.test_chemistries.to_numpy()
    print(f"  test rows: {len(X_te):,}")
    print(f"  unique chemistries in test: {sorted(set(chems_te))}")
    print()

    # Sanity check: per-row dispatch matches per-chemistry-slice prediction
    print("Sanity check — vectorized dispatch == per-slice prediction?")
    t0 = time.time()
    pred_router = router.predict_soh(X_te, chems_te)
    elapsed_router = time.time() - t0
    print(f"  router predict took {elapsed_router:.2f}s for {len(X_te):,} rows")

    pred_slice = np.full_like(y_te, np.nan)
    for chem, model in router.models.items():
        mask = chems_te == chem
        if mask.any():
            pred_slice[mask] = model.predict(X_te[mask]).astype(np.float32)
    sanity_ok = np.allclose(pred_router, pred_slice, equal_nan=True)
    print(f"  vectorized vs per-slice match: {sanity_ok}\n")

    # Headline comparison: router vs global audited model
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    grade_router = router.predict_grade(X_te, chems_te)
    grade_true   = soh_to_grade(y_te)
    router_grade_acc = float((grade_router == grade_true).mean())
    router_r2 = float(r2_score(y_te, pred_router))
    router_rmse = float(np.sqrt(mean_squared_error(y_te, pred_router)))
    router_mae = float(mean_absolute_error(y_te, pred_router))

    # Global audited model on the same test set (for delta)
    global_path = PROJECT_ROOT / "models" / "xgboost_soh" / "xgboost_soh_audited.json"
    if global_path.exists():
        gm = xgb.XGBRegressor()
        gm.load_model(str(global_path))
        pred_global = gm.predict(X_te)
        global_grade_acc = float((soh_to_grade(pred_global) == grade_true).mean())
        global_r2 = float(r2_score(y_te, pred_global))
        global_rmse = float(np.sqrt(mean_squared_error(y_te, pred_global)))
        global_mae = float(mean_absolute_error(y_te, pred_global))
    else:
        global_grade_acc = global_r2 = global_rmse = global_mae = float("nan")

    print("Headline comparison:")
    print(f"  {'metric':25s}  {'global':>12s}  {'router':>12s}  {'Δ':>10s}")
    print(f"  {'grade-routing accuracy':25s}  {global_grade_acc*100:11.2f}%  "
          f"{router_grade_acc*100:11.2f}%  {(router_grade_acc-global_grade_acc)*100:+9.2f}pp")
    print(f"  {'SoH R²':25s}  {global_r2:12.4f}  {router_r2:12.4f}  "
          f"{router_r2-global_r2:+10.4f}")
    print(f"  {'SoH RMSE %':25s}  {global_rmse:12.3f}  {router_rmse:12.3f}  "
          f"{router_rmse-global_rmse:+10.3f}")
    print(f"  {'SoH MAE %':25s}  {global_mae:12.3f}  {router_mae:12.3f}  "
          f"{router_mae-global_mae:+10.3f}")
    print()

    # Per-chemistry breakdown
    print("Per-chemistry grade-routing accuracy (test):")
    rows = []
    for chem in sorted(set(chems_te)):
        mask = chems_te == chem
        if not mask.any():
            continue
        n = int(mask.sum())
        rt_acc = float((grade_router[mask] == grade_true[mask]).mean())
        gl_acc = float((soh_to_grade(pred_global)[mask] == grade_true[mask]).mean()) \
                 if global_path.exists() else float("nan")
        used = "router" if chem in router.models else "fallback"
        rows.append({
            "chemistry": chem, "n_test": n,
            "router_grade_acc": rt_acc,
            "global_grade_acc": gl_acc,
            "delta_pp": (rt_acc - gl_acc) * 100,
            "dispatch": used,
        })
        print(f"  {chem:10s}  n={n:>7,d}  router={rt_acc*100:5.2f}%  "
              f"global={gl_acc*100:5.2f}%  Δ={(rt_acc-gl_acc)*100:+5.2f}pp  ({used})")

    pd.DataFrame(rows).to_csv(OUT_TBL / "per_chemistry.csv", index=False)

    payload = {
        "iter": 3,
        "router_manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
        "n_test_rows": int(len(X_te)),
        "n_chemistries_routed": len(router.models),
        "vectorized_dispatch_sanity_ok": bool(sanity_ok),
        "router": {
            "grade_acc": router_grade_acc, "r2": router_r2,
            "rmse": router_rmse, "mae": router_mae,
        },
        "global_audited": {
            "grade_acc": global_grade_acc, "r2": global_r2,
            "rmse": global_rmse, "mae": global_mae,
        },
        "delta": {
            "grade_acc_pp": (router_grade_acc - global_grade_acc) * 100,
            "r2": router_r2 - global_r2,
            "rmse": router_rmse - global_rmse,
            "mae": router_mae - global_mae,
        },
    }
    with open(OUT_TBL / "eval.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Save] {OUT_TBL.relative_to(PROJECT_ROOT)}/eval.json + per_chemistry.csv")
    print("Done.")


if __name__ == "__main__":
    main()
