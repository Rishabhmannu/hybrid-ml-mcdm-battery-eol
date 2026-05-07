"""
Iteration-3 — Per-chemistry XGBoost SoH submodels.

Hypothesis: a single global model is ill-posed when the corpus spans 7
chemistries with very different degradation physics (NMC, LFP, NCA, LCO,
Zn-ion, Na-ion, "other"). Per-chemistry submodels should outperform the
global model on minority chemistries (LCO, Zn-ion, Na-ion) where the
global learner is dominated by majority-NMC behaviour.

For each chemistry C with sufficient training rows:
  1. slice the train/val/test bundles to chemistry == C
  2. train XGBoost SoH on the audited feature set (capacity columns OUT)
  3. evaluate on the chemistry's test slice
  4. compare against the global audited model's per-chemistry numbers
     (reads from results/tables/xgboost_soh/test_metrics_per_chemistry_audited.csv
     and results/tables/grade_classifier/test_grade_metrics_per_chemistry_audited.csv)

Usage
-----
    python scripts/train_xgboost_per_chemistry.py --smoke
    python scripts/train_xgboost_per_chemistry.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from src.data.training_data import load_feature_bundle, soh_to_grade
from src.utils.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TARGETS, XGBOOST_CONFIG
from src.utils.metrics import regression_metrics

OUT_TBL = RESULTS_DIR / "tables" / "per_chemistry_submodels"
OUT_FIG = RESULTS_DIR / "figures" / "per_chemistry_submodels"
OUT_MODELS = MODELS_DIR / "per_chemistry"


def _grade_accuracy(y_true_pct: np.ndarray, y_pred_pct: np.ndarray) -> float:
    return float((soh_to_grade(y_true_pct) == soh_to_grade(y_pred_pct)).mean())


def _slice_to_chemistry(bundle, chem: str):
    tr = bundle.train_chemistries == chem
    va = bundle.val_chemistries == chem
    te = bundle.test_chemistries == chem
    return {
        "X_tr": bundle.X_train[tr.to_numpy()], "y_tr": bundle.y_train_soh[tr.to_numpy()],
        "X_va": bundle.X_val[va.to_numpy()],   "y_va": bundle.y_val_soh[va.to_numpy()],
        "X_te": bundle.X_test[te.to_numpy()],  "y_te": bundle.y_test_soh[te.to_numpy()],
    }


def _global_per_chemistry_baseline() -> pd.DataFrame:
    """Read the audited global model's per-chemistry metrics as the comparison row."""
    soh_csv = (RESULTS_DIR / "tables" / "xgboost_soh"
               / "test_metrics_per_chemistry_audited.csv")
    grade_csv = (RESULTS_DIR / "tables" / "grade_classifier"
                 / "test_grade_metrics_per_chemistry_audited.csv")
    soh = pd.read_csv(soh_csv) if soh_csv.exists() else pd.DataFrame()
    grade = pd.read_csv(grade_csv) if grade_csv.exists() else pd.DataFrame()
    if soh.empty or grade.empty:
        return pd.DataFrame()
    out = soh[["stratum", "n", "r2", "rmse", "mae"]].rename(columns={
        "stratum": "chemistry", "r2": "global_r2",
        "rmse": "global_rmse", "mae": "global_mae",
    })
    g = grade[["stratum", "accuracy", "f1_macro"]].rename(columns={
        "stratum": "chemistry", "accuracy": "global_grade_acc",
        "f1_macro": "global_f1_macro",
    })
    return out.merge(g, on="chemistry", how="left")


def main():
    p = argparse.ArgumentParser(description="Per-chemistry XGBoost submodels")
    p.add_argument("--smoke", action="store_true",
                   help="Quick run on smoke bundle, 50 trees")
    p.add_argument("--min-train-rows", type=int, default=2000,
                   help="Skip chemistries with fewer training rows than this")
    args = p.parse_args()

    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_MODELS.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Per-chemistry submodels  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)

    bundle = load_feature_bundle(smoke=args.smoke, exclude_capacity_features=True)

    # Shared scaler + feature names (one of each, applied across all chemistries
    # — consistent with how training_data.py builds the bundle). Saving them
    # once at the router level keeps the per-chemistry artefacts small.
    if not args.smoke:
        joblib.dump(bundle.scaler, OUT_MODELS / "feature_scaler.pkl")
        with open(OUT_MODELS / "feature_names.json", "w") as f:
            json.dump(bundle.feature_names, f, indent=2)
    chemistries = sorted(bundle.train_chemistries.dropna().unique().tolist())
    print(f"\nChemistries available: {chemistries}")

    base_params = {**XGBOOST_CONFIG, "early_stopping_rounds": 30}
    if args.smoke:
        base_params.update({"n_estimators": 50, "max_depth": 4})

    rows = []
    router_entries = []  # for the router_manifest.json
    t0 = time.time()
    for chem in chemistries:
        sl = _slice_to_chemistry(bundle, chem)
        n_tr, n_va, n_te = len(sl["y_tr"]), len(sl["y_va"]), len(sl["y_te"])
        if n_tr < args.min_train_rows:
            print(f"  [{chem:10s}]  skipped (n_train={n_tr} < {args.min_train_rows})")
            continue
        if n_te < 50:
            print(f"  [{chem:10s}]  skipped (n_test={n_te} < 50)")
            continue

        t_inner = time.time()
        model = xgb.XGBRegressor(**base_params)
        model.fit(sl["X_tr"], sl["y_tr"],
                  eval_set=[(sl["X_va"], sl["y_va"])],
                  verbose=False)
        pred = model.predict(sl["X_te"])
        m = regression_metrics(sl["y_te"], pred)
        grade_acc = _grade_accuracy(sl["y_te"], pred)
        elapsed = time.time() - t_inner

        # Save per-chemistry model + manifest entry. Router can then load
        # `router_manifest.json` and dispatch to the right .json by chemistry.
        if not args.smoke:
            chem_dir = OUT_MODELS / chem
            chem_dir.mkdir(parents=True, exist_ok=True)
            model_path = chem_dir / "xgboost_soh_audited.json"
            model.save_model(str(model_path))
            router_entries.append({
                "chemistry": chem,
                "model_path": str(model_path.relative_to(MODELS_DIR.parent)),
                "n_train": int(n_tr),
                "test_r2": float(m["r2"]),
                "test_rmse": float(m["rmse"]),
                "test_grade_acc": float(grade_acc),
            })

        rows.append({
            "chemistry": chem,
            "n_train": int(n_tr), "n_val": int(n_va), "n_test": int(n_te),
            "submodel_r2": m["r2"], "submodel_rmse": m["rmse"],
            "submodel_mae": m["mae"], "submodel_grade_acc": grade_acc,
            "fit_time_s": round(elapsed, 1),
        })
        print(f"  [{chem:10s}]  n_train={n_tr:>7,d}  n_test={n_te:>7,d}  "
              f"R²={m['r2']:+.4f}  RMSE={m['rmse']:.2f}%  "
              f"grade_acc={grade_acc*100:5.2f}%  ({elapsed:.1f}s)")

    total = time.time() - t0
    df = pd.DataFrame(rows)

    # ---- Compare against the global audited model's per-chemistry slice ----
    baseline = _global_per_chemistry_baseline()
    if not baseline.empty:
        df = df.merge(baseline, on="chemistry", how="left")
        df["delta_r2"] = df["submodel_r2"] - df["global_r2"]
        df["delta_rmse"] = df["submodel_rmse"] - df["global_rmse"]
        df["delta_grade_acc"] = df["submodel_grade_acc"] - df["global_grade_acc"]

    suffix = "_smoke" if args.smoke else ""
    csv_path = OUT_TBL / f"results{suffix}.csv"
    df.to_csv(csv_path, index=False)

    # ---- Router manifest -----------------------------------------------
    # Single-file index for downstream `ChemistryRouter.load(...)` to
    # discover models keyed by chemistry. Lists each per-chemistry model's
    # path + headline metrics + the shared scaler/feature names location.
    if router_entries:
        manifest = {
            "iter": 3,
            "feature_excludes_capacity": True,
            "shared_scaler": str((OUT_MODELS / "feature_scaler.pkl").relative_to(MODELS_DIR.parent)),
            "shared_feature_names": str((OUT_MODELS / "feature_names.json").relative_to(MODELS_DIR.parent)),
            "chemistries": router_entries,
            "fallback_global_model": "models/xgboost_soh/xgboost_soh_audited.json",
        }
        manifest_path = OUT_MODELS / "router_manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\n[router] manifest → {manifest_path.relative_to(PROJECT_ROOT)}")
        print(f"         {len(router_entries)} per-chemistry models saved under "
              f"{OUT_MODELS.relative_to(PROJECT_ROOT)}/")

    print()
    print("=" * 70)
    print(f"Wall-clock: {total:.1f}s  ·  results → {csv_path.relative_to(PROJECT_ROOT)}")

    if "delta_grade_acc" in df.columns:
        print()
        print("Per-chemistry submodel vs global audited model:")
        print(f"  {'chemistry':10s}  {'sub_acc':>9s}  {'glob_acc':>9s}  "
              f"{'Δ_acc':>7s}  {'sub_rmse':>9s}  {'glob_rmse':>10s}  {'Δ_rmse':>8s}")
        for _, r in df.iterrows():
            sub = r["submodel_grade_acc"] * 100
            glob = (r.get("global_grade_acc") or 0) * 100
            sub_rmse = r["submodel_rmse"]
            glob_rmse = r.get("global_rmse")
            d_acc = (r.get("delta_grade_acc") or 0) * 100
            d_rmse = r.get("delta_rmse") or 0
            print(f"  {r['chemistry']:10s}  {sub:>8.2f}%  {glob:>8.2f}%  "
                  f"{d_acc:>+6.2f}pp  {sub_rmse:>8.2f}%  "
                  f"{glob_rmse if glob_rmse is None else f'{glob_rmse:>9.2f}%'}  "
                  f"{d_rmse:>+7.2f}pp")

    # ---- Findings markdown ----
    md_lines = [
        "# Per-chemistry XGBoost submodels (audited features)",
        "",
        f"_Mode: {'SMOKE' if args.smoke else 'FULL'}  ·  "
        f"min_train_rows={args.min_train_rows}  ·  "
        f"chemistries trained: {len(df)}_",
        "",
        "## Comparison vs global audited model",
        "",
    ]
    if "delta_grade_acc" in df.columns:
        md_lines.extend([
            "| Chemistry | n train | n test | Submodel R² | Global R² | ΔR² | "
            "Submodel grade acc | Global grade acc | Δ acc |",
            "|---|---|---|---|---|---|---|---|---|",
        ])
        for _, r in df.iterrows():
            md_lines.append(
                f"| {r['chemistry']} | {r['n_train']:,} | {r['n_test']:,} | "
                f"{r['submodel_r2']:+.4f} | "
                f"{r.get('global_r2', float('nan')):+.4f} | "
                f"{r.get('delta_r2', 0):+.4f} | "
                f"{r['submodel_grade_acc']*100:.2f}% | "
                f"{(r.get('global_grade_acc') or 0)*100:.2f}% | "
                f"{(r.get('delta_grade_acc') or 0)*100:+.2f}pp |"
            )
        # Verdict
        n_better = (df["delta_grade_acc"] > 0.005).sum() if "delta_grade_acc" in df else 0
        n_worse  = (df["delta_grade_acc"] < -0.005).sum() if "delta_grade_acc" in df else 0
        md_lines.extend([
            "",
            "## Verdict",
            "",
            f"- Chemistries where the **submodel beats the global** by ≥0.5pp grade-acc: **{n_better} / {len(df)}**",
            f"- Chemistries where the **global beats the submodel** by ≥0.5pp grade-acc: **{n_worse} / {len(df)}**",
            "- A small minority chemistry (LCO, Zn-ion, Na-ion) lifting from a submodel "
            "demonstrates that the global model is paying a cross-chemistry-mixing tax. "
            "If the submodels do not beat the global on any chemistry, the global model "
            "is already chemistry-aware via its `chemistry_*` one-hot features and a "
            "router-of-submodels would just add deployment complexity for no gain.",
        ])
    else:
        md_lines.append(
            "_Global-model baseline tables not found — run `train_xgboost_soh.py "
            "--exclude-capacity` and `eval_grade_classifier.py --exclude-capacity` "
            "first to enable the side-by-side comparison._"
        )

    md_path = OUT_TBL / f"findings{suffix}.md"
    md_path.write_text("\n".join(md_lines) + "\n")
    print(f"findings → {md_path.relative_to(PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
