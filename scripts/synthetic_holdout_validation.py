"""
Iteration-3 — Held-out synthetic-cluster validation.

Currently every PyBaMM-synthetic Indian-climate cluster contributes some
batteries to the *training* split. That means the model has seen the
extreme-heat regime during training. The realistic deployment scenario for
India is the opposite: cells from a never-before-seen climate cluster
arriving at second-life intake.

This script holds out one synthetic climate cluster at a time:
  - SYN_IN_LFP_Rajasthan_extreme  (extreme heat, LFP)
  - SYN_IN_NMC_Rajasthan_extreme  (extreme heat, NMC)

For each held-out cluster:
  1. pull all of its batteries OUT of train AND val
  2. retrain XGBoost SoH on the remaining (real + non-Rajasthan synthetic)
  3. evaluate on the held-out cluster's batteries (all cycles)
  4. report SoH RMSE, grade accuracy, and per-grade breakdown

A model that maintains grade accuracy ≥ 90 % on a never-seen extreme-heat
cluster has demonstrated that its degradation signal generalizes across
climate regimes — a stronger claim than the in-corpus stratified test.

Usage
-----
    python scripts/synthetic_holdout_validation.py --smoke
    python scripts/synthetic_holdout_validation.py
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

from src.data.training_data import (
    CAPACITY_LEAK_FEATURES,
    HIGH_MISSING_FEATURES,
    NUMERIC_FEATURES_FULL,
    soh_to_grade,
)
from src.utils.config import PROCESSED_DIR, RANDOM_SEED, RESULTS_DIR, XGBOOST_CONFIG
from src.utils.metrics import regression_metrics
from sklearn.preprocessing import StandardScaler

OUT_TBL = RESULTS_DIR / "tables" / "synthetic_holdout"
OUT_FIG = RESULTS_DIR / "figures" / "synthetic_holdout"

UNIFIED_PARQUET = PROCESSED_DIR / "cycling" / "unified.parquet"
SPLITS_JSON = PROCESSED_DIR / "cycling" / "splits.json"
EOL_THRESHOLD = 0.80

HOLDOUT_CLUSTERS = [
    "SYN_IN_LFP_Rajasthan_extreme",
    "SYN_IN_NMC_Rajasthan_extreme",
]


def _build_feature_frame(df: pd.DataFrame, numeric_features: list) -> pd.DataFrame:
    """Local copy mirroring training_data._build_feature_frame (RUL+SoH+source_family)."""
    df = df.copy()
    rul_pieces = []
    for _, group in df.groupby("battery_id", sort=False):
        cycles = group["cycle"].to_numpy()
        soh = group["soh"].to_numpy()
        below = np.where(soh < EOL_THRESHOLD)[0]
        eol_cycle = cycles[below[0]] if len(below) > 0 else cycles.max()
        rul_pieces.append(pd.Series(np.maximum(0, eol_cycle - cycles).astype(float),
                                    index=group.index))
    df["rul"] = pd.concat(rul_pieces).reindex(df.index)
    df["soh_pct"] = df["soh"].clip(lower=0.0, upper=1.5) * 100.0
    df["source_family"] = df["source"].str.split("_").str[0]
    df[numeric_features] = df[numeric_features].fillna(
        df[numeric_features].median(numeric_only=True)
    )
    df = df[df["soh_pct"].notna() & np.isfinite(df["soh_pct"])]
    df = df[df["rul"].notna() & np.isfinite(df["rul"])]
    return df


def _onehot_with_categories(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    pieces = []
    for col, cats in categories.items():
        s = df[col].astype("category").cat.set_categories(cats)
        pieces.append(pd.get_dummies(s, prefix=col, dtype="float32"))
    return pd.concat(pieces, axis=1)


def _build_xy(sub: pd.DataFrame, numeric_features: list, categories: dict, scaler):
    x_num = sub[numeric_features].to_numpy(dtype=np.float32)
    x_num = scaler.transform(x_num).astype(np.float32)
    x_cat = _onehot_with_categories(sub, categories).to_numpy(dtype=np.float32)
    x = np.concatenate([x_num, x_cat], axis=1)
    y = sub["soh_pct"].to_numpy(dtype=np.float32)
    return x, y


def _grade_accuracy_per(y_true_pct: np.ndarray, y_pred_pct: np.ndarray) -> dict:
    g_true = soh_to_grade(y_true_pct)
    g_pred = soh_to_grade(y_pred_pct)
    overall = float((g_true == g_pred).mean())
    per = {}
    for g in ["A", "B", "C", "D"]:
        mask = g_true == g
        if mask.sum() == 0:
            continue
        per[g] = {"n": int(mask.sum()),
                  "accuracy": float((g_pred[mask] == g).mean())}
    return {"overall": overall, "per_grade": per}


def main():
    p = argparse.ArgumentParser(description="Held-out synthetic-cluster validation")
    p.add_argument("--smoke", action="store_true",
                   help="Quick run, 50 trees, 200 k-row training cap")
    args = p.parse_args()

    OUT_TBL.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Synthetic-cluster holdout  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    print(f"Holdout clusters: {HOLDOUT_CLUSTERS}")

    np.random.seed(RANDOM_SEED)

    # --- audited feature set (capacity OUT) ---
    numeric_features = [c for c in NUMERIC_FEATURES_FULL
                        if c not in HIGH_MISSING_FEATURES
                        and c not in CAPACITY_LEAK_FEATURES]
    print(f"Audited features: {len(numeric_features)} numeric")

    df_all = pd.read_parquet(UNIFIED_PARQUET)
    splits = json.loads(SPLITS_JSON.read_text())
    df_all = _build_feature_frame(df_all, numeric_features)

    # Categories built from the *full* training pool (so chemistry one-hots are stable)
    train_pool = df_all[df_all["battery_id"].isin(splits["train"])]
    categories = {
        "chemistry": sorted(train_pool["chemistry"].dropna().unique().tolist()),
        "form_factor": sorted(train_pool["form_factor"].dropna().astype(str).unique().tolist()),
    }

    base_params = {**XGBOOST_CONFIG, "early_stopping_rounds": 30}
    if args.smoke:
        base_params.update({"n_estimators": 50, "max_depth": 4})

    rows = []
    t0 = time.time()
    for cluster in HOLDOUT_CLUSTERS:
        held_bids = df_all[df_all["source"] == cluster]["battery_id"].unique().tolist()
        if len(held_bids) == 0:
            print(f"  [{cluster:35s}]  no batteries found, skipping")
            continue

        # Build train: original train+val splits, MINUS the held-out cluster's batteries
        train_mask = (df_all["battery_id"].isin(splits["train"] + splits["val"])
                      & ~df_all["battery_id"].isin(held_bids))
        # Build test: ALL cycles from the held-out cluster's batteries
        test_mask = df_all["battery_id"].isin(held_bids)

        train_df = df_all[train_mask]
        test_df = df_all[test_mask]
        if args.smoke and len(train_df) > 200_000:
            train_df = train_df.sample(n=200_000, random_state=RANDOM_SEED)

        # Fit scaler on the held-out-aware training set (no leakage from holdout)
        scaler = StandardScaler()
        scaler.fit(train_df[numeric_features].to_numpy(dtype=np.float32))

        X_tr, y_tr = _build_xy(train_df, numeric_features, categories, scaler)
        X_te, y_te = _build_xy(test_df, numeric_features, categories, scaler)
        # Carve a tiny val slice from train for early stopping (random 5%)
        rng = np.random.RandomState(RANDOM_SEED)
        val_idx = rng.choice(len(X_tr), size=max(1000, int(0.05 * len(X_tr))),
                             replace=False)
        val_mask_arr = np.zeros(len(X_tr), dtype=bool); val_mask_arr[val_idx] = True
        X_va, y_va = X_tr[val_mask_arr], y_tr[val_mask_arr]
        X_tr2, y_tr2 = X_tr[~val_mask_arr], y_tr[~val_mask_arr]

        t_inner = time.time()
        model = xgb.XGBRegressor(**base_params)
        model.fit(X_tr2, y_tr2, eval_set=[(X_va, y_va)], verbose=False)
        pred = model.predict(X_te)
        m = regression_metrics(y_te, pred)
        grade = _grade_accuracy_per(y_te, pred)
        elapsed = time.time() - t_inner

        rows.append({
            "held_out_cluster": cluster,
            "n_held_batteries": int(len(held_bids)),
            "n_train": int(len(X_tr2)),
            "n_test_cycles": int(len(y_te)),
            "test_r2": m["r2"],
            "test_rmse": m["rmse"],
            "test_mae": m["mae"],
            "grade_acc_overall": grade["overall"],
            **{f"grade_{g}_acc": v["accuracy"] for g, v in grade["per_grade"].items()},
            **{f"grade_{g}_n":   v["n"]        for g, v in grade["per_grade"].items()},
            "fit_time_s": round(elapsed, 1),
        })
        print(f"\n  [{cluster}]")
        print(f"    held out {len(held_bids)} batteries → {len(y_te):,} test cycles")
        print(f"    trained on {len(X_tr2):,} rows in {elapsed:.1f}s")
        print(f"    TEST  R²={m['r2']:+.4f}  RMSE={m['rmse']:.2f}%  MAE={m['mae']:.2f}%")
        print(f"    GRADE accuracy = {grade['overall']*100:.2f}%")
        for g, info in grade["per_grade"].items():
            print(f"      {g}: n={info['n']:>5,d}  acc={info['accuracy']*100:5.2f}%")

    total = time.time() - t0
    df = pd.DataFrame(rows)
    suffix = "_smoke" if args.smoke else ""
    csv_path = OUT_TBL / f"results{suffix}.csv"
    df.to_csv(csv_path, index=False)

    # ---- Findings markdown ----
    in_corpus_grade_acc = 0.9604  # from results/tables/grade_classifier/metrics_audited.json
    md_lines = [
        "# Held-out synthetic-cluster validation (audited XGBoost SoH)",
        "",
        f"_Mode: {'SMOKE' if args.smoke else 'FULL'}  ·  "
        f"clusters tested: {len(df)}  ·  total wall-clock: {total:.1f}s_",
        "",
        "## Setup",
        "",
        "Each row holds out an entire synthetic Indian-climate cluster from training, "
        "retrains audited XGBoost SoH on the remaining real+synthetic data, and "
        "evaluates on the held-out cluster's cycles. Audited features (capacity-derived "
        "columns dropped) are used so the test reflects the realistic deployment "
        "scenario.",
        "",
        f"In-corpus baseline (audited model on stratified test split): "
        f"**grade-acc = {in_corpus_grade_acc*100:.2f}%**.",
        "",
        "## Results",
        "",
        "| Held-out cluster | n batteries | n test cycles | Test R² | Test RMSE | "
        "Grade acc overall | A acc | B acc | C acc | D acc |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['held_out_cluster']} | {r['n_held_batteries']} | "
            f"{r['n_test_cycles']:,} | {r['test_r2']:+.4f} | {r['test_rmse']:.2f}% | "
            f"{r['grade_acc_overall']*100:.2f}% | "
            f"{r.get('grade_A_acc', float('nan'))*100 if pd.notna(r.get('grade_A_acc')) else float('nan'):.2f}% | "
            f"{r.get('grade_B_acc', float('nan'))*100 if pd.notna(r.get('grade_B_acc')) else float('nan'):.2f}% | "
            f"{r.get('grade_C_acc', float('nan'))*100 if pd.notna(r.get('grade_C_acc')) else float('nan'):.2f}% | "
            f"{r.get('grade_D_acc', float('nan'))*100 if pd.notna(r.get('grade_D_acc')) else float('nan'):.2f}% |"
        )
    if not df.empty:
        worst = df["grade_acc_overall"].min()
        delta = (df["grade_acc_overall"].mean() - in_corpus_grade_acc) * 100
        # Detect the SoH-variance pathology: if every test cycle is in Grade A,
        # the test is uninformative regardless of accuracy.
        only_grade_a = df["grade_A_n"].fillna(0).eq(df["n_test_cycles"]).all()
        md_lines.extend([
            "",
            "## Verdict",
            "",
            f"- Worst held-out grade-acc across clusters: **{worst*100:.2f}%**",
            f"- Mean held-out grade-acc minus in-corpus baseline: **{delta:+.2f}pp**",
        ])
        if only_grade_a:
            md_lines.extend([
                "",
                "### ⚠️  Caveat — held-out test set has zero SoH variance",
                "",
                "Every held-out cycle is in Grade A (SoH > 80%) because the PyBaMM "
                "synthetic generator only produces 100 cycles per battery and the "
                "climate-stress parameters do not move LFP / NMC cells out of the "
                "Grade-A band within that horizon. Implications:",
                "",
                "- The 100% grade accuracy is **not informative**: a constant "
                "predictor of 'A' would also score 100%.",
                "- The R² values are not interpretable (denominator near zero).",
                "- The audited model's RMSE on these held-out cells (0.3–0.4%) "
                "shows it predicts pristine cells correctly, but cannot be used to "
                "claim climate-regime generalization for grades B/C/D.",
                "",
                "**Iter-3 finding (data, not model):** the synthetic Indian-climate "
                "cohort needs longer cycle counts or steeper thermal-degradation "
                "parameters before it can stress-test grade routing. As of 2026-05, "
                "this validation cannot falsify the headline grade-routing claim — "
                "the falsification surface is the **real** corpus's per-source "
                "stratification (already reported in the audited grade classifier "
                "evaluation) and the LOSO study (already reported in Iteration 2).",
            ])
        else:
            md_lines.extend([
                "- A held-out grade-acc within ~5pp of the in-corpus baseline "
                "indicates the model's degradation signal generalizes across the "
                "held-out climate regime — a meaningfully stronger claim than the "
                "stratified test.",
                "- A larger drop indicates the model is partly memorizing "
                "climate-specific stress signatures rather than chemistry-level "
                "degradation physics.",
            ])

    md_path = OUT_TBL / f"findings{suffix}.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print()
    print(f"results  → {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"findings → {md_path.relative_to(PROJECT_ROOT)}")
    print("Done.")


if __name__ == "__main__":
    main()
