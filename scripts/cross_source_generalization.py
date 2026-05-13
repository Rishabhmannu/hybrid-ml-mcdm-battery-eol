"""
Cross-source generalization study (Leave-One-Source-Out, LOSO).

For each source S (above a row-count threshold):
  - train XGBoost on all rows from sources != S
  - evaluate on rows from S (using the test split's batteries only)

Supports both SoH and RUL targets (Iter-3 §3.12 RUL replacement extension).
For RUL, mirrors the production training config: audited feature set
(--exclude-capacity), trained on uncensored cells only (--exclude-censored
semantics), evaluation reports both RMSE in cycles and RMSE as % of train RUL
range to match the 2 % gate convention.

Surfaces how much of the headline R² is driven by within-source memorization vs
genuine cross-dataset generalization. Companion to the EDA finding that
`BL_ISU_ILCC` dominates 54 % of the corpus.

Usage
-----
    python scripts/cross_source_generalization.py                          # SoH (default)
    python scripts/cross_source_generalization.py --target rul             # XGBoost-RUL LOSO
    python scripts/cross_source_generalization.py --smoke
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

from src.data.training_data import load_feature_bundle
from src.utils.config import RANDOM_SEED, RESULTS_DIR, TARGETS, XGBOOST_CONFIG
from src.utils.metrics import regression_metrics
from src.utils.plots import _PALETTE, apply_theme

OUT_FIG = RESULTS_DIR / "figures" / "cross_source"
OUT_TBL = RESULTS_DIR / "tables" / "cross_source"


def main():
    p = argparse.ArgumentParser(description="Leave-One-Source-Out generalization")
    p.add_argument("--smoke", action="store_true",
                   help="Quick run on top-3 sources, 50 trees each")
    p.add_argument("--top-n", type=int, default=10,
                   help="Run LOSO on the top-N sources by sample count")
    p.add_argument("--min-rows", type=int, default=5000,
                   help="Skip sources with fewer than this many test rows")
    p.add_argument("--exclude-synthetic", action="store_true",
                   help="Exclude PyBaMM-synthetic Indian cells from the training "
                        "pool. Pair this with the default run to quantify the "
                        "PyBaMM-as-domain-bridge contribution.")
    p.add_argument("--target", choices=["soh", "rul"], default="soh",
                   help="Regression target. SoH (default) for the original LOSO "
                        "study; RUL replicates the production XGBoost-RUL config "
                        "(audited features, uncensored-only train) on each fold.")
    p.add_argument("--tag", default="",
                   help="Suffix appended to output filenames (e.g. '_no_synth').")
    args = p.parse_args()

    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TBL.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Cross-source LOSO  [target={args.target.upper()}]"
          f" ({'SMOKE' if args.smoke else 'FULL'})"
          f"{' [no synthetic]' if args.exclude_synthetic else ''}")
    print("=" * 70)
    # For RUL, use the production config: audited features + uncensored-only training.
    # Pool train+val for the LOSO training; test partition stays for held-out eval.
    if args.target == "rul":
        bundle = load_feature_bundle(
            smoke=args.smoke,
            exclude_capacity_features=True,
            exclude_censored_batteries=True,
        )
    else:
        bundle = load_feature_bundle(smoke=args.smoke)

    # Pool train/val/test (LOSO study uses its own split: held-out source vs rest).
    X_pool = np.concatenate([bundle.X_train, bundle.X_val, bundle.X_test])
    y_attr = "y_train_rul" if args.target == "rul" else "y_train_soh"
    y_attr_v = y_attr.replace("train", "val")
    y_attr_t = y_attr.replace("train", "test")
    y_pool = np.concatenate([
        getattr(bundle, y_attr),
        getattr(bundle, y_attr_v),
        getattr(bundle, y_attr_t),
    ])
    src_pool = pd.concat([
        bundle.train_sources, bundle.val_sources, bundle.test_sources
    ], ignore_index=True)
    chem_pool = pd.concat([
        bundle.train_chemistries, bundle.val_chemistries, bundle.test_chemistries
    ], ignore_index=True)
    # Train RUL range (used to convert RMSE → % of range, matching the 2 % gate).
    target_range = float(np.ptp(getattr(bundle, y_attr))) if args.target == "rul" else None

    if args.exclude_synthetic:
        keep = ~src_pool.str.startswith("SYN_IN").to_numpy()
        n_dropped = (~keep).sum()
        X_pool = X_pool[keep]; y_pool = y_pool[keep]
        src_pool = src_pool[keep].reset_index(drop=True)
        chem_pool = chem_pool[keep].reset_index(drop=True)
        print(f"Dropped {n_dropped:,} synthetic rows; "
              f"pool now {len(src_pool):,} rows from {src_pool.nunique()} sources.")

    src_counts = src_pool.value_counts()
    candidates = src_counts[src_counts >= args.min_rows].head(
        3 if args.smoke else args.top_n
    ).index.tolist()
    print(f"\nLOSO over {len(candidates)} sources: {candidates}")

    base_params = {**XGBOOST_CONFIG}
    if args.smoke:
        base_params.update({"n_estimators": 50, "max_depth": 4})

    rows = []
    t0 = time.time()
    for held_out in candidates:
        mask_train = (src_pool != held_out).to_numpy()
        mask_test  = (src_pool == held_out).to_numpy()
        X_tr = X_pool[mask_train]; y_tr = y_pool[mask_train]
        X_te = X_pool[mask_test];  y_te = y_pool[mask_test]
        chem_distribution = chem_pool[mask_test].value_counts(normalize=True)
        top_chem = chem_distribution.idxmax() if len(chem_distribution) else "—"
        top_chem_pct = chem_distribution.max() * 100 if len(chem_distribution) else 0

        t_inner = time.time()
        model = xgb.XGBRegressor(**base_params)
        model.fit(X_tr, y_tr, verbose=False)
        pred = model.predict(X_te)
        m = regression_metrics(y_te, pred)
        elapsed = time.time() - t_inner
        row = {
            "held_out_source": held_out,
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "dominant_chemistry": top_chem,
            "dominant_chem_pct": round(top_chem_pct, 1),
            "r2": m["r2"], "rmse": m["rmse"],
            "mae": m["mae"], "mape": m["mape"],
            "fit_time_s": round(elapsed, 1),
        }
        if args.target == "rul" and target_range:
            row["rmse_pct_of_range"] = m["rmse"] / target_range * 100.0
            row["mae_pct_of_range"]  = m["mae"]  / target_range * 100.0
        rows.append(row)
        if args.target == "rul":
            print(f"  {held_out:30s}  n_test={len(y_te):>7,d}  "
                  f"R²={m['r2']:+.4f}  RMSE={m['rmse']:.1f} cyc "
                  f"({row['rmse_pct_of_range']:.2f}% of range)  ({elapsed:.1f}s)")
        else:
            print(f"  {held_out:30s}  n_test={len(y_te):>7,d}  "
                  f"R²={m['r2']:+.4f}  RMSE={m['rmse']:.2f}%  ({elapsed:.1f}s)")

    total = time.time() - t0

    # ---- Save tables ----
    df = pd.DataFrame(rows).sort_values("r2")
    suffix_parts = []
    if args.target == "rul":
        suffix_parts.append("_rul")
    if args.smoke:
        suffix_parts.append("_smoke")
    if args.tag:
        suffix_parts.append(args.tag)
    suffix = "".join(suffix_parts)
    csv_path = OUT_TBL / f"results{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWall-clock: {total:.1f}s  ·  results → {csv_path.relative_to(PROJECT_ROOT)}")

    # ---- Plot: held-out R² vs in-corpus baseline ----
    apply_theme()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(df))))
    colors = [_PALETTE["val"] if r2 < 0.7 else _PALETTE["train"]
              for r2 in df["r2"].values]
    ax.barh(df["held_out_source"], df["r2"], color=colors, alpha=0.9)
    ax.axvline(TARGETS["soh_r2"], color="#7F8C8D", linestyle="--", linewidth=0.7,
               label=f"R² = {TARGETS['soh_r2']:.2f} reference")
    ax.axvline(0, color="#1A1A1A", linewidth=0.5)
    for i, (r2, n) in enumerate(zip(df["r2"], df["n_test"])):
        ax.text(r2, i, f"  R²={r2:+.3f} (n={n:,})", va="center", fontsize=8)
    ax.set_xlabel("R² (held-out source)")
    title_target = "XGBoost RUL (audited+uncensored)" if args.target == "rul" else "XGBoost SoH"
    ax.set_title(f"Leave-One-Source-Out generalization — {title_target}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    plot_path = OUT_FIG / f"loso_r2{suffix}.png"
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"figure → {plot_path.relative_to(PROJECT_ROOT)}")

    # ---- Headline report fragment ----
    median_r2 = df["r2"].median()
    n_fail = (df["r2"] < 0).sum()
    md_path = OUT_TBL / f"findings{suffix}.md"
    if args.target == "rul":
        n_pass_gate = (df["rmse_pct_of_range"] < TARGETS["rul_rmse"]).sum()
        median_rmse_pct = df["rmse_pct_of_range"].median()
        title = "Cross-source LOSO generalization (XGBoost RUL — audited+uncensored)"
        header = "| Held-out source | n train | n test | dominant chem | R² | RMSE (cyc) | RMSE (% range) | MAE (cyc) |"
        sep = "|---|---|---|---|---|---|---|---|"
        body_rows = [f"| {r['held_out_source']} | {r['n_train']:,} | {r['n_test']:,} | "
                     f"{r['dominant_chemistry']} ({r['dominant_chem_pct']:.0f}%) | "
                     f"{r['r2']:+.4f} | {r['rmse']:.1f} | "
                     f"{r['rmse_pct_of_range']:.2f}% | {r['mae']:.1f} |"
                     for _, r in df.iterrows()]
        headline = [
            f"- Median LOSO R² = **{median_r2:.3f}**",
            f"- Median LOSO RMSE = **{median_rmse_pct:.2f} % of range**",
            f"- Sources passing the **2 % RMSE-of-range gate** on held-out: **{n_pass_gate} / {len(df)}**",
            f"- Sources with **negative R²**: **{n_fail} / {len(df)}**",
            "",
            ("Compare against the in-corpus uncensored-test RMSE 1.92 % in "
             "`results/tables/xgboost_rul/metrics_audited_uncensored.json` — the gap between "
             "in-corpus and held-out RMSE quantifies cross-source RUL transfer difficulty."),
        ]
    else:
        n_pass_gate = (df["r2"] >= TARGETS["soh_r2"]).sum()
        title = "Cross-source LOSO generalization (XGBoost SoH)"
        header = "| Held-out source | n train | n test | dominant chem | R² | RMSE | MAE |"
        sep = "|---|---|---|---|---|---|---|"
        body_rows = [f"| {r['held_out_source']} | {r['n_train']:,} | {r['n_test']:,} | "
                     f"{r['dominant_chemistry']} ({r['dominant_chem_pct']:.0f}%) | "
                     f"{r['r2']:+.4f} | {r['rmse']:.2f}% | {r['mae']:.2f}% |"
                     for _, r in df.iterrows()]
        headline = [
            f"- Median LOSO R² = **{median_r2:.3f}**",
            f"- Sources passing the **0.95 R² gate** on held-out: **{n_pass_gate} / {len(df)}**",
            f"- Sources with **negative R²**: **{n_fail} / {len(df)}**",
            "",
            ("Compare against the in-corpus stratified test R² in "
             "`results/tables/xgboost_soh/test_metrics_per_source.csv` — the gap between the two "
             "is the share of headline R² that is within-source memorization vs cross-dataset transfer."),
        ]
    md_path.write_text("\n".join([
        f"# {title}",
        "",
        f"_Sources held out: {len(df)}  ·  generated in {total:.1f}s_",
        "",
        header,
        sep,
        *body_rows,
        "",
        "## Headline",
        "",
        *headline,
    ]) + "\n")
    print(f"report → {md_path.relative_to(PROJECT_ROOT)}")
    if args.target == "rul":
        print(f"\nMedian LOSO R²: {median_r2:.3f}  ·  median RMSE: "
              f"{df['rmse_pct_of_range'].median():.2f}%  ·  "
              f"pass 2% gate: {n_pass_gate}/{len(df)}  ·  negative R²: {n_fail}")
    else:
        print(f"\nMedian LOSO R²: {median_r2:.3f}  ·  pass 0.95 R² gate: "
              f"{n_pass_gate}/{len(df)}  ·  negative R²: {n_fail}")
    print("Done.")


if __name__ == "__main__":
    main()
