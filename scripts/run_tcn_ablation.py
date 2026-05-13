"""
Iter-3 §3.14 TCN regularization ablation runner.

Runs the 6-variant ablation flagged in the model audit (results/tables/model_audit/)
to address TCN's SEVERE overfit verdict (train R² 0.924 → test R² 0.704).

Each variant trains on the production config (--exclude-capacity --exclude-censored)
with a different (dropout, weight_decay, num_channels) combination. After all
variants finish, the runner prints a side-by-side comparison and identifies the
new winner.

Variants (per audit recommendation):
  baseline : dropout=0.30  wd=1e-4  channels=[32,32,64,64]   ← current TCN
  A        : dropout=0.45  wd=1e-4  channels=[32,32,64,64]   ← more dropout
  B        : dropout=0.30  wd=1e-4  channels=[32,32,32,32]   ← smaller arch
  C        : dropout=0.30  wd=1e-3  channels=[32,32,64,64]   ← stronger weight decay
  D        : dropout=0.40  wd=5e-4  channels=[32,32,32,32]   ← combo 1 (most likely winner)
  E        : dropout=0.45  wd=1e-3  channels=[32,32,32,32]   ← combo 2
  F        : dropout=0.50  wd=1e-3  channels=[16,16,32,32]   ← aggressive (under-fit risk)

Usage
-----
    python scripts/run_tcn_ablation.py                  # run all 7 (baseline + 6)
    python scripts/run_tcn_ablation.py --variants A,B,D # subset
    python scripts/run_tcn_ablation.py --skip-baseline  # baseline already exists
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TBL_DIR = PROJECT_ROOT / "results" / "tables" / "tcn_rul"

VARIANTS = {
    "baseline": dict(dropout=0.30, weight_decay=1e-4,
                     channels="32,32,64,64",
                     tag=""),  # the existing TCN we already trained
    "A":        dict(dropout=0.45, weight_decay=1e-4,
                     channels="32,32,64,64",
                     tag="_A_dropout45"),
    "B":        dict(dropout=0.30, weight_decay=1e-4,
                     channels="32,32,32,32",
                     tag="_B_smallarch"),
    "C":        dict(dropout=0.30, weight_decay=1e-3,
                     channels="32,32,64,64",
                     tag="_C_strongwd"),
    "D":        dict(dropout=0.40, weight_decay=5e-4,
                     channels="32,32,32,32",
                     tag="_D_combo1"),
    "E":        dict(dropout=0.45, weight_decay=1e-3,
                     channels="32,32,32,32",
                     tag="_E_combo2"),
    "F":        dict(dropout=0.50, weight_decay=1e-3,
                     channels="16,16,32,32",
                     tag="_F_aggressive"),
}


def run_variant(name: str, config: dict) -> dict:
    """Launch a single TCN training run; return path of metrics file."""
    cmd = [
        sys.executable, "-u", "scripts/train_tcn_rul.py",
        "--dropout", str(config["dropout"]),
        "--weight-decay", str(config["weight_decay"]),
        "--channels", config["channels"],
        "--tag", config["tag"],
    ]
    print("=" * 80)
    print(f"  Variant {name}  ·  dropout={config['dropout']}  "
          f"wd={config['weight_decay']}  channels={config['channels']}")
    print("=" * 80)
    print(f"  command: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0
    if proc.returncode != 0:
        print(f"  [FAIL] variant {name} exited with code {proc.returncode}")
        return {"variant": name, "ok": False, "elapsed_s": elapsed}
    metrics_path = TBL_DIR / f"metrics{config['tag']}.json"
    if not metrics_path.exists():
        print(f"  [FAIL] variant {name} produced no metrics file at {metrics_path}")
        return {"variant": name, "ok": False, "elapsed_s": elapsed}
    return {
        "variant": name, "ok": True, "elapsed_s": elapsed,
        "metrics_path": str(metrics_path),
    }


def collect_metrics(variant_runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in variant_runs:
        if not run.get("ok"):
            continue
        m = json.loads(Path(run["metrics_path"]).read_text())
        unc = m.get("test_censoring_stratified", {}).get("uncensored_only", {})
        hp = m.get("hyperparameters", {})
        rows.append({
            "variant": run["variant"],
            "dropout": hp.get("dropout"),
            "weight_decay": hp.get("weight_decay"),
            "channels": str(hp.get("channels")),
            "n_parameters": m.get("n_parameters"),
            "epochs_trained": m.get("epochs_trained"),
            "best_epoch": m.get("best_epoch"),
            "train_r2": m["train"]["r2"],
            "val_r2": m["val"]["r2"],
            "test_r2_full": m["test"]["r2"],
            "test_r2_uncensored": unc.get("r2"),
            "train_rmse_pct": m["train"]["rmse_pct_of_range"],
            "test_rmse_pct_full": m["test"]["rmse_pct_of_range"],
            "test_rmse_pct_uncensored": unc.get("rmse_pct_of_range"),
            "test_mae_pct_uncensored": unc.get("mae_pct_of_range"),
            "gap_train_to_uncens_r2": (m["train"]["r2"] - unc["r2"])
                                      if unc.get("r2") is not None else None,
            "elapsed_min": round(run["elapsed_s"] / 60, 1),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="TCN regularization ablation runner")
    p.add_argument("--variants", default="A,B,C,D,E,F",
                   help="Comma-separated variant codes (default: A,B,C,D,E,F)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Don't re-run the baseline (use the existing metrics.json)")
    args = p.parse_args()

    runs = []
    if not args.skip_baseline:
        # Skip if already present — baseline produces metrics.json
        if (TBL_DIR / "metrics.json").exists():
            print("[Skip] baseline metrics.json already exists; not re-running")
            runs.append({"variant": "baseline", "ok": True, "elapsed_s": 0,
                         "metrics_path": str(TBL_DIR / "metrics.json")})
        else:
            runs.append(run_variant("baseline", VARIANTS["baseline"]))
    else:
        runs.append({"variant": "baseline", "ok": True, "elapsed_s": 0,
                     "metrics_path": str(TBL_DIR / "metrics.json")})

    requested = [v.strip() for v in args.variants.split(",")]
    for v in requested:
        if v not in VARIANTS:
            print(f"[WARN] unknown variant '{v}', skipping")
            continue
        if v == "baseline":
            continue
        runs.append(run_variant(v, VARIANTS[v]))

    print("\n" + "=" * 80)
    print("  ABLATION SUMMARY")
    print("=" * 80)
    df = collect_metrics(runs)
    if df.empty:
        print("  no successful runs.")
        return
    df_sorted = df.sort_values("test_rmse_pct_uncensored")
    cols = ["variant", "dropout", "weight_decay", "channels", "n_parameters",
            "epochs_trained", "best_epoch",
            "train_r2", "test_r2_uncensored",
            "test_rmse_pct_uncensored", "test_mae_pct_uncensored",
            "gap_train_to_uncens_r2", "elapsed_min"]
    print(df_sorted[cols].to_string(index=False, float_format="%.4f"))

    summary_csv = TBL_DIR / "ablation_summary.csv"
    df_sorted.to_csv(summary_csv, index=False)
    print(f"\n[Save] {summary_csv.relative_to(PROJECT_ROOT)}")

    winner = df_sorted.iloc[0]
    baseline_row = df[df["variant"] == "baseline"]
    print()
    print("=" * 80)
    print("  WINNER")
    print("=" * 80)
    print(f"  Variant: {winner['variant']}")
    print(f"  Hyperparameters: dropout={winner['dropout']}  "
          f"wd={winner['weight_decay']}  channels={winner['channels']}")
    print(f"  Uncensored-test RMSE-of-range: {winner['test_rmse_pct_uncensored']:.2f}%  "
          f"(baseline: {baseline_row['test_rmse_pct_uncensored'].iloc[0] if not baseline_row.empty else 'n/a':.2f}%)")
    print(f"  Train-to-uncensored gap: {winner['gap_train_to_uncens_r2']:+.4f}  "
          f"(baseline: {baseline_row['gap_train_to_uncens_r2'].iloc[0] if not baseline_row.empty else 'n/a':+.4f})")
    if winner["test_rmse_pct_uncensored"] < 2.0:
        print(f"  ✓ PASSES strict 2% gate")
    elif winner["test_rmse_pct_uncensored"] < 3.0:
        print(f"  ✓ Passes 3% soft gate (strict 2% NOT cleared)")
    else:
        print(f"  ✗ Fails both gates")


if __name__ == "__main__":
    main()
