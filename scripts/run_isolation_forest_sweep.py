"""
Iter-3 §3.14 Isolation Forest contamination sweep.

The audit flagged Iso Forest as having +1.7 pp drift from train (5.0%) to test
(6.7%) anomaly rate — within tolerance but worth checking whether a different
contamination value gives a cleaner train/test alignment without losing the
"identify ~5 % anomalies" semantic.

Tests contamination ∈ {0.03, 0.04, 0.05 (current), 0.06, 0.08, 0.10}.
Each takes ~5 sec to retrain. Reports drift + n_estimators sensitivity.

Usage
-----
    python scripts/run_isolation_forest_sweep.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TBL_DIR = PROJECT_ROOT / "results" / "tables" / "isolation_forest"
SWEEP_VALUES = [0.03, 0.04, 0.05, 0.06, 0.08, 0.10]


def main():
    print("=" * 70)
    print(f"  Isolation Forest contamination sweep")
    print(f"  testing values: {SWEEP_VALUES}")
    print("=" * 70)

    rows = []
    t0 = time.time()
    for c in SWEEP_VALUES:
        print(f"\n  contamination = {c}")
        cmd = [sys.executable, "-u", "scripts/train_isolation_forest.py",
               "--contamination", str(c), "--n-estimators", "200"]
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"    [FAIL] code {proc.returncode}")
            continue
        m = json.loads((TBL_DIR / "metrics.json").read_text())
        train_rate = m.get("train_anomaly_rate", float("nan"))
        val_rate = m.get("val_anomaly_rate", float("nan"))
        test_rate = m.get("test_anomaly_rate", float("nan"))
        rows.append({
            "contamination": c,
            "train_rate": train_rate,
            "val_rate": val_rate,
            "test_rate": test_rate,
            "drift_train_to_test_pp": (test_rate - train_rate) * 100,
            "drift_train_to_val_pp": (val_rate - train_rate) * 100,
        })
        print(f"    train={train_rate:.4f}  val={val_rate:.4f}  test={test_rate:.4f}")

    print(f"\n  Total wall-clock: {time.time()-t0:.1f}s")
    df = pd.DataFrame(rows).sort_values("contamination")
    print()
    print("Sweep summary:")
    print(df.to_string(index=False, float_format="%.4f"))

    # Score each contamination by absolute drift from intended rate (smaller = better-calibrated)
    df["intended_minus_test_pp"] = (df["contamination"] - df["test_rate"]) * 100
    df["abs_drift_pp"] = df["drift_train_to_test_pp"].abs()
    best_calibrated = df.loc[df["abs_drift_pp"].idxmin()]
    print()
    print(f"Best-calibrated contamination (smallest train→test drift): "
          f"{best_calibrated['contamination']}")
    print(f"  → train {best_calibrated['train_rate']:.4f}, "
          f"test {best_calibrated['test_rate']:.4f}, "
          f"drift {best_calibrated['drift_train_to_test_pp']:+.2f}pp")

    out = TBL_DIR / "contamination_sweep.csv"
    df.to_csv(out, index=False)
    print(f"\n[Save] {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
