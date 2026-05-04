"""
Stage 9 — IsolationForest baseline anomaly detector.

Cheap unsupervised baseline alongside the VAE. Ranks cycle records by isolation
depth; flags the top contamination% as anomalies.

Usage
-----
    python scripts/train_isolation_forest.py --smoke
    python scripts/train_isolation_forest.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import load_feature_bundle
from src.utils.config import MODELS_DIR, RANDOM_SEED, RESULTS_DIR
from src.utils.plots import plot_anomaly_scores

OUT_DIR = MODELS_DIR / "isolation_forest"
FIG_DIR = RESULTS_DIR / "figures" / "isolation_forest"
TBL_DIR = RESULTS_DIR / "tables" / "isolation_forest"


def main():
    p = argparse.ArgumentParser(description="Train IsolationForest anomaly detector")
    p.add_argument("--smoke", action="store_true", help="Smoke run on subsampled data")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=200)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"IsolationForest training  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    bundle = load_feature_bundle(smoke=args.smoke)

    n_estimators = 50 if args.smoke else args.n_estimators
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=args.contamination,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    t0 = time.time()
    model.fit(bundle.X_train)
    fit_time = time.time() - t0
    print(f"\n[Train] n_estimators={n_estimators}  contamination={args.contamination}  "
          f"({fit_time:.1f}s)")

    # Anomaly score = -decision_function (higher = more anomalous)
    scores_train = -model.decision_function(bundle.X_train)
    scores_val   = -model.decision_function(bundle.X_val)
    scores_test  = -model.decision_function(bundle.X_test)

    threshold = float(np.percentile(scores_train, 100 * (1 - args.contamination)))
    train_anom = int((scores_train > threshold).sum())
    val_anom = int((scores_val > threshold).sum())
    test_anom = int((scores_test > threshold).sum())

    print(f"\n[Anomaly] threshold (p{int((1-args.contamination)*100)} of train) = {threshold:.4f}")
    print(f"  train: {train_anom}/{len(scores_train)} ({train_anom/len(scores_train)*100:.2f}%)")
    print(f"  val:   {val_anom}/{len(scores_val)} ({val_anom/len(scores_val)*100:.2f}%)")
    print(f"  test:  {test_anom}/{len(scores_test)} ({test_anom/len(scores_test)*100:.2f}%)")

    metrics = {
        "smoke": args.smoke,
        "n_estimators": n_estimators,
        "contamination": args.contamination,
        "n_features": int(bundle.X_train.shape[1]),
        "fit_time_s": round(fit_time, 2),
        "threshold": threshold,
        "train_anomaly_rate": train_anom / max(len(scores_train), 1),
        "val_anomaly_rate":   val_anom / max(len(scores_val), 1),
        "test_anomaly_rate":  test_anom / max(len(scores_test), 1),
        "score_quartiles_test": {
            "p25": float(np.percentile(scores_test, 25)),
            "p50": float(np.percentile(scores_test, 50)),
            "p75": float(np.percentile(scores_test, 75)),
            "p99": float(np.percentile(scores_test, 99)),
        },
    }
    with open(TBL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    joblib.dump(model, OUT_DIR / "isolation_forest.pkl")
    print(f"\n[Save] model   → {(OUT_DIR / 'isolation_forest.pkl').relative_to(PROJECT_ROOT)}")
    print(f"[Save] metrics → {(TBL_DIR / 'metrics.json').relative_to(PROJECT_ROOT)}")

    plot_anomaly_scores(scores_test, threshold,
                        out_path=FIG_DIR / "anomaly_score_distribution.png",
                        title="IsolationForest — Test isolation-depth distribution")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
