"""
Stage 9 orchestrator — runs every training script back-to-back.

In smoke mode (default if `--smoke`) each underlying script trains briefly so
you can verify wiring end-to-end in <2 minutes. In full mode, run on a beefy
terminal — total wall-clock ~30-90 min depending on hardware.

Usage
-----
    python scripts/train_all_models.py --smoke
    python scripts/train_all_models.py --full
    python scripts/train_all_models.py --full --tune-xgb
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _run(label: str, cmd: list[str]):
    print("\n" + "=" * 76)
    print(f">>> {label}")
    print(f"    cmd: {' '.join(cmd)}")
    print("=" * 76)
    t0 = time.time()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    dt = time.time() - t0
    status = "OK" if res.returncode == 0 else f"FAILED (exit {res.returncode})"
    print(f"<<< {label} — {status}  ({dt:.1f}s)")
    return res.returncode == 0


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke", action="store_true", help="Quick wiring check (~2 min)")
    g.add_argument("--full",  action="store_true", help="Full training (~30-90 min)")
    p.add_argument("--tune-xgb", action="store_true", help="Run Optuna HP search for XGBoost")
    p.add_argument("--skip", nargs="*", default=[],
                   choices=["xgb", "lstm", "vae", "iforest", "grade"],
                   help="Skip individual stages")
    args = p.parse_args()

    smoke_flag = ["--smoke"] if args.smoke else []
    py = sys.executable

    plan = []
    if "xgb" not in args.skip:
        cmd = [py, str(SCRIPTS / "train_xgboost_soh.py"), *smoke_flag]
        if args.tune_xgb and args.full:
            cmd += ["--tune"]
        plan.append(("XGBoost SoH", cmd))
    if "lstm" not in args.skip:
        plan.append(("LSTM RUL", [py, str(SCRIPTS / "train_lstm_rul.py"), *smoke_flag]))
    if "vae" not in args.skip:
        plan.append(("VAE Anomaly", [py, str(SCRIPTS / "train_vae_anomaly.py"), *smoke_flag]))
    if "iforest" not in args.skip:
        plan.append(("IsolationForest", [py, str(SCRIPTS / "train_isolation_forest.py"), *smoke_flag]))
    if "grade" not in args.skip:
        # Grade evaluation depends on XGB output; only meaningful after xgb runs.
        plan.append(("Grade Classifier Eval", [py, str(SCRIPTS / "eval_grade_classifier.py"), *smoke_flag]))

    print(f"Stage 9 orchestrator — mode = {'SMOKE' if args.smoke else 'FULL'}")
    print(f"Will run {len(plan)} stages: " + ", ".join(s[0] for s in plan))

    results = {}
    t0 = time.time()
    for label, cmd in plan:
        results[label] = _run(label, cmd)
    total = time.time() - t0

    print("\n" + "=" * 76)
    print("ORCHESTRATOR SUMMARY")
    print("=" * 76)
    for label, ok in results.items():
        print(f"  {'✅' if ok else '❌'}  {label}")
    print(f"\nTotal wall-clock: {total/60:.1f} min")
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
