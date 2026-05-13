"""
Iter-3 §3.14 VAE 4-variant ablation runner.

The model audit flagged VAE as having val_loss volatility (best epoch 8 of 18,
val_loss rose 9 times after best). The audit recommendation: "Default β=1.0
over-regularizes reconstruction; should anneal β from 0 → 1 over training,
or just use β=0.5. Reduce latent_dim to 8."

Variants tested:
  baseline : β=1.0 const,   latent=12   ← current VAE
  V1       : β=0.5 const,   latent=12   ← relax KL constraint
  V2       : β anneal 0→1.0, latent=12  ← proper KL warmup (Bowman 2016)
  V3       : β=1.0 const,   latent=8    ← smaller capacity
  V4       : β anneal 0→1.0, latent=8   ← combo (most-likely winner)

Each variant trains for up to 200 epochs with patience=10 (default config).
Estimated wall-clock: ~60-70 min per variant → ~4-5 hours total.

Usage
-----
    python scripts/run_vae_ablation.py
    python scripts/run_vae_ablation.py --variants V1,V4
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
TBL_DIR = PROJECT_ROOT / "results" / "tables" / "vae_anomaly"

VARIANTS = {
    "baseline": dict(beta=1.0, beta_anneal=False, latent=12, tag=""),
    "V1": dict(beta=0.5, beta_anneal=False, latent=12, tag="_V1_beta05"),
    "V2": dict(beta=1.0, beta_anneal=True,  latent=12, tag="_V2_betaanneal"),
    "V3": dict(beta=1.0, beta_anneal=False, latent=8,  tag="_V3_latent8"),
    "V4": dict(beta=1.0, beta_anneal=True,  latent=8,  tag="_V4_combo"),
}


def run_variant(name: str, config: dict) -> dict:
    cmd = [
        sys.executable, "-u", "scripts/train_vae_anomaly.py",
        "--beta", str(config["beta"]),
        "--latent-dim", str(config["latent"]),
        "--tag", config["tag"],
    ]
    if config["beta_anneal"]:
        cmd.append("--beta-anneal")
    print("=" * 80)
    print(f"  VAE Variant {name}  ·  β={config['beta']}  "
          f"anneal={config['beta_anneal']}  latent={config['latent']}")
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
        print(f"  [FAIL] no metrics file at {metrics_path}")
        return {"variant": name, "ok": False, "elapsed_s": elapsed}
    return {"variant": name, "ok": True, "elapsed_s": elapsed,
            "metrics_path": str(metrics_path)}


def collect_metrics(variant_runs: list[dict]) -> pd.DataFrame:
    rows = []
    for run in variant_runs:
        if not run.get("ok"):
            continue
        m = json.loads(Path(run["metrics_path"]).read_text())
        hp = m.get("hyperparameters", {})
        train_rate = m.get("train_anomaly_rate", float("nan"))
        val_rate = m.get("val_anomaly_rate", float("nan"))
        test_rate = m.get("test_anomaly_rate", float("nan"))
        rows.append({
            "variant": run["variant"],
            "beta": hp.get("beta_target", 1.0),
            "beta_anneal": hp.get("beta_anneal", False),
            "latent": hp.get("latent_dim", 12),
            "epochs_trained": m.get("epochs_trained"),
            "best_epoch": m.get("best_epoch"),
            "train_anomaly_rate": train_rate,
            "val_anomaly_rate": val_rate,
            "test_anomaly_rate": test_rate,
            "drift_train_to_test_pp": (test_rate - train_rate) * 100,
            "drift_train_to_val_pp": (val_rate - train_rate) * 100,
            "elapsed_min": round(run["elapsed_s"] / 60, 1),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description="VAE β/latent ablation runner")
    p.add_argument("--variants", default="V1,V2,V3,V4",
                   help="Comma-separated variant codes (default: V1,V2,V3,V4)")
    p.add_argument("--skip-baseline", action="store_true",
                   help="Don't re-train the baseline (already exists at metrics.json)")
    args = p.parse_args()

    runs = []
    if not args.skip_baseline:
        if (TBL_DIR / "metrics.json").exists():
            print("[Skip] baseline metrics.json exists; not re-running")
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
    print("  VAE ABLATION SUMMARY (smaller train→test drift = better calibrated)")
    print("=" * 80)
    df = collect_metrics(runs)
    if df.empty:
        print("  no successful runs.")
        return
    df["abs_drift"] = df["drift_train_to_test_pp"].abs()
    df_sorted = df.sort_values("abs_drift")
    cols = ["variant", "beta", "beta_anneal", "latent", "epochs_trained",
            "best_epoch", "train_anomaly_rate", "val_anomaly_rate",
            "test_anomaly_rate", "drift_train_to_test_pp",
            "drift_train_to_val_pp", "elapsed_min"]
    print(df_sorted[cols].to_string(index=False, float_format="%.4f"))

    summary_csv = TBL_DIR / "ablation_summary.csv"
    df_sorted.to_csv(summary_csv, index=False)
    print(f"\n[Save] {summary_csv.relative_to(PROJECT_ROOT)}")

    winner = df_sorted.iloc[0]
    print()
    print("=" * 80)
    print(f"  Best calibrated VAE: variant {winner['variant']}")
    print(f"  β={winner['beta']}  anneal={winner['beta_anneal']}  latent={winner['latent']}")
    print(f"  Drift train→test: {winner['drift_train_to_test_pp']:+.2f} pp")
    print("=" * 80)


if __name__ == "__main__":
    main()
