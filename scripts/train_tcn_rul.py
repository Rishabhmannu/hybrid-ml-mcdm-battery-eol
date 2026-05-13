"""
Iter-3 §3.13 — Train TCN RUL forecaster (secondary deep-learning baseline).

Sister script to `scripts/train_lstm_rul.py` and `scripts/train_xgboost_rul.py`.
Uses the same per-battery cycle-sequence pipeline (`load_battery_sequences`)
so the TCN, LSTM, and XGBoost RUL numbers are apples-to-apples.

Per Iter-3 §3.13 plan:
- Default config: audited features (`--exclude-capacity`) + uncensored-train
  (`--exclude-censored`), matching the production XGBoost-RUL configuration.
- Soft gate per professor's guidance: test RMSE-of-range ∈ [1 %, 3 %].
- Censoring-stratified test eval (uncensored-test = gate-relevant subset).

References
----------
- Bai et al. 2018 (arXiv:1803.01271) — TCN foundation
- TCN-LSTM hybrid for SoH+RUL — *J. Energy Storage* 2024
- TCN-spatial-attention RUL — *Sci. Reports* 2025
- Cycle-based MLP/GRU/LSTM/TCN bake-off — *Sci. Reports* 2025

Usage
-----
    python scripts/train_tcn_rul.py --smoke                            # 5 epochs, 30 batteries
    python scripts/train_tcn_rul.py --exclude-capacity --exclude-censored  # production config
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import compute_battery_censoring, load_battery_sequences
from src.models.lstm_rul import BatterySequenceDataset
from src.models.tcn_rul import BatteryTCN
from src.utils.config import DEVICE, RANDOM_SEED, RESULTS_DIR, MODELS_DIR, TARGETS, TCN_CONFIG
from src.utils.plots import (
    plot_loss_curves,
    plot_overfit_check,
    plot_predicted_vs_actual,
    plot_residuals,
)
from src.utils.training_callbacks import (
    CSVLogger,
    CosineLRScheduler,
    EarlyStopping,
    ModelCheckpoint,
    TorchTrainer,
)

OUT_DIR = MODELS_DIR / "tcn_rul"
FIG_DIR = RESULTS_DIR / "figures" / "tcn_rul"
TBL_DIR = RESULTS_DIR / "tables" / "tcn_rul"


def _eval(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            p = model(x).cpu().numpy()
            preds.append(p); targets.append(y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(targets)


def _metrics(y_true, y_pred, target_range: float) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else float("nan"),
        "rmse_cycles": rmse,
        "mae_cycles": mae,
        "rmse_pct_of_range": rmse / max(target_range, 1.0) * 100.0,
        "mae_pct_of_range": mae / max(target_range, 1.0) * 100.0,
    }


def main():
    p = argparse.ArgumentParser(description="Train TCN RUL forecaster")
    p.add_argument("--smoke", action="store_true", help="5 epochs, 30 batteries")
    p.add_argument("--epochs", type=int, default=TCN_CONFIG["epochs"])
    p.add_argument("--batch-size", type=int, default=TCN_CONFIG["batch_size"])
    p.add_argument("--patience", type=int, default=TCN_CONFIG["patience"])
    p.add_argument("--seq-len", type=int, default=TCN_CONFIG["sequence_length"])
    p.add_argument("--dropout", type=float, default=TCN_CONFIG["dropout"],
                   help="Dropout rate inside each TemporalBlock (default: %(default)s; "
                        "Iter-3 §3.14 ablation values 0.30 / 0.40 / 0.45 / 0.50)")
    p.add_argument("--weight-decay", type=float,
                   default=TCN_CONFIG.get("weight_decay", 1e-4),
                   help="AdamW weight_decay (default: %(default)s; "
                        "Iter-3 §3.14 ablation values 1e-4 / 5e-4 / 1e-3)")
    p.add_argument("--channels", type=str, default=None,
                   help="Comma-separated TCN channel widths "
                        "(default: from TCN_CONFIG = '32,32,64,64'). "
                        "Try '32,32,32,32' or '16,16,32,32' to reduce capacity.")
    p.add_argument("--tag", default="",
                   help="Suffix appended to model + metrics + figure paths "
                        "(e.g. '_dropout45_wd1e3_ch32x4')")
    p.add_argument("--exclude-capacity", action="store_true", default=True,
                   help="Audited mode (default ON to match production RUL config). "
                        "Drops capacity_Ah + 5 derived features.")
    p.add_argument("--include-capacity", dest="exclude_capacity", action="store_false",
                   help="Disable the audited mode (research / ablation only).")
    p.add_argument("--exclude-censored", action="store_true", default=True,
                   help="Drop right-censored cells from train/val (default ON to match "
                        "production RUL Exp 2 winner). Test stays full.")
    p.add_argument("--include-censored", dest="exclude_censored", action="store_false",
                   help="Disable the uncensored-train mode (research / ablation only).")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"TCN RUL training  ({'SMOKE' if args.smoke else 'FULL'})")
    print(f"  audited={args.exclude_capacity}  uncensored-train={args.exclude_censored}")
    print(f"  device={DEVICE}")
    print("=" * 70)

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sequences, n_features = load_battery_sequences(
        sequence_length=args.seq_len,
        smoke=args.smoke,
        exclude_capacity_features=args.exclude_capacity,
    )
    X_tr, y_tr, ids_tr = sequences["train"]
    X_va, y_va, ids_va = sequences["val"]
    X_te, y_te, ids_te = sequences["test"]

    # Apply uncensored-train policy (Exp 2 winner from Iter-3 §3.12)
    if args.exclude_censored:
        df_full = pd.read_parquet(
            PROJECT_ROOT / "data" / "processed" / "cycling" / "unified.parquet"
        )
        censoring_map = compute_battery_censoring(df_full)
        keep_tr = ~np.array([censoring_map.get(b, False) for b in ids_tr.values])
        keep_va = ~np.array([censoring_map.get(b, False) for b in ids_va.values])
        n_drop_tr, n_drop_va = (~keep_tr).sum(), (~keep_va).sum()
        X_tr, y_tr = X_tr[keep_tr], y_tr[keep_tr]
        X_va, y_va = X_va[keep_va], y_va[keep_va]
        ids_tr = ids_tr[keep_tr].reset_index(drop=True)
        ids_va = ids_va[keep_va].reset_index(drop=True)
        print(f"\n[uncensored-train] dropped {n_drop_tr:,} train and {n_drop_va:,} val "
              f"sequences from right-censored cells. Test unchanged.")

    if len(X_tr) == 0:
        raise SystemExit("No training sequences built — check sequence length vs cycle counts.")

    train_ds = BatterySequenceDataset(X_tr, y_tr)
    val_ds   = BatterySequenceDataset(X_va, y_va)
    test_ds  = BatterySequenceDataset(X_te, y_te)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    target_range = float(np.ptp(y_tr)) if len(y_tr) > 0 else 1.0
    print(f"\n[Data] n_features={n_features}  seq_len={args.seq_len}  "
          f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}")
    print(f"[Data] RUL range (train) = {target_range:.0f} cycles")

    channels = ([int(c) for c in args.channels.split(",")]
                if args.channels else TCN_CONFIG["num_channels"])
    model = BatteryTCN(
        input_size=n_features,
        num_channels=channels,
        dropout=args.dropout,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] BatteryTCN  ·  channels={channels}  "
          f"kernel={TCN_CONFIG['kernel_size']}  dropout={args.dropout}")
    print(f"        receptive field = {model.receptive_field} cycles  ·  "
          f"params = {n_params:,}")
    print(f"[Train] AdamW  ·  weight_decay={args.weight_decay}  "
          f"·  lr={TCN_CONFIG['learning_rate']}  ·  patience={args.patience}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TCN_CONFIG["learning_rate"],
        weight_decay=args.weight_decay,
    )
    loss_fn = nn.MSELoss()

    epochs = 5 if args.smoke else args.epochs
    suffix = args.tag or ""
    callbacks = [
        ModelCheckpoint(out_dir=OUT_DIR, monitor="val_loss",
                        filename=f"best{suffix}.pt"),
        EarlyStopping(monitor="val_loss", patience=args.patience),
        CSVLogger(csv_path=TBL_DIR / f"training_log{suffix}.csv"),
        CosineLRScheduler(T_max=epochs),
    ]
    trainer = TorchTrainer(
        model, optimizer, loss_fn,
        device=DEVICE,
        train_loader=train_dl, val_loader=val_dl,
        callbacks=callbacks,
    )

    t0 = time.time()
    history = trainer.fit(epochs=epochs)
    train_time = time.time() - t0
    print(f"\n[Train] {len(history)} epochs  ·  {train_time:.1f}s")

    ckpt = torch.load(OUT_DIR / f"best{suffix}.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[Eval]  loaded best checkpoint @ epoch {ckpt['epoch']} "
          f"({ckpt.get('monitor', 'val_loss')}={ckpt.get('monitor_value', float('nan')):.5f})")

    pred_tr, true_tr = _eval(model, train_dl, DEVICE)
    pred_va, true_va = _eval(model, val_dl, DEVICE)
    pred_te, true_te = _eval(model, test_dl, DEVICE)

    train_metrics = _metrics(true_tr, pred_tr, target_range)
    val_metrics   = _metrics(true_va, pred_va, target_range)
    test_metrics  = _metrics(true_te, pred_te, target_range)

    print("\n[Eval]")
    for name, m in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        print(f"  {name:5s}  R²={m['r2']:.4f}  "
              f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
              f"MAE={m['mae_cycles']:.1f} cyc ({m['mae_pct_of_range']:.2f}% of range)")

    # ---- Censoring-stratified test eval (matches train_xgboost_rul.py) ----
    df_full = pd.read_parquet(
        PROJECT_ROOT / "data" / "processed" / "cycling" / "unified.parquet"
    )
    censoring = compute_battery_censoring(df_full)
    test_censored_mask = np.array([censoring.get(b, False) for b in ids_te.values])
    print("\n[Eval] Censoring-stratified test metrics:")
    censoring_stratified = {}
    for label, mask in [("uncensored_only", ~test_censored_mask),
                        ("censored_only",   test_censored_mask)]:
        if mask.sum() > 0:
            m = _metrics(true_te[mask], pred_te[mask], target_range)
            censoring_stratified[label] = {"n": int(mask.sum()), **m}
            print(f"  {label:18s}  n={int(mask.sum()):>7,d}  "
                  f"R²={m['r2']:.4f}  "
                  f"RMSE={m['rmse_cycles']:.1f} cyc ({m['rmse_pct_of_range']:.2f}% of range)  "
                  f"MAE={m['mae_cycles']:.1f} cyc ({m['mae_pct_of_range']:.2f}% of range)")

    # Soft gate per professor: 1-3% RMSE-of-range on uncensored test.
    unc_rmse_pct = censoring_stratified.get("uncensored_only", {}).get("rmse_pct_of_range", 1e9)
    gates = {
        "rmse_pct_of_range < 2% strict (uncensored test)": unc_rmse_pct < TARGETS["rul_rmse"],
        "rmse_pct_of_range < 3% soft   (uncensored test)": unc_rmse_pct < 3.0,
        "rmse_pct_of_range < 5% widely-cited literature": unc_rmse_pct < 5.0,
    }
    print("[Gates]")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")

    # ---- Save metrics ----
    metrics_payload = {
        "smoke": args.smoke,
        "audited": args.exclude_capacity,
        "uncensored_train": args.exclude_censored,
        "epochs_trained": len(history),
        "train_time_s": round(train_time, 2),
        "best_epoch": int(ckpt["epoch"]),
        "n_features": int(n_features),
        "n_parameters": int(n_params),
        "receptive_field": int(model.receptive_field),
        "seq_len": args.seq_len,
        "target_range_cycles": target_range,
        "hyperparameters": {
            "channels": channels,
            "kernel_size": TCN_CONFIG["kernel_size"],
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "learning_rate": TCN_CONFIG["learning_rate"],
            "batch_size": args.batch_size,
            "patience": args.patience,
            "sequence_length": args.seq_len,
            "tag": args.tag,
        },
        "train": train_metrics,
        "val":   val_metrics,
        "test":  test_metrics,
        "test_censoring_stratified": censoring_stratified,
        "gates": gates,
    }
    with open(TBL_DIR / f"metrics{suffix}.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[Save] metrics → {(TBL_DIR / f'metrics{suffix}.json').relative_to(PROJECT_ROOT)}")

    # ---- Plots ----
    history_df = pd.DataFrame(history)
    plot_loss_curves(history_df, out_path=FIG_DIR / f"loss_curves{suffix}.png",
                     title=f"TCN RUL{suffix} — Train vs Val MSE", metric_name="loss")
    plot_overfit_check(history_df, out_path=FIG_DIR / f"overfit_check{suffix}.png",
                       title=f"TCN RUL{suffix} — overfit/underfit diagnostic")
    metric_text = (f"R²={test_metrics['r2']:.3f}\n"
                   f"RMSE={test_metrics['rmse_cycles']:.1f} cyc\n"
                   f"({test_metrics['rmse_pct_of_range']:.2f}% of range)")
    plot_predicted_vs_actual(true_te, pred_te,
                             out_path=FIG_DIR / f"predicted_vs_actual_test{suffix}.png",
                             title=f"TCN RUL{suffix} — Test set",
                             units="(cycles)", metric_text=metric_text)
    plot_residuals(true_te, pred_te,
                   out_path=FIG_DIR / f"residuals_test{suffix}.png",
                   title=f"TCN RUL{suffix} — Test residuals",
                   units="(cycles)")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
