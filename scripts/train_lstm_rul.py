"""
Stage 9 — Train LSTM RUL forecaster on per-battery cycle sequences.

Target per [src/utils/config.py::TARGETS](../src/utils/config.py): RMSE < 2 % RUL
(percent of training-set RUL range).

Usage
-----
    python scripts/train_lstm_rul.py --smoke          # 5 epochs, 30 batteries
    python scripts/train_lstm_rul.py                  # full training
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

from src.data.training_data import load_battery_sequences
from src.models.lstm_rul import BatteryLSTM, BatterySequenceDataset
from src.utils.config import DEVICE, LSTM_CONFIG, MODELS_DIR, RANDOM_SEED, RESULTS_DIR, TARGETS
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

OUT_DIR = MODELS_DIR / "lstm_rul"
FIG_DIR = RESULTS_DIR / "figures" / "lstm_rul"
TBL_DIR = RESULTS_DIR / "tables" / "lstm_rul"


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
        "r2": float(r2_score(y_true, y_pred)),
        "rmse_cycles": rmse,
        "mae_cycles": mae,
        "rmse_pct_of_range": rmse / max(target_range, 1.0) * 100.0,
        "mae_pct_of_range": mae / max(target_range, 1.0) * 100.0,
    }


def main():
    p = argparse.ArgumentParser(description="Train LSTM RUL forecaster")
    p.add_argument("--smoke", action="store_true", help="5 epochs, 30 batteries")
    p.add_argument("--epochs", type=int, default=LSTM_CONFIG["epochs"])
    p.add_argument("--batch-size", type=int, default=LSTM_CONFIG["batch_size"])
    p.add_argument("--patience", type=int, default=LSTM_CONFIG["patience"])
    p.add_argument("--seq-len", type=int, default=LSTM_CONFIG["sequence_length"])
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"LSTM RUL training  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    sequences, n_features = load_battery_sequences(
        sequence_length=args.seq_len,
        smoke=args.smoke,
    )
    X_tr, y_tr, _ = sequences["train"]
    X_va, y_va, _ = sequences["val"]
    X_te, y_te, ids_te = sequences["test"]

    if len(X_tr) == 0:
        raise SystemExit("No training sequences built — check sequence length vs cycle counts.")

    train_ds = BatterySequenceDataset(X_tr, y_tr)
    val_ds   = BatterySequenceDataset(X_va, y_va)
    test_ds  = BatterySequenceDataset(X_te, y_te)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    target_range = float(np.ptp(y_tr)) if len(y_tr) > 0 else 1.0
    print(f"\n[Data] n_features={n_features}  seq_len={args.seq_len}  "
          f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")
    print(f"[Data] RUL range (train) = {target_range:.0f} cycles")

    model = BatteryLSTM(input_size=n_features)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LSTM_CONFIG["learning_rate"],
        weight_decay=LSTM_CONFIG.get("weight_decay", 0.0),
    )
    loss_fn = nn.MSELoss()

    epochs = 5 if args.smoke else args.epochs
    callbacks = [
        ModelCheckpoint(out_dir=OUT_DIR, monitor="val_loss", filename="best.pt"),
        EarlyStopping(monitor="val_loss", patience=args.patience),
        CSVLogger(csv_path=TBL_DIR / "training_log.csv"),
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

    ckpt = torch.load(OUT_DIR / "best.pt", map_location=DEVICE, weights_only=False)
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
              f"MAE={m['mae_cycles']:.1f} cyc")

    gates = {
        "rmse_pct_of_range < 2%": test_metrics["rmse_pct_of_range"] < TARGETS["rul_rmse"],
    }
    print("[Gates] " + " | ".join(f"{k}: {'PASS' if v else 'FAIL'}" for k, v in gates.items()))

    metrics_payload = {
        "smoke": args.smoke,
        "epochs_trained": len(history),
        "train_time_s": round(train_time, 2),
        "best_epoch": int(ckpt["epoch"]),
        "n_features": int(n_features),
        "seq_len": args.seq_len,
        "target_range_cycles": target_range,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "gates": gates,
        "config": LSTM_CONFIG,
    }
    with open(TBL_DIR / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[Save] metrics → {(TBL_DIR / 'metrics.json').relative_to(PROJECT_ROOT)}")

    history_df = pd.DataFrame(history)
    plot_loss_curves(history_df, out_path=FIG_DIR / "loss_curves.png",
                     title="LSTM RUL — Train vs Val MSE", metric_name="loss")
    plot_overfit_check(history_df, out_path=FIG_DIR / "overfit_check.png",
                       title="LSTM RUL — overfit/underfit diagnostic")
    metric_text = (f"R²={test_metrics['r2']:.3f}\n"
                   f"RMSE={test_metrics['rmse_cycles']:.1f} cyc\n"
                   f"({test_metrics['rmse_pct_of_range']:.2f}% of range)")
    plot_predicted_vs_actual(true_te, pred_te,
                             out_path=FIG_DIR / "predicted_vs_actual_test.png",
                             title="LSTM RUL — Test set",
                             units="(cycles)", metric_text=metric_text)
    plot_residuals(true_te, pred_te,
                   out_path=FIG_DIR / "residuals_test.png",
                   title="LSTM RUL — Test residuals",
                   units="(cycles)")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")

    print("\nDone.")


if __name__ == "__main__":
    main()
