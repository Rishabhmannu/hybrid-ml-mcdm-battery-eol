"""
Stage 9 — Train VAE for cycle-level anomaly detection.

The VAE learns the manifold of healthy cycle records; high reconstruction
error flags noisy/defective cycles to feed downstream filters.

Usage
-----
    python scripts/train_vae_anomaly.py --smoke      # 3 epochs, sub-sample
    python scripts/train_vae_anomaly.py              # full training
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.training_data import load_feature_bundle
from src.models.vae import VAE, compute_reconstruction_error
from src.utils.config import DEVICE, MODELS_DIR, RANDOM_SEED, RESULTS_DIR, VAE_CONFIG
from src.utils.plots import (
    plot_anomaly_scores,
    plot_loss_curves,
    plot_overfit_check,
)
from src.utils.training_callbacks import (
    Callback,
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    TorchTrainer,
)

OUT_DIR = MODELS_DIR / "vae_anomaly"
FIG_DIR = RESULTS_DIR / "figures" / "vae_anomaly"
TBL_DIR = RESULTS_DIR / "tables" / "vae_anomaly"


# Module-level β controller (set in main, read by _vae_step + _ComponentLogger)
_BETA_STATE = {"current": VAE_CONFIG["beta"]}


def _vae_step(model, batch, _loss_fn, device, train: bool):
    """Custom step for the trainer — returns dict with `loss` and components."""
    x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
    recon, mu, log_var = model(x)
    loss, recon_l, kl_l = VAE.loss_function(
        recon, x, mu, log_var, beta=_BETA_STATE["current"]
    )
    return {"loss": loss, "recon": float(recon_l.detach()), "kl": float(kl_l.detach())}


class _ComponentLogger(Callback):
    """Capture last batch's recon/kl into history (placed last in callback list)."""

    def __init__(self):
        self.last = {}

    def on_epoch_end(self, trainer, epoch, logs):
        # The components from custom_step aren't carried up — recompute from val loader
        model = trainer.model
        model.eval()
        recon_total = 0.0
        kl_total = 0.0
        n = 0
        with torch.no_grad():
            for batch in trainer.val_loader:
                x = batch[0].to(trainer.device)
                recon, mu, lv = model(x)
                _, r, k = VAE.loss_function(recon, x, mu, lv, beta=_BETA_STATE["current"])
                recon_total += float(r) * x.size(0)
                kl_total += float(k) * x.size(0)
                n += x.size(0)
        logs["val_recon"] = recon_total / max(n, 1)
        logs["val_kl"] = kl_total / max(n, 1)
        logs["beta_used"] = _BETA_STATE["current"]
        return False


class _BetaAnnealer(Callback):
    """Linearly anneal β from 0 → target_beta over the first `anneal_epochs`."""

    def __init__(self, target_beta: float, anneal_epochs: int):
        self.target_beta = float(target_beta)
        self.anneal_epochs = max(1, int(anneal_epochs))

    def on_epoch_begin(self, trainer, epoch: int):
        if epoch < self.anneal_epochs:
            beta = self.target_beta * (epoch + 1) / self.anneal_epochs
        else:
            beta = self.target_beta
        _BETA_STATE["current"] = beta


def main():
    p = argparse.ArgumentParser(description="Train VAE anomaly detector")
    p.add_argument("--smoke", action="store_true", help="3 epochs, sub-sampled data")
    p.add_argument("--epochs", type=int, default=VAE_CONFIG["epochs"])
    p.add_argument("--batch-size", type=int, default=VAE_CONFIG["batch_size"])
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--beta", type=float, default=VAE_CONFIG["beta"],
                   help="β-VAE KL weight (default %(default)s; "
                        "Iter-3 §3.14 ablation tries β=0.5 to relax KL constraint)")
    p.add_argument("--beta-anneal", action="store_true",
                   help="Linearly anneal β from 0 to --beta over the first --anneal-epochs "
                        "(Iter-3 §3.14: prevents KL collapse at start of training).")
    p.add_argument("--anneal-epochs", type=int, default=10,
                   help="Number of epochs over which β anneals 0 → target value (default %(default)s)")
    p.add_argument("--latent-dim", type=int, default=VAE_CONFIG["latent_dim"],
                   help="VAE latent dimension (default %(default)s; "
                        "Iter-3 §3.14 ablation tries 8 to reduce capacity)")
    p.add_argument("--tag", default="",
                   help="Suffix for output files (e.g. '_beta05_latent8')")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TBL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"VAE anomaly training  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    bundle = load_feature_bundle(smoke=args.smoke)
    X_tr = bundle.X_train
    X_va = bundle.X_val
    X_te = bundle.X_test
    n_features = X_tr.shape[1]

    train_ds = TensorDataset(torch.from_numpy(X_tr).float())
    val_ds   = TensorDataset(torch.from_numpy(X_va).float())
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Set the module-level β state for the loss function
    if args.beta_anneal:
        _BETA_STATE["current"] = 0.0
        print(f"[β] Annealing 0 → {args.beta} over first {args.anneal_epochs} epochs, "
              f"then constant {args.beta}")
    else:
        _BETA_STATE["current"] = args.beta
        print(f"[β] Constant β = {args.beta}")

    model = VAE(input_dim=n_features, latent_dim=args.latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=VAE_CONFIG["learning_rate"])

    epochs = 3 if args.smoke else args.epochs
    suffix = args.tag or ""
    callbacks = [
        ModelCheckpoint(out_dir=OUT_DIR, monitor="val_loss",
                        filename=f"best{suffix}.pt"),
        EarlyStopping(monitor="val_loss", patience=args.patience),
        CSVLogger(csv_path=TBL_DIR / f"training_log{suffix}.csv"),
        _ComponentLogger(),
    ]
    if args.beta_anneal:
        callbacks.append(_BetaAnnealer(args.beta, args.anneal_epochs))
    trainer = TorchTrainer(
        model, optimizer, loss_fn=None,
        device=DEVICE, train_loader=train_dl, val_loader=val_dl,
        callbacks=callbacks,
        custom_step=_vae_step,
    )

    t0 = time.time()
    history = trainer.fit(epochs=epochs)
    train_time = time.time() - t0
    print(f"\n[Train] {len(history)} epochs · {train_time:.1f}s")

    ckpt = torch.load(OUT_DIR / f"best{suffix}.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[Eval] loaded best @ epoch {ckpt['epoch']} (val_loss={ckpt['monitor_value']:.5f})")

    train_errors = compute_reconstruction_error(model, torch.from_numpy(X_tr).float())
    val_errors = compute_reconstruction_error(model, torch.from_numpy(X_va).float())
    test_errors = compute_reconstruction_error(model, torch.from_numpy(X_te).float())

    threshold = float(np.percentile(train_errors, VAE_CONFIG["anomaly_percentile"]))
    train_anom = int((train_errors > threshold).sum())
    val_anom = int((val_errors > threshold).sum())
    test_anom = int((test_errors > threshold).sum())

    print(f"\n[Anomaly] threshold = p{VAE_CONFIG['anomaly_percentile']} of train recon error = {threshold:.5f}")
    print(f"  train: {train_anom}/{len(train_errors)} flagged ({train_anom/len(train_errors)*100:.2f}%)")
    print(f"  val:   {val_anom}/{len(val_errors)} flagged ({val_anom/len(val_errors)*100:.2f}%)")
    print(f"  test:  {test_anom}/{len(test_errors)} flagged ({test_anom/len(test_errors)*100:.2f}%)")

    metrics = {
        "smoke": args.smoke,
        "epochs_trained": len(history),
        "train_time_s": round(train_time, 2),
        "best_epoch": int(ckpt["epoch"]),
        "n_features": int(n_features),
        "anomaly_threshold": threshold,
        "train_anomaly_rate": train_anom / max(len(train_errors), 1),
        "val_anomaly_rate":   val_anom / max(len(val_errors), 1),
        "test_anomaly_rate":  test_anom / max(len(test_errors), 1),
        "train_recon_err_mean": float(train_errors.mean()),
        "val_recon_err_mean":   float(val_errors.mean()),
        "test_recon_err_mean":  float(test_errors.mean()),
        "hyperparameters": {
            "beta_target": args.beta,
            "beta_anneal": args.beta_anneal,
            "anneal_epochs": args.anneal_epochs if args.beta_anneal else None,
            "latent_dim": args.latent_dim,
            "tag": args.tag,
        },
        "config": VAE_CONFIG,
    }
    with open(TBL_DIR / f"metrics{suffix}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[Save] metrics → {(TBL_DIR / f'metrics{suffix}.json').relative_to(PROJECT_ROOT)}")

    # Plots
    history_df = pd.DataFrame(history)
    plot_loss_curves(history_df, out_path=FIG_DIR / f"loss_curves{suffix}.png",
                     title=f"VAE{suffix} — Train vs Val ELBO loss", metric_name="loss")
    plot_overfit_check(history_df, out_path=FIG_DIR / f"overfit_check{suffix}.png",
                       title=f"VAE{suffix} — overfit/underfit diagnostic")
    plot_anomaly_scores(test_errors, threshold,
                        out_path=FIG_DIR / f"anomaly_score_distribution{suffix}.png",
                        title=f"VAE{suffix} — Test reconstruction-error distribution")
    print(f"[Save] figures → {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    print("\nDone.")


if __name__ == "__main__":
    main()
