"""
Lightweight PyTorch training callbacks: early stopping, model checkpointing,
CSV logging, learning-rate scheduling. Single dependency on torch + tqdm.

Usage
-----
    trainer = TorchTrainer(model, optimizer, loss_fn, device=DEVICE,
                           train_loader=train_dl, val_loader=val_dl,
                           callbacks=[
                               EarlyStopping(patience=15, mode="min"),
                               ModelCheckpoint(out_dir, monitor="val_loss"),
                               CSVLogger(out_dir / "training_log.csv"),
                           ])
    history = trainer.fit(epochs=200)
"""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
from tqdm.auto import tqdm

_TQDM_INTERACTIVE = sys.stderr.isatty()


# --------------------------------------------------------------------------
# Callback protocol
# --------------------------------------------------------------------------

class Callback:
    """Subclass and override any subset of the hooks."""

    def on_train_begin(self, trainer): ...
    def on_train_end(self, trainer): ...
    def on_epoch_begin(self, trainer, epoch: int): ...
    def on_epoch_end(self, trainer, epoch: int, logs: dict): ...
    # If a callback returns True from on_epoch_end, training stops.


@dataclass
class EarlyStopping(Callback):
    monitor: str = "val_loss"
    mode: str = "min"          # min or max
    patience: int = 15
    min_delta: float = 1e-5
    verbose: bool = True

    _best: float = field(default=float("inf"), init=False)
    _wait: int = field(default=0, init=False)

    def on_train_begin(self, trainer):
        self._best = float("inf") if self.mode == "min" else -float("inf")
        self._wait = 0

    def _improved(self, current: float) -> bool:
        if self.mode == "min":
            return current < self._best - self.min_delta
        return current > self._best + self.min_delta

    def on_epoch_end(self, trainer, epoch: int, logs: dict) -> bool:
        current = logs.get(self.monitor)
        if current is None:
            return False
        if self._improved(current):
            self._best = current
            self._wait = 0
            return False
        self._wait += 1
        if self._wait >= self.patience:
            if self.verbose:
                print(f"[EarlyStopping] no improvement on {self.monitor} "
                      f"for {self.patience} epochs. Stopping at epoch {epoch}.")
            return True
        return False


@dataclass
class ModelCheckpoint(Callback):
    out_dir: Path
    monitor: str = "val_loss"
    mode: str = "min"
    filename: str = "best.pt"
    save_last: bool = True
    verbose: bool = True

    _best: float = field(default=float("inf"), init=False)

    def on_train_begin(self, trainer):
        self.out_dir = Path(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._best = float("inf") if self.mode == "min" else -float("inf")

    def _improved(self, current: float) -> bool:
        if self.mode == "min":
            return current < self._best
        return current > self._best

    def on_epoch_end(self, trainer, epoch: int, logs: dict) -> bool:
        current = logs.get(self.monitor)
        if current is not None and self._improved(current):
            self._best = current
            torch.save({
                "epoch": epoch,
                "state_dict": trainer.model.state_dict(),
                "logs": logs,
                "monitor": self.monitor,
                "monitor_value": current,
            }, self.out_dir / self.filename)
            if self.verbose:
                print(f"[ModelCheckpoint] new best {self.monitor}={current:.5f} "
                      f"(epoch {epoch}) -> {self.filename}")

        if self.save_last:
            torch.save({
                "epoch": epoch,
                "state_dict": trainer.model.state_dict(),
                "logs": logs,
            }, self.out_dir / "last.pt")
        return False


@dataclass
class CSVLogger(Callback):
    csv_path: Path
    _writer: object = field(default=None, init=False)
    _file: object = field(default=None, init=False)
    _fieldnames: list = field(default_factory=list, init=False)

    def on_train_begin(self, trainer):
        self.csv_path = Path(self.csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.csv_path, "w", newline="")

    def on_epoch_end(self, trainer, epoch: int, logs: dict) -> bool:
        row = {"epoch": epoch, **logs}
        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
        # Add new fields that appear later
        for k in row:
            if k not in self._fieldnames:
                # csv DictWriter doesn't support dynamic fields; rebuild.
                self._fieldnames.append(k)
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self._file.flush()
        return False

    def on_train_end(self, trainer):
        if self._file:
            self._file.close()


@dataclass
class CosineLRScheduler(Callback):
    """Wraps torch.optim.lr_scheduler.CosineAnnealingLR — stepped per epoch."""
    T_max: int = 200
    eta_min: float = 0.0
    _sched: object = field(default=None, init=False)

    def on_train_begin(self, trainer):
        self._sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=self.T_max, eta_min=self.eta_min,
        )

    def on_epoch_end(self, trainer, epoch: int, logs: dict) -> bool:
        if self._sched is not None:
            self._sched.step()
            logs["lr"] = self._sched.get_last_lr()[0]
        return False


# --------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------

class TorchTrainer:
    """Minimal training loop: epochs × (train_step → val_step) → callbacks."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable,
        *,
        device: torch.device,
        train_loader,
        val_loader,
        callbacks: list[Callback] | None = None,
        custom_step: Callable | None = None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks or []
        self.custom_step = custom_step
        self.history: list[dict] = []

    def _do_step(self, batch, train: bool):
        if self.custom_step:
            return self.custom_step(self.model, batch, self.loss_fn, self.device, train)
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)
        out = self.model(x)
        loss = self.loss_fn(out, y)
        return {"loss": loss}

    def _epoch(self, loader, *, train: bool, desc: str):
        self.model.train(train)
        total_loss = 0.0
        n = 0
        bar = tqdm(
            loader, desc=desc, leave=False,
            disable=(loader is None or len(loader) == 0 or not _TQDM_INTERACTIVE),
            mininterval=1.0,
        )
        for batch in bar:
            if train:
                self.optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                step_out = self._do_step(batch, train=train)
                loss = step_out["loss"]
            if train:
                loss.backward()
                self.optimizer.step()
            bs = batch[0].size(0) if isinstance(batch, (list, tuple)) else 1
            total_loss += float(loss.detach().cpu()) * bs
            n += bs
            bar.set_postfix(loss=f"{total_loss/max(n,1):.4f}")
        return total_loss / max(n, 1)

    def fit(self, *, epochs: int) -> list[dict]:
        for cb in self.callbacks:
            cb.on_train_begin(self)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)

            train_loss = self._epoch(self.train_loader, train=True,
                                     desc=f"Epoch {epoch}/{epochs} [train]")
            val_loss = self._epoch(self.val_loader, train=False,
                                   desc=f"Epoch {epoch}/{epochs} [val]")
            logs = {
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "epoch_time_s": round(time.time() - t0, 2),
            }
            self.history.append({"epoch": epoch, **logs})

            stop = False
            for cb in self.callbacks:
                if cb.on_epoch_end(self, epoch, logs):
                    stop = True
            print(f"Epoch {epoch}/{epochs}  "
                  f"train_loss={logs['train_loss']:.5f}  "
                  f"val_loss={logs['val_loss']:.5f}  "
                  f"({logs['epoch_time_s']}s)")
            if stop:
                break

        for cb in self.callbacks:
            cb.on_train_end(self)
        return self.history
