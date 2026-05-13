"""
Comprehensive model performance audit — Iter-3 §3.14.

Pulls every saved metrics file + training log, computes train/val/test gaps
consistently, identifies overfitting and under-tuning signals, and prints a
report grouped by severity.

Usage
-----
    python scripts/model_audit.py
    python scripts/model_audit.py --json     # also save audit_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TBL_ROOT = PROJECT_ROOT / "results" / "tables"
OUT_DIR = TBL_ROOT / "model_audit"


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _safe(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def _gap(train_val: float | None, test_val: float | None) -> float | None:
    """train R² - test R² (positive = train better, suggests overfit)."""
    if train_val is None or test_val is None:
        return None
    if isinstance(train_val, (int, float)) and isinstance(test_val, (int, float)):
        return float(train_val) - float(test_val)
    return None


def _verdict_overfit(gap_r2: float | None, gap_rmse_pp: float | None,
                     model_name: str = "") -> str:
    """Classify overfit severity from train-test gap."""
    if gap_r2 is None and gap_rmse_pp is None:
        return "n/a"
    g = gap_r2 if gap_r2 is not None else 0.0
    r = gap_rmse_pp if gap_rmse_pp is not None else 0.0
    if g >= 0.20 or r >= 4.0:
        return "SEVERE"
    if g >= 0.10 or r >= 2.0:
        return "MODERATE"
    if g >= 0.05 or r >= 1.0:
        return "MILD"
    return "negligible"


def _curve_audit(csv_path: Path) -> dict:
    """Read training log and assess val-loss behaviour."""
    if not csv_path.exists():
        return {"available": False}
    df = pd.read_csv(csv_path)
    if df.empty or len(df) < 3:
        return {"available": False}
    # Pick the right loss column
    val_col = next((c for c in df.columns if "val" in c.lower() and "loss" in c.lower()), None)
    if val_col is None:
        val_col = next((c for c in df.columns if c.lower() == "val_rmse"), None)
    if val_col is None:
        val_col = next((c for c in df.columns if c.startswith("val_aft")), None)
    train_col = next((c for c in df.columns if "train" in c.lower() and "loss" in c.lower()), None)
    if train_col is None:
        train_col = next((c for c in df.columns if c.lower() == "train_rmse"), None)
    if train_col is None:
        train_col = next((c for c in df.columns if c.startswith("train_aft")), None)
    if val_col is None or train_col is None:
        return {"available": False}

    val = df[val_col].to_numpy()
    train = df[train_col].to_numpy()
    n_epochs = len(val)
    best_idx = int(np.argmin(val))
    best_val = float(val[best_idx])
    end_val = float(val[-1])
    end_train = float(train[-1])
    final_gap_pct = (end_val - end_train) / max(abs(end_train), 1e-9) * 100.0

    # Volatility: epochs after best where val rises again
    rises_after_best = int(np.sum(val[best_idx + 1:] > best_val * 1.02))
    plateau_after_best = (n_epochs - best_idx - 1) - rises_after_best

    return {
        "available": True,
        "epochs_total": n_epochs,
        "best_epoch": best_idx + 1,                 # 1-indexed
        "best_val_loss": best_val,
        "final_train_loss": end_train,
        "final_val_loss": end_val,
        "final_train_to_val_gap_pct": float(final_gap_pct),
        "rises_after_best_epoch": rises_after_best,
        "plateau_after_best_epoch": plateau_after_best,
        "early_stopped": (best_idx + 1) < n_epochs,
        "best_to_end_ratio": (best_idx + 1) / n_epochs,
    }


# -------------------------------------------------------------------------
# Per-model audit functions
# -------------------------------------------------------------------------

def audit_xgboost_soh() -> dict:
    m = json.loads((TBL_ROOT / "xgboost_soh" / "metrics_audited.json").read_text())
    curve = _curve_audit(TBL_ROOT / "xgboost_soh" / "training_log_audited.csv")
    train_r2 = _safe(m, "train", "r2")
    val_r2 = _safe(m, "val", "r2")
    test_r2 = _safe(m, "test", "r2")
    train_rmse = _safe(m, "train", "rmse")
    test_rmse = _safe(m, "test", "rmse")
    gap_r2 = _gap(train_r2, test_r2)
    gap_rmse_pp = _gap(test_rmse, train_rmse)  # positive = test worse
    return {
        "name": "XGBoost SoH (audited)",
        "type": "regression",
        "tuned": m.get("tuned", False),
        "train_r2": train_r2, "val_r2": val_r2, "test_r2": test_r2,
        "train_rmse_pct": train_rmse, "val_rmse_pct": _safe(m, "val", "rmse"),
        "test_rmse_pct": test_rmse,
        "test_mae_pct": _safe(m, "test", "mae"),
        "gap_r2": gap_r2, "gap_rmse_pp": gap_rmse_pp,
        "gates": m.get("gates", {}),
        "verdict_overfit": _verdict_overfit(gap_r2, gap_rmse_pp),
        "curve_audit": curve,
        "best_iter": m.get("best_iteration"),
        "n_features": m.get("n_features"),
        "tuning_done": "Optuna" if m.get("tuned") else "default config only",
    }


def audit_xgboost_soh_unaudited() -> dict:
    m = json.loads((TBL_ROOT / "xgboost_soh" / "metrics.json").read_text())
    curve = _curve_audit(TBL_ROOT / "xgboost_soh" / "training_log.csv")
    train_r2 = _safe(m, "train", "r2"); test_r2 = _safe(m, "test", "r2")
    train_rmse = _safe(m, "train", "rmse"); test_rmse = _safe(m, "test", "rmse")
    return {
        "name": "XGBoost SoH (unaudited, capacity included)",
        "type": "regression",
        "tuned": m.get("tuned", False),
        "train_r2": train_r2, "val_r2": _safe(m, "val", "r2"), "test_r2": test_r2,
        "train_rmse_pct": train_rmse, "test_rmse_pct": test_rmse,
        "test_mae_pct": _safe(m, "test", "mae"),
        "gap_r2": _gap(train_r2, test_r2),
        "gap_rmse_pp": _gap(test_rmse, train_rmse),
        "gates": m.get("gates", {}),
        "verdict_overfit": _verdict_overfit(_gap(train_r2, test_r2),
                                            _gap(test_rmse, train_rmse)),
        "curve_audit": curve,
        "best_iter": m.get("best_iteration"),
        "tuning_done": "Optuna" if m.get("tuned") else "default config only",
    }


def audit_grade_classifier() -> dict:
    m = json.loads((TBL_ROOT / "grade_classifier" / "metrics_audited.json").read_text())
    train_acc = _safe(m, "train", "accuracy")
    val_acc = _safe(m, "val", "accuracy")
    test_acc = _safe(m, "test", "accuracy")
    train_f1 = _safe(m, "train", "f1_macro")
    test_f1 = _safe(m, "test", "f1_macro")
    return {
        "name": "Grade Classifier (audited)",
        "type": "classification",
        "train_acc": train_acc, "val_acc": val_acc, "test_acc": test_acc,
        "train_f1macro": train_f1, "test_f1macro": test_f1,
        "gap_acc": _gap(train_acc, test_acc),
        "gap_f1": _gap(train_f1, test_f1),
        "gates": m.get("gates", {}),
        "verdict_overfit": _verdict_overfit(_gap(train_acc, test_acc),
                                            None,
                                            "Grade"),
        "tuning_done": "n/a (derived from XGBoost SoH thresholds)",
    }


def audit_chemistry_router() -> dict:
    m = json.loads((TBL_ROOT / "chemistry_router" / "eval.json").read_text())
    return {
        "name": "ChemistryRouter (deployable)",
        "type": "regression+routing",
        "router_grade_acc": _safe(m, "router", "grade_acc"),
        "router_r2": _safe(m, "router", "r2"),
        "router_rmse_pct": _safe(m, "router", "rmse"),
        "global_grade_acc": _safe(m, "global_audited", "grade_acc"),
        "global_rmse_pct": _safe(m, "global_audited", "rmse"),
        "delta_grade_pp": _safe(m, "delta", "grade_acc_pp"),
        "delta_rmse_pp": _safe(m, "delta", "rmse"),
        "n_chemistries": m.get("n_chemistries_routed"),
        "verdict_overfit": "n/a (router is composite — see per-chemistry submodels)",
        "tuning_done": "default XGBoost config; per-chemistry submodels untuned",
    }


def audit_per_chemistry() -> dict:
    df = pd.read_csv(TBL_ROOT / "per_chemistry_submodels" / "results.csv")
    cols = ["chemistry", "n_train", "submodel_grade_acc",
            "global_grade_acc", "delta_grade_acc"]
    have = [c for c in cols if c in df.columns]
    return {
        "name": "Per-chemistry submodels (router constituents)",
        "type": "regression",
        "n_chemistries": len(df),
        "summary": df[have].to_dict("records") if have else df.to_dict("records"),
        "tuning_done": "default XGBoost config — no per-chemistry tuning",
    }


def audit_isolation_forest() -> dict:
    m = json.loads((TBL_ROOT / "isolation_forest" / "metrics.json").read_text())
    train_rate = _safe(m, "train_anomaly_rate")
    test_rate = _safe(m, "test_anomaly_rate")
    return {
        "name": "IsolationForest anomaly",
        "type": "anomaly",
        "train_anomaly_rate": train_rate,
        "val_anomaly_rate": _safe(m, "val_anomaly_rate"),
        "test_anomaly_rate": test_rate,
        "drift_pp": ((test_rate - train_rate) * 100.0)
                    if (train_rate is not None and test_rate is not None) else None,
        "verdict_overfit": "drift only — not classical overfit (unsupervised)",
        "tuning_done": "default contamination (0.05); no tuning attempted",
    }


def audit_vae() -> dict:
    m = json.loads((TBL_ROOT / "vae_anomaly" / "metrics.json").read_text())
    curve = _curve_audit(TBL_ROOT / "vae_anomaly" / "training_log.csv")
    return {
        "name": "VAE anomaly",
        "type": "anomaly (deep)",
        "train_anomaly_rate": _safe(m, "train_anomaly_rate"),
        "val_anomaly_rate": _safe(m, "val_anomaly_rate"),
        "test_anomaly_rate": _safe(m, "test_anomaly_rate"),
        "best_epoch": _safe(m, "best_epoch"),
        "epochs_trained": _safe(m, "epochs_trained"),
        "curve_audit": curve,
        "verdict_overfit": ("val volatile" if curve.get("available") and
                            curve.get("rises_after_best_epoch", 0) >= 3
                            else "stable"),
        "tuning_done": "default (latent_dim=12, beta=1.0, lr=1e-3); no tuning",
    }


def audit_lstm_rul() -> dict:
    m = json.loads((TBL_ROOT / "lstm_rul" / "metrics.json").read_text())
    curve = _curve_audit(TBL_ROOT / "lstm_rul" / "training_log.csv")
    train_r2 = _safe(m, "train", "r2"); test_r2 = _safe(m, "test", "r2")
    train_rmse = _safe(m, "train", "rmse_pct_of_range")
    test_rmse = _safe(m, "test", "rmse_pct_of_range")
    return {
        "name": "LSTM RUL (replaced baseline)",
        "type": "rul-sequence",
        "train_r2": train_r2, "val_r2": _safe(m, "val", "r2"), "test_r2": test_r2,
        "train_rmse_pct": train_rmse, "test_rmse_pct": test_rmse,
        "gap_r2": _gap(train_r2, test_r2),
        "gap_rmse_pp": _gap(test_rmse, train_rmse),
        "best_epoch": _safe(m, "best_epoch"),
        "epochs_trained": _safe(m, "epochs_trained"),
        "gates": m.get("gates", {}),
        "verdict_overfit": _verdict_overfit(_gap(train_r2, test_r2),
                                            _gap(test_rmse, train_rmse)),
        "curve_audit": curve,
        "tuning_done": "Iter-2 regularisation recipe (Hong/Tang/Chen) applied; "
                       "architecture not tuned; AdamW + dropout 0.35 + patience 8 fixed",
    }


def audit_xgboost_rul(suffix: str, label: str) -> dict:
    m = json.loads((TBL_ROOT / "xgboost_rul" / f"metrics{suffix}.json").read_text())
    curve = _curve_audit(TBL_ROOT / "xgboost_rul" / f"training_log{suffix}.csv")
    train_r2 = _safe(m, "train", "r2"); test_r2 = _safe(m, "test", "r2")
    train_rmse_pct = _safe(m, "train", "rmse_pct_of_range")
    test_rmse_pct = _safe(m, "test", "rmse_pct_of_range")
    unc = _safe(m, "test_censoring_stratified", "uncensored_only", default={})
    out = {
        "name": label,
        "type": "rul-tabular",
        "tuned": m.get("tuned", False),
        "train_r2": train_r2, "val_r2": _safe(m, "val", "r2"), "test_r2": test_r2,
        "train_rmse_pct": train_rmse_pct, "test_rmse_pct": test_rmse_pct,
        "test_uncensored_r2": unc.get("r2") if isinstance(unc, dict) else None,
        "test_uncensored_rmse_pct": unc.get("rmse_pct_of_range") if isinstance(unc, dict) else None,
        "gap_r2": _gap(train_r2, test_r2),
        "gap_r2_uncensored": _gap(train_r2, unc.get("r2") if isinstance(unc, dict) else None),
        "gap_rmse_pp": _gap(test_rmse_pct, train_rmse_pct),
        "gap_rmse_pp_uncensored": _gap(unc.get("rmse_pct_of_range") if isinstance(unc, dict) else None,
                                        train_rmse_pct),
        "gates": m.get("gates", {}),
        "best_iter": _safe(m, "best_iteration"),
        "verdict_overfit": _verdict_overfit(_gap(train_r2, test_r2),
                                            _gap(test_rmse_pct, train_rmse_pct)),
        "curve_audit": curve,
        "tuning_done": "Optuna 30 trials" if m.get("tuned") else "default XGBoost config only",
    }
    return out


def audit_xgboost_aft() -> dict:
    m = json.loads((TBL_ROOT / "xgboost_rul_aft" / "metrics_audited.json").read_text())
    curve = _curve_audit(TBL_ROOT / "xgboost_rul_aft" / "training_log_audited.csv")
    return {
        "name": "XGBoost RUL AFT (negative result)",
        "type": "rul-survival",
        "objective": _safe(m, "objective"),
        "aft_distribution": _safe(m, "aft_distribution"),
        "aft_sigma": _safe(m, "aft_sigma"),
        "train_uncens_r2": _safe(m, "train_uncensored", "r2"),
        "test_uncens_r2": _safe(m, "test_uncensored", "r2"),
        "test_uncens_rmse_pct": _safe(m, "test_uncensored", "rmse_pct_of_range"),
        "gap_r2_uncensored": _gap(_safe(m, "train_uncensored", "r2"),
                                   _safe(m, "test_uncensored", "r2")),
        "gates": m.get("gates", {}),
        "verdict_overfit": _verdict_overfit(
            _gap(_safe(m, "train_uncensored", "r2"),
                 _safe(m, "test_uncensored", "r2")),
            None,
        ),
        "curve_audit": curve,
        "tuning_done": "tested 4 (distribution × σ) combos; none of them work",
    }


def audit_tcn_rul() -> dict:
    m = json.loads((TBL_ROOT / "tcn_rul" / "metrics.json").read_text())
    curve = _curve_audit(TBL_ROOT / "tcn_rul" / "training_log.csv")
    train_r2 = _safe(m, "train", "r2"); test_r2 = _safe(m, "test", "r2")
    train_rmse = _safe(m, "train", "rmse_pct_of_range")
    test_rmse = _safe(m, "test", "rmse_pct_of_range")
    unc = _safe(m, "test_censoring_stratified", "uncensored_only", default={})
    return {
        "name": "TCN RUL (Bai 2018, secondary DL baseline)",
        "type": "rul-sequence",
        "train_r2": train_r2, "val_r2": _safe(m, "val", "r2"), "test_r2": test_r2,
        "train_rmse_pct": train_rmse, "test_rmse_pct": test_rmse,
        "test_uncensored_r2": unc.get("r2") if isinstance(unc, dict) else None,
        "test_uncensored_rmse_pct": unc.get("rmse_pct_of_range") if isinstance(unc, dict) else None,
        "gap_r2": _gap(train_r2, test_r2),
        "gap_r2_uncensored": _gap(train_r2, unc.get("r2") if isinstance(unc, dict) else None),
        "gap_rmse_pp": _gap(test_rmse, train_rmse),
        "gap_rmse_pp_uncensored": _gap(unc.get("rmse_pct_of_range") if isinstance(unc, dict) else None,
                                        train_rmse),
        "best_epoch": _safe(m, "best_epoch"),
        "epochs_trained": _safe(m, "epochs_trained"),
        "n_parameters": _safe(m, "n_parameters"),
        "receptive_field": _safe(m, "receptive_field"),
        "gates": m.get("gates", {}),
        "verdict_overfit": _verdict_overfit(_gap(train_r2, test_r2),
                                            _gap(test_rmse, train_rmse)),
        "curve_audit": curve,
        "tuning_done": "default literature-recipe hyperparameters; no tuning attempted",
    }


# -------------------------------------------------------------------------
# Pretty-print
# -------------------------------------------------------------------------

def fmt(v, fmt_str=".4f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{v:{fmt_str}}"
    return str(v)


def print_block(audit: dict, header: str):
    print("=" * 78)
    print(f"  {header}")
    print("=" * 78)
    name = audit.get("name", "—")
    print(f"  Model: {name}")
    print(f"  Tuning: {audit.get('tuning_done', '—')}")

    # Regression / RUL block
    if audit.get("type") in ("regression", "rul-tabular", "rul-sequence", "rul-survival"):
        tr_r2 = audit.get("train_r2")
        va_r2 = audit.get("val_r2")
        te_r2 = audit.get("test_r2")
        unc_r2 = audit.get("test_uncensored_r2") or audit.get("test_uncens_r2")
        tr_rmse = audit.get("train_rmse_pct")
        te_rmse = audit.get("test_rmse_pct")
        unc_rmse = audit.get("test_uncensored_rmse_pct") or audit.get("test_uncens_rmse_pct")
        print(f"  R² (train / val / test): {fmt(tr_r2)} / {fmt(va_r2)} / {fmt(te_r2)}")
        if unc_r2 is not None:
            print(f"  R² uncensored test     : {fmt(unc_r2)}")
        print(f"  RMSE %% (train / test) : {fmt(tr_rmse, '.2f')} / {fmt(te_rmse, '.2f')}")
        if unc_rmse is not None:
            print(f"  RMSE %% uncensored test: {fmt(unc_rmse, '.2f')}")
        gap_r2 = audit.get("gap_r2")
        gap_r2_u = audit.get("gap_r2_uncensored")
        gap_rmse = audit.get("gap_rmse_pp")
        gap_rmse_u = audit.get("gap_rmse_pp_uncensored")
        print(f"  Train-test gap (R²):     {fmt(gap_r2, '+.4f')}"
              f"{(' uncensored: ' + fmt(gap_r2_u, '+.4f')) if gap_r2_u is not None else ''}")
        print(f"  Train-test gap (RMSE pp):{fmt(gap_rmse, '+.2f')}"
              f"{(' uncensored: ' + fmt(gap_rmse_u, '+.2f')) if gap_rmse_u is not None else ''}")
    elif audit.get("type") == "classification":
        print(f"  Acc (train / val / test): {fmt(audit.get('train_acc'))} / "
              f"{fmt(audit.get('val_acc'))} / {fmt(audit.get('test_acc'))}")
        print(f"  F1 macro (train / test): {fmt(audit.get('train_f1macro'))} / "
              f"{fmt(audit.get('test_f1macro'))}")
        print(f"  Train-test gap (acc):    {fmt(audit.get('gap_acc'), '+.4f')}")
    elif audit.get("type") == "anomaly" or audit.get("type") == "anomaly (deep)":
        print(f"  Anomaly rates (train/val/test): "
              f"{fmt(audit.get('train_anomaly_rate'), '.4f')} / "
              f"{fmt(audit.get('val_anomaly_rate'), '.4f')} / "
              f"{fmt(audit.get('test_anomaly_rate'), '.4f')}")
        print(f"  Drift (test−train, pp):  {fmt(audit.get('drift_pp'), '+.2f')}")
    elif audit.get("type") == "regression+routing":
        print(f"  Router grade-acc:        {fmt(audit.get('router_grade_acc'))}")
        print(f"  Router test RMSE %% :   {fmt(audit.get('router_rmse_pct'), '.2f')}")
        print(f"  Δ vs global (grade-acc):{fmt(audit.get('delta_grade_pp'), '+.4f')}")

    # Curve audit
    c = audit.get("curve_audit", {}) or {}
    if c.get("available"):
        print(f"  Loss curve: {c.get('epochs_total')} epochs, best at {c.get('best_epoch')} "
              f"({100 * c.get('best_to_end_ratio', 0):.0f}% in)  ·  "
              f"early-stopped: {fmt(c.get('early_stopped'))}")
        print(f"              rises_after_best={c.get('rises_after_best_epoch')}  "
              f"plateau_after_best={c.get('plateau_after_best_epoch')}")

    # Verdict
    print(f"  Overfit verdict: {audit.get('verdict_overfit', 'n/a')}")

    # Gates
    g = audit.get("gates", {}) or {}
    if g:
        print(f"  Gates:")
        for k, v in g.items():
            print(f"    [{('PASS' if v else 'FAIL')}] {k}")

    print()


def main():
    p = argparse.ArgumentParser(description="Comprehensive model audit")
    p.add_argument("--json", action="store_true", help="Save audit_report.json")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 78)
    print("  COMPREHENSIVE MODEL PERFORMANCE AUDIT")
    print("  All results pulled from results/tables/*/metrics*.json + training_log*.csv")
    print("=" * 78)

    audits = {
        "xgboost_soh_audited":          audit_xgboost_soh(),
        "xgboost_soh_unaudited":        audit_xgboost_soh_unaudited(),
        "grade_classifier":             audit_grade_classifier(),
        "chemistry_router":             audit_chemistry_router(),
        "per_chemistry":                audit_per_chemistry(),
        "isolation_forest":             audit_isolation_forest(),
        "vae_anomaly":                  audit_vae(),
        "lstm_rul":                     audit_lstm_rul(),
        "xgboost_rul_baseline":         audit_xgboost_rul("",                       "XGBoost RUL (baseline)"),
        "xgboost_rul_audited":          audit_xgboost_rul("_audited",               "XGBoost RUL audited"),
        "xgboost_rul_audited_uncens":   audit_xgboost_rul("_audited_uncensored",    "XGBoost RUL audited+uncensored (PRODUCTION)"),
        "xgboost_rul_audited_imputed":  audit_xgboost_rul("_audited_imputed",       "XGBoost RUL audited+imputed (negative)"),
        "xgboost_rul_aft":              audit_xgboost_aft(),
        "tcn_rul":                      audit_tcn_rul(),
    }

    print_block(audits["xgboost_soh_audited"],         "1.  XGBoost SoH — audited (Iter-3 production)")
    print_block(audits["xgboost_soh_unaudited"],       "2.  XGBoost SoH — unaudited (capacity included, leakage variant)")
    print_block(audits["grade_classifier"],            "3.  Grade Classifier (audited)")
    print_block(audits["chemistry_router"],            "4.  ChemistryRouter (deployable artefact)")
    print_block(audits["isolation_forest"],            "5.  Isolation Forest (anomaly, primary)")
    print_block(audits["vae_anomaly"],                 "6.  VAE (anomaly, secondary)")
    print_block(audits["lstm_rul"],                    "7.  LSTM RUL (REPLACED baseline)")
    print_block(audits["xgboost_rul_baseline"],        "8a. XGBoost RUL — default config")
    print_block(audits["xgboost_rul_audited"],         "8b. XGBoost RUL — audited (full train)")
    print_block(audits["xgboost_rul_audited_uncens"],  "8c. XGBoost RUL — audited + uncensored TRAIN (PRODUCTION)")
    print_block(audits["xgboost_rul_audited_imputed"], "8d. XGBoost RUL — imputed labels (negative result)")
    print_block(audits["xgboost_rul_aft"],             "9.  XGBoost RUL AFT survival (negative result)")
    print_block(audits["tcn_rul"],                     "10. TCN RUL (Bai 2018, secondary DL baseline)")

    # ------------------------ summary ----------------------------------
    print("=" * 78)
    print("  SUMMARY — overfit verdicts, sorted by severity")
    print("=" * 78)
    rows = []
    for k, a in audits.items():
        v = a.get("verdict_overfit", "n/a")
        rows.append({
            "model": a.get("name", k),
            "verdict": v,
            "tuning": a.get("tuning_done", "n/a")[:60],
        })
    severity_rank = {"SEVERE": 0, "MODERATE": 1, "MILD": 2,
                     "negligible": 3, "n/a": 4}
    rows.sort(key=lambda r: severity_rank.get(r["verdict"], 5))
    for r in rows:
        print(f"  [{r['verdict']:11s}] {r['model']:55s}  ·  tuning: {r['tuning']}")

    print()

    # ------------------------ flagged interventions --------------------
    print("=" * 78)
    print("  FLAGGED FOR OPTIMISATION ROUND")
    print("=" * 78)
    flags = []
    for k, a in audits.items():
        v = a.get("verdict_overfit", "n/a")
        tuning = a.get("tuning_done", "")
        if v in ("SEVERE", "MODERATE"):
            flags.append((k, a, "overfit"))
        elif "default" in tuning.lower() and a.get("type") in ("rul-tabular", "rul-sequence"):
            # untuned RUL models — opportunity to tune
            flags.append((k, a, "untuned"))
    if not flags:
        print("  No models flagged. All within acceptable bounds.")
    else:
        for k, a, reason in flags:
            print(f"  - {a['name']:55s}  reason: {reason}")
            print(f"      verdict={a.get('verdict_overfit')}  tuning={a.get('tuning_done')}")

    if args.json:
        with open(OUT_DIR / "audit_report.json", "w") as f:
            json.dump(audits, f, indent=2, default=str)
        print(f"\n[Save] {(OUT_DIR / 'audit_report.json').relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
