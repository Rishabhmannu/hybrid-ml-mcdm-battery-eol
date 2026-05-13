"""Section 2 — Anomaly gate.

Runs both anomaly models on the selected cell/cycle and renders a side-by-side
verdict:

  Isolation Forest  ←  unsupervised tree ensemble
  VAE (β-anneal)    ←  reconstruction-error against the healthy manifold

A plain-language verdict at the bottom translates the two scores into a
"proceed to health assessment" / "interpret downstream predictions with caution"
recommendation.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from frontend.lib.featurize_single_cell import featurize_single_cell
from frontend.lib.hf_resolve import hf_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Metrics live in the GitHub repo (small JSON snapshots); model weights live on HF.
VAE_METRICS_PATH = PROJECT_ROOT / "results" / "tables" / "vae_anomaly" / "metrics.json"
ISO_METRICS_PATH = PROJECT_ROOT / "results" / "tables" / "isolation_forest" / "metrics.json"


@st.cache_resource
def _load_shared_scaler():
    return joblib.load(hf_path("anomaly_shared/feature_scaler.pkl"))


@st.cache_resource
def _load_isolation_forest():
    model = joblib.load(hf_path("isolation_forest/isolation_forest.pkl"))
    metrics = json.loads(ISO_METRICS_PATH.read_text())
    return model, metrics


@st.cache_resource
def _load_vae():
    from src.models.vae import VAE  # imported lazily so cold-start doesn't pay for torch
    ckpt = torch.load(hf_path("vae_anomaly/best.pt"),
                      map_location="cpu", weights_only=False)
    metrics = json.loads(VAE_METRICS_PATH.read_text())
    model = VAE(input_dim=metrics["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, metrics


def _vae_recon_error(model, x) -> float:
    """Single-sample CPU-only reconstruction error (decoupled from src/models/vae.DEVICE)."""
    with torch.no_grad():
        t = torch.from_numpy(x).float()
        recon, _, _ = model(t)
        return float(F.mse_loss(recon, t, reduction="none").mean(dim=1).item())


def _verdict_card(col, label: str, score: float, threshold: float,
                  flagged: bool, hint: str, help_text: str = "") -> None:
    col.markdown(f"**{label}**", help=help_text or None)
    col.metric(
        label="score" if not flagged else "score ⚠",
        value=f"{score:.4f}",
        delta=f"threshold {threshold:.4f}",
        delta_color="off",
    )
    col.markdown(
        f"<span style='color:{'#B33' if flagged else '#1A1A1A'};font-weight:600;'>"
        f"{'Flagged as anomalous' if flagged else 'Within healthy manifold'}"
        "</span>",
        unsafe_allow_html=True,
    )
    col.caption(hint)


def render(cell_df: pd.DataFrame, cycle_n: int) -> dict:
    """Render the anomaly panel. Returns the verdict dict for downstream use."""
    with st.container(border=True):
        st.subheader("2 · Anomaly gate")
        st.caption(
            "Two unsupervised checks run before health assessment. If either flags "
            "the cell at this cycle, downstream predictions should be interpreted "
            "with caution."
        )
        st.markdown(
            "<span style='color:#666;font-style:italic;font-size:0.85rem;'>"
            "_An **anomaly gate** quality-checks the cell before any predictions. "
            "Two unsupervised models — one tree-based, one neural — vote "
            "independently on whether this cell looks like the ones the framework "
            "was trained on._"
            "</span>",
            unsafe_allow_html=True,
        )

        scaler = _load_shared_scaler()
        iso_model, iso_metrics = _load_isolation_forest()
        vae_model, vae_metrics = _load_vae()
        x = featurize_single_cell(cell_df, cycle_n, scaler, mode="full")

        iso_pred = int(iso_model.predict(x)[0])  # +1 normal, -1 anomaly
        iso_score = float(-iso_model.decision_function(x)[0])
        iso_threshold = float(iso_metrics.get("threshold", 0.0))
        iso_flagged = iso_pred == -1

        vae_error = _vae_recon_error(vae_model, x)
        vae_threshold = float(vae_metrics["anomaly_threshold"])
        vae_flagged = vae_error > vae_threshold

        c1, c2 = st.columns(2)
        _verdict_card(
            c1, "Isolation Forest",
            score=iso_score, threshold=iso_threshold, flagged=iso_flagged,
            hint=("Threshold tuned to flag ~5 % of cells. "
                  "Higher score = more unusual cell."),
            help_text=("Tree-based outlier detector. Builds many random splits "
                       "and measures how easily a sample gets isolated — outliers "
                       "are isolated quickly."),
        )
        _verdict_card(
            c2, "VAE (neural-network gate)",
            score=vae_error, threshold=vae_threshold, flagged=vae_flagged,
            hint=("Reconstruction error vs a threshold set so ~5 % of training "
                  "cells would be flagged. Higher score = more unusual cell."),
            help_text=("Variational Autoencoder — a neural network that learns to "
                       "compress and reconstruct healthy cells. Cells with high "
                       "reconstruction error don't look like training data."),
        )

        agree = (iso_flagged == vae_flagged)
        if iso_flagged and vae_flagged:
            verdict = "Both models flag this cell — downstream predictions are unreliable."
            color = "#B33"
        elif iso_flagged or vae_flagged:
            which = "Isolation Forest" if iso_flagged else "VAE"
            verdict = (f"Only {which} flags this cell — borderline case, "
                       "downstream predictions may still be usable with caveats.")
            color = "#C70"
        else:
            verdict = "Both models pass — proceed to health assessment with normal confidence."
            color = "#080"

        st.markdown("---")
        st.markdown(
            f"<div style='color:{color};font-weight:600;'>{verdict}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Model agreement: **{'yes' if agree else 'no'}** "
            f"(Jaccard 1.0 / 0.0 for a single-cell snapshot)"
        )

    return {
        "iso": {"score": iso_score, "threshold": iso_threshold, "flagged": iso_flagged},
        "vae": {"score": vae_error, "threshold": vae_threshold, "flagged": vae_flagged},
        "agree": agree,
    }
