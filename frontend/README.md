# Streamlit demo — Hybrid ML-MCDM Battery EoL

[![Live Demo](https://img.shields.io/badge/Live%20Demo-click%20here-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://hybrid-ml-mcdm-battery-eol.streamlit.app)
[![Model Weights](https://img.shields.io/badge/Model%20Weights-Hugging%20Face-FFD21E?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/cmpunkmannu/hybrid-ml-mcdm-battery-eol)

Interactive web app that demonstrates the full Hybrid ML-MCDM pipeline on 19 curated battery cells across 7 chemistries. Each cell moves through the same six stages used in the manuscript — anomaly gating, State-of-Health prediction, Remaining-Useful-Life estimation, regulatory-aware MCDM routing, and Digital Product Passport emission — with a downloadable 8-page PDF cell report.

## Run locally

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

First launch downloads ~90 MB of model weights from the Hugging Face Hub and caches them at `~/.cache/huggingface/hub/`. Subsequent launches are instant.

## Structure

```
app.py            Streamlit entry point
components/       Per-section UI modules (sidebar + 6 panels)
lib/              Featurizer, HF-path resolver, PDF report builder
data/             Curated 19-cell demo parquet + manifest + cell stories
.streamlit/       Theme overrides (sans-serif, neutral palette)
requirements.txt  Inference-only dependencies
```

## How the live demo runs

- Model weights live on the [Hugging Face Hub](https://huggingface.co/cmpunkmannu/hybrid-ml-mcdm-battery-eol) and are pulled at first load via `huggingface_hub.hf_hub_download`.
- Streamlit Community Cloud auto-redeploys on every push to `main` — no build configuration beyond `frontend/requirements.txt` and the deploy form's `frontend/app.py` entry-point.
- Per-cell inference is live. Selecting a different cell or moving the cycle slider triggers a fresh pass through all five ML models, the MCDM ranking, and the DPP construction — nothing is replayed from saved predictions.

## Editing the demo

- Section renderers live in `components/<section>.py` and each return a dict downstream sections consume.
- Adding or replacing a demo cell: edit `scripts/prepare_demo_cells.py`, re-run it, commit the regenerated `data/demo_cells.parquet` + `data/demo_cells_manifest.json`.
- Refreshing model weights after a retrain: re-run `scripts/upload_models_to_hf.py` — the demo picks up new versions on next cold start.
