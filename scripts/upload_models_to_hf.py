"""
Upload the demo-required model artefacts to the Hugging Face Hub.

Why: the demo frontend deploys to Streamlit Community Cloud, which clones the
GitHub repo at boot. Trained model weights (~55 MB across XGBoost / TCN / VAE /
Isolation Forest) are too bulky and dataset-shaped to live in the code repo,
so they're released as a public HF model repo and fetched on first load via
`huggingface_hub.hf_hub_download`.

This script is idempotent — `upload_folder` diffs against the remote and only
re-pushes files that have changed. Safe to re-run after any retrain.

Usage
-----
    HF_TOKEN=hf_xxx python scripts/upload_models_to_hf.py
    # or, with .env populated:
    python scripts/upload_models_to_hf.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
HF_REPO_ID = "cmpunkmannu/hybrid-ml-mcdm-battery-eol"
MODEL_CARD_PATH = PROJECT_ROOT / "scripts" / "_hf_model_card.md"

# The exact subset of /models/** the demo needs at inference time. Anything
# not in this list (LSTM legacy, ablation variants, smoke-test checkpoints,
# Optuna studies) stays local. Keep this list narrow — any new file pushed
# to HF should be deliberate.
DEMO_FILES = [
    # Audited global SoH regressor — largest single file (~46 MB)
    "xgboost_soh/xgboost_soh_audited.json",
    "xgboost_soh/feature_scaler_audited.pkl",
    "xgboost_soh/feature_names_audited.json",
    # Chemistry-router: 7 per-chemistry specialists + shared scaler + manifest
    "per_chemistry/router_manifest.json",
    "per_chemistry/feature_scaler.pkl",
    "per_chemistry/feature_names.json",
    "per_chemistry/LCO/xgboost_soh_audited.json",
    "per_chemistry/LFP/xgboost_soh_audited.json",
    "per_chemistry/NCA/xgboost_soh_audited.json",
    "per_chemistry/NMC/xgboost_soh_audited.json",
    "per_chemistry/Na-ion/xgboost_soh_audited.json",
    "per_chemistry/Zn-ion/xgboost_soh_audited.json",
    "per_chemistry/other/xgboost_soh_audited.json",
    # XGBoost RUL — production (audited + uncensored-train winner)
    "xgboost_rul/xgboost_rul_audited_uncensored.json",
    "xgboost_rul/feature_scaler_audited_uncensored.pkl",
    "xgboost_rul/feature_names_audited_uncensored.json",
    # TCN RUL (deep-learning RUL)
    "tcn_rul/best.pt",
    "tcn_rul/feature_scaler.pkl",
    "tcn_rul/feature_meta.json",
    # Anomaly gate — IsoForest + VAE share a 38-dim featurizer
    "isolation_forest/isolation_forest.pkl",
    "vae_anomaly/best.pt",
    "anomaly_shared/feature_scaler.pkl",
    "anomaly_shared/feature_meta.json",
]


def _load_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in env or .env")


def _check_files_exist() -> None:
    missing = [f for f in DEMO_FILES if not (MODELS_DIR / f).exists()]
    if missing:
        print("Missing files (will skip):")
        for f in missing:
            print(f"  · {f}")
        raise SystemExit("aborting — fix missing files first")


def main() -> None:
    from huggingface_hub import HfApi

    token = _load_hf_token()
    api = HfApi(token=token)

    me = api.whoami()
    print(f"Authenticated as: {me['name']} (token role: write)")
    print(f"Target repo:      {HF_REPO_ID}")
    print(f"Local /models/:   {MODELS_DIR}")
    print()

    _check_files_exist()

    total_bytes = sum((MODELS_DIR / f).stat().st_size for f in DEMO_FILES)
    print(f"Uploading {len(DEMO_FILES)} files · {total_bytes / 1024 / 1024:.1f} MB total")
    for f in DEMO_FILES:
        size_kb = (MODELS_DIR / f).stat().st_size / 1024
        print(f"  · {f}  ({size_kb:.0f} KB)")
    print()

    print("Starting upload (idempotent — only changed files are re-pushed)...")
    api.upload_folder(
        folder_path=str(MODELS_DIR),
        repo_id=HF_REPO_ID,
        repo_type="model",
        allow_patterns=DEMO_FILES,
        commit_message="Upload demo-required model artefacts",
    )

    if MODEL_CARD_PATH.exists():
        print("\nUploading model card (README.md)...")
        api.upload_file(
            path_or_fileobj=str(MODEL_CARD_PATH),
            path_in_repo="README.md",
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="Update model card",
        )
    else:
        print(f"\n[skip] no model card found at {MODEL_CARD_PATH.relative_to(PROJECT_ROOT)}")

    print(f"\n✓ Done. View at https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
