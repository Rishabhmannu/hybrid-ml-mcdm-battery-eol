"""Hugging Face Hub path resolver for model artefacts.

All trained model weights live in the public HF repo
`cmpunkmannu/hybrid-ml-mcdm-battery-eol`. This module wraps `hf_hub_download`
with a small in-process cache so multiple loaders sharing a file (e.g. the
ChemistryRouter looking up seven specialists) don't re-hit HF.

`hf_hub_download` itself caches downloaded files on disk at
`~/.cache/huggingface/hub/`, so the network round-trip happens once per
file per machine — first run downloads ~55 MB, subsequent runs are instant.

The local `/models/` directory is no longer needed at runtime. The training
scripts still write there, but the live demo (locally or on Streamlit Cloud)
loads from HF.
"""
from __future__ import annotations

from functools import lru_cache

from huggingface_hub import hf_hub_download

HF_REPO_ID = "cmpunkmannu/hybrid-ml-mcdm-battery-eol"


@lru_cache(maxsize=None)
def hf_path(filename: str) -> str:
    """Return the local cache path for a file in the HF model repo.

    `filename` is the path within the HF repo, e.g. "xgboost_soh/xgboost_soh_audited.json".
    Downloads on first call, returns the cached path on subsequent calls.
    """
    return hf_hub_download(repo_id=HF_REPO_ID, filename=filename)
