"""NASA Randomized v1 (Kaggle, RW_Skewed + Uniform_Distribution) loader — STUB.

Source: data/raw/nasa_kaggle/dataset/dataset/{Battery_Uniform_Distribution_*,
        RW_Skewed_*}/

Each sub-experiment folder contains 11 files: 10 .mat data files plus a
README.html / .Rmd. The .mat files have a top-level dict with key 'data' that
unpacks into {step, procedure, description}, where 'step' is an array of step
records describing each test step in the cycler protocol.

Parsing the step array correctly requires understanding the experimental
protocol per sub-experiment (different load profiles per RW vs Uniform).

Status: ❌ Not implemented for v1. This is a bonus / supplementary dataset
that does not block the unified parquet.
"""
from __future__ import annotations

import pandas as pd
from src.data.loaders.schema import UNIFIED_COLUMNS


def load() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_COLUMNS)
