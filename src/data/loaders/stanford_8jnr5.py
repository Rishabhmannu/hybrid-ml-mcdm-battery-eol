"""Stanford OSF 8jnr5 (Data in Brief 2024) sister-dataset loader — STUB.

The 8jnr5 archive contains 6 INR21700-M50T cells with diagnostic_tests/RPT_*
folders and Capacity_test_with_pulses subfolders, each holding Channel_N xlsx
files exported from Arbin/Maccor cyclers. Roughly 3,282 entries in the zip.

Status: ❌ Not implemented for v1. BatteryLife coverage is sufficient.
"""
from __future__ import annotations

import pandas as pd
from src.data.loaders.schema import UNIFIED_COLUMNS


def load() -> pd.DataFrame:
    return pd.DataFrame(columns=UNIFIED_COLUMNS)
