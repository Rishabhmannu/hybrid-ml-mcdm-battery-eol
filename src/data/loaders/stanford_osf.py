"""Stanford OSF (Onori Lab) Second-Life dataset loader — STUB.

NOTE: this loader is intentionally a stub. The Stanford OSF Aging dataset
contains 8 retired Nissan Leaf cells, each archived as a zip with many xlsx
files representing test sessions. Building a robust loader requires:

  1) Unzipping each Cell {n}p{m}.zip into a working directory
  2) Parsing each xlsx (different cycler software → different sheet structures)
  3) Detecting cycle boundaries from the test-step indicators
  4) Joining HPPC / Cby20 / OCV characterization tests as supplementary features

For the project's first end-to-end pass, BatteryLife's coverage is sufficient
(MATR + HUST + Stanford_2 alone provide ~12 GB of cycling data with well-defined
schemas). The Stanford OSF data adds value for the *second-life routing*
validation but is not strictly required for the initial unified parquet.

To complete this loader:
  - Implement `unzip_to_workdir()` that lazily extracts on first call
  - Implement `parse_aging_xlsx()` per Cell folder
  - Map Stanford's cycle-step structure to our per-cycle schema
  - Set chemistry=NMC (LMO/graphite per Cui et al. 2024), form_factor=18650,
    nominal_Ah=4.85 (28.6 Wh / 3.6 V nominal), second_life=True

Status: ❌ Not implemented for v1 of unified.parquet.
"""
from __future__ import annotations

import pandas as pd

from src.data.loaders.schema import UNIFIED_COLUMNS


def load() -> pd.DataFrame:
    """Returns an empty DataFrame for now. See module docstring."""
    return pd.DataFrame(columns=UNIFIED_COLUMNS)
