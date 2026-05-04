"""Per-source loaders that emit a unified per-cycle schema.

All loaders return a pandas DataFrame with the columns defined in
src/data/loaders/schema.py.
"""
from src.data.loaders.schema import UNIFIED_COLUMNS, normalize_chemistry, normalize_form_factor
