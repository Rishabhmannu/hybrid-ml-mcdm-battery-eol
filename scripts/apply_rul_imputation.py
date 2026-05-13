"""
Apply best-validated RUL imputation methods to right-censored cells.

Per `scripts/rul_imputation_validation.py` results: at truncation SoH = 0.815
(matching our actual censored cells' median min-SoH = 0.817), the top three
methods are:

  Method      Median rel-err   Convergence
  GP           1.41 %           49 %
  Poly2        1.64 %           95 %
  Postknee     2.68 %          100 %

This script runs an **ENSEMBLE** of (GP, Poly2, Postknee) on each of the 299
censored cells in `unified.parquet`, takes the median of converged predictions
as the imputed EoL cycle, and uses the spread (min/max of converged
predictions) as a confidence band. Falls back to Linear (100 % conv) if all
three fail.

Output
------
data/processed/cycling/imputed_rul_labels.csv with columns:
  battery_id, chemistry, source, n_observed_cycles, max_observed_cycle,
  min_observed_soh, imputed_eol_cycle, imputed_eol_lower, imputed_eol_upper,
  n_methods_converged, primary_method, status

The downstream training pipeline reads this file when
`use_imputed_rul=True` is passed to `load_feature_bundle(...)`.

Usage
-----
    python scripts/apply_rul_imputation.py
    python scripts/apply_rul_imputation.py --methods gp,poly2,postknee,linear
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.rul_imputation import (
    BaseImputer, CellTrajectory, EOL_THRESHOLD, cells_from_parquet, make_imputer,
)
from src.utils.config import PROCESSED_DIR

UNIFIED_PARQUET = PROCESSED_DIR / "cycling" / "unified.parquet"
OUT_CSV = PROCESSED_DIR / "cycling" / "imputed_rul_labels.csv"

# Default ensemble — top 3 by held-out median rel err at truncation 0.815, plus
# Linear as last-resort 100%-convergence fallback.
DEFAULT_METHODS = ("gp", "poly2", "postknee", "linear")


def main():
    p = argparse.ArgumentParser(description="Apply RUL imputation ensemble to censored cells")
    p.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS),
                   help=f"Comma-separated method names (default: {DEFAULT_METHODS})")
    p.add_argument("--output", type=str, default=str(OUT_CSV))
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",")]
    print("=" * 70)
    print(f"Applying RUL imputation ensemble  (methods={methods})")
    print("=" * 70)

    print(f"\n[Load] {UNIFIED_PARQUET.relative_to(PROJECT_ROOT)}")
    df = pd.read_parquet(UNIFIED_PARQUET)
    cells = cells_from_parquet(df, min_observed=5)
    print(f"  {len(cells):,} cells loaded")

    censored = [c for c in cells if c.is_censored]
    uncensored = [c for c in cells if not c.is_censored]
    print(f"  {len(uncensored):,} uncensored cells (population for fitting)")
    print(f"  {len(censored):,} censored cells (target for imputation)")

    # Fit population-aware imputers on ALL uncensored cells
    print(f"\n[Fit] Fitting ensemble methods on {len(uncensored)} uncensored cells ...")
    imputers: list[BaseImputer] = []
    for m in methods:
        imp = make_imputer(m)
        t0 = time.time()
        imp.fit(uncensored)
        print(f"  {m:14s}  fit time {time.time()-t0:.2f}s")
        imputers.append(imp)

    # Run ensemble per censored cell
    rows = []
    print(f"\n[Impute] Running ensemble on {len(censored)} censored cells ...")
    t0 = time.time()
    for i, cell in enumerate(censored):
        per_method = {}
        for imp in imputers:
            try:
                r = imp.impute(cell)
                if r.converged and not np.isnan(r.imputed_eol_cycle):
                    per_method[imp.name] = r.imputed_eol_cycle
            except Exception:
                pass
        # Drop the linear fallback from primary aggregation; keep it only if
        # the more accurate methods all failed.
        primary_methods = [m for m in per_method if m != "linear"]
        if primary_methods:
            ensemble_vals = np.array([per_method[m] for m in primary_methods])
            ensemble_eol = float(np.median(ensemble_vals))
            ensemble_lo = float(ensemble_vals.min())
            ensemble_hi = float(ensemble_vals.max())
            primary = ",".join(primary_methods)
            status = "ensemble"
        elif "linear" in per_method:
            ensemble_eol = float(per_method["linear"])
            ensemble_lo = ensemble_eol
            ensemble_hi = ensemble_eol
            primary = "linear (fallback)"
            status = "linear_fallback"
        else:
            ensemble_eol = float("nan")
            ensemble_lo = ensemble_hi = float("nan")
            primary = "none"
            status = "FAILED"

        rows.append({
            "battery_id":             cell.battery_id,
            "chemistry":              cell.chemistry,
            "source":                 cell.source,
            "n_observed_cycles":      int(cell.n_observed),
            "max_observed_cycle":     int(cell.max_cycle),
            "min_observed_soh":       float(cell.min_soh),
            "imputed_eol_cycle":      ensemble_eol,
            "imputed_eol_lower":      ensemble_lo,
            "imputed_eol_upper":      ensemble_hi,
            "n_methods_converged":    int(len(per_method)),
            "primary_method":         primary,
            "status":                 status,
            **{f"eol_{m}": per_method.get(m, float("nan")) for m in methods},
        })

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(censored)} cells imputed ({time.time()-t0:.1f}s)")

    print(f"  Total wall-clock: {time.time()-t0:.1f}s")

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    print(f"\n[Save] {Path(args.output).relative_to(PROJECT_ROOT)}  ({len(out):,} rows)")

    # Summary
    print(f"\n[Summary]")
    print(f"  Total censored cells:           {len(out):,}")
    print(f"  Ensemble (≥1 primary converged): "
          f"{(out['status']=='ensemble').sum():,} "
          f"({(out['status']=='ensemble').mean()*100:.1f}%)")
    print(f"  Linear fallback:                "
          f"{(out['status']=='linear_fallback').sum():,}")
    print(f"  FAILED:                         "
          f"{(out['status']=='FAILED').sum():,}")

    print(f"\n  Imputed-EoL summary stats:")
    desc = out["imputed_eol_cycle"].describe()
    print(f"    min/median/max:   {desc['min']:.0f} / {desc['50%']:.0f} / {desc['max']:.0f} cycles")
    print(f"    mean ± std:       {desc['mean']:.0f} ± {desc['std']:.0f} cycles")

    print(f"\n  Per-chemistry breakdown:")
    for chem, grp in out.groupby("chemistry"):
        n = len(grp)
        n_ens = (grp["status"] == "ensemble").sum()
        med_eol = grp["imputed_eol_cycle"].median()
        med_max = grp["max_observed_cycle"].median()
        med_extension = (grp["imputed_eol_cycle"] - grp["max_observed_cycle"]).median()
        print(f"    {chem:10s}  n={n:>3d}  ensemble={n_ens:>3d}  "
              f"med_obs_max={med_max:>5.0f}  med_imputed_eol={med_eol:>5.0f}  "
              f"med_extension=+{med_extension:>5.0f} cyc")

    print("\nDone.")


if __name__ == "__main__":
    main()
