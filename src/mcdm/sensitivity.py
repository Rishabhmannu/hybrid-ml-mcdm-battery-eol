"""
Sensitivity analysis for MCDM route rankings (canonical 6-criterion stack).

Runs Fuzzy BWM-TOPSIS across five weight scenarios and quantifies the
robustness of the route-recommendation as a function of weight choice.

Operationalises RQ2: how do BWMR-aligned and EU-aligned weights produce
different end-of-life route rankings? Reports:
- per-scenario rankings for each grade (A/B/C/D)
- pairwise Spearman ρ stability across all scenarios
- a dedicated BWMR-vs-EU comparison with H2 falsification check
  (if mean ρ < 0.20 across grades → H2 rejected: regimes converge)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.mcdm.topsis import (
    CANONICAL_CRITERIA,
    CANONICAL_TYPES,
    CANONICAL_ALTERNATIVES,
    build_canonical_decision_matrix,
    topsis_rank,
)
from src.utils.config import PROCESSED_DIR, SENSITIVITY_SCENARIOS, TARGETS

GRADES = ["A", "B", "C", "D"]
H2_FALSIFICATION_THRESHOLD = 0.20  # mean ρ < 0.20 between BWMR and EU rejects H2


def _literature_weights(weights_csv: Path | None = None) -> np.ndarray:
    """Load defuzzified normalized weights from fuzzy_bwm_input.csv."""
    weights_csv = weights_csv or (
        PROCESSED_DIR / "mcdm_weights" / "fuzzy_bwm_input.csv"
    )
    df = pd.read_csv(weights_csv).set_index("criterion")
    w = []
    for c in CANONICAL_CRITERIA:
        mid = df.loc[c, "tfn_middle_normalized"]
        w.append(0.0 if pd.isna(mid) else float(mid))
    arr = np.asarray(w, dtype=float)
    if arr.sum() <= 0:
        raise ValueError("All literature weights are zero — check fuzzy_bwm_input.csv")
    return arr / arr.sum()


def _resolve_scenarios(weights_csv: Path | None = None) -> dict[str, np.ndarray]:
    """Materialize SENSITIVITY_SCENARIOS, computing 'Literature Mean' on demand."""
    out: dict[str, np.ndarray] = {}
    for name, w in SENSITIVITY_SCENARIOS.items():
        if w is None:
            out[name] = _literature_weights(weights_csv)
        else:
            arr = np.asarray(w, dtype=float)
            if not np.isclose(arr.sum(), 1.0, atol=1e-3):
                arr = arr / arr.sum()
            out[name] = arr
    return out


def _topsis_for_grade(grade: str, weights: np.ndarray) -> pd.DataFrame:
    """Run TOPSIS and return ranked alternatives for a single grade."""
    dm = build_canonical_decision_matrix(grade)
    res = topsis_rank(dm, weights, CANONICAL_TYPES)
    return pd.DataFrame({
        "alternative": CANONICAL_ALTERNATIVES,
        "closeness": res["closeness"],
        "rank": res["ranking"],
    })


def run_sensitivity(weights_csv: Path | None = None) -> dict:
    """Run the full sweep. Returns a dict of DataFrames + summary stats."""
    scenarios = _resolve_scenarios(weights_csv)

    # Per-scenario × per-grade rankings
    long_rows = []
    for scenario, w in scenarios.items():
        for g in GRADES:
            df = _topsis_for_grade(g, w)
            for _, row in df.iterrows():
                long_rows.append({
                    "scenario": scenario,
                    "grade": g,
                    "alternative": row["alternative"],
                    "closeness": float(row["closeness"]),
                    "rank": int(row["rank"]),
                })
    rankings = pd.DataFrame(long_rows)

    # Recommended route per (scenario, grade)
    recos = (rankings[rankings["rank"] == 1]
             .pivot(index="scenario", columns="grade", values="alternative")
             .reindex(scenarios.keys())[GRADES])

    # Pairwise Spearman ρ across scenarios
    scenario_names = list(scenarios.keys())
    rho_matrix = pd.DataFrame(
        np.zeros((len(scenario_names), len(scenario_names))),
        index=scenario_names, columns=scenario_names
    )
    grade_rho = {g: pd.DataFrame(
        np.zeros((len(scenario_names), len(scenario_names))),
        index=scenario_names, columns=scenario_names
    ) for g in GRADES}

    for s1 in scenario_names:
        for s2 in scenario_names:
            per_grade = []
            for g in GRADES:
                r1 = (rankings[(rankings["scenario"] == s1) & (rankings["grade"] == g)]
                      .sort_values("alternative")["rank"].to_numpy())
                r2 = (rankings[(rankings["scenario"] == s2) & (rankings["grade"] == g)]
                      .sort_values("alternative")["rank"].to_numpy())
                rho, _ = spearmanr(r1, r2)
                per_grade.append(1.0 if np.isnan(rho) else float(rho))
                grade_rho[g].loc[s1, s2] = per_grade[-1]
            rho_matrix.loc[s1, s2] = float(np.mean(per_grade))

    # RQ2-specific BWMR-vs-EU comparison
    bwmr_eu_per_grade = {}
    for g in GRADES:
        rho = grade_rho[g].loc["BWMR-Heavy", "EU-Heavy"]
        bwmr_route = recos.loc["BWMR-Heavy", g]
        eu_route = recos.loc["EU-Heavy", g]
        bwmr_eu_per_grade[g] = {
            "spearman_rho": float(rho),
            "bwmr_recommended": str(bwmr_route),
            "eu_recommended": str(eu_route),
            "agree": bool(bwmr_route == eu_route),
        }
    mean_bwmr_eu_rho = float(np.mean([v["spearman_rho"] for v in bwmr_eu_per_grade.values()]))
    h2_rejected = mean_bwmr_eu_rho < H2_FALSIFICATION_THRESHOLD

    # Stability vs target gate (TARGETS["topsis_stability"] = 0.8)
    target_rho = TARGETS.get("topsis_stability", 0.8)
    rho_lower = rho_matrix.to_numpy()[np.tril_indices(len(scenario_names), k=-1)]
    overall_min = float(rho_lower.min())

    return {
        "scenarios": {k: v.tolist() for k, v in scenarios.items()},
        "rankings": rankings,
        "recommendations": recos,
        "rho_matrix_mean": rho_matrix,
        "rho_matrix_per_grade": grade_rho,
        "bwmr_eu_per_grade": bwmr_eu_per_grade,
        "mean_bwmr_eu_rho": mean_bwmr_eu_rho,
        "h2_rejected": h2_rejected,
        "h2_threshold": H2_FALSIFICATION_THRESHOLD,
        "overall_min_rho": overall_min,
        "stability_target": target_rho,
    }
