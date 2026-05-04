"""
TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)
for ranking end-of-life routes per battery grade.

References:
- Aishwarya et al. (2025), CLSC — Fuzzy AHP-TOPSIS for EV battery SC
- ACS IECR (2025) — BWM-TOPSIS for sustainable Li-ion SC
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import MCDM_CRITERIA, MCDM_CRITERIA_TYPES, MCDM_ALTERNATIVES


def normalize_matrix(decision_matrix: np.ndarray) -> np.ndarray:
    """Vector normalization of the decision matrix."""
    norms = np.sqrt(np.sum(decision_matrix ** 2, axis=0))
    norms[norms == 0] = 1  # Avoid division by zero
    return decision_matrix / norms


def topsis_rank(decision_matrix: np.ndarray, weights: np.ndarray,
                criteria_types: list) -> dict:
    """
    Run TOPSIS ranking.

    Parameters:
        decision_matrix: (n_alternatives x n_criteria) array
        weights: criteria weight vector (sums to 1)
        criteria_types: list of 'benefit' or 'cost' per criterion

    Returns:
        dict with rankings, closeness coefficients, and ideal solutions
    """
    n_alt, n_crit = decision_matrix.shape

    # Step 1: Normalize
    normalized = normalize_matrix(decision_matrix)

    # Step 2: Apply weights
    weighted = normalized * weights

    # Step 3: Determine ideal solutions
    pis = np.zeros(n_crit)  # Positive Ideal Solution
    nis = np.zeros(n_crit)  # Negative Ideal Solution

    for j in range(n_crit):
        if criteria_types[j] == "benefit":
            pis[j] = weighted[:, j].max()
            nis[j] = weighted[:, j].min()
        else:  # cost
            pis[j] = weighted[:, j].min()
            nis[j] = weighted[:, j].max()

    # Step 4: Calculate distances
    d_pos = np.sqrt(np.sum((weighted - pis) ** 2, axis=1))
    d_neg = np.sqrt(np.sum((weighted - nis) ** 2, axis=1))

    # Step 5: Relative closeness
    closeness = d_neg / (d_pos + d_neg + 1e-10)

    # Step 6: Rank (higher closeness = better)
    ranking = np.argsort(-closeness) + 1  # 1-indexed ranks

    return {
        "closeness": closeness,
        "ranking": ranking,
        "d_positive": d_pos,
        "d_negative": d_neg,
        "pis": pis,
        "nis": nis,
    }


def build_decision_matrix_for_grade(grade: str) -> np.ndarray:
    """
    Build the decision matrix for a specific battery grade.
    Scores are derived from literature and domain knowledge.

    Criteria: [Technical, Economic, Environmental, Compliance, Safety(cost)]
    Alternatives: [Grid ESS, Home ESS, Component Reuse, Direct Recycling]

    Returns:
        4x5 numpy array
    """
    # Score matrices per grade (1-10 scale)
    # These are initial estimates; to be refined during implementation
    matrices = {
        "A": np.array([
            # Tech   Econ   Env    Compl  Safety
            [9.0,   8.5,   8.0,   7.0,   3.0],   # Grid ESS
            [8.0,   7.5,   7.5,   7.0,   2.5],   # Home ESS
            [5.0,   5.0,   6.0,   6.0,   4.0],   # Component Reuse
            [3.0,   4.0,   5.0,   9.0,   5.0],   # Direct Recycling
        ]),
        "B": np.array([
            [6.0,   6.0,   7.0,   7.0,   4.0],
            [8.0,   7.5,   7.5,   7.5,   3.0],
            [6.0,   6.0,   6.5,   6.5,   3.5],
            [4.0,   5.0,   5.5,   9.0,   5.0],
        ]),
        "C": np.array([
            [3.0,   3.0,   5.0,   5.0,   6.0],
            [4.0,   4.0,   5.5,   5.5,   5.0],
            [7.0,   6.5,   7.0,   7.0,   4.0],
            [6.0,   7.0,   6.0,   9.0,   4.5],
        ]),
        "D": np.array([
            [1.0,   1.0,   3.0,   3.0,   8.0],
            [2.0,   2.0,   3.5,   3.5,   7.0],
            [4.0,   4.0,   5.0,   5.0,   5.5],
            [9.0,   8.5,   7.0,   9.5,   3.0],
        ]),
    }

    return matrices.get(grade, matrices["D"])


CANONICAL_CRITERIA = ["SoH", "Value", "Carbon", "Compliance", "Safety", "EPR Return"]
CANONICAL_TYPES = ["benefit", "benefit", "cost", "benefit", "cost", "benefit"]
CANONICAL_ALTERNATIVES = [
    "Grid-scale ESS",
    "Home/Distributed ESS",
    "Component Reuse",
    "Direct Recycling",
]


def build_canonical_decision_matrix(grade: str) -> np.ndarray:
    """
    Decision matrix on the canonical 6 criteria (matches fuzzy_bwm_input.csv).

    Columns: SoH | Value | Carbon | Compliance | Safety | EPR Return
    Rows:    Grid ESS | Home ESS | Component Reuse | Direct Recycling

    Carbon and Safety are cost-type (lower is better). All others benefit.
    Scores are 1-10 literature-anchored estimates per grade tier.
    """
    matrices = {
        "A": np.array([
            # SoH  Value  Carbon  Compl  Safety  EPR
            [9.0,  8.5,   3.0,    7.0,   3.0,    7.5],   # Grid ESS
            [8.5,  7.5,   3.5,    7.0,   2.5,    7.0],   # Home ESS
            [6.0,  5.0,   4.5,    6.0,   4.0,    6.0],   # Component Reuse
            [3.0,  4.0,   6.0,    9.0,   5.0,    9.0],   # Direct Recycling
        ]),
        "B": np.array([
            [7.0,  6.5,   4.0,    7.0,   4.0,    7.0],
            [8.0,  7.5,   3.5,    7.5,   3.0,    7.0],
            [6.5,  6.0,   5.0,    6.5,   3.5,    6.0],
            [4.0,  5.0,   6.0,    9.0,   5.0,    9.0],
        ]),
        "C": np.array([
            [4.0,  3.5,   5.5,    5.0,   6.0,    5.5],
            [5.0,  4.5,   5.0,    5.5,   5.0,    5.5],
            [7.0,  6.5,   4.0,    7.0,   4.0,    6.5],
            [6.5,  7.0,   5.5,    9.0,   4.5,    9.0],
        ]),
        "D": np.array([
            [1.5,  1.5,   7.5,    3.0,   8.0,    3.0],
            [2.5,  2.5,   7.0,    3.5,   7.0,    3.5],
            [4.5,  4.0,   6.0,    5.0,   5.5,    5.0],
            [9.0,  8.5,   4.0,    9.5,   3.0,    9.5],
        ]),
    }
    return matrices.get(grade, matrices["D"])


def run_canonical_topsis(grade: str, weights_csv: Path | None = None) -> dict:
    """
    Stage-10/12 canonical-criteria TOPSIS using fuzzy_bwm_input.csv weights.

    Returns dict with `weights`, `criteria`, and `ranked` (list of
    {alternative, closeness, rank} dicts).
    """
    weights_csv = weights_csv or (
        PROJECT_ROOT / "data" / "processed" / "mcdm_weights" / "fuzzy_bwm_input.csv"
    )
    df = pd.read_csv(weights_csv)
    df = df.set_index("criterion")

    weights = []
    for c in CANONICAL_CRITERIA:
        row = df.loc[c]
        mid = row["tfn_middle_normalized"]
        if pd.isna(mid):
            weights.append(0.0)
        else:
            weights.append(float(mid))
    w = np.array(weights, dtype=float)
    if w.sum() <= 0:
        raise ValueError("All canonical weights are zero - check fuzzy_bwm_input.csv")
    w = w / w.sum()

    dm = build_canonical_decision_matrix(grade)
    result = topsis_rank(dm, w, CANONICAL_TYPES)

    ranked = []
    for i, alt in enumerate(CANONICAL_ALTERNATIVES):
        ranked.append({
            "alternative": alt,
            "closeness": float(result["closeness"][i]),
            "rank": int(result["ranking"][i]),
        })
    ranked.sort(key=lambda r: r["rank"])

    return {
        "grade": grade,
        "criteria": CANONICAL_CRITERIA,
        "weights": dict(zip(CANONICAL_CRITERIA, w.tolist())),
        "ranked": ranked,
        "decision_matrix": dm.tolist(),
    }


def run_topsis_all_grades(weights: np.ndarray) -> pd.DataFrame:
    """Run TOPSIS for all 4 grades and return combined results."""
    criteria_types = MCDM_CRITERIA_TYPES
    results = []

    for grade in ["A", "B", "C", "D"]:
        dm = build_decision_matrix_for_grade(grade)
        topsis_result = topsis_rank(dm, weights, criteria_types)

        for i, alt in enumerate(MCDM_ALTERNATIVES):
            results.append({
                "grade": grade,
                "alternative": alt,
                "closeness": topsis_result["closeness"][i],
                "rank": topsis_result["ranking"][i],
            })

    df = pd.DataFrame(results)

    print("\nTOPSIS Route Rankings per Grade:")
    print("=" * 60)
    for grade in ["A", "B", "C", "D"]:
        grade_df = df[df["grade"] == grade].sort_values("rank")
        print(f"\nGrade {grade}:")
        for _, row in grade_df.iterrows():
            print(f"  Rank {int(row['rank'])}: {row['alternative']:25s} (C={row['closeness']:.4f})")

    return df
