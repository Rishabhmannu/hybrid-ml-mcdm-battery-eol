"""PyBaMM validation gate per DATASET_IMPLEMENTATION_PLAN.md §4.2 / IMPLEMENTATION_PLAN.md §8.5.

Goal: confirm PyBaMM reproduces real cell degradation closely enough to trust
synthetic Indian-context generation downstream.

Acceptance gate:
  RMSE < 5 % SoH    AND    Pearson r > 0.90  (over the per-cycle SoH trajectory)

Reference cell (chemistry-matched per IMPLEMENTATION_PLAN §8.5):
  BL_SNL_SNL_18650_NMC_25C_0-100_0.5-2C_b
  · 18650 cylindrical · NMC · 3.0 Ah nominal · 4.2/2.0 V
  · Cycling: 0-100% SoC at 25°C, charge 0.5C / discharge 2C
  · 1,321 experimental cycles available, SoH 0.934 → 0.738
  · Source: Sandia National Labs via BatteryArchive, integrated into BatteryLife

Param set: Mohtat2020 (NMC 18650, full degradation params: SEI + Li-plating + LAM)

Runtime:
  --quick (default): 50 cycles, ~5-10 min on M4 Pro CPU. Trajectory shape check.
  --full:           500 cycles, ~30-90 min on M4 Pro. The actual acceptance gate.

Usage:
    conda activate Eco-Research
    python scripts/pybamm_validation.py --quick      # 50-cycle shape check
    python scripts/pybamm_validation.py --full       # 500-cycle acceptance gate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIFIED = PROJECT_ROOT / "data/processed/cycling/unified.parquet"
OUT_DIR = PROJECT_ROOT / "results/tables"
FIG_DIR = PROJECT_ROOT / "results/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# NMC 18650 from Sandia National Labs (via BatteryLife/BatteryArchive)
REFERENCE_CELL = "BL_SNL_SNL_18650_NMC_25C_0-100_0.5-2C_b"
REFERENCE_AMBIENT_K = 298.15      # 25 °C
REFERENCE_NOMINAL_AH = 3.0        # 3.0 Ah per .pkl metadata
REFERENCE_V_MAX = 4.2
REFERENCE_V_MIN = 2.0
REFERENCE_CHARGE_C = 0.5          # 0.5C charge
REFERENCE_DISCHARGE_C = 2.0       # 2C discharge

# PyBaMM parameter set — OKane2022 is the canonical full-degradation stack
# per PyBaMM's coupled-degradation tutorial. It is a superset of Chen2020,
# calibrated against an LG M50T cell (NMC811/graphite-SiOx — same chemistry
# family as our SNL_18650 reference), with all rate constants needed for
# SEI growth + partially-reversible Li-plating + LAM stress-driven model.
# Was Mohtat2020; switched 2026-04-28 — see WIP_PYBAMM_OKANE2022_RECALIBRATION.md
PARAM_SET = "OKane2022"

# Acceptance thresholds (project gate per IMPLEMENTATION_PLAN §8.5)
RMSE_GATE = 5.0     # % SoH
PEARSON_R_GATE = 0.90


def load_real_soh(cell_id: str = REFERENCE_CELL) -> pd.DataFrame:
    """Load reference cell from unified.parquet and normalize SoH to start at 1.0.

    SoH is renormalized so cycle-1 SoH = 1.0, matching the PyBaMM sim's
    relative-to-first-cycle convention. Without this, a constant offset
    (e.g. real cell starts at 0.93 if its capacity_at_cycle_1 < nominal)
    would dominate RMSE comparisons against a fresh-cell sim.
    """
    df = pd.read_parquet(UNIFIED, columns=["battery_id", "cycle", "soh", "capacity_Ah"])
    cell = df[df.battery_id == cell_id].sort_values("cycle").reset_index(drop=True)
    if cell.empty:
        raise SystemExit(f"Reference cell {cell_id} not found in unified parquet")
    # Renormalize SoH relative to first cycle's capacity
    first_cap = float(cell["capacity_Ah"].dropna().iloc[0])
    cell["soh_norm"] = cell["capacity_Ah"] / first_cap
    cell["soh_original"] = cell["soh"]
    cell["soh"] = cell["soh_norm"]
    return cell


def run_pybamm_sim(n_cycles: int, ambient_K: float = REFERENCE_AMBIENT_K,
                   nominal_Ah: float = REFERENCE_NOMINAL_AH,
                   v_max: float = REFERENCE_V_MAX, v_min: float = REFERENCE_V_MIN,
                   charge_C: float = REFERENCE_CHARGE_C,
                   discharge_C: float = REFERENCE_DISCHARGE_C,
                   with_degradation: bool = True) -> pd.DataFrame:
    """Run PyBaMM DFN with the canonical OKane2022 full-degradation stack.

    Per PyBaMM's coupled-degradation tutorial (cited in WIP_PYBAMM_OKANE2022_
    RECALIBRATION.md), the canonical stack is:
      - SEI growth (solvent-diffusion limited)
      - Lithium plating (partially reversible — recommended; "irreversible" is
        documented to produce unrealistic decay rates)
      - Particle mechanics (swelling and cracking on negative; swelling only on
        positive, since OKane2022 lacks positive-electrode crack params)
      - SEI on cracks (true) — accelerates aging on cracked surfaces
      - Stress-driven loss of active material
      - Lumped thermal coupling
    OKane2022 is calibrated against the LG M50T cell (NMC811/graphite-SiOx).
    """
    import pybamm
    parameter_values = pybamm.ParameterValues(PARAM_SET)
    parameter_values["Ambient temperature [K]"] = ambient_K
    parameter_values["Nominal cell capacity [A.h]"] = nominal_Ah

    if with_degradation:
        model = pybamm.lithium_ion.DFN(options={
            "SEI": "solvent-diffusion limited",
            "lithium plating": "partially reversible",
            "particle mechanics": ("swelling and cracking", "swelling only"),
            "SEI on cracks": "true",
            "loss of active material": "stress-driven",
            "thermal": "lumped",
        })
    else:
        model = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
    # SNL_18650_NMC protocol: charge 0.5C CC → 4.2V CV → discharge 2C → 2.0V
    # (see filename "0-100_0.5-2C" → full DoD, charge 0.5C / discharge 2C)
    experiment = pybamm.Experiment(
        [(
            f"Discharge at {discharge_C}C until {v_min} V",
            "Rest for 30 minutes",
            f"Charge at {charge_C}C until {v_max} V",
            f"Hold at {v_max} V until C/50",
            "Rest for 30 minutes",
        )] * n_cycles,
        termination="80% capacity",
    )
    sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
    print(f"  Solving {n_cycles} cycles...")
    solution = sim.solve()

    rows = []
    for cycle_idx, sub in enumerate(solution.cycles, start=1):
        if sub is None:
            continue
        try:
            qd = sub["Discharge capacity [A.h]"].entries
            cap = float(np.max(np.abs(qd)))
            rows.append({"cycle": cycle_idx, "capacity_Ah_sim": cap})
        except Exception:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        # Standard SoH definition: capacity at cycle t relative to start-of-life capacity.
        # We use cycle-1 simulated capacity as the reference rather than the
        # user-provided nominal_Ah, because PyBaMM parameter sets define their
        # own electrode geometry whose actual capacity may differ from the
        # `Nominal cell capacity` parameter (which only flags rated capacity).
        sol_nominal = float(df["capacity_Ah_sim"].iloc[0])
        df["soh_sim"] = df["capacity_Ah_sim"] / sol_nominal
    return df


def compare(real: pd.DataFrame, sim: pd.DataFrame) -> dict:
    """Compute RMSE and Pearson r between real and sim SoH on overlapping cycles."""
    merged = pd.merge(real[["cycle", "soh"]], sim[["cycle", "soh_sim"]], on="cycle", how="inner")
    if len(merged) < 3:
        return {"n_overlap": len(merged), "rmse_pct": float("nan"),
                "pearson_r": float("nan"), "passed": False, "note": "too few overlapping cycles"}
    err = (merged["soh"] - merged["soh_sim"]) * 100  # in % SoH points
    rmse_pct = float(np.sqrt((err ** 2).mean()))
    r = float(merged["soh"].corr(merged["soh_sim"])) if merged["soh"].std() > 0 and merged["soh_sim"].std() > 0 else float("nan")
    passed = (rmse_pct < RMSE_GATE) and (np.isfinite(r) and r > PEARSON_R_GATE)
    return {
        "n_overlap": int(len(merged)),
        "rmse_pct": round(rmse_pct, 3),
        "pearson_r": round(r, 4) if np.isfinite(r) else None,
        "rmse_gate": RMSE_GATE,
        "pearson_r_gate": PEARSON_R_GATE,
        "passed": bool(passed),
    }


def plot(real: pd.DataFrame, sim: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100)
    ax.plot(real["cycle"], real["soh"], label="CALCE CS2_35 (real)", color="C0", linewidth=2)
    ax.plot(sim["cycle"], sim["soh_sim"], label="PyBaMM Marquis2019 (sim)", color="C1", linestyle="--")
    ax.axhline(0.80, color="grey", linewidth=0.8, linestyle=":", label="80% SoH (EoL threshold)")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SoH")
    ax.set_title("PyBaMM-CALCE validation — CS2_35")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.relative_to(PROJECT_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="Run full validation (~30-90 min)")
    ap.add_argument("--quick", action="store_true", help="Smoke test only (~2 min, default)")
    ap.add_argument("--cycles", type=int, default=None, help="Override cycle count")
    args = ap.parse_args()

    n_cycles = args.cycles or (500 if args.full else 10)
    print(f"=== PyBaMM-CALCE Validation ===")
    print(f"Reference cell: {REFERENCE_CELL}")
    print(f"Cycles to simulate: {n_cycles}")
    print(f"Acceptance gate: RMSE < {RMSE_GATE}% AND Pearson r > {PEARSON_R_GATE}")
    print()

    print("[1/3] Loading real SoH trajectory...")
    real = load_real_soh()
    print(f"  {len(real)} real cycles for {REFERENCE_CELL}")
    print(f"  SoH range: {real.soh.min():.3f} → {real.soh.max():.3f}")

    print(f"\n[2/3] Running PyBaMM simulation ({n_cycles} cycles)...")
    sim = run_pybamm_sim(n_cycles=n_cycles)
    print(f"  Simulation produced {len(sim)} cycles before termination")
    if not sim.empty:
        print(f"  Sim SoH range: {sim.soh_sim.min():.3f} → {sim.soh_sim.max():.3f}")

    print(f"\n[3/3] Comparing trajectories...")
    res = compare(real, sim)
    print(f"  Overlap cycles: {res['n_overlap']}")
    print(f"  RMSE: {res['rmse_pct']}% (gate: < {RMSE_GATE}%)")
    print(f"  Pearson r: {res['pearson_r']} (gate: > {PEARSON_R_GATE})")
    print(f"  PASSED: {res['passed']}")

    # Write artifacts
    label = "full" if args.full else "quick"
    sim.to_csv(OUT_DIR / f"pybamm_calce_cs2_35_{label}_sim.csv", index=False)
    (OUT_DIR / f"pybamm_validation_{label}.json").write_text(json.dumps(res, indent=2))
    plot(real, sim, FIG_DIR / f"pybamm_validation_calce_cs2_35_{label}.png")

    print(f"\nArtifacts written to {OUT_DIR.relative_to(PROJECT_ROOT)}/ and {FIG_DIR.relative_to(PROJECT_ROOT)}/")
    return 0 if res["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
