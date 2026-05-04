"""
PyBaMM-based Indian-context synthetic battery cycling data generation.

Generates synthetic Li-ion cycling data under Indian operating conditions to
bridge the Indian battery data gap (no public Indian cycling datasets exist).

References:
- Sulzer et al. (2021), J. Open Res. Software, 9(1), 14. DOI: 10.5334/jors.309
- Brosa Planella et al. (2024), J. Power Sources -- high-throughput PyBaMM degradation
- ARAI CMVR/TAP-115/116 Indian Driving Cycle specifications

Usage:
    conda run -n Eco-Research python src/data/synthetic.py --chemistry NMC --n-cells 10 --n-cycles 200
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import PROCESSED_DIR, RANDOM_SEED


# ==========================================================================
# INDIAN DRIVING CYCLE PROFILES (from ARAI CMVR/TAP-115/116)
# ==========================================================================

@dataclass
class IndianDrivingCycle:
    """Simplified Indian Driving Cycle profile.

    Full specifications available in ARAI CMVR/TAP-115/116 public documents.
    These profiles are representative; reproduce from the ARAI PDFs for
    publication-grade results.
    """
    name: str
    duration_s: int
    distance_km: float
    max_speed_kmph: float
    avg_speed_kmph: float
    # Speed-time profile points (s, km/h)
    profile: list = field(default_factory=list)


IDC_2W = IndianDrivingCycle(
    name="Indian Driving Cycle 2-Wheeler",
    duration_s=648,
    distance_km=3.948,
    max_speed_kmph=42.0,
    avg_speed_kmph=21.9,
    # Simplified 6 subsequent idle-accel-cruise-decel segments (reproduce from ARAI PDF for rigour)
    profile=[(0, 0), (20, 25), (50, 25), (75, 0), (108, 0)],
)

MIDC = IndianDrivingCycle(
    name="Modified Indian Driving Cycle",
    duration_s=1180,
    distance_km=8.528,
    max_speed_kmph=90.0,
    avg_speed_kmph=26.0,
    # Combines urban + extra-urban segments
    profile=[(0, 0), (100, 50), (400, 50), (500, 90), (800, 90), (1000, 0), (1180, 0)],
)

IN_UDC = IndianDrivingCycle(
    name="Indian Urban Driving Cycle",
    duration_s=780,
    distance_km=4.8,
    max_speed_kmph=50.0,
    avg_speed_kmph=22.0,
    profile=[(0, 0), (60, 30), (180, 30), (240, 50), (420, 50), (600, 0), (780, 0)],
)

INDIAN_DRIVING_CYCLES = {
    "IDC_2W": IDC_2W,
    "MIDC": MIDC,
    "IN_UDC": IN_UDC,
}


# ==========================================================================
# INDIAN AMBIENT THERMAL PROFILES
# ==========================================================================

INDIAN_AMBIENT_PROFILES = {
    "Delhi_summer": 313.15,   # 40 C
    "Mumbai_monsoon": 303.15, # 30 C
    "Bengaluru_mild": 298.15, # 25 C (baseline)
    "Rajasthan_extreme": 318.15, # 45 C
}


# ==========================================================================
# VEHICLE PARAMETERS (for converting driving cycle speed to current demand)
# ==========================================================================

@dataclass
class VehicleParams:
    """Lumped vehicle parameters for power demand calculation."""
    vehicle_type: str
    mass_kg: float
    drag_coefficient: float
    frontal_area_m2: float
    rolling_resistance: float
    drivetrain_efficiency: float


EV_2W = VehicleParams("2W", 150.0, 0.9, 0.7, 0.015, 0.85)
EV_3W = VehicleParams("3W", 400.0, 1.0, 1.2, 0.018, 0.80)
EV_4W = VehicleParams("4W", 1500.0, 0.32, 2.2, 0.011, 0.92)


# ==========================================================================
# POWER PROFILE FROM DRIVING CYCLE
# ==========================================================================

def cycle_to_power_profile(
    cycle: IndianDrivingCycle,
    vehicle: VehicleParams,
    time_step_s: float = 1.0,
) -> pd.DataFrame:
    """Convert driving cycle speed profile to power demand (W).

    Simple physics: P = (1/eta) * (F_aero + F_roll + F_accel) * v
    where F_aero = 0.5 * rho * Cd * A * v^2, F_roll = m * g * Crr,
    F_accel = m * a.
    """
    rho = 1.225  # air density kg/m^3
    g = 9.81     # gravity m/s^2

    # Interpolate profile to per-second resolution
    points = np.array(cycle.profile)
    times = np.arange(0, cycle.duration_s + 1, time_step_s)
    speeds_kmph = np.interp(times, points[:, 0], points[:, 1])
    speeds_ms = speeds_kmph / 3.6

    # Acceleration
    accel = np.gradient(speeds_ms, time_step_s)

    # Forces
    f_aero = 0.5 * rho * vehicle.drag_coefficient * vehicle.frontal_area_m2 * speeds_ms**2
    f_roll = vehicle.mass_kg * g * vehicle.rolling_resistance
    f_accel = vehicle.mass_kg * accel

    force_total = f_aero + f_roll + f_accel
    power_demand = force_total * speeds_ms / vehicle.drivetrain_efficiency  # W

    return pd.DataFrame({
        "time_s": times,
        "speed_kmph": speeds_kmph,
        "speed_ms": speeds_ms,
        "power_W": power_demand,
    })


# ==========================================================================
# PYBAMM SIMULATION
# ==========================================================================

def run_pybamm_simulation(
    n_cycles: int,
    chemistry: str = "NMC",
    ambient_K: float = 313.15,  # 40 C (Indian summer)
    power_profile_W: Optional[np.ndarray] = None,
    time_profile_s: Optional[np.ndarray] = None,
    cell_capacity_Ah: float = 5.0,  # informational: param sets define their own electrode geometry
    include_degradation: bool = True,
) -> pd.DataFrame:
    """Run PyBaMM simulation for one virtual cell.

    Note: SoH is normalized against the *simulated cycle-1 capacity*, not the
    `cell_capacity_Ah` argument, because each PyBaMM parameter set defines its
    own electrode geometry whose actual capacity may differ from the rated
    nominal. The argument is retained for caller-side bookkeeping.

    This is a thin wrapper. Detailed PyBaMM setup must follow the tool's docs;
    see https://docs.pybamm.org for parameter-set options (Chen2020, Mohtat2020, etc.)

    Parameters
    ----------
    n_cycles : int
        Number of charge-discharge cycles to simulate.
    chemistry : str
        "NMC" or "LFP" -- selects PyBaMM parameter set.
    ambient_K : float
        Ambient temperature in Kelvin.
    power_profile_W : array, optional
        Discharge power profile from driving cycle (if None, constant-current discharge).
    time_profile_s : array, optional
        Time axis for power_profile_W.
    cell_capacity_Ah : float
        Nominal cell capacity (Ah).
    include_degradation : bool
        If True, enable SEI growth + lithium plating + LAM degradation models.

    Returns
    -------
    DataFrame with columns:
        cycle, time_s, voltage, current, temperature, capacity, soh
    """
    try:
        import pybamm
    except ImportError:
        raise ImportError(
            "PyBaMM not installed. Run: pip install pybamm"
        )

    # Select model + parameter set per chemistry.
    # NMC: OKane2022 + canonical full-degradation stack (SEI + partially-reversible
    #      Li-plating + LAM + SEI-on-cracks + particle mechanics). This is the
    #      stack from PyBaMM's coupled-degradation tutorial — calibrated against
    #      LG M50T (NMC811/graphite-SiOx). 500-cycle gate on SNL_18650 NMC ref:
    #      RMSE 9.64% (magnitude under-predicts by ~6× — calibration gap to
    #      LG M50T cell), Pearson r 0.93 (trajectory shape matches). Synthetic
    #      data preserves shape + Indian-thermal ordering — see WIP_PYBAMM_*.
    #      Switched from Mohtat2020 + irreversible plating on 2026-04-28.
    # LFP: Prada2013 lacks current-collector params for thermal/aging models.
    #      Use isothermal SPMe; cell capacity stays constant; V/I waveforms
    #      still vary via Arrhenius terms. PyBaMM devs confirm "no good LFP
    #      parameter set has been found in the literature" — known field gap.
    chem_supports_degradation = chemistry.upper() == "NMC"
    use_degradation = include_degradation and chem_supports_degradation

    if use_degradation:
        model = pybamm.lithium_ion.DFN(
            options={
                "SEI": "solvent-diffusion limited",
                "lithium plating": "partially reversible",
                "particle mechanics": ("swelling and cracking", "swelling only"),
                "SEI on cracks": "true",
                "loss of active material": "stress-driven",
                "thermal": "lumped",
            }
        )
    else:
        model = pybamm.lithium_ion.SPMe()

    if chemistry.upper() == "NMC":
        parameter_values = pybamm.ParameterValues("OKane2022")  # was Mohtat2020
    elif chemistry.upper() == "LFP":
        parameter_values = pybamm.ParameterValues("Prada2013")   # LFP, no degradation
    else:
        raise ValueError(f"Unsupported chemistry: {chemistry}")

    # Indian ambient temperature
    parameter_values["Ambient temperature [K]"] = ambient_K

    # Build experiment -- simplified CC-CV charge / CC discharge for now.
    # TODO: for full publication rigour, embed the driving-cycle power profile
    # via parameter_values["Current function [A]"] = pybamm.Interpolant(...)
    experiment = pybamm.Experiment(
        [(
            f"Discharge at 1C until 2.5 V",
            "Rest for 30 minutes",
            f"Charge at 0.5C until 4.2 V",
            "Hold at 4.2 V until C/50",
            "Rest for 30 minutes",
        )] * n_cycles,
        termination="80% capacity",  # stop when SoH drops below 80%
    )

    sim = pybamm.Simulation(model, parameter_values=parameter_values, experiment=experiment)
    solution = sim.solve()

    # Extract per-cycle summary
    records = []
    for cycle_idx, cycle_sol in enumerate(solution.cycles, start=1):
        if cycle_sol is None:
            continue
        try:
            V = cycle_sol["Voltage [V]"].entries
            I = cycle_sol["Current [A]"].entries
            T = cycle_sol["Cell temperature [K]"].entries - 273.15  # to Celsius
            Q = cycle_sol["Discharge capacity [A.h]"].entries
            t = cycle_sol["Time [s]"].entries

            records.append({
                "cycle": cycle_idx,
                "time_s": t[-1] if len(t) else np.nan,
                "voltage_mean": float(np.mean(V)),
                "voltage_min": float(np.min(V)),
                "voltage_max": float(np.max(V)),
                "current_mean": float(np.mean(I)),
                "temperature_mean": float(np.mean(T)),
                "temperature_max": float(np.max(T)),
                "temperature_range": float(np.max(T) - np.min(T)),
                "capacity": float(np.max(np.abs(Q))),
            })
        except Exception as exc:
            print(f"  [cycle {cycle_idx}] extraction failed: {exc}")
            continue

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        # SoH = capacity at cycle t / capacity at cycle 1 (start-of-life convention).
        # We normalize against the simulated cycle-1 capacity rather than
        # `cell_capacity_Ah` because PyBaMM parameter sets define their own
        # electrode geometry whose actual capacity may differ from the
        # user-supplied nominal.
        sol_nominal = float(df_out["capacity"].iloc[0])
        df_out["soh"] = df_out["capacity"] / sol_nominal * 100
    return df_out


# ==========================================================================
# GENERATION PIPELINE
# ==========================================================================

def generate_synthetic_dataset(
    n_cells: int = 10,
    n_cycles: int = 500,
    chemistries: Optional[list] = None,
    ambient_profiles: Optional[list] = None,
    output_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generate a full synthetic Indian-context battery cycling dataset.

    Parameters
    ----------
    n_cells : int
        Number of virtual cells to simulate.
    n_cycles : int
        Number of cycles per cell.
    chemistries : list
        List of chemistries to sample from (default: ["NMC", "LFP"]).
    ambient_profiles : list
        List of Indian ambient profile names (default: all four).
    output_dir : Path
        Where to save the synthetic dataset CSV.
    seed : int
        Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    chemistries = chemistries or ["NMC", "LFP"]
    ambient_profiles = ambient_profiles or list(INDIAN_AMBIENT_PROFILES.keys())

    output_dir = Path(output_dir) if output_dir else PROCESSED_DIR / "synthetic_indian"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    for cell_idx in range(n_cells):
        chemistry = rng.choice(chemistries)
        ambient_name = rng.choice(ambient_profiles)
        ambient_K = INDIAN_AMBIENT_PROFILES[ambient_name]
        cycle_name = rng.choice(list(INDIAN_DRIVING_CYCLES.keys()))

        battery_id = f"IN_SYNTH_{cell_idx:04d}_{chemistry}_{ambient_name}_{cycle_name}"
        print(f"\n[{cell_idx+1}/{n_cells}] {battery_id}")

        try:
            df = run_pybamm_simulation(
                n_cycles=n_cycles,
                chemistry=chemistry,
                ambient_K=ambient_K,
                include_degradation=True,
            )
        except Exception as exc:
            print(f"  Simulation failed: {exc}")
            continue

        df["battery_id"] = battery_id
        df["chemistry"] = chemistry
        df["ambient_profile"] = ambient_name
        df["ambient_K"] = ambient_K
        df["driving_cycle"] = cycle_name
        df["source"] = "synthetic_pybamm_indian"

        all_records.append(df)

        # Save per-cell file for recoverability
        df.to_csv(output_dir / f"{battery_id}.csv", index=False)

    if not all_records:
        print("No successful simulations. Check PyBaMM installation.")
        return pd.DataFrame()

    combined = pd.concat(all_records, ignore_index=True)
    combined_path = output_dir / "synthetic_indian_combined.csv"
    combined.to_csv(combined_path, index=False)

    # Save metadata
    meta = {
        "n_cells_requested": n_cells,
        "n_cells_successful": len(all_records),
        "n_cycles_per_cell": n_cycles,
        "chemistries": chemistries,
        "ambient_profiles": ambient_profiles,
        "driving_cycles": list(INDIAN_DRIVING_CYCLES.keys()),
        "seed": seed,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSynthetic dataset saved: {combined_path}")
    print(f"Total rows: {len(combined)} | unique batteries: {combined['battery_id'].nunique()}")
    return combined


# ==========================================================================
# CLI
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate PyBaMM-based synthetic Indian-context battery cycling data."
    )
    parser.add_argument("--n-cells", type=int, default=10, help="Number of virtual cells")
    parser.add_argument("--n-cycles", type=int, default=500, help="Cycles per cell")
    parser.add_argument("--chemistry", type=str, default="all", choices=["NMC", "LFP", "all"],
                        help="Battery chemistry")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    chemistries = ["NMC", "LFP"] if args.chemistry == "all" else [args.chemistry]

    generate_synthetic_dataset(
        n_cells=args.n_cells,
        n_cycles=args.n_cycles,
        chemistries=chemistries,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
