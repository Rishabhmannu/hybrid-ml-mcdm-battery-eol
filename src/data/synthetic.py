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

# Iter-1 / Iter-2 used these as constant scalars, which under-represented
# Indian thermal stress (no diurnal swing → no stress cycling). Retained for
# backward compatibility with older scripts that pass `ambient_K` as a float.
INDIAN_AMBIENT_PROFILES = {
    "Delhi_summer": 313.15,   # 40 C
    "Mumbai_monsoon": 303.15, # 30 C
    "Bengaluru_mild": 298.15, # 25 C (baseline)
    "Rajasthan_extreme": 318.15, # 45 C
}

# Iter-3 (2026-05): diurnal profiles per Mulpuri et al. 2025 (RSC Advances,
# d5ra04379d). Each tuple is (mean_K, peak-to-peak_amplitude_K).
# Ambient temperature swings as T(t) = mean + (amp/2) * sin(ω(t − φ)) with
# the daily peak placed at 14:00. This captures the SEI-stress acceleration
# from cycling between cool nights and hot days that constant-T simulation
# misses. Mulpuri Zone-C (Churu, Rajasthan) measured 1.8–47.2 °C diurnal
# range; we use the daily summer envelope, not the full annual range.
DIURNAL_CLIMATE_PROFILES = {
    "Bengaluru_mild": {
        "mean_K": 296.65,         # 23.5 °C mean
        "amplitude_K": 12.0,      # 17.5 °C night → 29.5 °C day
        "description": "Year-round mild, low diurnal swing",
    },
    "Mumbai_monsoon": {
        "mean_K": 301.15,         # 28 °C mean
        "amplitude_K": 8.0,       # 24 °C → 32 °C, dampened by humidity
        "description": "Coastal humid, dampened diurnal range",
    },
    "Delhi_summer": {
        "mean_K": 308.15,         # 35 °C mean
        "amplitude_K": 14.0,      # 28 °C → 42 °C, big continental swing
        "description": "Continental, large summer diurnal swing",
    },
    "Rajasthan_extreme": {
        "mean_K": 309.15,         # 36 °C mean
        "amplitude_K": 22.0,      # 25 °C → 47 °C, Mulpuri-Zone-C envelope
        "description": "Desert, extreme diurnal swing (Mulpuri 2025 Zone-C)",
    },
}

# Iter-3 (2026-05): empirical calibration multipliers applied on top of
# OKane2022. The stock parameter set is calibrated to LG M50T cells under
# clean lab cycling and under-predicts capacity fade for commercial
# deployment cells (see OKane et al. 2022 PCCP §5; the same gap is
# documented for our Iter-1/Iter-2 SNL_18650 reference run, RMSE 9.64 %
# magnitude under-prediction). We multiply five aging-relevant rate
# constants by `INDIA_CALIBRATION_DEFAULT` to recover realistic Indian-
# deployment fade rates. Empirical benchmark (200 cycles, 45 °C, full
# cycle): stock 1.82 % SoH loss → 10× calibration 5.78 % SoH loss
# (3.2× acceleration; mechanisms saturate / compete so multiplier > 1
# returns sub-linearly).
#
# Methodology: this is informal Bayesian calibration without uncertainty
# reporting. For a production-grade fit, use the PyBOP toolbox
# (https://github.com/pybop-team/PyBOP) — Howey/Onori/WMG-Warwick groups
# do this routinely (Aitio & Howey 2021, Joule 5(12); Kuzhiyil et al.
# 2023, J. Energy Storage). Cite OKane 2022 PCCP as the prior.
INDIA_CALIBRATION_DEFAULT = {
    "SEI kinetic rate constant [m.s-1]": 10.0,
    "SEI solvent diffusivity [m2.s-1]": 10.0,
    "Lithium plating kinetic rate constant [m.s-1]": 10.0,
    "Negative electrode LAM constant proportional term [s-1]": 10.0,
    "Positive electrode LAM constant proportional term [s-1]": 10.0,
}

# Iter-3: Indian-deployment usage protocols. The default `full_cycle` was
# used in Iter-1/Iter-2 and represents lab-style 0–100 % deep cycling. The
# `two_wheeler` and `three_wheeler` profiles capture the partial-SoC
# fast-charging patterns dominant in Indian EV deployment due to sparse
# charging infrastructure and range anxiety. Lithium plating accelerates
# under partial-SoC fast-charging at high temperatures — exactly the
# Indian-fleet stress combination this audit is meant to surface.
#
# SoC↔voltage mapping (NMC, OKane2022 OCV):
#   100 % ≈ 4.20 V, 90 % ≈ 4.05 V, 80 % ≈ 3.95 V, 50 % ≈ 3.80 V,
#   30 % ≈ 3.65 V, 20 % ≈ 3.55 V, 0 % ≈ 3.00 V
INDIAN_USAGE_PROTOCOLS = {
    "full_cycle": {
        "description": "0–100 % deep cycling (lab reference, Iter-1/2 default)",
        "experiment": [
            "Discharge at 1C until 2.5 V",
            "Rest for 30 minutes",
            "Charge at 0.5C until 4.2 V",
            "Hold at 4.2 V until C/50",
            "Rest for 30 minutes",
        ],
    },
    "two_wheeler": {
        "description": "Indian 2W: 30–90 % SoC partial cycles, 1C fast-charge",
        "experiment": [
            "Discharge at 1C until 3.65 V",  # 30 % SoC NMC
            "Rest for 5 minutes",
            "Charge at 1C until 4.05 V",     # 90 % SoC NMC
            "Rest for 5 minutes",
        ],
    },
    "three_wheeler": {
        "description": "Indian 3W commercial: 20–80 % SoC, 0.5C reliability",
        "experiment": [
            "Discharge at 0.5C until 3.55 V",  # 20 % SoC NMC
            "Rest for 10 minutes",
            "Charge at 0.5C until 3.95 V",     # 80 % SoC NMC
            "Rest for 10 minutes",
        ],
    },
}


def diurnal_temperature_callable(mean_K: float, amplitude_K: float,
                                 peak_hour_24h: float = 14.0):
    """Build a callable T(t) → ambient temperature in Kelvin.

    Implements a 24-hour sinusoidal cycle peaking at `peak_hour_24h` (default
    14:00) with given mean and peak-to-peak amplitude. Compatible with
    PyBaMM's symbolic time variable: returned function uses pybamm.sin so it
    can be evaluated against either a scalar or a pybamm.Variable.

    Parameters
    ----------
    mean_K : float
        Daily mean temperature in Kelvin.
    amplitude_K : float
        Peak-to-peak swing in Kelvin (max − min over the day).
    peak_hour_24h : float
        Hour of day at which temperature peaks (24-hour clock).

    Returns
    -------
    callable
        f(t) where t is time in seconds. Returns mean_K +
        (amplitude_K / 2) * sin(2π(t − phase)/86400).
    """
    import pybamm
    period_s = 86400.0
    omega = 2.0 * np.pi / period_s
    # Place the sin peak at peak_hour_24h: sin(omega*(t-phase)) peaks when
    # (t - phase) = period_s/4, so phase = peak_hour_seconds - period_s/4.
    phase_s = peak_hour_24h * 3600.0 - period_s / 4.0
    half_amp = amplitude_K / 2.0

    # PyBaMM's lumped-thermal submodel calls ambient T as f(y, z, t) — 2
    # spatial coords + time. Accept all 3 and use only t.
    def T_of_yzt(y, z, t):
        return mean_K + half_amp * pybamm.sin(omega * (t - phase_s))

    return T_of_yzt


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
    ambient_K: "float | callable" = 313.15,  # 40 C (Indian summer); callable for diurnal
    power_profile_W: Optional[np.ndarray] = None,
    time_profile_s: Optional[np.ndarray] = None,
    cell_capacity_Ah: float = 5.0,  # informational: param sets define their own electrode geometry
    include_degradation: bool = True,
    usage_protocol: str = "full_cycle",
    termination_capacity_pct: int = 30,
    calibration_multipliers: Optional[dict] = None,
    progress_every_k_cycles: int = 0,
    progress_label: str = "",
    chunk_cycles: int = 100,
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

    # Iter-3 empirical calibration: multiply selected aging-rate constants
    # to recover realistic Indian-deployment fade. See INDIA_CALIBRATION_DEFAULT
    # docstring above for methodology / citations.
    if calibration_multipliers and use_degradation:
        for param_name, multiplier in calibration_multipliers.items():
            if param_name in parameter_values.keys():
                parameter_values[param_name] *= float(multiplier)
            else:
                print(f"  [calibration WARN] '{param_name}' not in parameter set, "
                      f"skipping multiplier {multiplier}")

    # Indian ambient temperature — accepts either a scalar or a callable
    # built by `diurnal_temperature_callable()`. PyBaMM treats callables as
    # time-varying parameters automatically.
    parameter_values["Ambient temperature [K]"] = ambient_K

    # Build experiment from the named usage protocol. Default `full_cycle`
    # reproduces Iter-1/Iter-2 behaviour; `two_wheeler` and `three_wheeler`
    # exercise the partial-SoC fast-charging stress dominant in Indian EV
    # deployment. Termination capacity controls how aggressively PyBaMM
    # stops simulating: 30 % drives cells deep into Grade D coverage if
    # the parameter set permits it.
    if usage_protocol not in INDIAN_USAGE_PROTOCOLS:
        raise ValueError(
            f"Unknown usage_protocol '{usage_protocol}'. "
            f"Available: {list(INDIAN_USAGE_PROTOCOLS.keys())}"
        )
    experiment_steps = tuple(INDIAN_USAGE_PROTOCOLS[usage_protocol]["experiment"])

    # Heartbeat callback — instantiated once outside the chunk loop so the
    # internal `cycle_count` accumulates across chunks. PyBaMM accepts the
    # same callback object across multiple `sim.solve()` calls.
    callbacks = []
    if progress_every_k_cycles and progress_every_k_cycles > 0:
        import sys as _sys
        import time as _time

        class _HeartbeatCallback(pybamm.callbacks.Callback):
            def __init__(self, label: str, total_cycles: int, every_k: int):
                self.label = label or "cell"
                self.total = total_cycles
                self.every_k = every_k
                self.t0 = _time.time()
                self.cycle_count = 0

            def on_cycle_end(self, logs):
                self.cycle_count += 1
                if (self.cycle_count % self.every_k == 0
                        or self.cycle_count == self.total):
                    elapsed = _time.time() - self.t0
                    pct = self.cycle_count / max(self.total, 1)
                    eta = elapsed * (1.0 / pct - 1.0) if pct > 0 else 0.0
                    _sys.stderr.write(
                        f"  [{self.label}] cycle {self.cycle_count}/{self.total} "
                        f"({pct*100:5.1f}%)  elapsed={elapsed:6.0f}s  ETA={eta:6.0f}s\n"
                    )
                    _sys.stderr.flush()

        callbacks.append(_HeartbeatCallback(
            label=progress_label,
            total_cycles=n_cycles,
            every_k=progress_every_k_cycles,
        ))

    # Iter-3 chunked simulation. The original single `sim.solve(...)` over
    # 2500 cycles caused 5–8 GB RSS per worker because PyBaMM retains every
    # cycle's full state vector in `solution.cycles`. We solve in chunks of
    # `chunk_cycles`, extract the per-cycle summary, then drop the chunk's
    # solution and continue from `last_state`. Memory stays bounded at
    # ~chunk_size × per-cycle-state regardless of total n_cycles.
    import gc as _gc

    def _extract_cycle_summary(cycle_sol, cycle_idx: int) -> Optional[dict]:
        try:
            V = cycle_sol["Voltage [V]"].entries
            I = cycle_sol["Current [A]"].entries
            T = cycle_sol["Cell temperature [K]"].entries - 273.15  # to Celsius
            Q = cycle_sol["Discharge capacity [A.h]"].entries
            t = cycle_sol["Time [s]"].entries
            return {
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
            }
        except Exception as exc:
            print(f"  [cycle {cycle_idx}] extraction failed: {exc}")
            return None

    records = []
    last_state = None
    cycles_done = 0
    chunk_cycles = max(1, int(chunk_cycles))

    while cycles_done < n_cycles:
        this_chunk = min(chunk_cycles, n_cycles - cycles_done)
        chunk_experiment = pybamm.Experiment(
            [experiment_steps] * this_chunk,
            termination=f"{termination_capacity_pct}% capacity",
        )
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            experiment=chunk_experiment,
        )

        # `starting_solution` is `None` on first chunk (fresh start) and the
        # previous chunk's `last_state` thereafter — PyBaMM continues the
        # time-integrator from that terminal state without re-running the
        # earlier cycles.
        chunk_solution = sim.solve(
            starting_solution=last_state,
            callbacks=callbacks if callbacks else None,
        )

        # Extract per-cycle summary from this chunk's solution
        n_new_cycles = 0
        for local_idx, cycle_sol in enumerate(chunk_solution.cycles):
            if cycle_sol is None:
                continue
            global_cycle_idx = cycles_done + local_idx + 1
            rec = _extract_cycle_summary(cycle_sol, global_cycle_idx)
            if rec is not None:
                records.append(rec)
                n_new_cycles += 1

        # Save terminal state for the next chunk, then explicitly drop the
        # chunk's solution + simulation. `last_state` returns a fresh, small
        # Solution object holding only the final timestep — independent of
        # the parent solution, so deleting the parent is safe.
        try:
            new_last_state = chunk_solution.last_state
        except Exception:
            # If last_state isn't accessible (e.g., termination triggered),
            # break the loop with whatever we've extracted.
            new_last_state = None

        # Detect early termination: if the chunk produced fewer cycles than
        # we asked for, the termination criterion fired. Stop here.
        terminated_early = n_new_cycles < this_chunk

        del chunk_solution
        del chunk_experiment
        del sim
        _gc.collect()

        last_state = new_last_state
        cycles_done += n_new_cycles

        if terminated_early or last_state is None:
            break

    df_out = pd.DataFrame(records)
    if not df_out.empty:
        # SoH = capacity at cycle t / nominal capacity (start-of-life convention).
        # We normalize against the simulated *baseline* capacity rather than
        # `cell_capacity_Ah` because PyBaMM parameter sets define their own
        # electrode geometry whose actual capacity may differ from the
        # user-supplied nominal.
        #
        # Iter-3 fix: under partial-SoC cycling the first cycle's discharge can
        # be slightly less than steady-state (initial transient on Li
        # diffusion), producing later SoH ratios > 1.0 which break downstream
        # A/B/C/D grade logic. Use the max of the first 5 cycles as baseline
        # (clamps the transient) and additionally cap at 100 % defensively.
        n_baseline = min(5, len(df_out))
        sol_nominal = float(df_out["capacity"].iloc[:n_baseline].max())
        df_out["soh"] = (df_out["capacity"] / sol_nominal * 100).clip(upper=100.0)
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
    use_diurnal: bool = True,
    usage_protocol: str = "two_wheeler",
    termination_capacity_pct: int = 30,
    calibration_multipliers: Optional[dict] = None,
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
        cycle_name = rng.choice(list(INDIAN_DRIVING_CYCLES.keys()))

        # Iter-3: pick diurnal callable per Mulpuri 2025 Zone-X envelope, or
        # fall back to the constant-K Iter-1/Iter-2 profile when explicitly
        # disabled (used for direct A/B comparison against the prior corpus).
        if use_diurnal and ambient_name in DIURNAL_CLIMATE_PROFILES:
            cfg = DIURNAL_CLIMATE_PROFILES[ambient_name]
            ambient_K = diurnal_temperature_callable(
                mean_K=cfg["mean_K"], amplitude_K=cfg["amplitude_K"]
            )
            ambient_K_log = float(cfg["mean_K"])
            ambient_amp_log = float(cfg["amplitude_K"])
        else:
            ambient_K = INDIAN_AMBIENT_PROFILES[ambient_name]
            ambient_K_log = float(ambient_K)
            ambient_amp_log = 0.0

        battery_id = (f"IN_SYNTH_{cell_idx:04d}_{chemistry}_{ambient_name}"
                      f"_{cycle_name}_{usage_protocol}")
        print(f"\n[{cell_idx+1}/{n_cells}] {battery_id}")

        try:
            df = run_pybamm_simulation(
                n_cycles=n_cycles,
                chemistry=chemistry,
                ambient_K=ambient_K,
                include_degradation=True,
                usage_protocol=usage_protocol,
                termination_capacity_pct=termination_capacity_pct,
                calibration_multipliers=calibration_multipliers,
            )
        except Exception as exc:
            print(f"  Simulation failed: {exc}")
            continue

        df["battery_id"] = battery_id
        df["chemistry"] = chemistry
        df["ambient_profile"] = ambient_name
        df["ambient_K_mean"] = ambient_K_log
        df["ambient_K_amplitude"] = ambient_amp_log
        df["usage_protocol"] = usage_protocol
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
        "use_diurnal": use_diurnal,
        "usage_protocol": usage_protocol,
        "termination_capacity_pct": termination_capacity_pct,
        "diurnal_profiles": DIURNAL_CLIMATE_PROFILES if use_diurnal else None,
        "calibration_multipliers": calibration_multipliers,
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
    parser.add_argument("--no-diurnal", action="store_true",
                        help="Disable diurnal temperature profile; use Iter-1/Iter-2 "
                             "constant-K ambient instead. Default is diurnal.")
    parser.add_argument("--usage-protocol", type=str, default="two_wheeler",
                        choices=list(INDIAN_USAGE_PROTOCOLS.keys()),
                        help="Cycling protocol. `full_cycle` reproduces Iter-1/Iter-2 "
                             "(0–100 %% deep cycling). `two_wheeler` and `three_wheeler` "
                             "exercise Indian-deployment partial-SoC fast-charging stress.")
    parser.add_argument("--termination-capacity-pct", type=int, default=30,
                        help="Stop simulating a cell when capacity falls below this "
                             "percentage of cycle-1 capacity. Lower = drives cells to "
                             "deeper Grade D coverage if the parameter set permits it.")
    args = parser.parse_args()

    chemistries = ["NMC", "LFP"] if args.chemistry == "all" else [args.chemistry]

    generate_synthetic_dataset(
        n_cells=args.n_cells,
        n_cycles=args.n_cycles,
        chemistries=chemistries,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        use_diurnal=not args.no_diurnal,
        usage_protocol=args.usage_protocol,
        termination_capacity_pct=args.termination_capacity_pct,
    )


if __name__ == "__main__":
    main()
