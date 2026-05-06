"""
Iter-3 — Generate synthetic Indian-deployment LFP corpus via NREL BLAST-Lite.

Why this script exists
----------------------
PyBaMM's open-source LFP parameter set (Prada2013) has no aging model, so
our existing PyBaMM-based Iter-3 corpus has zero LFP coverage — a real gap
given that the Indian EV market is heavily LFP-leaning (Tata Nexon EV,
Tata Tigor EV, MG ZS EV, BYD Atto 3, Mahindra eVerito etc.).

NREL BLAST-Lite (Smith et al. 2017 ACC, Gasper et al. 2023 JES) ships a
peer-reviewed semi-empirical aging model for 250 Ah prismatic LFP-Gr cells
(`Lfp_Gr_250AhPrismatic`, fit to test data published in
J. Energy Storage 2024, doi:10.1016/j.est.2023.109042). Calendar+cycle
aging, calibrated to 10–45 °C, DoD 80–100 %, charge-rate ≤0.65 C —
operating envelope a good match for Indian fleet conditions, with the
Rajasthan-extreme 47 °C peak being a noted ~2 °C extrapolation.

Pipeline
--------
1. Build a per-cell time series spanning ~3 years of Indian-climate
   diurnal temperature + 2-wheeler partial-SoC duty cycling (30–90 % SoC,
   ~4 cycles/day).
2. Run BLAST-Lite `simulate_battery_life` with `threshold_efc=...` to get
   the capacity-fade trajectory under Indian conditions.
3. Sample the trajectory at the requested number of cycles, synthesize
   per-cycle V/I/T summary features using LFP-realistic OCV curves, and
   write per-cell CSVs matching the NMC Iter-3 schema.

Why BLAST-Lite alone (no PyBaMM-SPM hybrid):
- LFP voltage curves are characteristically flat (~3.05 V – 3.40 V across
  most of SoC), so V_min/V_max/V_mean don't vary much cycle-to-cycle —
  this stability is a *real* LFP feature, not an artefact of our pipeline.
- The aging signal (capacity, SoH, charge/discharge time) is what
  downstream ML actually consumes; BLAST-Lite gives us this directly.
- Avoids 100k+ PyBaMM-SPM cycles of compute we don't need.

Output schema matches `data/processed/synthetic_indian_iter3/IN_SYNTH_*.csv`
exactly so the unify pipeline merges them automatically.

Usage
-----
    python scripts/generate_lfp_blast.py --smoke              # 2 cells × 100 EFC
    python scripts/generate_lfp_blast.py                      # 40 cells × 2500 EFC
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import DIURNAL_CLIMATE_PROFILES, INDIAN_AMBIENT_PROFILES
from src.utils.config import PROCESSED_DIR, RANDOM_SEED


# ---------------------------------------------------------------------------
# LFP voltage curves (semi-empirical, characteristic of commercial LFP-Gr)
# ---------------------------------------------------------------------------
# LFP open-circuit voltage is famously flat. Published curves (A123, BYD,
# CATL prismatic) show a plateau ~3.30 V across SoC 10-90 %, sharp drop
# below 5 % to ~2.5 V, and rise to 3.45 V above 95 %. This is the dominant
# *physical signature* of LFP — it's why grade routing sometimes uses dQ/dV
# peak position rather than absolute voltage to discriminate aged cells.
#
# Aging affects the curve mildly: SEI growth raises internal resistance,
# which DEPRESSES voltage during discharge and ELEVATES it during charge.
# We model this as a small SoH-linked offset on top of the OCV plateau.

LFP_V_PLATEAU = 3.30        # V, mid-SoC plateau
LFP_V_LOW_DROP = 2.50       # V, deep discharge floor
LFP_V_HIGH_RISE = 3.45      # V, top-of-charge ceiling
LFP_NOMINAL_AH = 250.0      # commercial prismatic cell rating
TWO_WHEELER_C_RATE = 1.0    # 1C 2W partial-SoC fast charging


def _two_wheeler_voltage_envelope(soh_pct: float) -> tuple[float, float, float]:
    """Return (V_min, V_max, V_mean) for one 2W partial-SoC LFP cycle.

    SoC swings 30→90 % at 1C. Voltage at 30 % SoC under 1C discharge is
    ~3.05 V (small sag from OCV ~3.27); at 90 % SoC during 1C charge it's
    ~3.40 V (small overshoot). Mean tracks the plateau ~3.25 V.
    Aging adds a small IR-driven offset (~0.02 V at 60 % SoH).
    """
    aging_offset = 0.05 * (1.0 - soh_pct / 100.0)
    v_min = 3.05 - aging_offset
    v_max = 3.40 + aging_offset
    v_mean = 3.25
    return v_min, v_max, v_mean


def _two_wheeler_cycle_durations(soh_pct: float) -> tuple[float, float, float]:
    """Return (charge_time_s, discharge_time_s, total_cycle_time_s) for a
    2W partial-SoC LFP cycle, given current SoH.

    Charge phase: 30→90 % SoC at 1C → 0.6 h × (1C reciprocal in s)
    Discharge phase: 90→30 % SoC at 1C → same
    Plus 5 min rests on each side (matching INDIAN_USAGE_PROTOCOLS['two_wheeler']).
    Aging-induced capacity loss reduces charge/discharge times proportionally
    when current draw is in absolute amps (which is the realistic 2W case).
    """
    soh = soh_pct / 100.0
    base_charge_s = 0.6 * 3600.0  # 60 % SoC at 1C → 36 minutes
    base_discharge_s = 0.6 * 3600.0
    charge_s = base_charge_s * soh
    discharge_s = base_discharge_s * soh
    rest_s = 600.0  # 5 min × 2 rests
    return charge_s, discharge_s, charge_s + discharge_s + rest_s


# ---------------------------------------------------------------------------
# Diurnal temperature trajectory matching DIURNAL_CLIMATE_PROFILES
# ---------------------------------------------------------------------------

def _diurnal_temperature_celsius(t_s: np.ndarray, mean_K: float,
                                 amplitude_K: float, peak_hour: float = 14.0) -> np.ndarray:
    """24-hour sinusoidal temperature profile in Celsius.

    Mirrors the diurnal_temperature_callable() used by the PyBaMM pipeline,
    so LFP and NMC corpora share comparable thermal stress envelopes.
    """
    period_s = 86400.0
    omega = 2.0 * np.pi / period_s
    phase_s = peak_hour * 3600.0 - period_s / 4.0
    T_K = mean_K + (amplitude_K / 2.0) * np.sin(omega * (t_s - phase_s))
    return T_K - 273.15


# ---------------------------------------------------------------------------
# Per-cell BLAST-Lite simulation
# ---------------------------------------------------------------------------

@dataclass
class CellPlan:
    cell_idx: int
    chemistry: str               # always "LFP" for this script
    ambient_profile_name: str
    ambient_mean_K: float
    ambient_amplitude_K: float
    n_cycles: int                # target cycle count to record (matches NMC corpus)
    output_dir: str
    seed: int = RANDOM_SEED

    @property
    def battery_id(self) -> str:
        return (
            f"IN_SYNTH_LFP_{self.cell_idx:04d}_{self.ambient_profile_name}_two_wheeler"
        )


def _build_input_timeseries(plan: CellPlan, n_cycles: int) -> pd.DataFrame:
    """Build the (Time_s, SOC, Temperature_C) input for BLAST-Lite.

    Encodes ~4 cycles per day (one cycle ≈ 5,400 s including rests + active
    cycling) — typical for an Indian 2-wheeler in commercial use. Total
    horizon = n_cycles cycles ÷ 4 cycles/day. Adds small per-cell noise
    (seeded by cell_idx) so cells in the same climate aren't bit-identical.

    SoC swings 30 % → 90 % linearly each cycle; BLAST-Lite uses these for
    its DoD + average-SoC stressors. Temperature comes from the diurnal
    profile keyed off the wall-clock-equivalent timestamp.
    """
    rng = np.random.default_rng(plan.seed)
    cycle_dur_s = 5400.0  # ~90 minutes per partial-SoC cycle
    total_s = n_cycles * cycle_dur_s
    # Sample once per cycle midpoint — BLAST-Lite handles per-cycle stressors
    t_s = np.arange(0.0, total_s, cycle_dur_s) + cycle_dur_s / 2.0

    # Triangular SoC trajectory: 30 % at t=0, 90 % at midpoint, 30 % at end of cycle
    # We sample at the midpoint so SOC = 90 % (top of cycle). For BLAST-Lite's
    # cycle-aging model, what matters is avg SOC, DOD, and Crate — we drive
    # the model at avg ≈ 60 %.
    soc = np.full(t_s.shape, 0.6) + rng.uniform(-0.02, 0.02, size=t_s.shape)
    soc = np.clip(soc, 0.30, 0.90)

    # Diurnal temperature at this timestamp
    T_C = _diurnal_temperature_celsius(t_s, plan.ambient_mean_K,
                                       plan.ambient_amplitude_K)

    return pd.DataFrame({"Time_s": t_s, "SOC": soc, "Temperature_C": T_C})


def _run_one_cell(plan: CellPlan) -> dict:
    """Top-level so it pickles for ProcessPoolExecutor.

    Resumes if a complete CSV already exists for this cell.
    """
    t0 = time.time()
    out_csv = Path(plan.output_dir) / f"{plan.battery_id}.csv"
    if out_csv.exists() and out_csv.stat().st_size > 0:
        try:
            existing = pd.read_csv(out_csv)
            if len(existing) >= 50 and "soh" in existing.columns:
                return {
                    "cell_idx": plan.cell_idx,
                    "battery_id": plan.battery_id,
                    "status": "skipped",
                    "n_cycles_completed": int(len(existing)),
                    "soh_start": float(existing["soh"].iloc[0]),
                    "soh_end": float(existing["soh"].iloc[-1]),
                    "soh_min": float(existing["soh"].min()),
                    "elapsed_s": 0.0,
                }
        except Exception:
            pass

    try:
        # Lazy import so workers don't all initialise BLAST at module load
        from blast.models import Lfp_Gr_250AhPrismatic
        bm = Lfp_Gr_250AhPrismatic()
        df_input = _build_input_timeseries(plan, plan.n_cycles)
        bm.simulate_battery_life(
            input_timeseries=df_input,
            threshold_efc=plan.n_cycles + 50,  # +50 buffer over target
        )
    except Exception as exc:
        return {
            "cell_idx": plan.cell_idx,
            "battery_id": plan.battery_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - t0, 1),
        }

    # ---- Extract SoH trajectory from BLAST-Lite output -------------------
    # bm.outputs["q"] is the capacity ratio at each breakpoint (1.0 = pristine)
    # bm.stressors["efc"] is the cumulative EFC at each breakpoint
    q_traj = np.asarray(bm.outputs["q"])
    efc_traj = np.asarray(bm.stressors["efc"])

    if len(q_traj) < 2:
        return {
            "cell_idx": plan.cell_idx,
            "battery_id": plan.battery_id,
            "status": "empty",
            "elapsed_s": round(time.time() - t0, 1),
        }

    # Resample onto exactly n_cycles cycles via linear interpolation in EFC
    target_efc = np.linspace(0, min(efc_traj[-1], plan.n_cycles), plan.n_cycles + 1)
    q_resampled = np.interp(target_efc, efc_traj, q_traj)
    soh_pct = q_resampled * 100.0  # SoH as percent

    # ---- Synthesize per-cycle V/I/T features matching NMC schema ---------
    cycle_dur_s = 5400.0
    records = []
    for cycle_idx, soh in enumerate(soh_pct, start=0):
        t_in_year = cycle_idx * cycle_dur_s + cycle_dur_s / 2.0
        T_C = float(_diurnal_temperature_celsius(
            np.array([t_in_year]),
            plan.ambient_mean_K,
            plan.ambient_amplitude_K,
        )[0])
        v_min, v_max, v_mean = _two_wheeler_voltage_envelope(soh)
        capacity_Ah = LFP_NOMINAL_AH * (soh / 100.0)
        # 2W cycle: equal charge + discharge → mean current ≈ 0
        # but we record the magnitude during active phase
        i_mean_active = TWO_WHEELER_C_RATE * capacity_Ah
        _, _, total_cycle_s = _two_wheeler_cycle_durations(soh)

        records.append({
            "cycle": cycle_idx,
            "time_s": float(total_cycle_s),
            "voltage_mean": float(v_mean),
            "voltage_min": float(v_min),
            "voltage_max": float(v_max),
            "current_mean": float(i_mean_active),
            "temperature_mean": T_C,
            "temperature_max": T_C + 2.0,  # small thermal rise during 1C phase
            "temperature_range": 2.0,
            "capacity": float(capacity_Ah),
            "soh": float(np.clip(soh, 0.0, 100.0)),
        })

    df = pd.DataFrame(records)
    df["battery_id"] = plan.battery_id
    df["chemistry"] = "LFP"
    df["ambient_profile"] = plan.ambient_profile_name
    df["ambient_K_mean"] = plan.ambient_mean_K
    df["ambient_K_amplitude"] = plan.ambient_amplitude_K
    df["usage_protocol"] = "two_wheeler"
    df["source"] = "synthetic_blast_indian_iter3_lfp"

    Path(plan.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    soh_min = float(df["soh"].min())
    soh_end = float(df["soh"].iloc[-1])
    soh_start = float(df["soh"].iloc[0])
    grade_thresholds = {"A": 80, "B": 60, "C": 40}
    coverage = {f"reaches_grade_{g}_or_lower": bool((df["soh"] < t).any())
                for g, t in grade_thresholds.items()}
    coverage["reaches_grade_D"] = bool((df["soh"] < 40).any())

    result = {
        "cell_idx": plan.cell_idx,
        "battery_id": plan.battery_id,
        "status": "ok",
        "n_cycles_completed": int(len(df)),
        "soh_start": soh_start,
        "soh_end": soh_end,
        "soh_min": soh_min,
        **coverage,
        "elapsed_s": round(time.time() - t0, 1),
    }
    del df
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _plan_corpus(
    n_cells: int,
    n_cycles: int,
    output_dir: Path,
    ambient_profiles: Optional[list] = None,
) -> list[CellPlan]:
    """Round-robin assign cells across the 4 Indian climate clusters.

    Each cell's `seed = RANDOM_SEED + cell_idx` introduces small SoC noise
    in `_build_input_timeseries`, giving different BLAST-Lite trajectories
    even within the same climate cluster.
    """
    ambient_profiles = ambient_profiles or list(DIURNAL_CLIMATE_PROFILES.keys())
    plans = []
    for i in range(n_cells):
        amb_name = ambient_profiles[i % len(ambient_profiles)]
        cfg = DIURNAL_CLIMATE_PROFILES[amb_name]
        plans.append(CellPlan(
            cell_idx=i,
            chemistry="LFP",
            ambient_profile_name=amb_name,
            ambient_mean_K=float(cfg["mean_K"]),
            ambient_amplitude_K=float(cfg["amplitude_K"]),
            n_cycles=n_cycles,
            output_dir=str(output_dir),
            seed=RANDOM_SEED + i,
        ))
    return plans


def main():
    p = argparse.ArgumentParser(description="Iter-3 LFP corpus via BLAST-Lite")
    p.add_argument("--smoke", action="store_true",
                   help="2 cells × 100 EFC, sequential")
    p.add_argument("--n-cells", type=int, default=40)
    p.add_argument("--n-cycles", type=int, default=12000,
                   help="Target EFCs per cell. Default 12,000 — empirically "
                        "needed for the LFP-Gr 250Ah model to drive cells from "
                        "Grade A in mild climates down to Grade D in Rajasthan-"
                        "extreme heat. LFP is genuinely more durable than NMC, "
                        "so this is ~5× the NMC corpus's 2500-cycle target.")
    p.add_argument("--output-dir", type=str,
                   default=str(PROCESSED_DIR / "synthetic_indian_iter3_lfp"))
    p.add_argument("--ambient-profiles", type=str, nargs="+",
                   default=list(DIURNAL_CLIMATE_PROFILES.keys()))
    args = p.parse_args()

    if args.smoke:
        args.n_cells = 2
        args.n_cycles = 100

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = _plan_corpus(
        n_cells=args.n_cells, n_cycles=args.n_cycles,
        output_dir=output_dir, ambient_profiles=args.ambient_profiles,
    )

    print("=" * 72)
    print(f"Iter-3 LFP corpus via BLAST-Lite  ({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 72)
    print(f"  cells          : {args.n_cells}")
    print(f"  EFCs/cell      : {args.n_cycles}")
    print(f"  output_dir     : {output_dir}")
    plan_summary = pd.DataFrame([
        {"climate": p.ambient_profile_name} for p in plans
    ]).groupby("climate").size().rename("n_cells").reset_index()
    print()
    print("Planned cell distribution:")
    print(plan_summary.to_string(index=False))
    print()

    # BLAST-Lite is fast (~seconds per cell) and uses little RAM (<200 MB
    # per worker). Sequential is fine for the full 40-cell run.
    log_rows = []
    t_start = time.time()
    with tqdm(total=len(plans), desc="LFP cells", unit="cell",
              file=sys.stderr, mininterval=0.5, dynamic_ncols=True) as pbar:
        for plan in plans:
            result = _run_one_cell(plan)
            log_rows.append(result)
            line = (f"  [{result['battery_id']}] status={result['status']}  "
                    f"elapsed={result['elapsed_s']}s")
            if result["status"] == "ok":
                line += (f"  SoH {result['soh_start']:.2f}→"
                         f"{result['soh_end']:.2f}%  min={result['soh_min']:.2f}%")
            elif "error" in result:
                line += f"  error={result['error'][:80]}"
            sys.stderr.write(line + "\n")
            sys.stderr.flush()
            pbar.update(1)

    total_elapsed = time.time() - t_start
    log_df = pd.DataFrame(log_rows).sort_values("cell_idx")
    log_df.to_csv(output_dir / "regeneration_log.csv", index=False)

    n_ok = log_df["status"].isin(["ok", "skipped"]).sum()
    n_skipped = (log_df["status"] == "skipped").sum()
    n_fresh = (log_df["status"] == "ok").sum()
    print()
    print("=" * 72)
    print(f"Done. {n_ok}/{len(log_df)} cells valid  ({n_fresh} fresh + {n_skipped} resumed)  ·  "
          f"wall-clock {total_elapsed/60:.1f} min")
    print("=" * 72)

    # Combined CSV
    if n_ok > 0:
        all_dfs = []
        for path in sorted(output_dir.glob("IN_SYNTH_LFP_*.csv")):
            all_dfs.append(pd.read_csv(path))
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(output_dir / "synthetic_indian_lfp_combined.csv", index=False)
            print(f"  combined → {output_dir/'synthetic_indian_lfp_combined.csv'}  "
                  f"({len(combined):,} rows, {combined['battery_id'].nunique()} cells)")

        coverage_cols = ["reaches_grade_A_or_lower", "reaches_grade_B_or_lower",
                         "reaches_grade_C_or_lower", "reaches_grade_D"]
        present_cols = [c for c in coverage_cols if c in log_df.columns]
        if present_cols:
            print("\nGrade coverage:")
            for c in present_cols:
                pct = log_df[log_df["status"] == "ok"][c].mean() * 100
                print(f"  {c:35s}  {pct:5.1f}% of cells")

    # Metadata
    meta = {
        "iter": 3,
        "chemistry": "LFP",
        "generator": "NREL BLAST-Lite (Lfp_Gr_250AhPrismatic, J. Energy Storage 2024)",
        "smoke": args.smoke,
        "n_cells_succeeded": int(n_ok),
        "n_cells_failed": int(len(log_df) - n_ok),
        "n_cycles_per_cell": args.n_cycles,
        "ambient_profiles": args.ambient_profiles,
        "diurnal_profiles": DIURNAL_CLIMATE_PROFILES,
        "lfp_voltage_curve_assumptions": {
            "v_plateau_V": LFP_V_PLATEAU,
            "v_low_drop_V": LFP_V_LOW_DROP,
            "v_high_rise_V": LFP_V_HIGH_RISE,
            "nominal_capacity_Ah": LFP_NOMINAL_AH,
            "two_wheeler_C_rate": TWO_WHEELER_C_RATE,
        },
        "methodology_note": (
            "BLAST-Lite produces capacity-fade trajectories from semi-empirical "
            "calendar+cycle aging fits to commercial 250 Ah prismatic LFP cells "
            "(experimental data: Smith/Gasper et al. NREL, J. Energy Storage 2024 "
            "doi:10.1016/j.est.2023.109042). Per-cycle V/I/T summary stats are "
            "synthesised from LFP-realistic voltage curves and the diurnal "
            "Indian-climate temperature at each cycle's wall-clock timestamp. "
            "This pairs aging realism (BLAST) with feature realism (LFP voltage "
            "plateau is genuinely flat, and is itself a key diagnostic signature)."
        ),
        "total_wall_clock_min": round(total_elapsed / 60, 2),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nlog      → {output_dir/'regeneration_log.csv'}")
    print(f"metadata → {output_dir/'metadata.json'}")


if __name__ == "__main__":
    os.environ["PYTHONUNBUFFERED"] = "1"
    # BLAST-Lite emits harmless RuntimeWarnings during numerical integration
    # at the edges of test bins (zero-length intervals → divide-by-zero in
    # np.trapz averaging). The fade trajectory is unaffected. Silence them
    # so the heartbeat output stays readable.
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning,
                            module=r"blast\..*")
    main()
