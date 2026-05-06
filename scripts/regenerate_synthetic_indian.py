"""
Iter-3 — Regenerate the synthetic Indian-context cycling corpus with the
calibrated PyBaMM pipeline (10× SEI/LAM/plating multipliers + diurnal
climate profiles + Indian partial-SoC usage protocols).

Per the Iter-2 audit, the original 50-cell × 100-cycle corpus stayed at
SoH ≈ 100 % regardless of climate cluster, making the synthetic-holdout
test uninformative. This script produces a replacement corpus that spans
Grade A → D coverage under realistic Indian-deployment stress.

Outputs (parquet + per-cell CSV + metadata.json):
    data/processed/synthetic_indian_iter3/
        synthetic_indian_combined.csv       — long-form per-cycle records
        IN_SYNTH_*.csv                      — one file per cell (recoverable)
        metadata.json                       — full provenance + multipliers
        regeneration_log.csv                — wall-clock + SoH range per cell

Smoke test (2 cells × 100 cycles, ≈ 4 minutes):
    conda run -n Eco-Research python scripts/regenerate_synthetic_indian.py --smoke

Full run (40 NMC cells × 2,500 cycles, parallelised, ≈ 5–7 hours):
    conda run -n Eco-Research python scripts/regenerate_synthetic_indian.py \\
        --n-cells 40 --n-cycles 2500 --n-workers 6
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil
from tqdm import tqdm

# OOM-safety: with chunked solving (chunk_cycles=100), PyBaMM DFN + full
# degradation stack settles at 3.0–3.6 GB per worker — verified empirically
# on M2 Pro / 24 GB hardware via the 200-cycle × 2-worker smoke run on
# 2026-05-05. Without chunking the same run can hit 5–8 GB.
EST_MEMORY_PER_WORKER_GB = 3.0
# Reserve this much for the OS, IDE, browser, and the orchestrator itself.
# 6 GB is a conservative floor on a 24 GB box; bump up if you have heavier
# background apps.
RESERVED_MEMORY_GB = 6.0
# Hard floor on per-worker RAM regardless of estimate. Matches the verified
# steady-state RSS so we don't squeeze workers below their working set.
HARD_MIN_GB_PER_WORKER = 3.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.synthetic import (
    DIURNAL_CLIMATE_PROFILES,
    INDIAN_USAGE_PROTOCOLS,
    INDIA_CALIBRATION_DEFAULT,
    INDIAN_AMBIENT_PROFILES,
    diurnal_temperature_callable,
    run_pybamm_simulation,
)
from src.utils.config import PROCESSED_DIR, RANDOM_SEED


@dataclass
class CellPlan:
    """One cell's full configuration. Pickle-safe for ProcessPoolExecutor."""
    cell_idx: int
    chemistry: str
    ambient_profile_name: str
    ambient_mean_K: float
    ambient_amplitude_K: float
    use_diurnal: bool
    usage_protocol: str
    n_cycles: int
    termination_capacity_pct: int
    calibration_multipliers: dict
    output_dir: str  # str so it pickles cleanly
    heartbeat_every_k: int = 50
    chunk_cycles: int = 100

    @property
    def battery_id(self) -> str:
        return (
            f"IN_SYNTH_{self.cell_idx:04d}_{self.chemistry}_"
            f"{self.ambient_profile_name}_{self.usage_protocol}"
        )


def _plan_corpus(
    n_cells: int,
    n_cycles: int,
    usage_protocol: str,
    use_diurnal: bool,
    termination_capacity_pct: int,
    calibration_multipliers: dict,
    output_dir: Path,
    chemistries: Optional[list] = None,
    ambient_profiles: Optional[list] = None,
    heartbeat_every_k: int = 50,
    chunk_cycles: int = 100,
    randomize_after_cell: int = 0,
) -> list[CellPlan]:
    """Round-robin assign cells to (chemistry × climate) combinations.

    Round-robin rather than random-sample so we get balanced cluster
    coverage even at small n_cells (avoids the chance of zero LFP cells
    or zero Rajasthan cells in a 10-cell smoke run).

    Cells with `cell_idx >= randomize_after_cell` get per-cell perturbation
    of calibration multipliers + diurnal envelope (seeded reproducibly by
    RANDOM_SEED + cell_idx). Cells with smaller idx stay on the canonical
    deterministic config — useful when extending an existing corpus where
    the first N cells already exist on disk.
    """
    chemistries = chemistries or ["NMC"]
    ambient_profiles = ambient_profiles or list(DIURNAL_CLIMATE_PROFILES.keys())

    plans = []
    for i in range(n_cells):
        chem = chemistries[i % len(chemistries)]
        amb_name = ambient_profiles[i % len(ambient_profiles)]
        if use_diurnal and amb_name in DIURNAL_CLIMATE_PROFILES:
            cfg = DIURNAL_CLIMATE_PROFILES[amb_name]
            mean_K = float(cfg["mean_K"])
            amp_K = float(cfg["amplitude_K"])
        else:
            mean_K = float(INDIAN_AMBIENT_PROFILES.get(amb_name, 313.15))
            amp_K = 0.0

        cell_multipliers = dict(calibration_multipliers)

        # Iter-3 Option C: per-cell randomization past `randomize_after_cell`.
        # Each cell's perturbation is reproducibly seeded so re-running this
        # command yields the same configs (resume-logic skips already-done
        # cells without re-rolling).
        if i >= randomize_after_cell and randomize_after_cell >= 0:
            seed = RANDOM_SEED + i
            cell_multipliers, mean_K, amp_K = _perturb_config(
                base_multipliers=calibration_multipliers,
                base_mean_K=mean_K,
                base_amplitude_K=amp_K,
                seed=seed,
            )

        plans.append(CellPlan(
            cell_idx=i,
            chemistry=chem,
            ambient_profile_name=amb_name,
            ambient_mean_K=mean_K,
            ambient_amplitude_K=amp_K,
            use_diurnal=use_diurnal,
            usage_protocol=usage_protocol,
            n_cycles=n_cycles,
            termination_capacity_pct=termination_capacity_pct,
            calibration_multipliers=cell_multipliers,
            output_dir=str(output_dir),
            heartbeat_every_k=heartbeat_every_k,
            chunk_cycles=chunk_cycles,
        ))
    return plans


def _perturb_config(
    base_multipliers: dict,
    base_mean_K: float,
    base_amplitude_K: float,
    seed: int,
) -> tuple[dict, float, float]:
    """Per-cell randomization for Iter-3 Option C.

    Generates a unique calibration multiplier set + diurnal climate envelope
    for each cell, seeded reproducibly by (RANDOM_SEED + cell_idx). Used
    only for cells past `--randomize-after-cell`; earlier cells stay on the
    canonical config so they match existing CSVs and the resume-logic
    skips them.

    Perturbation amplitudes (chosen to span a meaningful spread without
    producing physically implausible cells):
        - calibration multipliers: × Uniform(0.7, 1.4)  → spread roughly
          7× to 14× around the 10× canonical
        - ambient mean K:          + Uniform(-1.5, 1.5)  → ~3 K spread
        - ambient amplitude K:     + Uniform(-2.0, 2.0)  → ~4 K spread

    The same seed produces the same perturbation, so re-running the
    extension command yields identical cells and resume-skips them.
    """
    import random as _random
    rng = _random.Random(seed)
    perturbed_multipliers = {
        k: float(v) * rng.uniform(0.7, 1.4)
        for k, v in base_multipliers.items()
    }
    perturbed_mean_K = base_mean_K + rng.uniform(-1.5, 1.5)
    perturbed_amp_K = max(0.5, base_amplitude_K + rng.uniform(-2.0, 2.0))
    return perturbed_multipliers, perturbed_mean_K, perturbed_amp_K


def _recommend_workers(requested: int, est_per_worker_gb: float = EST_MEMORY_PER_WORKER_GB,
                       reserved_gb: float = RESERVED_MEMORY_GB) -> tuple[int, dict]:
    """Pre-flight memory check. Caps worker count to what current free RAM
    can safely sustain.

    Returns the safe worker count and a diagnostics dict showing the
    calculation, so the orchestrator can print a readable summary before
    starting a long-running job.
    """
    vm = psutil.virtual_memory()
    available_gb = vm.available / 1e9
    total_gb = vm.total / 1e9
    usable_gb = max(available_gb - reserved_gb, 0.5)
    # Two ceilings: (a) usable / estimated, (b) usable / HARD_MIN. Use the
    # smaller — protects against an aggressive estimate that would still
    # squeeze workers below CasADi's transient compile floor.
    est_workers = max(1, int(usable_gb // est_per_worker_gb))
    hard_workers = max(1, int(usable_gb // HARD_MIN_GB_PER_WORKER))
    safe_workers = min(est_workers, hard_workers)
    capped = min(requested, safe_workers)
    return capped, {
        "total_gb": round(total_gb, 1),
        "available_gb": round(available_gb, 1),
        "reserved_gb": reserved_gb,
        "usable_for_workers_gb": round(usable_gb, 1),
        "est_per_worker_gb": est_per_worker_gb,
        "hard_min_gb_per_worker": HARD_MIN_GB_PER_WORKER,
        "safe_workers": safe_workers,
        "requested_workers": requested,
        "final_workers": capped,
        "downgraded": capped < requested,
    }


def _check_memory_pressure(threshold_pct: float = 90.0) -> Optional[float]:
    """Return current memory-used % if it exceeds the threshold, else None.

    Used during the run as a soft guardrail — printed to stderr so the user
    can see if the machine is approaching paging territory.
    """
    pct = psutil.virtual_memory().percent
    return pct if pct >= threshold_pct else None


def _safe_rel(p: Path) -> Path | str:
    """Display a path relative to PROJECT_ROOT when possible, else absolute.

    Saves us a bunch of try/except blocks when --output-dir points
    outside the project (typical during smoke tests using /tmp).
    """
    try:
        return p.relative_to(PROJECT_ROOT)
    except ValueError:
        return p


def _print_cell_summary(result: dict) -> None:
    """Emit a one-line per-cell completion record to stderr.

    Goes to stderr alongside tqdm + heartbeat lines so everything ends up
    in the same log stream when the user redirects via `2>&1 | tee log`.
    """
    line = (f"  [{result['battery_id']}] status={result['status']} "
            f"elapsed={result['elapsed_s']}s")
    if "rss_end_gb" in result:
        line += f"  RSS={result['rss_end_gb']:.1f}GB"
        if "rss_delta_gb" in result:
            line += f" (Δ{result['rss_delta_gb']:+.1f})"
    if result["status"] == "ok":
        line += (f"  SoH {result.get('soh_start', float('nan')):.2f}→"
                 f"{result.get('soh_end', float('nan')):.2f}%  "
                 f"min={result.get('soh_min', float('nan')):.2f}%")
    elif "error" in result:
        line += f"  error={result['error'][:80]}"
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _run_one_cell(plan: CellPlan) -> dict:
    """Top-level so it pickles for multiprocessing.

    Returns a summary dict (timing + SoH coverage + per-worker RSS) suitable
    for the regeneration log; per-cell parquet/CSV is written as a side effect.

    Resume behaviour: if a complete CSV already exists for this cell's
    `battery_id`, the cell is skipped and a status='skipped' result is
    returned. A "complete" file has the expected schema and at least
    50 rows (anything smaller is treated as a partial / corrupt write
    from a previous interrupted run and is overwritten).
    """
    t0 = time.time()
    proc = psutil.Process(os.getpid())
    rss_start_gb = proc.memory_info().rss / 1e9

    # ---- Resume-from-checkpoint --------------------------------------------
    # Reuse work from any previous run that completed this cell. The
    # script's writer puts the CSV down only after the simulation fully
    # finishes (last line in `_run_one_cell`'s success branch), so a
    # well-formed file means the cell is done.
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
                    "rss_start_gb": round(rss_start_gb, 2),
                    "rss_end_gb": round(rss_start_gb, 2),
                    "rss_delta_gb": 0.0,
                    "note": "resumed-from-existing-csv",
                }
        except Exception:
            # Corrupted CSV — fall through and rerun the cell
            pass

    try:
        if plan.use_diurnal and plan.ambient_amplitude_K > 0:
            ambient = diurnal_temperature_callable(
                mean_K=plan.ambient_mean_K,
                amplitude_K=plan.ambient_amplitude_K,
            )
        else:
            ambient = plan.ambient_mean_K

        df = run_pybamm_simulation(
            n_cycles=plan.n_cycles,
            chemistry=plan.chemistry,
            ambient_K=ambient,
            include_degradation=True,
            usage_protocol=plan.usage_protocol,
            termination_capacity_pct=plan.termination_capacity_pct,
            calibration_multipliers=plan.calibration_multipliers,
            progress_every_k_cycles=plan.heartbeat_every_k,
            progress_label=plan.battery_id,
            chunk_cycles=plan.chunk_cycles,
        )
    except Exception as exc:
        rss_end_gb = proc.memory_info().rss / 1e9
        return {
            "cell_idx": plan.cell_idx,
            "battery_id": plan.battery_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_s": round(time.time() - t0, 1),
            "rss_start_gb": round(rss_start_gb, 2),
            "rss_end_gb": round(rss_end_gb, 2),
        }

    if df.empty:
        rss_end_gb = proc.memory_info().rss / 1e9
        return {
            "cell_idx": plan.cell_idx,
            "battery_id": plan.battery_id,
            "status": "empty",
            "elapsed_s": round(time.time() - t0, 1),
            "rss_start_gb": round(rss_start_gb, 2),
            "rss_end_gb": round(rss_end_gb, 2),
        }

    df["battery_id"] = plan.battery_id
    df["chemistry"] = plan.chemistry
    df["ambient_profile"] = plan.ambient_profile_name
    df["ambient_K_mean"] = plan.ambient_mean_K
    df["ambient_K_amplitude"] = plan.ambient_amplitude_K
    df["usage_protocol"] = plan.usage_protocol
    df["source"] = "synthetic_pybamm_indian_iter3"

    out_dir = Path(plan.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{plan.battery_id}.csv", index=False)

    # Compute Grade coverage to surface in the per-cell log
    grade_thresholds = {"A": 80, "B": 60, "C": 40}  # > = grade
    soh = df["soh"].to_numpy()
    coverage = {}
    for label, thresh in grade_thresholds.items():
        coverage[f"reaches_grade_{label}_or_lower"] = bool((soh < thresh).any())
    coverage["reaches_grade_D"] = bool((soh < 40).any())

    rss_end_gb = proc.memory_info().rss / 1e9
    result = {
        "cell_idx": plan.cell_idx,
        "battery_id": plan.battery_id,
        "status": "ok",
        "n_cycles_completed": int(len(df)),
        "soh_start": float(df["soh"].iloc[0]),
        "soh_end": float(df["soh"].iloc[-1]),
        "soh_min": float(df["soh"].min()),
        **coverage,
        "elapsed_s": round(time.time() - t0, 1),
        "rss_start_gb": round(rss_start_gb, 2),
        "rss_end_gb": round(rss_end_gb, 2),
        "rss_delta_gb": round(rss_end_gb - rss_start_gb, 2),
    }
    # Force-release PyBaMM internals (Casadi solver state, sparse matrices,
    # cycle-by-cycle solution objects). Without this each worker's RSS
    # creeps up over multiple cells and you can OOM 4 hours into a long run.
    del df
    gc.collect()
    return result


def main():
    p = argparse.ArgumentParser(
        description="Iter-3 calibrated PyBaMM synthetic Indian-corpus regeneration"
    )
    p.add_argument("--smoke", action="store_true",
                   help="2 cells × 100 cycles, sequential. Validates wiring.")
    p.add_argument("--n-cells", type=int, default=40,
                   help="Number of cells to generate (default 40)")
    p.add_argument("--n-cycles", type=int, default=2500,
                   help="Cycles per cell (default 2500 — should reach Grade D "
                        "with calibration multipliers)")
    p.add_argument("--n-workers", type=int, default=1,
                   help="Parallel worker processes. Tune to physical CPU cores. "
                        "PyBaMM uses BLAS internally — don't oversubscribe; on an "
                        "M-series Mac with 8-12 cores, 4-6 workers is the sweet spot.")
    p.add_argument("--usage-protocol", type=str, default="two_wheeler",
                   choices=list(INDIAN_USAGE_PROTOCOLS.keys()),
                   help="Cycling protocol (default two_wheeler — Indian 2W partial-SoC)")
    p.add_argument("--no-diurnal", action="store_true",
                   help="Disable diurnal temperature profile (use constant-K Iter-1/2 "
                        "ambient instead, for direct A/B comparison)")
    p.add_argument("--no-calibration", action="store_true",
                   help="Disable the 10× aging calibration multipliers (recovers the "
                        "stock OKane2022 Iter-1/Iter-2 fade rate)")
    p.add_argument("--termination-capacity-pct", type=int, default=30)
    p.add_argument("--heartbeat-every", type=int, default=50,
                   help="Print a per-cell heartbeat line (cycle K/total + ETA) "
                        "every N cycles. Set to 0 to disable. Default 50 — "
                        "with ~0.5 s/cycle that's a heartbeat every ~25 s.")
    p.add_argument("--ignore-memory-guard", action="store_true",
                   help="Skip the pre-flight memory check that auto-caps "
                        "--n-workers. Use only if you are sure you have headroom.")
    p.add_argument("--chunk-cycles", type=int, default=100,
                   help="Solve the simulation in chunks of N cycles, extracting "
                        "the per-cycle summary and discarding the chunk's full "
                        "solution before continuing. Bounds RSS at "
                        "~chunk_cycles × per-cycle-state regardless of total "
                        "n_cycles. Lower = lower memory + slightly slower; "
                        "higher = higher memory + slightly faster. Default 100.")
    p.add_argument("--randomize-after-cell", type=int, default=0,
                   help="Apply per-cell randomization (perturbed calibration "
                        "multipliers + diurnal envelope, seeded by cell_idx) "
                        "for cells with idx >= N. Default 0 = randomize all "
                        "cells. Set to e.g. 40 to preserve an existing 40-cell "
                        "canonical corpus and only randomize cells 40+. "
                        "PyBaMM is deterministic — without this flag, all "
                        "cells in the same (chemistry, climate, protocol) "
                        "cluster produce bit-identical trajectories.")
    p.add_argument("--output-dir", type=str,
                   default=str(PROCESSED_DIR / "synthetic_indian_iter3"))
    p.add_argument("--chemistries", type=str, nargs="+", default=["NMC"],
                   help="Chemistries to generate. NMC supports degradation; LFP "
                        "(Prada2013) does not — see synthetic.py for details.")
    p.add_argument("--ambient-profiles", type=str, nargs="+",
                   default=list(DIURNAL_CLIMATE_PROFILES.keys()),
                   help="Climate profiles to round-robin over.")
    args = p.parse_args()

    if args.smoke:
        args.n_cells = 2
        args.n_cycles = 100
        args.n_workers = 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_multipliers = (
        {} if args.no_calibration else dict(INDIA_CALIBRATION_DEFAULT)
    )

    # ---- Pre-flight memory guard --------------------------------------
    safe_workers, mem_diag = _recommend_workers(args.n_workers)
    if not args.ignore_memory_guard and safe_workers < args.n_workers:
        print(f"\n[memory-guard]  --n-workers {args.n_workers} would need ~"
              f"{args.n_workers * mem_diag['est_per_worker_gb']:.1f} GB but only "
              f"{mem_diag['usable_for_workers_gb']:.1f} GB is safely usable "
              f"(after reserving {mem_diag['reserved_gb']:.0f} GB for the OS).")
        print(f"[memory-guard]  Capping to --n-workers {safe_workers} to "
              f"prevent OOM. Pass --ignore-memory-guard to override.\n")
        args.n_workers = safe_workers

    print("=" * 72)
    print(f"Iter-3 synthetic Indian-corpus regeneration  "
          f"({'SMOKE' if args.smoke else 'FULL'})")
    print("=" * 72)
    print(f"  cells          : {args.n_cells}")
    print(f"  cycles/cell    : {args.n_cycles}")
    print(f"  chunk_cycles   : {args.chunk_cycles}  "
          f"(per-cell solve runs in chunks of this size; bounds RSS)")
    print(f"  workers        : {args.n_workers}  "
          f"(memory: {mem_diag['available_gb']:.1f}/{mem_diag['total_gb']:.1f} GB available; "
          f"~{args.n_workers * mem_diag['est_per_worker_gb']:.1f} GB est for workers)")
    print(f"  protocol       : {args.usage_protocol}")
    print(f"  diurnal        : {not args.no_diurnal}")
    print(f"  calibration    : {calibration_multipliers if calibration_multipliers else 'OFF (stock OKane2022)'}")
    n_random = max(0, args.n_cells - args.randomize_after_cell)
    n_canonical = min(args.n_cells, args.randomize_after_cell)
    print(f"  randomization  : {n_canonical} canonical + {n_random} randomized cells "
          f"(--randomize-after-cell={args.randomize_after_cell})")
    print(f"  output_dir     : {_safe_rel(output_dir)}")
    print()

    plans = _plan_corpus(
        n_cells=args.n_cells,
        n_cycles=args.n_cycles,
        usage_protocol=args.usage_protocol,
        use_diurnal=not args.no_diurnal,
        termination_capacity_pct=args.termination_capacity_pct,
        calibration_multipliers=calibration_multipliers,
        output_dir=output_dir,
        chemistries=args.chemistries,
        ambient_profiles=args.ambient_profiles,
        heartbeat_every_k=args.heartbeat_every,
        chunk_cycles=args.chunk_cycles,
        randomize_after_cell=args.randomize_after_cell,
    )

    # Print the planned corpus so the user can sanity-check before a long run
    print("Planned cell distribution:")
    plan_summary = pd.DataFrame([
        {"chemistry": p.chemistry, "climate": p.ambient_profile_name,
         "diurnal": p.use_diurnal, "n_cycles": p.n_cycles}
        for p in plans
    ]).groupby(["chemistry", "climate"]).size().rename("n_cells").reset_index()
    print(plan_summary.to_string(index=False))
    print()

    # ---- Run cells (parallel if n_workers > 1, sequential otherwise) -------
    log_rows = []
    t_start = time.time()

    # tqdm and the per-cell heartbeat both go to stderr — together they form
    # a coherent log: the bar shows overall cell-level progress, the
    # heartbeat shows live cycle-level work. Workers write to the parent's
    # stderr so all lines accumulate in any redirected log file.
    tqdm_kwargs = dict(
        desc="Cells",
        unit="cell",
        file=sys.stderr,
        mininterval=1.0,
        dynamic_ncols=True,
    )

    def _maybe_warn_memory():
        pct = _check_memory_pressure(threshold_pct=88.0)
        if pct is not None:
            sys.stderr.write(
                f"  [memory-pressure WARN] system memory at {pct:.0f}% — "
                f"if this stays high for several cells, kill the run "
                f"and lower --n-workers.\n")
            sys.stderr.flush()

    if args.n_workers <= 1:
        # Sequential — useful for smoke test and easier-to-read logs
        with tqdm(total=len(plans), **tqdm_kwargs) as pbar:
            for plan in plans:
                result = _run_one_cell(plan)
                log_rows.append(result)
                _print_cell_summary(result)
                _maybe_warn_memory()
                pbar.update(1)
    else:
        # Parallel — use 'spawn' for macOS safety with PyBaMM/CasADi
        ctx = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(max_workers=args.n_workers, mp_context=ctx) as pool:
            futures = {pool.submit(_run_one_cell, p): p for p in plans}
            with tqdm(total=len(futures), **tqdm_kwargs) as pbar:
                for fut in as_completed(futures):
                    result = fut.result()
                    log_rows.append(result)
                    _print_cell_summary(result)
                    _maybe_warn_memory()
                    pbar.update(1)

    total_elapsed = time.time() - t_start

    # ---- Persist log + metadata + combined CSV -----------------------------
    log_df = pd.DataFrame(log_rows).sort_values("cell_idx")
    log_df.to_csv(output_dir / "regeneration_log.csv", index=False)

    # Treat both freshly-completed and resumed-skip as successful — the
    # underlying CSV is valid in either case.
    n_ok = log_df["status"].isin(["ok", "skipped"]).sum()
    n_skipped = (log_df["status"] == "skipped").sum()
    n_fresh = (log_df["status"] == "ok").sum()
    n_fail = len(log_df) - n_ok
    print()
    print("=" * 72)
    print(f"Done. {n_ok}/{len(log_df)} cells valid  "
          f"({n_fresh} fresh + {n_skipped} resumed)  ·  "
          f"total wall-clock {total_elapsed/60:.1f} min")
    print("=" * 72)

    # Always rebuild the combined CSV when at least one valid cell exists —
    # this matters for resumed runs where every cell was skipped but we
    # still want a fresh combined file in case the user added new
    # per-cell CSVs by hand.
    if n_ok > 0:
        all_dfs = []
        for path in sorted(output_dir.glob("IN_SYNTH_*.csv")):
            all_dfs.append(pd.read_csv(path))
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)
            combined.to_csv(output_dir / "synthetic_indian_combined.csv", index=False)
            print(f"  combined → {_safe_rel(output_dir/'synthetic_indian_combined.csv')}  "
                  f"({len(combined):,} rows, {combined['battery_id'].nunique()} cells)")

        # Grade-coverage summary across the corpus
        coverage_cols = ["reaches_grade_A_or_lower", "reaches_grade_B_or_lower",
                         "reaches_grade_C_or_lower", "reaches_grade_D"]
        present_cols = [c for c in coverage_cols if c in log_df.columns]
        if present_cols:
            print("\nGrade coverage across the regenerated corpus:")
            for c in present_cols:
                pct = log_df[log_df["status"] == "ok"][c].mean() * 100
                print(f"  {c:35s}  {pct:5.1f}% of cells")

    # Metadata
    meta = {
        "iter": 3,
        "smoke": args.smoke,
        "n_cells_requested": args.n_cells,
        "n_cells_succeeded": int(n_ok),
        "n_cells_failed": int(n_fail),
        "n_cycles_per_cell": args.n_cycles,
        "n_workers": args.n_workers,
        "usage_protocol": args.usage_protocol,
        "use_diurnal": not args.no_diurnal,
        "termination_capacity_pct": args.termination_capacity_pct,
        "chemistries": args.chemistries,
        "ambient_profiles": args.ambient_profiles,
        "diurnal_profiles": DIURNAL_CLIMATE_PROFILES if not args.no_diurnal else None,
        "calibration_multipliers": calibration_multipliers,
        "calibration_methodology": (
            "Empirical 10× multiplier on (SEI rate, SEI diffusivity, Li plating "
            "rate, +/- electrode LAM proportional terms). Cite OKane et al. 2022 "
            "(PCCP, DOI 10.1039/D2CP00417H) as the prior; informal calibration "
            "without uncertainty quantification. For production-grade fitting, "
            "use PyBOP (https://github.com/pybop-team/PyBOP)."
        ),
        "total_wall_clock_min": round(total_elapsed / 60, 2),
        "seed": RANDOM_SEED,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nlog      → {_safe_rel(output_dir/'regeneration_log.csv')}")
    print(f"metadata → {_safe_rel(output_dir/'metadata.json')}")


if __name__ == "__main__":
    # ---- Output buffering fix (CRITICAL for live progress) -----------------
    # `python -u` only unbuffers the orchestrator. multiprocessing.spawn
    # workers are launched as fresh `python -c "..."` processes that do NOT
    # inherit -u, so their stderr is block-buffered when connected to a pipe
    # (e.g., `2>&1 | tee log.file`). Heartbeats and per-cell summaries from
    # workers can sit in libc buffers for minutes, making the run look
    # frozen. Setting PYTHONUNBUFFERED=1 in os.environ BEFORE spawning any
    # worker forces unbuffered Python in all child interpreters too — the
    # env var is inherited via fork-then-exec.
    os.environ["PYTHONUNBUFFERED"] = "1"
    # Avoid macOS multiprocessing quirks with PyBaMM + CasADi (BLAS oversubscription)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    main()
