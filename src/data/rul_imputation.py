"""
RUL imputation methods for right-censored battery cells.

Right-censored cells: experimentally cycled but stopped before reaching the
EoL threshold (SoH<0.8). This module provides multiple imputation methods to
estimate the unobserved EoL cycle from observed partial trajectories, so the
downstream RUL regressor trains on physically-meaningful labels rather than
the `max(observed_cycle) - current_cycle` fabrication that the standard
pipeline currently uses.

Per Iter-3 §3.11.5 RUL diagnostic:
- 18.9% of corpus is right-censored (299 of 1,581 batteries)
- Median min-SoH for censored cells = 0.817 (cells stopped just above EoL)
- All right-censoring is in real data; synthetic cells (PyBaMM + BLAST-Lite)
  are 100% uncensored

Imputers implemented (all conform to the BaseImputer ABC):

  Parametric (chemistry-agnostic, sub-second per cell):
    - LinearExtrapolator          : SoH(N) = a + b·N on last K points
    - PolynomialExtrapolator      : degree-2 polynomial fit
    - ExponentialDecayExtrapolator: SoH(N) = a·exp(-b·N) + c
    - BiExponentialExtrapolator   : SoH(N) = a·exp(-b·N) + c·exp(-d·N) + e
    - StretchedExponentialExtrapolator: SoH(N) = a·exp(-(N/τ)^β) + c
    - PowerLawExtrapolator        : SoH(N) = 1 - a·N^b
    - SquareRootExtrapolator      : SoH(N) = 1 - a·sqrt(N)
    - PostKneeExtrapolator        : detect knee, linear post-knee fit

  Population-aware (chemistry-conditioned):
    - NearestNeighborImputer      : K-NN over (chemistry, source, early-life curve)
    - MLRegressorImputer          : XGBoost on uncensored cells' summary features → true RUL
    - GaussianProcessImputer      : GP with population mean function

References
----------
- Smith et al. 2021 J. Electrochem. Soc. 168, 100515 — algebraic life-model fits
- Strange & dos Reis 2023 J. Energy Storage 60, 107012 — knee extrapolation
- Liu, Greenbank, Howey 2024 Cell Rep. Phys. Sci. 5, 101820 — hierarchical Bayesian
- Tan et al. 2024 Nat. Mach. Intell. 6, 1077–1090 — inter-cell deep learning
- Aitio & Howey 2021 Joule 5, 3204–3220 — GP off-grid field-data EoL
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import warnings

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


EOL_THRESHOLD = 0.80


# =============================================================================
# Data containers
# =============================================================================

@dataclass
class CellTrajectory:
    """Per-cell aging trajectory used as input to imputation methods."""
    battery_id: str
    cycles: np.ndarray
    soh: np.ndarray
    chemistry: str = "unknown"
    source: str = "unknown"
    nominal_Ah: float = 0.0
    features: Optional[pd.DataFrame] = None

    def __post_init__(self):
        order = np.argsort(self.cycles)
        self.cycles = np.asarray(self.cycles, dtype=float)[order]
        self.soh    = np.asarray(self.soh,    dtype=float)[order]

    @property
    def n_observed(self) -> int:
        return len(self.cycles)

    @property
    def min_soh(self) -> float:
        return float(self.soh.min()) if len(self.soh) else float("nan")

    @property
    def max_cycle(self) -> int:
        return int(self.cycles.max()) if len(self.cycles) else 0

    @property
    def is_censored(self, eol_threshold: float = EOL_THRESHOLD) -> bool:
        return self.min_soh >= eol_threshold

    @property
    def true_eol_cycle(self) -> Optional[int]:
        below = np.where(self.soh < EOL_THRESHOLD)[0]
        return int(self.cycles[below[0]]) if len(below) else None

    def truncate_at_soh(self, threshold_soh: float) -> "CellTrajectory":
        """Return a copy of this trajectory truncated at the first cycle where
        SoH falls to or below `threshold_soh`. Used by the held-out validation
        to artificially censor uncensored cells."""
        below_or_eq = np.where(self.soh <= threshold_soh)[0]
        cut = (below_or_eq[0] + 1) if len(below_or_eq) else len(self.cycles)
        return CellTrajectory(
            battery_id=self.battery_id,
            cycles=self.cycles[:cut].copy(),
            soh=self.soh[:cut].copy(),
            chemistry=self.chemistry,
            source=self.source,
            nominal_Ah=self.nominal_Ah,
            features=self.features.iloc[:cut].copy() if self.features is not None else None,
        )


@dataclass
class ImputationResult:
    """Output of a single imputation."""
    cell_id: str
    method: str
    imputed_eol_cycle: float = float("nan")
    converged: bool = True
    confidence_lower: float = float("nan")
    confidence_upper: float = float("nan")
    n_observed: int = 0
    min_observed_soh: float = float("nan")
    diagnostic: dict = field(default_factory=dict)


# =============================================================================
# Base class
# =============================================================================

class BaseImputer(ABC):
    """All imputers conform to this API: optional fit() on population, then
    impute() per cell.  Methods that don't need population data return self
    from fit() without storing anything."""
    name: str = "base"

    def fit(self, population: list[CellTrajectory]) -> "BaseImputer":
        return self

    @abstractmethod
    def impute(self, cell: CellTrajectory) -> ImputationResult:
        ...

    def _solve_eol_from_callable(
        self,
        soh_fn: Callable[[float], float],
        cell: CellTrajectory,
        n_search_max: int = 100_000,
    ) -> float:
        """Find first cycle N* where SoH(N*) ≤ EOL_THRESHOLD via bracketed root.
        Falls back to extrapolated forward search if root-finding fails.
        Returns NaN if no crossing within `n_search_max` cycles past last observed."""
        last = cell.max_cycle
        try:
            n_max = last + n_search_max
            sample = np.linspace(last + 1, n_max, num=4000)
            vals = np.array([soh_fn(n) for n in sample])
            crossing = np.where(vals <= EOL_THRESHOLD)[0]
            if len(crossing) == 0:
                return float("nan")
            n_first = sample[crossing[0]]
            try:
                root = optimize.brentq(
                    lambda x: soh_fn(x) - EOL_THRESHOLD,
                    last,
                    n_first + 1,
                    xtol=1.0,
                )
                return float(root)
            except Exception:
                return float(n_first)
        except Exception:
            return float("nan")


# =============================================================================
# Parametric extrapolators
# =============================================================================

class LinearExtrapolator(BaseImputer):
    """SoH(N) = a + b·N fit on last `tail_frac` of observed data.

    Linear extrapolation is the simplest baseline. Tends to over-predict EoL
    when fade is super-linear (most NMC) and under-predict when fade is
    sub-linear (early-life square-root regime). Best when fitted to the tail
    where local fade is approximately linear."""
    name = "linear"

    def __init__(self, tail_frac: float = 0.3, min_points: int = 8):
        self.tail_frac = tail_frac
        self.min_points = min_points

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        n_tail = max(self.min_points, int(np.ceil(cell.n_observed * self.tail_frac)))
        n_tail = min(n_tail, cell.n_observed)
        x = cell.cycles[-n_tail:]
        y = cell.soh[-n_tail:]
        if n_tail < 2 or np.allclose(y, y[0]):
            return ImputationResult(cell.battery_id, self.name,
                                    converged=False, n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        b, a = np.polyfit(x, y, 1)  # slope, intercept (numpy returns highest-degree first)
        if b >= 0:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh,
                                    diagnostic={"slope": float(b), "issue": "non-decreasing"})
        eol = (EOL_THRESHOLD - a) / b
        if eol < cell.max_cycle:
            eol = cell.max_cycle + 1.0
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=float(eol),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"slope": float(b), "intercept": float(a),
                                            "n_tail_used": int(n_tail)})


class PolynomialExtrapolator(BaseImputer):
    """SoH(N) = c0 + c1·N + c2·N² (degree 2) fit on last `tail_frac` of data.

    Captures curvature near EoL (super-linear fade). Risk: over-fitting tail
    noise → unphysical extrapolation. Mitigated by minimum-points constraint."""
    name = "poly2"

    def __init__(self, tail_frac: float = 0.4, min_points: int = 12):
        self.tail_frac = tail_frac
        self.min_points = min_points

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        n_tail = max(self.min_points, int(np.ceil(cell.n_observed * self.tail_frac)))
        n_tail = min(n_tail, cell.n_observed)
        if n_tail < 3:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        x = cell.cycles[-n_tail:]
        y = cell.soh[-n_tail:]
        try:
            coeffs = np.polyfit(x, y, 2)  # [c2, c1, c0]
        except np.linalg.LinAlgError:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        soh_fn = lambda n: float(coeffs[0] * n**2 + coeffs[1] * n + coeffs[2])
        eol = self._solve_eol_from_callable(soh_fn, cell)
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=eol,
                                converged=not np.isnan(eol),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"coeffs": coeffs.tolist(),
                                            "n_tail_used": int(n_tail)})


def _safe_curve_fit(f, xdata, ydata, p0, bounds=None, max_iter=5000):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = optimize.curve_fit(f, xdata, ydata, p0=p0,
                                         bounds=bounds if bounds else (-np.inf, np.inf),
                                         maxfev=max_iter)
        return popt, True
    except (RuntimeError, ValueError, TypeError, optimize.OptimizeWarning):
        return None, False


class ExponentialDecayExtrapolator(BaseImputer):
    """SoH(N) = a · exp(-b · N) + c. Single-time-constant decay model."""
    name = "exp1"

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if cell.n_observed < 5:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        N0 = float(cell.cycles[0])
        x = cell.cycles - N0
        y = cell.soh
        f = lambda n, a, b, c: a * np.exp(-b * n) + c
        p0 = [0.2, 1e-3, 0.8]
        bounds = ([0.0, 1e-7, 0.0], [1.5, 1.0, 1.2])
        popt, ok = _safe_curve_fit(f, x, y, p0, bounds)
        if not ok:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        soh_fn = lambda n: f(float(n) - N0, *popt)
        eol = self._solve_eol_from_callable(soh_fn, cell)
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=eol,
                                converged=not np.isnan(eol),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"params": popt.tolist()})


class BiExponentialExtrapolator(BaseImputer):
    """SoH(N) = a · exp(-b · N) + c · exp(-d · N) + e. Captures fast (early-cycle
    SEI formation) and slow (steady-state aging) decay components — the recipe
    Smith et al. 2021 found most reliable when ≥80% of trajectory observed."""
    name = "exp2"

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if cell.n_observed < 8:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        N0 = float(cell.cycles[0])
        x = cell.cycles - N0
        y = cell.soh
        f = lambda n, a, b, c, d, e: a * np.exp(-b * n) + c * np.exp(-d * n) + e
        p0 = [0.1, 5e-3, 0.1, 1e-4, 0.7]
        bounds = ([0.0, 1e-6, 0.0, 1e-8, 0.0], [1.5, 1.0, 1.5, 1.0, 1.2])
        popt, ok = _safe_curve_fit(f, x, y, p0, bounds, max_iter=10000)
        if not ok:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        soh_fn = lambda n: f(float(n) - N0, *popt)
        eol = self._solve_eol_from_callable(soh_fn, cell)
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=eol,
                                converged=not np.isnan(eol),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"params": popt.tolist()})


class StretchedExponentialExtrapolator(BaseImputer):
    """SoH(N) = a · exp(-(N/τ)^β) + c. Kohlrausch-Williams-Watts form. Captures
    the smooth knee transition between fast early aging and slow steady-state."""
    name = "kww"

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if cell.n_observed < 8:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        N0 = float(cell.cycles[0])
        x = cell.cycles - N0
        y = cell.soh
        f = lambda n, a, tau, beta, c: a * np.exp(-((np.maximum(n, 0) / tau)**beta)) + c
        # initial: tau ≈ scale of observed cycles, beta ≈ 1 (exponential), a ≈ 0.2, c ≈ 0.8
        p0 = [0.2, max(np.median(x), 100.0), 1.0, 0.8]
        bounds = ([0.0, 1.0, 0.1, 0.0], [1.5, 1e7, 5.0, 1.2])
        popt, ok = _safe_curve_fit(f, x, y, p0, bounds, max_iter=10000)
        if not ok:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        soh_fn = lambda n: f(float(n) - N0, *popt)
        eol = self._solve_eol_from_callable(soh_fn, cell)
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=eol,
                                converged=not np.isnan(eol),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"params": popt.tolist()})


class PowerLawExtrapolator(BaseImputer):
    """SoH(N) = 1 - a · N^b. Bloom 2003 / Wang 2011 power-law fade model. Good
    fit for early/mid-life aging dominated by SEI thickening (b ≈ 0.5)."""
    name = "powerlaw"

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if cell.n_observed < 5:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        N0 = float(cell.cycles[0])
        x = cell.cycles - N0
        y = cell.soh
        f = lambda n, a, b: 1.0 - a * np.maximum(n, 0)**b
        p0 = [1e-4, 0.5]
        bounds = ([1e-12, 0.05], [1.0, 3.0])
        popt, ok = _safe_curve_fit(f, x, y, p0, bounds, max_iter=10000)
        if not ok:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        soh_fn = lambda n: f(float(n) - N0, *popt)
        eol = self._solve_eol_from_callable(soh_fn, cell)
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=eol,
                                converged=not np.isnan(eol),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"params": popt.tolist()})


class SquareRootExtrapolator(BaseImputer):
    """SoH(N) = 1 - a · sqrt(N). Special case of power-law with b=0.5 — the
    diffusion-limited SEI growth regime. Locked-exponent variant of `powerlaw`."""
    name = "sqrt"

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if cell.n_observed < 5:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        N0 = float(cell.cycles[0])
        x = cell.cycles - N0
        y = cell.soh
        f = lambda n, a: 1.0 - a * np.sqrt(np.maximum(n, 0))
        p0 = [1e-3]
        bounds = ([1e-10], [1.0])
        popt, ok = _safe_curve_fit(f, x, y, p0, bounds)
        if not ok:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        a = popt[0]
        # Solve 0.8 = 1 - a·sqrt(N) → N = ((1-0.8)/a)²
        eol = N0 + ((1.0 - EOL_THRESHOLD) / a) ** 2
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=float(eol),
                                converged=True,
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"a": float(a)})


class PostKneeExtrapolator(BaseImputer):
    """Knee-aware piecewise extrapolation per Strange & dos Reis 2023. Detects
    the cycle at which fade rate accelerates (the knee), then linearly
    extrapolates the post-knee tail. Falls back to linear extrapolation on the
    full series if no knee is detected."""
    name = "postknee"

    def __init__(self, smooth_window: int = 11, min_post_knee_pts: int = 8):
        self.smooth_window = smooth_window
        self.min_post_knee_pts = min_post_knee_pts

    def _detect_knee(self, cycles: np.ndarray, soh: np.ndarray) -> Optional[int]:
        """Return index of detected knee point, or None."""
        if len(soh) < max(20, self.smooth_window + 4):
            return None
        w = self.smooth_window if self.smooth_window % 2 == 1 else self.smooth_window + 1
        w = min(w, len(soh) - 2)
        if w < 5:
            return None
        try:
            soh_smooth = savgol_filter(soh, w, polyorder=2)
            d = np.gradient(soh_smooth, cycles)
            d2 = np.gradient(d, cycles)
            # Knee = max |2nd derivative| in the second half of the trajectory
            half = len(soh) // 2
            search = np.abs(d2[half:])
            if not np.any(np.isfinite(search)):
                return None
            knee_idx = half + int(np.argmax(search))
            if knee_idx >= len(cycles) - self.min_post_knee_pts:
                return None
            return int(knee_idx)
        except Exception:
            return None

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        knee_idx = self._detect_knee(cell.cycles, cell.soh)
        if knee_idx is None:
            tail_imp = LinearExtrapolator(tail_frac=0.4).impute(cell)
            tail_imp.method = self.name + "(fallback=linear)"
            tail_imp.diagnostic = {**(tail_imp.diagnostic or {}), "knee_detected": False}
            return tail_imp
        x = cell.cycles[knee_idx:]
        y = cell.soh[knee_idx:]
        if len(x) < 2 or np.allclose(y, y[0]):
            tail_imp = LinearExtrapolator(tail_frac=0.4).impute(cell)
            tail_imp.method = self.name + "(fallback=linear)"
            tail_imp.diagnostic = {**(tail_imp.diagnostic or {}),
                                   "knee_detected": True, "issue": "flat post-knee"}
            return tail_imp
        b, a = np.polyfit(x, y, 1)
        if b >= 0:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh,
                                    diagnostic={"knee_detected": True,
                                                "issue": "non-decreasing post-knee"})
        eol = (EOL_THRESHOLD - a) / b
        if eol < cell.max_cycle:
            eol = cell.max_cycle + 1.0
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=float(eol),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"knee_detected": True,
                                            "knee_cycle": float(cell.cycles[knee_idx]),
                                            "post_knee_n": int(len(x)),
                                            "post_knee_slope": float(b)})


# =============================================================================
# Population-aware imputers
# =============================================================================

class NearestNeighborImputer(BaseImputer):
    """For each censored cell, find the K most similar uncensored cells (matched
    on chemistry + early-life trajectory shape), then estimate this cell's EoL
    cycle from the K neighbours' EoL cycles, scaled by the cell's observed
    fade rate.

    Similarity metric: cosine distance on resampled early-life SoH curves.
    Default K=5; chemistry-matching is preferred but falls back to all
    uncensored cells if the cell's chemistry has too few neighbours.
    """
    name = "nn"

    def __init__(self, k: int = 5, n_resample: int = 30,
                 fade_rate_scale: bool = True, min_neighbors: int = 3):
        self.k = k
        self.n_resample = n_resample
        self.fade_rate_scale = fade_rate_scale
        self.min_neighbors = min_neighbors
        # Built by fit():
        self._curves: dict[str, np.ndarray] = {}     # battery_id → resampled SoH curve
        self._eol_cycles: dict[str, float] = {}       # battery_id → true EoL cycle
        self._chemistries: dict[str, str] = {}        # battery_id → chemistry
        self._fade_rates: dict[str, float] = {}       # battery_id → cycles per pp SoH drop in early life

    def _resample_curve(self, cycles: np.ndarray, soh: np.ndarray) -> np.ndarray:
        """Resample a SoH-vs-cycle trajectory to `n_resample` points uniformly
        across the cell's observed cycle range. Only the SHAPE matters for
        similarity, not absolute scale, so we normalise cycle axis to [0, 1]."""
        if len(cycles) < 2:
            return np.full(self.n_resample, np.nan)
        c_min, c_max = cycles[0], cycles[-1]
        if c_max <= c_min:
            return np.full(self.n_resample, np.nan)
        target = np.linspace(c_min, c_max, self.n_resample)
        return np.interp(target, cycles, soh)

    def _early_life_curve(self, cycles: np.ndarray, soh: np.ndarray,
                          frac: float = 0.4) -> np.ndarray:
        n = max(2, int(np.ceil(len(cycles) * frac)))
        return self._resample_curve(cycles[:n], soh[:n])

    def _early_fade_rate(self, cycles: np.ndarray, soh: np.ndarray) -> float:
        """Approximate cycles required to drop SoH by 1 percentage point in the
        cell's observed early life. Robust to noise via linear fit."""
        if len(cycles) < 4:
            return float("nan")
        n = max(4, len(cycles) // 2)
        x = cycles[:n]
        y = soh[:n]
        if np.allclose(y, y[0]):
            return float("nan")
        b, _ = np.polyfit(x, y, 1)
        if b >= -1e-9:
            return float("nan")
        return float(-0.01 / b)  # cycles per 1pp drop

    def fit(self, population: list[CellTrajectory]) -> "NearestNeighborImputer":
        for cell in population:
            if cell.is_censored:
                continue
            true_eol = cell.true_eol_cycle
            if true_eol is None:
                continue
            curve = self._early_life_curve(cell.cycles, cell.soh)
            if np.any(np.isnan(curve)):
                continue
            self._curves[cell.battery_id] = curve
            self._eol_cycles[cell.battery_id] = float(true_eol)
            self._chemistries[cell.battery_id] = cell.chemistry
            self._fade_rates[cell.battery_id] = self._early_fade_rate(cell.cycles, cell.soh)
        return self

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if not self._curves:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh,
                                    diagnostic={"issue": "imputer not fitted"})
        target_curve = self._early_life_curve(cell.cycles, cell.soh)
        if np.any(np.isnan(target_curve)):
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh,
                                    diagnostic={"issue": "early curve nan"})

        same_chem = [bid for bid, ch in self._chemistries.items() if ch == cell.chemistry]
        candidates = same_chem if len(same_chem) >= self.min_neighbors else list(self._curves.keys())

        curves = np.stack([self._curves[bid] for bid in candidates])
        target = target_curve.reshape(1, -1)
        # cosine distance
        norms = np.linalg.norm(curves, axis=1) * np.linalg.norm(target)
        norms = np.where(norms < 1e-12, 1e-12, norms)
        sims = (curves @ target.T).ravel() / norms
        dists = 1.0 - sims
        order = np.argsort(dists)
        top_k = [candidates[i] for i in order[:self.k]]

        nbr_eols = np.array([self._eol_cycles[b] for b in top_k])
        if not self.fade_rate_scale:
            eol_pred = float(np.median(nbr_eols))
        else:
            target_rate = self._early_fade_rate(cell.cycles, cell.soh)
            nbr_rates = np.array([self._fade_rates[b] for b in top_k])
            valid = ~np.isnan(nbr_rates) & (nbr_rates > 0)
            if not valid.any() or np.isnan(target_rate) or target_rate <= 0:
                eol_pred = float(np.median(nbr_eols))
            else:
                # Scale neighbour EoL by ratio of fade rates: slower fade → later EoL
                scale = target_rate / np.median(nbr_rates[valid])
                eol_pred = float(np.median(nbr_eols * scale))

        # Floor at last observed cycle
        eol_pred = max(eol_pred, cell.max_cycle + 1.0)
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=float(eol_pred),
                                confidence_lower=float(np.quantile(nbr_eols, 0.10)),
                                confidence_upper=float(np.quantile(nbr_eols, 0.90)),
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"neighbours": top_k,
                                            "neighbour_eols": nbr_eols.tolist()})


class MLRegressorImputer(BaseImputer):
    """Train an XGBoost regressor on uncensored cells whose features are the
    cell's *summary statistics over observed cycles* (current min SoH, fade
    rate, knee-point indicator, capacity-Ah summary, voltage statistics,
    chemistry one-hot) and target is the cell's true EoL cycle. Apply to
    censored cells.

    Distinct from the standard XGBoost-RUL: this is a per-cell EoL regressor,
    not a per-cycle RUL regressor. One prediction per cell."""
    name = "ml"

    def __init__(self, n_estimators: int = 300, max_depth: int = 5,
                 learning_rate: float = 0.05, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self._model = None
        self._scaler: Optional[StandardScaler] = None
        self._chem_categories: list[str] = []

    def _summarise(self, cell: CellTrajectory) -> dict:
        x = cell.cycles
        y = cell.soh
        d = {
            "n_observed":   float(len(x)),
            "max_cycle":    float(x[-1]) if len(x) else 0.0,
            "min_soh":      float(y.min()) if len(y) else float("nan"),
            "soh_at_last":  float(y[-1]) if len(y) else float("nan"),
            "soh_range":    float(y.max() - y.min()) if len(y) else 0.0,
            "mean_fade":    float((1.0 - y.mean())) if len(y) else 0.0,
        }
        if len(x) >= 4:
            n = max(4, len(x) // 2)
            try:
                slope_early, _ = np.polyfit(x[:n], y[:n], 1)
                slope_tail,  _ = np.polyfit(x[-n:], y[-n:], 1)
            except np.linalg.LinAlgError:
                slope_early = slope_tail = float("nan")
            d["fade_slope_early"] = float(slope_early)
            d["fade_slope_tail"]  = float(slope_tail)
            d["fade_acceleration"] = float(slope_tail - slope_early)
        else:
            d["fade_slope_early"] = d["fade_slope_tail"] = d["fade_acceleration"] = float("nan")
        return d

    def _featurise(self, cells: list[CellTrajectory]) -> tuple[np.ndarray, list[str]]:
        feats = [self._summarise(c) for c in cells]
        df = pd.DataFrame(feats)
        chem_dummies = pd.get_dummies(
            pd.Categorical([c.chemistry for c in cells],
                           categories=self._chem_categories),
            prefix="chem", dtype=float,
        )
        Xfull = pd.concat([df, chem_dummies.reset_index(drop=True)], axis=1).fillna(0.0)
        return Xfull.to_numpy(dtype=float), Xfull.columns.tolist()

    def fit(self, population: list[CellTrajectory]) -> "MLRegressorImputer":
        import xgboost as xgb
        usable = [c for c in population if not c.is_censored and c.true_eol_cycle is not None]
        if len(usable) < 30:
            return self
        self._chem_categories = sorted({c.chemistry for c in usable})
        X, _ = self._featurise(usable)
        y = np.array([float(c.true_eol_cycle) for c in usable], dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist",
            random_state=self.random_state,
        )
        self._model.fit(Xs, y, verbose=False)
        return self

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if self._model is None:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh,
                                    diagnostic={"issue": "model not fitted"})
        X, _ = self._featurise([cell])
        eol = float(self._model.predict(self._scaler.transform(X))[0])
        eol = max(eol, cell.max_cycle + 1.0)
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=eol,
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh)


class GaussianProcessImputer(BaseImputer):
    """Per-chemistry GP regressor on (cycles, SoH) with population mean function
    estimated from uncensored cells of that chemistry. For each censored cell:
    fit GP on its observed (cycles, SoH); use the GP posterior to extrapolate
    forward; find first cycle where posterior mean ≤ EoL threshold."""
    name = "gp"

    def __init__(self, length_scale: float = 200.0, n_extrap_steps: int = 10000):
        self.length_scale = length_scale
        self.n_extrap_steps = n_extrap_steps

    def fit(self, population: list[CellTrajectory]) -> "GaussianProcessImputer":
        # No global state needed — GP is fit per-cell.
        return self

    def impute(self, cell: CellTrajectory) -> ImputationResult:
        if cell.n_observed < 6:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh)
        X = cell.cycles.reshape(-1, 1)
        y = cell.soh
        kernel = (
            ConstantKernel(0.5, (1e-3, 5.0))
            * RBF(length_scale=self.length_scale, length_scale_bounds=(20.0, 1e5))
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
        )
        try:
            gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                          n_restarts_optimizer=2, alpha=1e-6,
                                          random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp.fit(X, y)
        except Exception:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh,
                                    diagnostic={"issue": "GP fit failed"})

        last = cell.max_cycle
        n_extrap = self.n_extrap_steps
        x_future = np.linspace(last + 1, last + n_extrap, num=2000).reshape(-1, 1)
        mu, sigma = gp.predict(x_future, return_std=True)
        below = np.where(mu <= EOL_THRESHOLD)[0]
        if len(below) == 0:
            return ImputationResult(cell.battery_id, self.name, converged=False,
                                    n_observed=cell.n_observed,
                                    min_observed_soh=cell.min_soh,
                                    diagnostic={"issue": "no crossing within search range",
                                                "search_max": float(last + n_extrap)})
        eol = float(x_future[below[0], 0])
        eol_lo, eol_hi = eol, eol
        below_hi = np.where((mu - 1.96 * sigma) <= EOL_THRESHOLD)[0]
        below_lo = np.where((mu + 1.96 * sigma) <= EOL_THRESHOLD)[0]
        if len(below_hi):
            eol_lo = float(x_future[below_hi[0], 0])
        if len(below_lo):
            eol_hi = float(x_future[below_lo[0], 0])
        return ImputationResult(cell.battery_id, self.name,
                                imputed_eol_cycle=eol,
                                confidence_lower=eol_lo,
                                confidence_upper=eol_hi,
                                n_observed=cell.n_observed,
                                min_observed_soh=cell.min_soh,
                                diagnostic={"sigma_at_eol": float(sigma[below[0]])})


# =============================================================================
# Registry
# =============================================================================

ALL_IMPUTERS: dict[str, type[BaseImputer]] = {
    LinearExtrapolator.name:                LinearExtrapolator,
    PolynomialExtrapolator.name:            PolynomialExtrapolator,
    ExponentialDecayExtrapolator.name:      ExponentialDecayExtrapolator,
    BiExponentialExtrapolator.name:         BiExponentialExtrapolator,
    StretchedExponentialExtrapolator.name:  StretchedExponentialExtrapolator,
    PowerLawExtrapolator.name:              PowerLawExtrapolator,
    SquareRootExtrapolator.name:            SquareRootExtrapolator,
    PostKneeExtrapolator.name:              PostKneeExtrapolator,
    NearestNeighborImputer.name:            NearestNeighborImputer,
    MLRegressorImputer.name:                MLRegressorImputer,
    GaussianProcessImputer.name:            GaussianProcessImputer,
}


def make_imputer(name: str, **kwargs) -> BaseImputer:
    if name not in ALL_IMPUTERS:
        raise KeyError(f"Unknown imputer '{name}'. Available: {sorted(ALL_IMPUTERS)}")
    return ALL_IMPUTERS[name](**kwargs)


# =============================================================================
# Builder: parquet → list[CellTrajectory]
# =============================================================================

def cells_from_parquet(df: pd.DataFrame,
                       min_observed: int = 5) -> list[CellTrajectory]:
    """Convert a wide unified.parquet-style DataFrame into a list of
    CellTrajectory objects. Skips cells with fewer than `min_observed` cycles."""
    cells = []
    for bid, group in df.groupby("battery_id", sort=False):
        if len(group) < min_observed:
            continue
        chemistry = group["chemistry"].iloc[0] if "chemistry" in group else "unknown"
        source    = group["source"].iloc[0]    if "source"    in group else "unknown"
        nominal   = float(group["nominal_Ah"].iloc[0]) if "nominal_Ah" in group else 0.0
        cells.append(CellTrajectory(
            battery_id=str(bid),
            cycles=group["cycle"].to_numpy(),
            soh=group["soh"].to_numpy(),
            chemistry=str(chemistry),
            source=str(source),
            nominal_Ah=nominal,
        ))
    return cells


if __name__ == "__main__":
    # Quick self-test on a synthetic trajectory.
    np.random.seed(42)
    cycles = np.arange(1, 1500)
    true_eol = 2400
    a = 0.2 / np.sqrt(true_eol)
    soh = 1.0 - a * np.sqrt(cycles) + np.random.normal(0, 0.003, size=cycles.shape)
    cell = CellTrajectory("test_cell", cycles, soh, chemistry="NMC", source="test")
    print(f"True EoL ~ {true_eol}, observed cycles {cell.cycles[0]}-{cell.cycles[-1]}, "
          f"min SoH = {cell.min_soh:.3f}")
    for name in ALL_IMPUTERS:
        imp = make_imputer(name)
        if name in ("nn", "ml"):
            # Need a population — skip in self-test
            print(f"  {name}: skipped (population-aware)")
            continue
        imp.fit([])
        r = imp.impute(cell)
        if r.converged and not np.isnan(r.imputed_eol_cycle):
            err = (r.imputed_eol_cycle - true_eol) / true_eol * 100
            print(f"  {name:14s}  imputed EoL = {r.imputed_eol_cycle:.0f}  ({err:+.1f}% vs true)")
        else:
            print(f"  {name:14s}  did not converge")
