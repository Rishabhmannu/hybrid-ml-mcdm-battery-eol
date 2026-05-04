"""
Per-cycle dQ/dV peak feature extraction.

Computes Severson-2019-style protocol-invariant features from within-cycle
voltage and capacity waveforms. The motivation (per Roman 2021 Nat. Mach.
Intel., Greenbank & Howey 2022 IEEE T-II) is that raw cycle statistics
(`v_min`, `charge_time_s`, etc.) encode source-specific cycling-protocol
fingerprints. dQ/dV peak *position* and *height* are properties of the
cathode chemistry's intercalation thermodynamics and transfer cleanly
across labs/protocols; peak width carries kinetic information that does
shift with C-rate but in a chemistry-relative way.

Per-cycle features emitted:
  v_peak_dqdv_charge      : voltage at max dQ/dV during charge phase (V)
  dqdv_peak_height_charge : value of dQ/dV at that peak (Ah/V)
  dqdv_peak_width_charge  : full-width-at-half-maximum of charge peak (V)
  v_peak_dqdv_discharge   : same, for discharge phase
  dqdv_peak_height_discharge
  dqdv_peak_width_discharge
  q_at_v_lo, q_at_v_hi    : capacity at chemistry-relative low/high voltage points
                            (computed only when V band sufficient; NaN otherwise)

All features default to NaN if the input waveform is too short / non-monotonic
/ missing — the downstream training pipeline already handles NaN inputs.
"""
from __future__ import annotations

import numpy as np


# Default voltage band for spot-capacity features (Li-ion 18650 typical operating range).
# For other chemistries the function falls back to chemistry-relative percentiles.
LI_ION_V_LO = 3.5
LI_ION_V_HI = 3.9


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    """Centered moving average — smooths sensor noise before differentiation."""
    if y.size < window:
        return y
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.concatenate([np.full(pad, y[0]), y, np.full(pad, y[-1])])
    return np.convolve(padded, kernel, mode="valid")[: y.size]


def _peak_features(v: np.ndarray, q: np.ndarray) -> dict:
    """Compute (voltage of max dQ/dV, peak height, FWHM) for a single phase.

    `v` and `q` must have the same length and be sorted by V (monotonically
    increasing for charge or decreasing for discharge — caller's responsibility).
    Returns NaNs if data are insufficient.
    """
    nan_result = {"v_peak": float("nan"),
                  "peak_height": float("nan"),
                  "peak_width": float("nan")}
    if v.size < 10 or q.size != v.size:
        return nan_result

    # Sort by V monotonically increasing (handles either phase).
    order = np.argsort(v)
    v_s = v[order]
    q_s = q[order]
    # Drop duplicate V values (np.gradient needs strictly increasing x).
    keep = np.concatenate([[True], np.diff(v_s) > 1e-6])
    v_s = v_s[keep]
    q_s = q_s[keep]
    if v_s.size < 10:
        return nan_result

    q_smooth = _smooth(q_s, window=5)
    try:
        dqdv = np.gradient(q_smooth, v_s)
    except Exception:
        return nan_result
    dqdv_abs = np.abs(dqdv)
    if not np.isfinite(dqdv_abs).any():
        return nan_result

    idx = int(np.argmax(dqdv_abs))
    peak_height = float(dqdv_abs[idx])
    v_peak = float(v_s[idx])

    # Full-width at half-maximum: voltage span where |dQ/dV| ≥ peak/2.
    half = peak_height / 2.0
    above = dqdv_abs >= half
    if above.any():
        v_above = v_s[above]
        peak_width = float(v_above.max() - v_above.min())
    else:
        peak_width = float("nan")

    return {"v_peak": v_peak,
            "peak_height": peak_height,
            "peak_width": peak_width}


def _q_at_voltage(v: np.ndarray, q: np.ndarray, v_target: float) -> float:
    """Linearly interpolate capacity at a target voltage. NaN if out of range."""
    if v.size < 2:
        return float("nan")
    order = np.argsort(v)
    v_s = v[order]; q_s = q[order]
    if v_target < v_s[0] or v_target > v_s[-1]:
        return float("nan")
    return float(np.interp(v_target, v_s, q_s))


def _phase_indices(q: np.ndarray, min_run: int = 10) -> np.ndarray | None:
    """Identify the contiguous index range during which Q is monotonically
    increasing (the active phase — charge for `qc`, discharge for `qd`).

    Robust to BatteryLife's per-cell sign convention drift: rather than
    relying on the sign of the current, we infer the phase from which Q
    field is actually accumulating.

    Returns the index slice as a 1-D boolean mask, or None if no run of
    at least `min_run` increasing samples is found.
    """
    if q.size < min_run:
        return None
    # Smooth out single-sample noise
    dq = np.diff(q, prepend=q[0])
    increasing = dq > 1e-6
    if increasing.sum() < min_run:
        return None
    return increasing


def compute_dqdv_features(
    voltage: np.ndarray,
    current: np.ndarray,
    charge_capacity: np.ndarray,
    discharge_capacity: np.ndarray,
    *,
    v_lo: float = LI_ION_V_LO,
    v_hi: float = LI_ION_V_HI,
) -> dict:
    """Extract per-cycle dQ/dV peak features.

    Phase identification is convention-agnostic: the charge phase is the
    range of samples where `charge_capacity` is monotonically increasing,
    and the discharge phase is where `discharge_capacity` is monotonically
    increasing. This is robust to per-cell current-sign convention drift
    in BatteryLife and other corpora.

    All inputs are 1-D ndarrays of the same length representing the
    within-cycle samples. Returns a dict of 8 features (NaN where data
    are insufficient).
    """
    out = {
        "v_peak_dqdv_charge": float("nan"),
        "dqdv_peak_height_charge": float("nan"),
        "dqdv_peak_width_charge": float("nan"),
        "v_peak_dqdv_discharge": float("nan"),
        "dqdv_peak_height_discharge": float("nan"),
        "dqdv_peak_width_discharge": float("nan"),
        "q_at_v_lo": float("nan"),
        "q_at_v_hi": float("nan"),
    }
    voltage = np.asarray(voltage, dtype=float)
    current = np.asarray(current, dtype=float)  # currently unused but kept for future
    qc = np.asarray(charge_capacity, dtype=float)
    qd = np.asarray(discharge_capacity, dtype=float)
    if voltage.size < 10 or current.size != voltage.size:
        return out

    if qc.size != voltage.size:
        qc = np.full(voltage.size, np.nan)
    if qd.size != voltage.size:
        qd = np.full(voltage.size, np.nan)

    # CHARGE phase = where qc is increasing
    charge_mask = _phase_indices(qc)
    if charge_mask is not None and charge_mask.sum() >= 10:
        v_c = voltage[charge_mask]
        q_c = qc[charge_mask]
        ok = np.isfinite(v_c) & np.isfinite(q_c)
        if ok.sum() >= 10:
            feats_c = _peak_features(v_c[ok], q_c[ok])
            out["v_peak_dqdv_charge"] = feats_c["v_peak"]
            out["dqdv_peak_height_charge"] = feats_c["peak_height"]
            out["dqdv_peak_width_charge"] = feats_c["peak_width"]
            out["q_at_v_lo"] = _q_at_voltage(v_c[ok], q_c[ok], v_lo)
            out["q_at_v_hi"] = _q_at_voltage(v_c[ok], q_c[ok], v_hi)

    # DISCHARGE phase = where qd is increasing
    discharge_mask = _phase_indices(qd)
    if discharge_mask is not None and discharge_mask.sum() >= 10:
        v_d = voltage[discharge_mask]
        q_d = qd[discharge_mask]
        ok = np.isfinite(v_d) & np.isfinite(q_d)
        if ok.sum() >= 10:
            feats_d = _peak_features(v_d[ok], q_d[ok])
            out["v_peak_dqdv_discharge"] = feats_d["v_peak"]
            out["dqdv_peak_height_discharge"] = feats_d["peak_height"]
            out["dqdv_peak_width_discharge"] = feats_d["peak_width"]

    return out


# Column list for callers that need to splice these into a unified schema.
DQDV_COLUMNS = [
    "v_peak_dqdv_charge",
    "dqdv_peak_height_charge",
    "dqdv_peak_width_charge",
    "v_peak_dqdv_discharge",
    "dqdv_peak_height_discharge",
    "dqdv_peak_width_discharge",
    "q_at_v_lo",
    "q_at_v_hi",
]
