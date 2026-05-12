"""Closed-form efficient-coding solver.

Given signal power C(f,ν), input-noise power S(f,ν), output-noise power N(f,ν),
and response-power budget P0, solve for

    g(f,ν) = |v(f,ν)|².

The information density is

    log((g(C+S)+N)/(gS+N))

and the budget is

    sum g(C+S) weights = P0.

The pointwise KKT solution is analytic.  A scalar Lagrange multiplier λ is found
by bisection so that the total budget is exactly spent.
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq

from .noise import evaluate_noise, WhiteNoise
from .plotting import radial_weights, band_mask_radial


@dataclass
class Result:
    spectrum: object
    f: np.ndarray
    tf_hz: np.ndarray
    C: np.ndarray
    input_noise_power: np.ndarray
    output_noise_power: np.ndarray
    v_sq: np.ndarray
    lam: float
    I: float
    P0: float


def _as_power_array(x, shape, name):
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr), dtype=float)
    else:
        arr = np.broadcast_to(arr, shape).astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def optimal_filter_squared_magnitude(C, S, N, lam, band_mask=None):
    """Return optimal |v|² for a fixed λ."""
    C = np.asarray(C, dtype=float)
    S = _as_power_array(S, C.shape, "input noise S")
    N = _as_power_array(N, C.shape, "output noise N")
    lam = float(lam)

    if np.any(C < 0) or np.any(S < 0) or np.any(N <= 0):
        raise ValueError("C>=0, S>=0, and N>0 are required")

    active = C > 0
    if band_mask is not None:
        active &= np.asarray(band_mask, dtype=bool)

    g = np.zeros_like(C)
    if not np.any(active):
        return g

    Cg = C[active]
    Sg = S[active]
    Ng = N[active]

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Algebraically stable form of the KKT solution.
        x = 4.0 * Sg / (lam * Ng * Cg)
        bracket = (2.0 / (lam * Ng)) / (np.sqrt(1.0 + x) + 1.0) - 1.0
        g_active = (Ng / (Cg + Sg)) * np.maximum(bracket, 0.0)

    g[active] = np.where(np.isfinite(g_active), np.maximum(g_active, 0.0), 0.0)
    return g


def response_power_spend(C, v_sq, weights, input_noise):
    """sum |v|² (C+S) weights."""
    C = np.asarray(C, dtype=float)
    S = _as_power_array(input_noise, C.shape, "input_noise")
    return float(np.sum(np.asarray(v_sq, dtype=float) * (C + S) * np.asarray(weights, dtype=float)))


def mutual_information_density(C, v_sq, input_noise, output_noise):
    C = np.asarray(C, dtype=float)
    S = _as_power_array(input_noise, C.shape, "input_noise")
    N = _as_power_array(output_noise, C.shape, "output_noise")
    g = np.asarray(v_sq, dtype=float)
    return np.log((g * (C + S) + N) / (g * S + N))


def mutual_information(C, v_sq, weights, input_noise, output_noise):
    return float(np.sum(mutual_information_density(C, v_sq, input_noise, output_noise) * weights))


def find_lambda(C, S, N, weights, P0, band_mask=None):
    """Find λ by solving spend(λ)=P0.  Spend decreases monotonically in λ."""
    C = np.asarray(C, dtype=float)
    S = _as_power_array(S, C.shape, "input noise S")
    N = _as_power_array(N, C.shape, "output noise N")
    weights = np.asarray(weights, dtype=float)
    P0 = float(P0)

    if P0 <= 0:
        return 1e12

    def spend(lam):
        g = optimal_filter_squared_magnitude(C, S, N, lam, band_mask=band_mask)
        return np.sum(g * (C + S) * weights)

    def objective(lam):
        return spend(lam) - P0

    lo = 1e-14
    hi = 1e14
    while objective(lo) < 0 and lo > 1e-300:
        lo *= 0.01
    while objective(hi) > 0 and hi < 1e300:
        hi *= 100.0

    if objective(lo) <= 0:
        return lo
    if objective(hi) >= 0:
        return hi
    return float(brentq(objective, lo, hi, xtol=1e-13, rtol=1e-11, maxiter=300))


def solve_efficient_coding(C, input_noise, output_noise, weights, P0, band_mask=None):
    """Solve the efficient-coding problem for arrays C, S, N."""
    C = np.asarray(C, dtype=float)
    S = _as_power_array(input_noise, C.shape, "input_noise")
    N = _as_power_array(output_noise, C.shape, "output_noise")
    weights = np.asarray(weights, dtype=float)
    lam = find_lambda(C, S, N, weights, P0, band_mask=band_mask)
    v_sq = optimal_filter_squared_magnitude(C, S, N, lam, band_mask=band_mask)
    I = mutual_information(C, v_sq, weights, S, N)
    return v_sq, lam, I


def solve_on_grid(
    spectrum,
    f,
    tf_hz,
    *,
    P0=1.0,
    input_noise=None,
    output_noise=None,
    sigma_in=0.0,
    sigma_out=1.0,
    band=None,
):
    """Convenience wrapper: spectrum object -> Result.

    ``input_noise`` and ``output_noise`` may be Noise objects, scalars, arrays, or
    None.  If None, ``sigma_in``/``sigma_out`` are interpreted as white-noise
    standard deviations.
    """
    f = np.asarray(f, dtype=float)
    tf_hz = np.asarray(tf_hz, dtype=float)
    C = spectrum.C(f, tf_hz)
    S = evaluate_noise(input_noise, f, tf_hz, default_sigma=sigma_in)
    N = evaluate_noise(output_noise, f, tf_hz, default_sigma=sigma_out)
    weights = radial_weights(f, tf_hz)
    if band is None:
        band_mask = np.ones_like(C, dtype=bool)
    else:
        f_max, tf_min_hz, tf_max_hz = band
        band_mask = band_mask_radial(f, tf_hz, f_max, tf_min_hz, tf_max_hz)
    weights = weights * band_mask
    v_sq, lam, I = solve_efficient_coding(C, S, N, weights, P0, band_mask=band_mask)
    return Result(spectrum, f, tf_hz, C, S, N, v_sq, lam, I, float(P0))
