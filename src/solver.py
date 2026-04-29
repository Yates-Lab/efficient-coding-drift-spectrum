"""Linsker/Jun optimal-filter solver and mutual-information evaluation.

Solves the constrained optimization (eq. 20 in the appendix):

    max_{|v|^2 >= 0}  ∫_B Iθ(v; k, ω) dμ
    subject to        ∫_B |v|^2 (Cθ + σ_in^2) dμ = P_0

where dμ = d²k dω / (2π)^3 (or, for radial integrals, f df dω / (2π)^2),
and B is the band ΩNyq × Ωt.

Closed-form solution (eq. 23, numerically stabilized):

    |v*|^2 = σ_out^2 / (C + σ_in^2) * [ 2/(λ σ_out^2) / (sqrt(1+x)+1) - 1 ]_+
    x = 4 σ_in^2 / (λ σ_out^2 C)

with λ chosen by bisection so that the budget constraint is satisfied.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


__all__ = [
    "optimal_filter_squared_magnitude",
    "find_lambda",
    "mutual_information_density",
    "mutual_information",
    "solve_efficient_coding",
    "active_threshold_C",
]


# ---------------------------------------------------------------------------
# Closed-form filter solution
# ---------------------------------------------------------------------------

def optimal_filter_squared_magnitude(C, sigma_in, sigma_out, lam, band_mask=None):
    """Optimal |v(k,ω)|^2 (eq. 23) given Lagrange multiplier λ.

    Parameters
    ----------
    C : ndarray
        Input power spectrum Cθ(k, ω) on the analysis grid.
    sigma_in, sigma_out : float
        Input and output noise standard deviations.
    lam : float
        Lagrange multiplier. Smaller λ => more total response power.
    band_mask : ndarray of bool, optional
        Mask of frequencies inside the band B. Outside the band, |v|^2 is set to 0.

    Returns
    -------
    v_sq : ndarray, same shape as C
    """
    C = np.asarray(C, dtype=float)
    s_in2 = float(sigma_in) ** 2
    s_out2 = float(sigma_out) ** 2

    v_sq = np.zeros_like(C)

    # Only frequencies with strictly positive power can receive gain.
    if band_mask is None:
        active_grid = C > 0
    else:
        active_grid = (C > 0) & band_mask

    if not np.any(active_grid):
        return v_sq

    Cg = C[active_grid]

    if s_in2 == 0.0:
        # Water-filling: p*C = (1/λ - σ_out²)_+
        spend = np.maximum(1.0 / lam - s_out2, 0.0)
        v_sq_active = np.where(Cg > 0, spend / Cg, 0.0)
    else:
        # Stable form using sqrt(1+x)+1 in denominator.
        x = 4.0 * s_in2 / (lam * s_out2 * Cg)
        sqrt_term = np.sqrt(1.0 + x)
        bracket = (2.0 / (lam * s_out2)) / (sqrt_term + 1.0) - 1.0
        v_sq_active = (s_out2 / (Cg + s_in2)) * np.maximum(bracket, 0.0)

    v_sq[active_grid] = v_sq_active
    return v_sq


def active_threshold_C(sigma_in, sigma_out, lam):
    """Threshold C below which |v*|^2 = 0 (sigma_in > 0 case).

    From bracket > 0:  2/(λ σ_out²) > sqrt(1+x)+1  =>  x < (2/(λ σ_out²) - 1)^2 - 1
    => C > σ_in^2 λ σ_out² / (1 - λ σ_out²)
    """
    if sigma_in == 0:
        return 0.0
    s_out2 = sigma_out ** 2
    if lam * s_out2 >= 1.0:
        return np.inf  # No frequency is active
    return sigma_in ** 2 * lam * s_out2 / (1.0 - lam * s_out2)


# ---------------------------------------------------------------------------
# Lambda from budget
# ---------------------------------------------------------------------------

def _budget_spend(C, sigma_in, sigma_out, lam, weights, band_mask=None):
    v_sq = optimal_filter_squared_magnitude(C, sigma_in, sigma_out, lam, band_mask)
    return float(np.sum(v_sq * (C + sigma_in ** 2) * weights))


def find_lambda(C, sigma_in, sigma_out, P0, weights, band_mask=None,
                lam_lo=1e-12, lam_hi=1e12, xtol=1e-14, rtol=1e-12):
    """Solve for λ such that ∫ |v*|^2 (C + σ_in^2) dμ = P0.

    The budget spend is monotonically decreasing in λ, so we use bisection
    after expanding the bracket if needed.
    """
    P0 = float(P0)

    def f(lam):
        return _budget_spend(C, sigma_in, sigma_out, lam, weights, band_mask) - P0

    # Expand bracket: at small λ spend is huge, at large λ spend is 0.
    flo = f(lam_lo)
    while flo < 0 and lam_lo > 1e-30:
        lam_lo *= 0.01
        flo = f(lam_lo)
    fhi = f(lam_hi)
    while fhi > 0 and lam_hi < 1e30:
        lam_hi *= 100.0
        fhi = f(lam_hi)

    if flo <= 0:
        # Even at the smallest λ the spend is 0: degenerate case (no signal).
        return lam_lo
    if fhi >= 0:
        # Budget is unreachable from above: clamp.
        return lam_hi

    return brentq(f, lam_lo, lam_hi, xtol=xtol, rtol=rtol, maxiter=200)


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def mutual_information_density(C, v_sq, sigma_in, sigma_out):
    """Per-frequency MI density (eq. 18), in nats.

        I(k,ω) = log[(|v|^2(C + σ_in^2) + σ_out^2) / (|v|^2 σ_in^2 + σ_out^2)]
    """
    s_in2 = sigma_in ** 2
    s_out2 = sigma_out ** 2
    num = v_sq * (C + s_in2) + s_out2
    den = v_sq * s_in2 + s_out2
    return np.log(num / den)


def mutual_information(C, v_sq, sigma_in, sigma_out, weights):
    """Total MI integrated over the band, in nats."""
    return float(
        np.sum(mutual_information_density(C, v_sq, sigma_in, sigma_out) * weights)
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def solve_efficient_coding(C, sigma_in, sigma_out, P0, weights, band_mask=None):
    """Find optimal filter and compute the optimized mutual information.

    Returns
    -------
    v_sq : ndarray
    lam : float
    I : float
    """
    lam = find_lambda(C, sigma_in, sigma_out, P0, weights, band_mask)
    v_sq = optimal_filter_squared_magnitude(C, sigma_in, sigma_out, lam, band_mask)
    I = mutual_information(C, v_sq, sigma_in, sigma_out, weights)
    return v_sq, lam, I
