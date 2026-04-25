"""
Optimal filter magnitude (Eq. 10) and the lambda(D) constraint solver.

The filter magnitude at a single (k, omega) depends only on the local input
power Cx = C_D(k, omega) and the three scalars (sigma_in^2, sigma_out^2, lambda).
Everything is vectorized over (k, omega) grids.
"""

import numpy as np
from scipy.optimize import brentq


def v_star_sq(Cx, sigma_in_sq, sigma_out_sq, lam):
    """
    Optimal filter power |v*(k, omega)|^2 from Eq. (10), rectified.

    Parameters
    ----------
    Cx : array_like
        Input power spectrum C_x(k, omega). Any shape.
    sigma_in_sq, sigma_out_sq : float
        Input and output noise variances.
    lam : float
        Lagrange multiplier for the power constraint.

    Returns
    -------
    Array with the same shape as Cx. Zero where the active-set condition fails.
    """
    Cx = np.asarray(Cx, dtype=float)

    # Guard against Cx = 0 (would divide by zero in the sqrt term).
    # At Cx = 0 the inner expression is sqrt(inf) -> inf, and the factor
    # Cx/(Cx + sigma_in^2) -> 0, so the whole bracket -> 0 * inf - 1 = -1,
    # and [-1]_+ = 0. So the answer is just 0.
    out = np.zeros_like(Cx)
    mask = Cx > 0
    if not np.any(mask):
        return out

    Cxm = Cx[mask]
    gate = Cxm / (Cxm + sigma_in_sq)
    disc = 1.0 + 4.0 * sigma_in_sq / (lam * sigma_out_sq * Cxm)
    bracket = 0.5 * gate * (np.sqrt(disc) + 1.0) - 1.0
    val = (sigma_out_sq / sigma_in_sq) * bracket
    out[mask] = np.maximum(val, 0.0)
    return out


def active_threshold(sigma_in_sq, sigma_out_sq, lam):
    """
    Input-power threshold Cth(lambda) from Eq. (12).
    Frequencies with C_x(k, omega) > Cth are in the active set.
    Returns +inf if lam * sigma_out^2 >= 1 (everything is off).
    """
    denom = 1.0 - lam * sigma_out_sq
    if denom <= 0:
        return np.inf
    return lam * sigma_out_sq * sigma_in_sq / denom


def total_power(Cx, v2, weights):
    """
    Total response power P[v] from Eq. (6):
        P = integral |v|^2 * (C_x + sigma_in^2) dk domega / (2 pi)^(d+1)
    implemented as a discrete sum.

    Parameters
    ----------
    Cx : array
        C_x(k, omega) on the grid. sigma_in^2 should be added before calling
        (we pass Cx + sigma_in^2 from outside).
    v2 : array
        |v*|^2 on the same grid.
    weights : array (broadcastable to Cx)
        Grid-cell measure, including any polar/jacobian factors and the
        (2 pi)^-(d+1) Fourier convention factor.
    """
    return np.sum(v2 * Cx * weights)


def solve_lambda(
    Cx,
    sigma_in_sq,
    sigma_out_sq,
    P_target,
    weights,
    lam_lo=None,
    lam_hi=None,
    xtol=1e-10,
):
    """
    Find lambda such that total_power(|v*|^2, Cx + sigma_in^2) = P_target.

    total_power is monotonically decreasing in lambda:
      - lambda -> 0+ : |v*|^2 -> infinity, P -> infinity
      - lambda -> 1/sigma_out^2 : active set collapses, P -> 0

    We solve on log(lambda) for numerical conditioning.

    Parameters
    ----------
    Cx : array
        C_x on the grid.
    P_target : float
        Desired total power.
    weights : array
        Grid cell weights including Fourier factors.
    lam_lo, lam_hi : float or None
        Optional bracket for lambda. Defaults are 1e-12 and
        0.9999/sigma_out_sq.

    Returns
    -------
    lam : float
    """
    Cx_plus_n = Cx + sigma_in_sq

    if lam_lo is None:
        lam_lo = 1e-12
    if lam_hi is None:
        lam_hi = 0.9999 / sigma_out_sq

    def residual(log_lam):
        lam = np.exp(log_lam)
        v2 = v_star_sq(Cx, sigma_in_sq, sigma_out_sq, lam)
        P = total_power(Cx_plus_n, v2, weights)
        return P - P_target

    r_lo = residual(np.log(lam_lo))
    r_hi = residual(np.log(lam_hi))

    if r_lo < 0:
        raise RuntimeError(
            f"P(lam_lo={lam_lo:g}) = {r_lo + P_target:g} < P_target = {P_target:g}. "
            "Lower lam_lo or reduce P_target."
        )
    if r_hi > 0:
        raise RuntimeError(
            f"P(lam_hi={lam_hi:g}) = {r_hi + P_target:g} > P_target = {P_target:g}. "
            "This means even the maximum allowed lambda cannot kill enough power; "
            "something is off with the setup."
        )

    log_lam_star = brentq(residual, np.log(lam_lo), np.log(lam_hi), xtol=xtol)
    return float(np.exp(log_lam_star))
