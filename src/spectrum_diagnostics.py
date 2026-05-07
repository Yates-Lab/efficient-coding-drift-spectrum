"""Diagnostics for validating spatiotemporal spectra.

These metrics are meant to make unit/convention errors and separability
assumptions visible before optimal filters or cell classes are interpreted.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def temporal_centroid_by_spatial_frequency(
    C: Array,
    k: Array,
    omega: Array,
    *,
    omega_min: float = 0.0,
) -> tuple[Array, Array]:
    """Return |omega|-centroid of C(k,omega) at each spatial frequency.

    For a multiplicatively separable spectrum C(k,omega)=A(k)B(omega), this
    centroid is constant in k. Movement-generated spectra generally make the
    centroid depend on k.
    """
    C = np.asarray(C, dtype=float)
    k = np.asarray(k, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    if C.shape != (k.size, omega.size):
        raise ValueError("C must have shape (len(k), len(omega))")
    m = np.abs(omega) >= float(omega_min)
    Om = np.abs(omega[m])[None, :]
    Cp = np.maximum(C[:, m], 0.0)
    denom = np.trapezoid(Cp, omega[m], axis=1)
    numer = np.trapezoid(Cp * Om, omega[m], axis=1)
    centroid = numer / np.maximum(denom, 1e-300)
    return k, centroid


def temporal_centroid_log_slope(
    C: Array,
    k: Array,
    omega: Array,
    *,
    k_lo: float | None = None,
    k_hi: float | None = None,
    f_lo: float | None = None,
    f_hi: float | None = None,
    omega_min: float = 0.0,
) -> float:
    """Slope of log temporal centroid vs log spatial frequency.

    A separable stationary control should have slope near zero. Linear motion
    tends toward slope one; Brownian drift tends toward slope two before finite
    bandwidth truncation.
    """
    if f_lo is not None:
        k_lo = f_lo
    if f_hi is not None:
        k_hi = f_hi
    k, centroid = temporal_centroid_by_spatial_frequency(
        C, k, omega, omega_min=omega_min
    )
    m = np.isfinite(centroid) & (centroid > 0) & (k > 0)
    if k_lo is not None:
        m &= k >= float(k_lo)
    if k_hi is not None:
        m &= k <= float(k_hi)
    if np.count_nonzero(m) < 3:
        return float("nan")
    return float(np.polyfit(np.log(k[m]), np.log(centroid[m]), 1)[0])
