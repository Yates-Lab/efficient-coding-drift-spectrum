"""Diagnostics for validating spatiotemporal spectra.

These metrics are meant to make unit/convention errors and separability
assumptions visible before optimal filters or cell classes are interpreted.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def temporal_centroid_by_spatial_frequency(
    C: Array,
    f: Array,
    omega: Array,
    *,
    omega_min: float = 0.0,
) -> tuple[Array, Array]:
    """Return |omega|-centroid of C(f,omega) at each spatial frequency.

    For a multiplicatively separable spectrum C(f,omega)=A(f)B(omega), this
    centroid is constant in f. Movement-generated spectra generally make the
    centroid depend on f.
    """
    C = np.asarray(C, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    if C.shape != (f.size, omega.size):
        raise ValueError("C must have shape (len(f), len(omega))")
    m = np.abs(omega) >= float(omega_min)
    Om = np.abs(omega[m])[None, :]
    Cp = np.maximum(C[:, m], 0.0)
    denom = np.trapz(Cp, omega[m], axis=1)
    numer = np.trapz(Cp * Om, omega[m], axis=1)
    centroid = numer / np.maximum(denom, 1e-300)
    return f, centroid


def temporal_centroid_log_slope(
    C: Array,
    f: Array,
    omega: Array,
    *,
    f_lo: float | None = None,
    f_hi: float | None = None,
    omega_min: float = 0.0,
) -> float:
    """Slope of log temporal centroid vs log spatial frequency.

    A separable stationary control should have slope near zero. Linear motion
    tends toward slope one; Brownian drift tends toward slope two before finite
    bandwidth truncation.
    """
    f, centroid = temporal_centroid_by_spatial_frequency(
        C, f, omega, omega_min=omega_min
    )
    m = np.isfinite(centroid) & (centroid > 0) & (f > 0)
    if f_lo is not None:
        m &= f >= float(f_lo)
    if f_hi is not None:
        m &= f <= float(f_hi)
    if np.count_nonzero(m) < 3:
        return float("nan")
    return float(np.polyfit(np.log(f[m]), np.log(centroid[m]), 1)[0])
