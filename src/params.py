"""Shared band/grid parameters used across figures.

The band parameters are biologically motivated approximations:
  f_max  = 4   cycles/unit  (upper spatial frequency cells respond to)
  ω_min  = 0.5 rad/s ≈  0.08 Hz   (slow temporal cutoff)
  ω_max  = 400 rad/s ≈ 64    Hz   (fast temporal cutoff; matches the upper
                                    end of primate retinal ganglion temporal
                                    bandwidths).

Hz conversion: f_Hz = ω / (2π).
"""

from __future__ import annotations

import numpy as np


F_MAX = 6.0
OMEGA_MIN = 0.5
OMEGA_MAX = 400.0


def hi_res_grid():
    """Wide grid suitable for kernel reconstruction and 2D contours.

    n_omega = 2048, ω_max_grid = 800 rad/s ⇒
        Δω ≈ 0.78 rad/s,  Δt ≈ 3.9 ms,  T ≈ 8.0 s.
    The FFT Nyquist (800 rad/s) is 2× the band ω_max, so no aliasing inside
    the band.
    """
    f = np.geomspace(0.05, 5.0, 220)
    n_omega = 2048
    omega_max_grid = 800.0
    domega = 2.0 * omega_max_grid / n_omega
    omega = (np.arange(n_omega) - n_omega // 2) * domega
    return f, omega


def fast_grid():
    """Coarser grid suitable for I*(D) sweeps where many solves are needed.

    n_omega = 1024, ω_max_grid = 500 rad/s ⇒
        Δω ≈ 0.98 rad/s,  Δt ≈ 6.3 ms,  T ≈ 6.4 s.
    """
    f = np.geomspace(0.05, 5.0, 120)
    n_omega = 1024
    omega_max_grid = 500.0
    domega = 2.0 * omega_max_grid / n_omega
    omega = (np.arange(n_omega) - n_omega // 2) * domega
    return f, omega
