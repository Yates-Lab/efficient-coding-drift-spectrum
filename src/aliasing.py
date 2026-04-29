"""Spatial sampling and aliasing.

Two related notions:

1. Global Nyquist: the retina has a fixed mosaic with sampling vector k_s,
   and the represented spectrum is the periodic sum
       C^sample(k, ω) = Σ_m C(k + m k_s, ω),  k in Ω_Nyq.

2. Per-cell Nyquist (user request): assume that cells tuned to spatial
   frequency f_0 form a mosaic that just-barely tiles space so the Nyquist is
   at f_0. Then the sampling frequency seen by that cell type is k_s = 2 f_0.
   The aliased spectrum at the cell's preferred frequency is
       C^sample(f_0, ω; f_0) = Σ_m C(f_0 + m·2 f_0, ω)
                             = C(f_0, ω) + C(3 f_0, ω) + C(-f_0, ω)
                                + C(-3 f_0, ω) + ...
   In radial form, |k + m·2 f_0| folds to a series of magnitudes f_0,
   f_0, 3 f_0, 3 f_0, 5 f_0, 5 f_0, ... (counting both signs of m), i.e.
       C^sample(f_0, ω; f_0) = Σ_{q=0,1,2,...} 2 C((2q+1) f_0, ω) (for q>0; q=0 gives one copy).

The 1D radial reduction is approximate: in 2D the aliased copies form a
ring of points |k + m k_s|, only one of which lies at exactly f_0. We
implement both versions (1D radial estimate and full 2D) so that the
1D version can be cross-checked against the 2D one.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


__all__ = [
    "aliased_spectrum_radial",
    "aliased_spectrum_2d",
    "per_cell_aliased_spectrum",
]


def aliased_spectrum_radial(C_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                            f, omega, k_s, m_max=8):
    """1D radial aliasing approximation: sum C(|f + m k_s|, ω) over m=-m_max..m_max,
    where k_s is the (scalar) sampling-frequency magnitude.

    `C_func(f, ω)` must accept arrays.
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    out = np.zeros(np.broadcast_shapes(f.shape, omega.shape))
    for m in range(-m_max, m_max + 1):
        out = out + C_func(np.abs(f + m * k_s), omega)
    return out


def aliased_spectrum_2d(C_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                        kx, ky, omega, k_s_x, k_s_y, m_max=4):
    """2D aliased spectrum, summing over a square (2 m_max + 1)^2 grid of
    aliased copies.

    `C_func(f, ω)` is the rotationally symmetric input spectrum, with
    f = sqrt(k_x^2 + k_y^2).
    """
    kx = np.asarray(kx, dtype=float)
    ky = np.asarray(ky, dtype=float)
    omega = np.asarray(omega, dtype=float)
    out = np.zeros(np.broadcast_shapes(kx.shape, ky.shape, omega.shape))
    for mx in range(-m_max, m_max + 1):
        for my in range(-m_max, m_max + 1):
            kx_a = kx + mx * k_s_x
            ky_a = ky + my * k_s_y
            f_a = np.sqrt(kx_a ** 2 + ky_a ** 2)
            out = out + C_func(f_a, omega)
    return out


def per_cell_aliased_spectrum(C_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
                              f0, omega, m_max=8):
    """User-requested aliasing: at each spatial frequency f0, assume a mosaic
    whose Nyquist is f0 (so k_s = 2 f0). Compute the represented power at the
    cell's preferred spatial frequency f0:

        C^sample(f0, ω) = Σ_m C(|f0 + m·2 f0|, ω)
                       = C(f0, ω) + Σ_{q=1}^∞ 2 C((2q+1) f0, ω)

    f0 may be an array; omega may be an array.
    """
    f0 = np.asarray(f0, dtype=float)
    omega = np.asarray(omega, dtype=float)
    out = np.zeros(np.broadcast_shapes(f0.shape, omega.shape))
    for m in range(-m_max, m_max + 1):
        # |f0 + m * 2 f0| = |1 + 2m| f0
        coef = np.abs(1 + 2 * m)
        out = out + C_func(coef * f0, omega)
    return out
