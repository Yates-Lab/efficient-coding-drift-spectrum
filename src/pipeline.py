"""End-to-end pipeline: Spectrum -> Result.

Given a Spectrum instance, computes the optimal filter |v*(f, ω)|^2,
the optimized mutual information I*, and (optionally) the reconstructed
spatial and temporal kernels. Wraps the boilerplate that was previously
duplicated across figure scripts.

Conventions for the output Result:
- v_sq : (Nf, Nω) array of |v*|^2
- I    : float, mutual information in nats
- lam  : float, Lagrange multiplier
- spatial kernel: 1D radial cross-section v_s(r), with r in length units
- temporal kernel: 1D causal v_t(t), with t in seconds
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.spectra import Spectrum
from src.solver import solve_efficient_coding
from src.plotting import radial_weights, band_mask_radial
from src.kernels import (
    spatial_kernel_2d,
    radial_cross_section,
    minimum_phase_temporal_filter,
    soft_band_taper,
)
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, hi_res_grid, fast_grid


@dataclass
class Result:
    """Output of a single efficient-coding solve."""
    spectrum: Spectrum
    sigma_in: float
    sigma_out: float
    P0: float
    f: np.ndarray
    omega: np.ndarray
    C: np.ndarray
    v_sq: np.ndarray
    lam: float
    I: float
    # filled in lazily by extract_*; None until requested
    spatial_r: Optional[np.ndarray] = None
    spatial_v: Optional[np.ndarray] = None
    temporal_t: Optional[np.ndarray] = None
    temporal_v: Optional[np.ndarray] = None
    f_peak: Optional[float] = None  # spatial frequency at peak energy

    @property
    def label(self) -> str:
        return self.spectrum.describe()


def run(
    spectrum: Spectrum,
    sigma_in: float = 0.3,
    sigma_out: float = 1.0,
    P0: float = 50.0,
    grid: str = "fast",
    band: tuple = (F_MAX, OMEGA_MIN, OMEGA_MAX),
) -> Result:
    """Compute the optimal filter and I* for a Spectrum.

    Parameters
    ----------
    spectrum : Spectrum
    sigma_in, sigma_out, P0 : float
        Noise std (input, output) and total power budget.
    grid : "fast" or "hi_res"
        Which (f, ω) grid to use. fast is sufficient for I* sweeps;
        hi_res is required for kernel reconstruction.
    band : tuple
        (f_max, omega_min, omega_max) defining the analysis band.

    Returns
    -------
    Result with v_sq, I, lam, C, and the grid filled in. Spatial /
    temporal kernels are not extracted by run(); call extract_kernels()
    on the result.
    """
    if grid == "fast":
        f, omega = fast_grid()
    elif grid == "hi_res":
        f, omega = hi_res_grid()
    else:
        raise ValueError(f"Unknown grid {grid!r}; use 'fast' or 'hi_res'.")

    f_max, omega_min, omega_max = band
    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, f_max, omega_min, omega_max)
    weights_b = weights * mask

    C = spectrum.C(f, omega)
    v_sq, lam, I = solve_efficient_coding(
        C, sigma_in, sigma_out, P0, weights_b, band_mask=mask,
    )

    return Result(
        spectrum=spectrum,
        sigma_in=sigma_in, sigma_out=sigma_out, P0=P0,
        f=f, omega=omega, C=C, v_sq=v_sq, lam=lam, I=I,
    )


def extract_spatial_kernel(result: Result, k_max: float = 8.0,
                           n_k: int = 512, n_f_fine: int = 1024) -> Result:
    """Reconstruct the 1D radial spatial kernel from |v*|^2 and store it
    on the result. Returns the same result (mutated in place)."""
    v_sq = result.v_sq
    f = result.f
    omega = result.omega

    domega = np.gradient(omega)
    v_s_sq = np.sum(v_sq * np.abs(domega)[None, :], axis=1) / (2 * np.pi)
    v_s = np.sqrt(np.maximum(v_s_sq, 0.0))

    f_fine = np.linspace(0.0, f.max() * 1.2, n_f_fine)
    v_s_interp = np.interp(f_fine, f, v_s, left=v_s[0], right=0.0)

    def vmag(k):
        return np.interp(k, f_fine, v_s_interp, left=v_s_interp[0], right=0.0)

    rx, ry, v_xy = spatial_kernel_2d(vmag, k_max=k_max, n_k=n_k)
    r_radial, v_radial = radial_cross_section(v_xy, rx, ry)
    result.spatial_r = r_radial
    result.spatial_v = v_radial
    return result


def extract_temporal_kernel(result: Result,
                            taper_alpha: float = 0.25,
                            floor_rel: float = 1e-3) -> Result:
    """Reconstruct the causal min-phase temporal kernel at the spatial
    frequency where the filter has the most energy. Stores temporal_t,
    temporal_v, and f_peak on the result."""
    v_sq = result.v_sq
    f = result.f
    omega = result.omega

    domega = np.gradient(omega)
    energy_per_f = np.sum(v_sq * np.abs(domega)[None, :], axis=1)
    i_peak_f = int(np.argmax(energy_per_f))
    f_peak = float(f[i_peak_f])

    v_t_mag = np.sqrt(np.maximum(v_sq[i_peak_f, :], 0.0))
    taper = soft_band_taper(omega, OMEGA_MIN, OMEGA_MAX, alpha=taper_alpha)
    v_t_smooth = v_t_mag * taper
    floor = floor_rel * max(v_t_smooth.max(), 1e-30)
    v_t_smooth = np.maximum(v_t_smooth, floor)
    t, h_t, _ = minimum_phase_temporal_filter(v_t_smooth, omega)
    result.temporal_t = t
    result.temporal_v = h_t
    result.f_peak = f_peak
    return result


def extract_kernels(result: Result, **kw) -> Result:
    """Convenience: extract both spatial and temporal kernels."""
    extract_spatial_kernel(result, **{k: v for k, v in kw.items()
                                      if k in ("k_max", "n_k", "n_f_fine")})
    extract_temporal_kernel(result, **{k: v for k, v in kw.items()
                                       if k in ("taper_alpha", "floor_rel")})
    return result


__all__ = ["Result", "run", "extract_spatial_kernel",
           "extract_temporal_kernel", "extract_kernels"]
