"""M- and P-cell kernels from Casile, Victor, and Rucci (2019).

The model is a separable retinal ganglion-cell transfer function,

    RF_z(f, w) = K_z(f) H_z(w),

with a difference-of-Gaussians spatial term and a Victor/Benardete-Kaplan
temporal cascade.  The parameter values below are copied from Tables 1 and 2
of Casile et al., eLife 8:e40924.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class SpatialParams:
    """Difference-of-Gaussians spatial parameters."""

    rc: float
    Kc: float
    rs: float
    Ks: float


@dataclass(frozen=True)
class TemporalParams:
    """Victor/Benardete-Kaplan temporal cascade parameters."""

    N: int
    A: float
    D_ms: float
    Hs: float
    tau_L_ms: float
    tau_S_ms: float


SPATIAL_PARAMS: Dict[str, SpatialParams] = {
    "M": SpatialParams(rc=0.10, Kc=148.0, rs=0.72, Ks=1.1),
    "P": SpatialParams(rc=0.03, Kc=353.2, rs=0.18, Ks=4.4),
}


TEMPORAL_PARAMS: Dict[str, TemporalParams] = {
    "M": TemporalParams(
        N=30,
        A=499.77,
        D_ms=2.0,
        Hs=1.0,
        tau_L_ms=1.1,
        tau_S_ms=2.23,
    ),
    "P": TemporalParams(
        N=38,
        A=67.59,
        D_ms=3.5,
        Hs=0.69,
        tau_L_ms=1.27,
        tau_S_ms=29.36,
    ),
}


def _cell_key(cell: str) -> str:
    key = str(cell).upper()
    if key not in SPATIAL_PARAMS:
        raise ValueError("cell must be 'M' or 'P'.")
    return key


def spatial_frequency_kernel(
    sf_cpd: Array | float,
    cell: str,
    *,
    gamma: float = 0.5,
    gain: float = 1.0,
    dog_form: str = "fourier",
) -> Array:
    """Spatial frequency response ``K(f)`` in cycles/degree.

    ``dog_form="fourier"`` uses the conventional Fourier-domain Gaussian
    exponent, ``exp(-(pi r gamma f)^2)``.  ``dog_form="printed"`` keeps the
    exponent as typeset in the paper's PDF text.
    """

    p = SPATIAL_PARAMS[_cell_key(cell)]
    f_eff = gamma * np.asarray(sf_cpd, dtype=float)
    f2 = np.abs(f_eff) ** 2

    if dog_form == "fourier":
        center_exp = -((np.pi * p.rc) ** 2) * f2
        surround_exp = -((np.pi * p.rs) ** 2) * f2
    elif dog_form == "printed":
        center_exp = -np.pi * p.rc * f2
        surround_exp = -np.pi * p.rs * f2
    else:
        raise ValueError("dog_form must be 'fourier' or 'printed'.")

    center = p.Kc * np.pi * p.rc**2 * np.exp(center_exp)
    surround = p.Ks * np.pi * p.rs**2 * np.exp(surround_exp)
    return gain * (center - surround)


def spatial_kernel_profile(
    r_deg: Array | float,
    cell: str,
    *,
    gamma: float = 0.5,
    gain: float = 1.0,
) -> Array:
    """Real-space difference-of-Gaussians profile for the Fourier DOG form.

    The default ``gamma`` matches the paper's foveal scaling.  Amplitude scaling
    is included for completeness, although most figure panels normalize each
    curve by its own peak absolute value.
    """

    p = SPATIAL_PARAMS[_cell_key(cell)]
    r = np.asarray(r_deg, dtype=float)
    scale = gain / max(float(gamma) ** 2, 1e-300)
    center = p.Kc * np.exp(-(r / (gamma * p.rc)) ** 2)
    surround = p.Ks * np.exp(-(r / (gamma * p.rs)) ** 2)
    return scale * (center - surround)


def temporal_frequency_kernel(
    tf_hz: Array | float,
    cell: str,
    *,
    rho: float = 1.0 / 1.6,
    causal_delay: bool = True,
) -> Array:
    """Complex temporal frequency response ``H(w)`` at frequencies in Hz."""

    p = TEMPORAL_PARAMS[_cell_key(cell)]
    f = np.asarray(tf_hz, dtype=float)
    s = 1j * rho * 2.0 * np.pi * f

    delay_s = p.D_ms * 1e-3
    tau_L_s = p.tau_L_ms * 1e-3
    tau_S_s = p.tau_S_ms * 1e-3

    delay_sign = -1.0 if causal_delay else 1.0
    delay = np.exp(delay_sign * s * delay_s)
    high_pass = 1.0 - p.Hs / (1.0 + s * tau_S_s)
    low_pass_cascade = (1.0 / (1.0 + s * tau_L_s)) ** p.N
    return p.A * delay * high_pass * low_pass_cascade


def spatiotemporal_frequency_kernel(
    sf_cpd: Array,
    tf_hz: Array,
    cell: str,
    *,
    gamma: float = 0.5,
    rho: float = 1.0 / 1.6,
    dog_form: str = "fourier",
) -> Array:
    """Separable transfer function with shape ``(n_sf, n_tf)``."""

    K = spatial_frequency_kernel(
        sf_cpd,
        cell,
        gamma=gamma,
        dog_form=dog_form,
    )[:, None]
    H = temporal_frequency_kernel(tf_hz, cell, rho=rho)[None, :]
    return K * H


def spatiotemporal_filter_power(
    sf_cpd: Array,
    tf_hz: Array,
    cell: str,
    **kwargs,
) -> Array:
    """Return ``|K(f) H(w)|^2`` with shape ``(n_sf, n_tf)``."""

    RF = spatiotemporal_frequency_kernel(sf_cpd, tf_hz, cell, **kwargs)
    return np.abs(RF) ** 2


def temporal_impulse_response(
    cell: str,
    *,
    sample_rate_hz: float = 2048.0,
    n_samples: int = 4096,
    rho: float = 1.0 / 1.6,
) -> tuple[Array, Array]:
    """Numerically invert ``H(w)`` into a time-domain impulse response."""

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    df = float(sample_rate_hz) / int(n_samples)
    tf_centered = (np.arange(int(n_samples)) - int(n_samples) // 2) * df
    H_centered = temporal_frequency_kernel(tf_centered, cell, rho=rho)
    H_dft = np.fft.ifftshift(H_centered)
    h = np.fft.ifft(H_dft).real * int(n_samples) * df
    t = np.arange(int(n_samples), dtype=float) / float(sample_rate_hz)
    return t, h


def mp_kernel_curves(
    cells: Iterable[str] = ("M", "P"),
    *,
    r_max_deg: float = 2.0,
    n_r: int = 1201,
    sample_rate_hz: float = 2048.0,
    n_temporal_samples: int = 4096,
) -> dict[str, dict[str, Array]]:
    """Generate spatial and temporal plotting curves for M/P cells."""

    r = np.linspace(-float(r_max_deg), float(r_max_deg), int(n_r))
    curves: dict[str, dict[str, Array]] = {}
    for cell in cells:
        key = _cell_key(cell)
        t, h = temporal_impulse_response(
            key,
            sample_rate_hz=sample_rate_hz,
            n_samples=n_temporal_samples,
        )
        curves[key] = {
            "r": r,
            "spatial": spatial_kernel_profile(r, key),
            "t": t,
            "temporal": h,
        }
    return curves


__all__ = [
    "SpatialParams",
    "TemporalParams",
    "SPATIAL_PARAMS",
    "TEMPORAL_PARAMS",
    "spatial_frequency_kernel",
    "spatial_kernel_profile",
    "temporal_frequency_kernel",
    "spatiotemporal_frequency_kernel",
    "spatiotemporal_filter_power",
    "temporal_impulse_response",
    "mp_kernel_curves",
]
