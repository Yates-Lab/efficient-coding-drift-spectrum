"""Publication-style matplotlib config and grid/integration utilities."""

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

PUB_RC = {
    "font.size": 9,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.6,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.7,
    "ytick.minor.size": 1.7,
    "xtick.major.width": 0.55,
    "ytick.major.width": 0.55,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "figure.dpi": 110,
    "savefig.dpi": 320,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "lines.linewidth": 1.1,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def setup_style():
    mpl.rcParams.update(PUB_RC)


# Two-color sequential ordering for parameter sweeps
def parameter_palette(n, cmap="viridis", lo=0.10, hi=0.90):
    cm = plt.get_cmap(cmap)
    return [cm(x) for x in np.linspace(lo, hi, n)]


# ---------------------------------------------------------------------------
# Log contour plots
# ---------------------------------------------------------------------------

def log_contourf(ax, x, y, Z, n_levels=20, cmap="magma", vmin_floor=1e-6,
                 logx=True, logy=True, label=None, extend="both"):
    """contourf with logarithmic color scale and (optionally) log axes.

    Z is positive (or near-positive). Values <= 0 are floored to vmin_floor*max.
    """
    Zp = np.where(np.isfinite(Z) & (Z > 0), Z, np.nan)
    zmax = np.nanmax(Zp)
    if not np.isfinite(zmax) or zmax <= 0:
        zmax = 1.0
    floor = max(vmin_floor * zmax, 1e-300)
    Zc = np.where(np.isnan(Zp), floor, np.maximum(Zp, floor))
    levels = np.logspace(np.log10(floor), np.log10(zmax), n_levels)
    cf = ax.contourf(x, y, Zc, levels=levels,
                     norm=mpl.colors.LogNorm(vmin=floor, vmax=zmax),
                     cmap=cmap, extend=extend)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    return cf


# ---------------------------------------------------------------------------
# Integration weights for radial (f, ω) and 2D (k_x, k_y, ω) grids
# ---------------------------------------------------------------------------

def trapezoid_weights_1d(x):
    """1D trapezoidal weights for arbitrary 1D grid."""
    x = np.asarray(x, dtype=float)
    w = np.zeros_like(x)
    if x.size == 1:
        return np.ones_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return w


def radial_weights(f, omega):
    """Build integration weights for I = (1/(2π)^2) ∫ f df dω · g(f, ω).

    Returns weights w(f, ω) such that np.sum(g * w) ≈ I.
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    wf = trapezoid_weights_1d(f)
    ww = trapezoid_weights_1d(omega)
    W = (f * wf)[:, None] * ww[None, :] / (2.0 * np.pi) ** 2
    return W


def cartesian_weights(kx, ky, omega):
    """Weights for I = (1/(2π)^3) ∫ dk_x dk_y dω · g."""
    wkx = trapezoid_weights_1d(np.asarray(kx, dtype=float))
    wky = trapezoid_weights_1d(np.asarray(ky, dtype=float))
    ww = trapezoid_weights_1d(np.asarray(omega, dtype=float))
    W = wkx[:, None, None] * wky[None, :, None] * ww[None, None, :] / (2.0 * np.pi) ** 3
    return W


# ---------------------------------------------------------------------------
# Default analysis grids
# ---------------------------------------------------------------------------

def default_radial_grid(n_f=192, f_min=0.02, f_max=8.0,
                        n_omega=1024, omega_max=200.0):
    """Default (f, ω) grid for efficient-coding analysis.

    f is log-spaced (positive only), omega is linear and centered (uniform DFT
    grid for min-phase reconstruction).
    """
    # f log-spaced
    f = np.geomspace(f_min, f_max, n_f)
    # omega centered, uniform, even N for FFT
    if n_omega % 2:
        n_omega += 1
    domega = 2.0 * omega_max / n_omega
    omega = (np.arange(n_omega) - n_omega // 2) * domega
    return f, omega


def band_mask_radial(f, omega, f_max, omega_min, omega_max):
    """Boolean mask of (f, ω) inside the band B = [0, f_max] × {ω: ω_min <= |ω| <= ω_max}.

    Returns a 2D mask matching np.broadcast(f[:,None], ω[None,:]).
    """
    F = f[:, None]
    W = omega[None, :]
    return (F <= f_max) & (np.abs(W) >= omega_min) & (np.abs(W) <= omega_max)
