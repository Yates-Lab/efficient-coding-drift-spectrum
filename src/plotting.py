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
# Shared plotting primitives
# ---------------------------------------------------------------------------

def log_grid(x_min, x_max, n):
    """Return a positive logarithmic grid."""
    if x_min <= 0 or x_max <= 0:
        raise ValueError("log_grid bounds must be positive")
    if n < 2:
        raise ValueError("log_grid requires at least two points")
    return np.geomspace(float(x_min), float(x_max), int(n))


def radial_log_grid(
    n_f=200,
    n_omega=200,
    *,
    f_min=0.1,
    f_max=6.0,
    omega_min=0.25,
    omega_max=400.0,
):
    """Positive log-spaced (f, omega) display grid for spectrum panels."""
    return (
        log_grid(f_min, f_max, n_f),
        log_grid(omega_min, omega_max, n_omega),
    )


def positive_frequency(omega, *, temporal_hz=False):
    """Return a positive-frequency mask and display-axis values."""
    omega = np.asarray(omega, dtype=float)
    mask = omega > 0
    y = omega[mask]
    if temporal_hz:
        y = y / (2.0 * np.pi)
    return mask, y


def finite_positive_values(arrays):
    """Flatten one or more arrays to finite positive values."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    return vals[np.isfinite(vals) & (vals > 0)]


def shared_lims(arrays, *, floor=1e-6, percentile=None):
    """Return shared positive log-color limits for one or more arrays."""
    vals = finite_positive_values(arrays)
    if vals.size == 0:
        return float(floor), 1.0
    vmax = float(np.nanpercentile(vals, percentile)) if percentile is not None else float(vals.max())
    vmax = max(vmax, 1e-300)
    return max(float(floor) * vmax, 1e-300), vmax


def log_levels(vmin, vmax, n_levels):
    """Return log-spaced contour levels with safe positive bounds."""
    vmin = max(float(vmin), 1e-300)
    vmax = max(float(vmax), vmin * (1.0 + 1e-12))
    return np.geomspace(vmin, vmax, int(n_levels))


def add_log_colorbar(
    fig,
    rect,
    *,
    cmap="magma",
    vmin=1e-6,
    vmax=1.0,
    label=None,
    orientation="vertical",
    extend="min",
    tick_labelsize=7,
):
    """Add a standalone log-scaled colorbar at ``rect`` in figure coords."""
    cax = fig.add_axes(rect)
    vmin = max(float(vmin), 1e-300)
    vmax = max(float(vmax), vmin * (1.0 + 1e-12))
    cb = mpl.colorbar.ColorbarBase(
        cax,
        cmap=plt.get_cmap(cmap),
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        orientation=orientation,
        extend=extend,
    )
    cb.ax.tick_params(direction="out", labelsize=tick_labelsize)
    if label is not None:
        cb.set_label(label)
    return cb


def add_band_edges(
    ax,
    *,
    f_max=None,
    omega_min=None,
    omega_max=None,
    color="white",
    lw=0.5,
    ls=":",
    alpha=0.6,
):
    """Overlay standard spatial and temporal band edges on a panel."""
    if f_max is not None:
        ax.axvline(f_max, color=color, lw=lw, ls=ls, alpha=alpha)
    if omega_min is not None:
        ax.axhline(omega_min, color=color, lw=lw, ls=ls, alpha=alpha)
    if omega_max is not None:
        ax.axhline(omega_max, color=color, lw=lw, ls=ls, alpha=alpha)


# ---------------------------------------------------------------------------
# Log contour plots
# ---------------------------------------------------------------------------

def log_contourf(
    ax,
    x,
    y,
    Z,
    n_levels=20,
    cmap="magma",
    vmin_floor=1e-6,
    logx=True,
    logy=True,
    label=None,
    extend="both",
    vmin=None,
    vmax=None,
):
    """contourf with logarithmic color scale and (optionally) log axes.

    Z is positive (or near-positive). Values <= 0 are floored to vmin_floor*max.
    """
    Zp = np.where(np.isfinite(Z) & (Z > 0), Z, np.nan)
    zmax = float(vmax) if vmax is not None else np.nanmax(Zp)
    if not np.isfinite(zmax) or zmax <= 0:
        zmax = 1.0
    floor = max(float(vmin) if vmin is not None else vmin_floor * zmax, 1e-300)
    Zc = np.where(np.isnan(Zp), floor, np.maximum(Zp, floor))
    levels = log_levels(floor, zmax, n_levels)
    cf = ax.contourf(x, y, Zc, levels=levels,
                     norm=mpl.colors.LogNorm(vmin=floor, vmax=zmax),
                     cmap=cmap, extend=extend)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    return cf


def panel_loglog(
    ax,
    f,
    omega,
    C,
    vmin=None,
    vmax=None,
    n_levels=24,
    cmap="magma",
    f_min=0.1,
    f_max=6.0,
    omega_min=0.25,
    omega_max=400.0,
    *,
    positive_only=False,
    temporal_hz=False,
    transpose=True,
    extend="both",
):
    """Plot C with shape (Nf, Nomega) as a log-log contourf in (f, omega)."""
    y = np.asarray(omega, dtype=float)
    Z_source = np.asarray(C, dtype=float)
    if positive_only:
        mask, y = positive_frequency(y, temporal_hz=temporal_hz)
        Z_source = Z_source[:, mask]
    elif temporal_hz:
        y = y / (2.0 * np.pi)

    if vmin is None or vmax is None:
        auto_vmin, auto_vmax = shared_lims([Z_source], floor=1e-6)
        vmin = auto_vmin if vmin is None else vmin
        vmax = auto_vmax if vmax is None else vmax

    Z = Z_source.T if transpose else Z_source
    Z = np.where(np.isfinite(Z) & (Z > 0), Z, vmin)
    Z = np.maximum(Z, vmin)
    levels = log_levels(vmin, vmax, n_levels)
    cf = ax.contourf(
        f, y, Z, levels=levels,
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=cmap, extend=extend,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    if f_min is not None and f_max is not None:
        ax.set_xlim(f_min, f_max)
    if omega_min is not None and omega_max is not None:
        if temporal_hz:
            omega_min = omega_min / (2.0 * np.pi)
            omega_max = omega_max / (2.0 * np.pi)
        ax.set_ylim(omega_min, omega_max)
    return cf


# ---------------------------------------------------------------------------
# Integration weights for radial (f, omega) grids
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


def band_mask_radial(f, omega, f_max, omega_min, omega_max):
    """Boolean mask of (f, ω) inside the band B = [0, f_max] × {ω: ω_min <= |ω| <= ω_max}.

    Returns a 2D mask matching np.broadcast(f[:,None], ω[None,:]).
    """
    F = f[:, None]
    W = omega[None, :]
    return (F <= f_max) & (np.abs(W) >= omega_min) & (np.abs(W) <= omega_max)
