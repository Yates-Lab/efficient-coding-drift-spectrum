"""Figure 1: on-retina power spectra C_theta(f, omega).

Three companion figures:
  1a  main examples: drift sweep over D; saccade sweep over A.
  1b  analytic cycle selector: early vs late fixation.
  1c  spectrum library: Brownian drift, saccade, Dong-Atick separable
      approximation, and Dong-Atick linear velocity spread.

Every panel is a log-log contourf of C_theta(f, omega) on
f in [0.1, 4] cycles/unit and omega in [0.5, 400] rad/s. White overlays
mark the characteristic movement-induced cross-over (omega = D f^2 for
drift, omega = s f for linear motion) on panels where they are meaningful.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.spectra import (
    DriftSpectrum,
    SaccadeSpectrum,
)
from src.power_spectrum_library import (
    cycle_decomposition_panels,
    spectrum_library_panels,
    overlay_curve_hz,
)
from src.plotting import setup_style


setup_style()


F_MIN, F_MAX = 0.1, 6.0
OMEGA_MIN, OMEGA_MAX = 0.25, 400.0
CMAP = "magma"
N_LEVELS = 24
FLOOR = 1e-6  # vmin = FLOOR * shared_vmax


def make_grid(n_f=200, n_omega=200):
    f = np.geomspace(F_MIN, F_MAX, n_f)
    omega = np.geomspace(OMEGA_MIN, OMEGA_MAX, n_omega)
    return f, omega


def shared_lims(panels):
    v = np.concatenate([np.asarray(p).ravel() for p in panels])
    v = v[np.isfinite(v) & (v > 0)]
    vmax = float(v.max())
    return FLOOR * vmax, vmax


def panel_loglog(ax, f, omega, C, vmin, vmax):
    """Plot C with shape (Nf, Nomega) as a log-log contourf in (f, omega)."""
    Z = np.asarray(C).T
    Z = np.where(np.isfinite(Z) & (Z > 0), Z, vmin)
    Z = np.maximum(Z, vmin)
    levels = np.geomspace(vmin, vmax, N_LEVELS)
    cf = ax.contourf(
        f, omega, Z, levels=levels,
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=CMAP, extend="both",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(F_MIN, F_MAX)
    ax.set_ylim(OMEGA_MIN, OMEGA_MAX)
    return cf


def panel_loglog_hz(ax, panel, vmin, vmax):
    """Plot a shared SpectrumPanel on log f / log temporal-Hz axes."""
    Z = np.asarray(panel.C).T
    Z = np.where(np.isfinite(Z) & (Z > 0), Z, vmin)
    Z = np.maximum(Z, vmin)
    levels = np.geomspace(vmin, vmax, N_LEVELS)
    cf = ax.contourf(
        panel.f,
        panel.temporal_hz,
        Z,
        levels=levels,
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap=CMAP,
        extend="both",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(panel.f.min(), panel.f.max())
    ax.set_ylim(panel.temporal_hz.min(), panel.temporal_hz.max())
    overlay = overlay_curve_hz(panel)
    if overlay is not None:
        ax.plot(overlay[0], overlay[1], color="white", lw=0.8, alpha=0.6)
    return cf


def overlay_drift(ax, f, D):
    ax.plot(f, D * f ** 2, color="white", lw=0.8, alpha=0.6)


def overlay_drift_cycles(ax, f, D):
    ax.plot(f, D * (2.0 * np.pi * f) ** 2, color="white", lw=0.8, alpha=0.6)


def overlay_linear(ax, f, s):
    ax.plot(f, s * f, color="white", lw=0.8, alpha=0.6)


def add_colorbar(fig, rect, label):
    cax = fig.add_axes(rect)
    cb = mpl.colorbar.ColorbarBase(
        cax, cmap=plt.get_cmap(CMAP),
        norm=mpl.colors.LogNorm(vmin=FLOOR, vmax=1.0),
        orientation="vertical", extend="min",
    )
    cb.ax.tick_params(direction="out", labelsize=7)
    cb.set_label(label)


CBAR_LABEL = (
    r"$C_\theta(f,\omega) / \max_{\mathrm{fig}}\,C_\theta$"
    r"  (per-figure max normalization)"
)


def fig1a():
    """Main example spectra: 2 rows x 5 columns."""
    f, omega = make_grid()

    Ds = [0.05, 0.5, 2.0, 10.0, 50.0]
    As = [0.5, 1.0, 2.0, 4.0, 8.0]

    row_drift = [DriftSpectrum(D=D).C(f, omega) for D in Ds]
    row_sacc = [SaccadeSpectrum(A=A, lam=3.0).C(f, omega) for A in As]

    vmin, vmax = shared_lims(row_drift + row_sacc)

    fig, axes = plt.subplots(
        2, 5, figsize=(10.0, 4.6), sharex=True, sharey=True,
        gridspec_kw={"hspace": 0.45, "wspace": 0.18},
    )

    for ax, C, D in zip(axes[0], row_drift, Ds):
        panel_loglog(ax, f, omega, C, vmin, vmax)
        ax.set_title(rf"$D = {D:g}$", pad=2)
        overlay_drift(ax, f, D)

    for ax, C, A in zip(axes[1], row_sacc, As):
        panel_loglog(ax, f, omega, C, vmin, vmax)
        ax.set_title(rf"$A = {A:g}$", pad=2)

    for ax in axes[-1]:
        ax.set_xlabel(r"$f$ (cycles/unit)")
    for ax_row in axes:
        ax_row[0].set_ylabel(r"$\omega$ (rad/s)")

    fig.text(0.015, 0.715, "Diffusion\n(vary $D$)", rotation=90,
             fontsize=9, va="center", ha="center")
    fig.text(0.015, 0.305, "Saccades\n(vary $A$, $\\lambda=3$)",
             rotation=90, fontsize=9, va="center", ha="center")

    fig.subplots_adjust(left=0.07, right=0.91, top=0.91, bottom=0.10)
    fig.suptitle(
        r"Figure 1a  on-retina power spectra $C_\theta(f, \omega)$",
        y=0.985, fontsize=10.5,
    )
    add_colorbar(fig, [0.93, 0.13, 0.012, 0.74], CBAR_LABEL)

    out = "outputs/fig1a_main.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig1b():
    """Analytic cycle selector: early vs late fixation."""
    panels = cycle_decomposition_panels(normalize="panel")
    vmin, vmax = FLOOR, 1.0

    fig, axes = plt.subplots(
        1, 2, figsize=(6.0, 3.3), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.22},
    )

    for ax, panel in zip(axes, panels):
        panel_loglog_hz(ax, panel, vmin, vmax)
        ax.axhline(8.0, color="white", lw=0.6, ls=":", alpha=0.8)
        ax.set_title(panel.title, pad=4)

    for ax in axes:
        ax.set_xlabel(r"spatial frequency $f$ (cpd)")
    axes[0].set_ylabel("temporal frequency (Hz)")

    fig.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.16)
    fig.suptitle(
        r"Figure 1b  analytic cycle selector $C_\theta(f, \omega)$: "
        r"early vs late fixation",
        y=0.99, fontsize=10.5,
    )
    add_colorbar(
        fig,
        [0.90, 0.18, 0.018, 0.64],
        r"$C_\theta(f,\omega) / \max_\mathrm{panel}\,C_\theta$",
    )

    out = "outputs/fig1b_boi_cycle.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig1c():
    """Spectrum library: drift, saccade, separable approximation, linear velocity."""
    panels = spectrum_library_panels(normalize="panel")
    vmin, vmax = FLOOR, 1.0

    fig, axes = plt.subplots(
        1, 4, figsize=(10.2, 3.1), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.20},
    )

    for ax, panel in zip(axes, panels):
        panel_loglog_hz(ax, panel, vmin, vmax)
        ax.set_title(panel.title, pad=4)

    for ax in axes:
        ax.set_xlabel(r"spatial frequency $f$ (cpd)")
    axes[0].set_ylabel("temporal frequency (Hz)")

    fig.subplots_adjust(left=0.06, right=0.93, top=0.84, bottom=0.18)
    fig.suptitle(
        r"Figure 1c  spectrum library $C_\theta(f, \omega)$",
        y=0.99, fontsize=10.5,
    )
    add_colorbar(
        fig,
        [0.945, 0.20, 0.010, 0.60],
        r"$C_\theta(f,\omega) / \max_\mathrm{panel}\,C_\theta$",
    )

    out = "outputs/fig1c_library.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1a()
    fig1b()
    fig1c()
