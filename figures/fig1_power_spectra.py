"""Figure 1: on-retina power spectra C_theta(f, omega).

Three companion figures:
  1a  main examples: drift sweep over D; saccade sweep over A.
  1b  Boi cycle decomposition: early vs late fixation.
  1c  comprehensive library: diffusion, saccades, early/late fixation cycle,
      linear velocity distribution.

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
    LinearMotionSpectrum,
    BoiEarlyCleanApprox,
    BoiLateDriftApprox,
)
from src.plotting import setup_style


setup_style()


F_MIN, F_MAX = 0.1, 4.0
OMEGA_MIN, OMEGA_MAX = 0.5, 400.0
CMAP = "magma"
N_LEVELS = 24
FLOOR = 1e-5  # vmin = FLOOR * shared_vmax


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
    """Boi cycle decomposition: early vs late fixation."""
    f, omega = make_grid()

    D_late = 0.05
    early = BoiEarlyCleanApprox(
        mean_A=4.4, sd_A=1.3, A_min=1.0, A_max=10.0,
    ).C(f, omega)
    late = BoiLateDriftApprox(D=D_late).C(f, omega)

    panels = [
        ("Early fixation\n(saccade transient)", early, "none", None),
        (f"Late fixation\n(drift, $D={D_late:g}$)", late, "drift_cycles", D_late),
    ]

    vmin, vmax = shared_lims([p[1] for p in panels])

    fig, axes = plt.subplots(
        1, 2, figsize=(6.0, 3.3), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.22},
    )

    for ax, (title, C, kind, param) in zip(axes, panels):
        panel_loglog(ax, f, omega, C, vmin, vmax)
        ax.set_title(title, pad=4)
        if kind == "drift":
            overlay_drift(ax, f, param)
        elif kind == "drift_cycles":
            overlay_drift_cycles(ax, f, param)

    for ax in axes:
        ax.set_xlabel(r"$f$ (cycles/unit)")
    axes[0].set_ylabel(r"$\omega$ (rad/s)")

    fig.subplots_adjust(left=0.10, right=0.88, top=0.82, bottom=0.16)
    fig.suptitle(
        r"Figure 1b  Boi cycle decomposition $C_\theta(f, \omega)$: "
        r"early vs late fixation",
        y=0.99, fontsize=10.5,
    )
    add_colorbar(fig, [0.90, 0.18, 0.018, 0.64], CBAR_LABEL)

    out = "outputs/fig1b_boi_cycle.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def fig1c():
    """Comprehensive library: diffusion, saccades, early/late fixation, linear velocity."""
    f, omega = make_grid()

    D = 2.0
    D_late = 0.05
    s_lin = 1.0
    panels = [
        ("Diffusion",
         DriftSpectrum(D=D).C(f, omega),
         "drift", D),
        ("Saccades",
         SaccadeSpectrum(A=2.5, lam=3.0).C(f, omega),
         "none", None),
        ("Early fixation cycle",
         BoiEarlyCleanApprox(
             mean_A=4.4, sd_A=1.3, A_min=1.0, A_max=10.0,
         ).C(f, omega),
         "none", None),
        ("Late fixation cycle",
         BoiLateDriftApprox(D=D_late).C(f, omega),
         "drift_cycles", D_late),
        ("Linear velocity distribution",
         LinearMotionSpectrum(s=s_lin).C(f, omega),
         "linear", s_lin),
    ]

    vmin, vmax = shared_lims([p[1] for p in panels])

    fig, axes = plt.subplots(
        1, 5, figsize=(12.5, 3.1), sharex=True, sharey=True,
        gridspec_kw={"wspace": 0.20},
    )

    for ax, (title, C, kind, param) in zip(axes, panels):
        panel_loglog(ax, f, omega, C, vmin, vmax)
        ax.set_title(title, pad=4)
        if kind == "drift":
            overlay_drift(ax, f, param)
        elif kind == "drift_cycles":
            overlay_drift_cycles(ax, f, param)
        elif kind == "linear":
            overlay_linear(ax, f, param)

    for ax in axes:
        ax.set_xlabel(r"$f$ (cycles/unit)")
    axes[0].set_ylabel(r"$\omega$ (rad/s)")

    fig.subplots_adjust(left=0.06, right=0.93, top=0.84, bottom=0.18)
    fig.suptitle(
        r"Figure 1c  spectrum library $C_\theta(f, \omega)$",
        y=0.99, fontsize=10.5,
    )
    add_colorbar(fig, [0.945, 0.20, 0.010, 0.60], CBAR_LABEL)

    out = "outputs/fig1c_library.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1a()
    fig1b()
    fig1c()
