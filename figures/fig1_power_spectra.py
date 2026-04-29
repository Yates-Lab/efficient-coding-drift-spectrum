"""Figure 1: power spectra C_θ(f, ω).

Three rows × four columns:
  Row 1: Drift spectrum at varying D, fixed β.
  Row 2: Drift spectrum at varying β, fixed D.
  Row 3: Linear-motion spectrum (Gaussian velocity) at varying speed s.

Each panel: log-log contourf of C_θ in the (f, ω) plane (positive ω only,
since spectra are even in ω). The white line marks the characteristic
movement-induced cross-over (ω = D f^2 for drift, ω = s f for linear motion).
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/claude/efficient_coding")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.spectra import (
    drift_spectrum,
    linear_motion_spectrum_gaussian,
)
from src.plotting import setup_style


setup_style()


def make_grid():
    f = np.geomspace(0.05, 6.0, 220)
    omega = np.geomspace(0.05, 600.0, 220)
    F, W = np.meshgrid(f, omega, indexing="ij")
    return f, omega, F, W


def panel_loglog(ax, x, y, Z, vmin, vmax, n_levels=24, cmap="magma"):
    Zc = np.where(np.isfinite(Z) & (Z > 0), Z, vmin)
    Zc = np.maximum(Zc, vmin)
    levels = np.geomspace(vmin, vmax, n_levels)
    cf = ax.contourf(x, y, Zc, levels=levels,
                     norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                     cmap=cmap, extend="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    return cf


def fig1():
    f, omega, F, W = make_grid()

    fig, axes = plt.subplots(3, 4, figsize=(8.8, 7.0), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.42, "wspace": 0.18})

    # Compute all spectra once so we can choose row-shared color limits.
    Ds = [0.05, 0.5, 5.0, 50.0]
    row1 = [drift_spectrum(F, W, D=D, beta=2.0) for D in Ds]
    betas = [1.2, 1.8, 2.4, 3.0]
    row2 = [drift_spectrum(F, W, D=1.0, beta=b) for b in betas]
    s_list = [0.3, 1.0, 3.0, 10.0]
    row3 = [linear_motion_spectrum_gaussian(F, W, s=s, beta=2.0) for s in s_list]

    def row_lims(row, floor=1e-5):
        v = np.concatenate([r.flatten() for r in row])
        v = v[np.isfinite(v) & (v > 0)]
        return floor * v.max(), v.max()

    cmap = "magma"

    vmin1, vmax1 = row_lims(row1)
    for ax, C, D in zip(axes[0], row1, Ds):
        panel_loglog(ax, f, omega, C.T, vmin1, vmax1, cmap=cmap)
        ax.set_title(rf"$D = {D:g}$,  $\beta = 2$", pad=2)
        ax.plot(f, D * f ** 2, color="white", lw=0.8, alpha=0.55)

    vmin2, vmax2 = row_lims(row2)
    for ax, C, b in zip(axes[1], row2, betas):
        panel_loglog(ax, f, omega, C.T, vmin2, vmax2, cmap=cmap)
        ax.set_title(rf"$D = 1$,  $\beta = {b:g}$", pad=2)
        ax.plot(f, 1.0 * f ** 2, color="white", lw=0.8, alpha=0.55)

    vmin3, vmax3 = row_lims(row3)
    for ax, C, s in zip(axes[2], row3, s_list):
        panel_loglog(ax, f, omega, C.T, vmin3, vmax3, cmap=cmap)
        ax.set_title(rf"$s = {s:g}$,  $\beta = 2$", pad=2)
        ax.plot(f, s * f, color="white", lw=0.8, alpha=0.55)

    # Limit visible y-range to where the spectra have support
    for ax in axes.flat:
        ax.set_ylim(omega.min(), omega.max())
        ax.set_xlim(f.min(), f.max())

    for ax in axes[-1]:
        ax.set_xlabel(r"$f$ (cycles/unit)")
    for ax_row in axes:
        ax_row[0].set_ylabel(r"$\omega$ (rad/s)")

    # Row banners as text, placed to the left of each row of plots
    fig.text(0.02, 0.815, "Brownian drift\n(vary $D$)", rotation=90,
             fontsize=9, va="center", ha="center")
    fig.text(0.02, 0.515, "Brownian drift\n(vary $\\beta$)", rotation=90,
             fontsize=9, va="center", ha="center")
    fig.text(0.02, 0.215, "Linear motion\n(vary $s$)", rotation=90,
             fontsize=9, va="center", ha="center")

    # Single shared colorbar on the right (visualizes the colormap; per-row
    # scales are noted in the label). The contour fill uses log levels.
    cbar_ax = fig.add_axes([0.93, 0.13, 0.012, 0.74])
    cb = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=plt.get_cmap(cmap),
        norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1.0),
        orientation="vertical",
        extend="min",
    )
    cb.ax.tick_params(direction="out", labelsize=7)
    cb.set_label(r"$C_\theta(f,\omega)/\max C_\theta$  (per-row normalization)")

    fig.subplots_adjust(left=0.10, right=0.91, top=0.92, bottom=0.08)

    fig.suptitle(
        r"Movement-induced spatiotemporal power spectra  ($\omega$ in rad/s; $f_\mathrm{Hz}=\omega/2\pi$)",
        y=0.97, fontsize=10.5,
    )

    out = "/home/claude/efficient_coding/outputs/fig1_power_spectra.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1()

