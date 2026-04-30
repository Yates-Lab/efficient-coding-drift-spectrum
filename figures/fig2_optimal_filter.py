"""Figure 2: optimal filter |v*(f, ω)|^2 on the (f, ω) plane.

Two parameter sweeps for a fixed retinal budget P_0, fixed band B, and a
power-law image:
  Row 1: vary drift coefficient D at fixed σ_in.
  Row 2: vary input noise σ_in at fixed D.

Each panel reports the achieved I^* in nats; white dotted lines mark the
band edges.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.spectra import drift_spectrum
from src.solver import solve_efficient_coding
from src.plotting import setup_style, radial_weights, band_mask_radial
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, hi_res_grid

setup_style()


def _solve_grid(f, omega, D, beta, sigma_in, sigma_out, P0):
    F = f[:, None]
    W = omega[None, :]
    C = drift_spectrum(F, W, D=D, beta=beta)
    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    weights = weights * mask
    v_sq, lam, I = solve_efficient_coding(C, sigma_in, sigma_out, P0,
                                          weights, band_mask=mask)
    return v_sq, I, mask


def fig2():
    f, omega = hi_res_grid()
    omega_pos = omega[omega > 0]

    sigma_out = 1.0
    P0 = 50.0

    sigma_in_fixed = 1.0
    D_sweep = np.geomspace(0.05, 50.0, 6)

    D_fixed = 5.0
    sigma_in_sweep = np.geomspace(0.05, 2.0, 6)

    fig, axes = plt.subplots(2, 6, figsize=(12.5, 5.0), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.40, "wspace": 0.16,
                                          "left": 0.07, "right": 0.93,
                                          "top": 0.90, "bottom": 0.10})

    row1 = []
    for D in D_sweep:
        v_sq, I, mask = _solve_grid(f, omega, D, beta=2.0,
                                    sigma_in=sigma_in_fixed,
                                    sigma_out=sigma_out, P0=P0)
        row1.append((D, v_sq, I, mask))

    row2 = []
    for sin in sigma_in_sweep:
        v_sq, I, mask = _solve_grid(f, omega, D_fixed, beta=2.0,
                                    sigma_in=sin,
                                    sigma_out=sigma_out, P0=P0)
        row2.append((sin, v_sq, I, mask))

    cmap = "viridis"

    def lims(rows, floor=1e-5):
        v = np.concatenate([r[1].flatten() for r in rows])
        v = v[np.isfinite(v) & (v > 0)]
        return floor * v.max(), v.max()

    def panel(ax, v_sq, vmin, vmax, mask, title):
        v_disp = np.where(mask, v_sq, vmin)
        v_disp = np.maximum(v_disp, vmin)
        i_pos = omega > 0
        levels = np.geomspace(vmin, vmax, 24)
        ax.contourf(f, omega[i_pos], v_disp[:, i_pos].T,
                    levels=levels,
                    norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                    cmap=cmap, extend="both")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(f.min(), f.max())
        ax.set_ylim(omega_pos.min(), omega.max())
        ax.axvline(F_MAX, color="white", lw=0.5, ls=":")
        ax.axhline(OMEGA_MIN, color="white", lw=0.5, ls=":")
        ax.axhline(OMEGA_MAX, color="white", lw=0.5, ls=":")
        ax.set_title(title, pad=2, fontsize=8.5)

    vmin1, vmax1 = lims(row1)
    for ax, (D, v_sq, I, mask) in zip(axes[0], row1):
        panel(ax, v_sq, vmin1, vmax1, mask,
              rf"$D = {D:.2g}$,  $I^* = {I:.2f}$")

    vmin2, vmax2 = lims(row2)
    for ax, (sin, v_sq, I, mask) in zip(axes[1], row2):
        panel(ax, v_sq, vmin2, vmax2, mask,
              rf"$\sigma_\mathrm{{in}} = {sin:.2g}$,  $I^* = {I:.2f}$")

    for ax in axes[-1]:
        ax.set_xlabel(r"$f$ (cyc/u)")
    for row in axes:
        row[0].set_ylabel(r"$\omega$ (rad/s)")

    fig.text(0.02, 0.69, rf"vary $D$" + "\n" + rf"$\sigma_\mathrm{{in}}={sigma_in_fixed}$",
             rotation=90, fontsize=9, va="center", ha="center")
    fig.text(0.02, 0.27, rf"vary $\sigma_\mathrm{{in}}$" + "\n" + rf"$D={D_fixed}$",
             rotation=90, fontsize=9, va="center", ha="center")

    cbar_ax = fig.add_axes([0.945, 0.13, 0.012, 0.74])
    cb = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=plt.get_cmap(cmap),
        norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1.0),
        orientation="vertical",
        extend="min",
    )
    cb.set_label(r"$|v^\star|^2 / \max |v^\star|^2$ (per row)")
    cb.ax.tick_params(direction="out", labelsize=7)

    fig.suptitle(
        r"Optimal spatiotemporal filter $|v^\star(f,\omega)|^2$ under drift",
        y=0.97, fontsize=10.5,
    )

    out = "outputs/fig2_optimal_filter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig2()
