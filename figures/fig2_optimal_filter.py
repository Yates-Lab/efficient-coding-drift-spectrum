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

from src.pipeline import SolveConfig, run_many
from src.plotting import (
    add_band_edges,
    add_log_colorbar,
    band_mask_radial,
    panel_loglog,
    setup_style,
    shared_lims,
)
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX
from src.power_spectrum_library import drift_spectrum_specs

setup_style()


def fig2():
    sigma_out = 2.0
    P0 = 50.0

    sigma_in_fixed = 0.2
    D_sweep = np.geomspace(0.05, 50.0, 6)

    D_fixed = 5.0
    sigma_in_sweep = np.geomspace(0.05, 2.0, 6)

    fig, axes = plt.subplots(2, 6, figsize=(12.5, 5.0), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.40, "wspace": 0.16,
                                          "left": 0.07, "right": 0.93,
                                          "top": 0.90, "bottom": 0.10})

    row1_specs = drift_spectrum_specs(D_sweep)
    row1_results = run_many(
        row1_specs,
        SolveConfig(
            sigma_in=sigma_in_fixed, sigma_out=sigma_out, P0=P0,
            grid="hi_res",
        ),
    )
    row1 = [(spec.parameters["D"], result) for spec, result in zip(row1_specs, row1_results)]

    row2_specs = drift_spectrum_specs([D_fixed] * len(sigma_in_sweep))
    row2_results = [
        run_many(
            [spec],
            SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0, grid="hi_res"),
        )[0]
        for spec, sin in zip(row2_specs, sigma_in_sweep)
    ]
    row2 = [(sin, result) for sin, result in zip(sigma_in_sweep, row2_results)]

    f = row1_results[0].f
    omega = row1_results[0].omega
    omega_pos = omega[omega > 0]
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)

    cmap = "viridis"

    def row_vmax(rows):
        _, vmax = shared_lims([np.where(mask, r[1].v_sq, np.nan) for r in rows], floor=1e-5)
        return vmax

    def panel(ax, result, vmax, title):
        v_disp = np.where(mask, result.v_sq / vmax, 1e-5)
        panel_loglog(
            ax,
            f,
            omega,
            v_disp,
            1e-5,
            1.0,
            n_levels=24,
            cmap=cmap,
            f_min=f.min(),
            f_max=f.max(),
            omega_min=omega_pos.min(),
            omega_max=omega_pos.max(),
            positive_only=True,
        )
        add_band_edges(ax, f_max=F_MAX, omega_min=OMEGA_MIN, omega_max=OMEGA_MAX)
        ax.set_title(title, pad=2, fontsize=8.5)

    vmax1 = row_vmax(row1)
    for ax, (D, result) in zip(axes[0], row1):
        panel(ax, result, vmax1,
              rf"$D = {D:.2g}$,  $I^* = {result.I:.2f}$")

    vmax2 = row_vmax(row2)
    for ax, (sin, result) in zip(axes[1], row2):
        panel(ax, result, vmax2,
              rf"$\sigma_\mathrm{{in}} = {sin:.2g}$,  $I^* = {result.I:.2f}$")

    for ax in axes[-1]:
        ax.set_xlabel(r"$f$ (cyc/u)")
    for row in axes:
        row[0].set_ylabel(r"$\omega$ (rad/s)")

    fig.text(0.02, 0.69, rf"vary $D$" + "\n" + rf"$\sigma_\mathrm{{in}}={sigma_in_fixed}$",
             rotation=90, fontsize=9, va="center", ha="center")
    fig.text(0.02, 0.27, rf"vary $\sigma_\mathrm{{in}}$" + "\n" + rf"$D={D_fixed}$",
             rotation=90, fontsize=9, va="center", ha="center")

    add_log_colorbar(
        fig,
        [0.945, 0.13, 0.012, 0.74],
        cmap=cmap,
        vmin=1e-5,
        vmax=1.0,
        label=r"$|v^\star|^2 / \max |v^\star|^2$ (per row)",
    )

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
