"""
Fig 1: Input spectrum C_D (top row) and optimal filter |v*_D|^2 (bottom
row) for a row of D values.

The filter is approximately inverse to the input (whitening), so they
appear visually complementary. As D grows, C_D broadens in omega
(Lorentzian bandwidth D k^2), and the filter correspondingly adjusts.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from grids import build_radial_weights, omega_grid, radial_grid
from optimizer import solve_lambda, v_star_sq
from spectrum import C_D
from style import set_defaults


def run():
    set_defaults()

    # ----- Parameters -----
    A, beta, k0 = 1.0, 2.0, 0.02
    sigma_in_sq = 1e-3
    sigma_out_sq = 1e-3
    P_target = 0.05
    D_values = [0.08, 0.8, 8.0, 80.0]

    # Grid.
    kmax, Nk = 3.0, 400
    wmax, Nw = 80.0, 801
    k, wk = radial_grid(kmax, Nk, kmin=k0)
    omega, dw = omega_grid(wmax, Nw)
    weights = build_radial_weights(wk, dw)

    # Display window.
    k_plot_min, k_plot_max = 0.04, 2.5
    w_plot_min, w_plot_max = 0.1, 60.0

    # ----- Compute -----
    cd_grid, v2_grid = [], []
    for D in D_values:
        Cd = C_D(k[:, None], omega[None, :], A=A, beta=beta, D=D, k0=k0)
        lam = solve_lambda(Cd, sigma_in_sq, sigma_out_sq, P_target, weights)
        v2 = v_star_sq(Cd, sigma_in_sq, sigma_out_sq, lam)
        cd_grid.append(Cd)
        v2_grid.append(v2)

    # Shared levels per row.
    cd_max = max(g.max() for g in cd_grid)
    cd_min = cd_max * 1e-12
    cd_levels = np.linspace(np.log10(cd_min), np.log10(cd_max), 21)

    v2_max = max(g.max() for g in v2_grid)
    v2_min = v2_max * 1e-8
    v2_levels = np.linspace(np.log10(v2_min), np.log10(v2_max), 21)

    # ----- Plot -----
    fig, axes = plt.subplots(
        2, len(D_values), figsize=(10.0, 5.0),
        sharey=True, constrained_layout=True,
    )
    K, W = np.meshgrid(k, omega, indexing="ij")

    # Top row: C_D.
    for ax, D, Cd in zip(axes[0], D_values, cd_grid):
        logC = np.log10(np.maximum(Cd, cd_min))
        cs_top = ax.contourf(K, W, logC, levels=cd_levels,
                             cmap="magma", extend="min")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(k_plot_min, k_plot_max)
        ax.set_ylim(w_plot_min, w_plot_max)
        ax.set_title(rf"$D={D:g}$")

    # Bottom row: |v*|^2.
    for ax, D, v2 in zip(axes[1], D_values, v2_grid):
        logv = np.log10(np.maximum(v2, v2_min))
        cs_bot = ax.contourf(K, W, logv, levels=v2_levels,
                             cmap="magma", extend="min")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(k_plot_min, k_plot_max)
        ax.set_ylim(w_plot_min, w_plot_max)
        ax.set_xlabel(r"spatial frequency $k$")

    axes[0, 0].set_ylabel(r"temporal frequency $\omega$")
    axes[1, 0].set_ylabel(r"temporal frequency $\omega$")

    # Separate colorbars for each row.
    cbar_top = fig.colorbar(cs_top, ax=axes[0], shrink=0.88, aspect=15,
                            pad=0.02)
    cbar_top.set_label(r"$\log_{10}\,C_D(k,\omega)$")
    cbar_bot = fig.colorbar(cs_bot, ax=axes[1], shrink=0.88, aspect=15,
                            pad=0.02)
    cbar_bot.set_label(r"$\log_{10}\,|v^\star_D(k,\omega)|^2$")

    os.makedirs("figures", exist_ok=True)
    out = "figures/fig1_input_and_filter.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
