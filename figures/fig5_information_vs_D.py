"""Figure 5: maximum mutual information I^*(D) for varying input noise σ_in.

Direct (solid) vs per-cell-Nyquist aliased (dashed) input spectra. The
inverted-U dependence on drift D is preserved under aliasing, but the
aliased input has higher total power per cell (folded copies sum) so the
curves are shifted upward. The peak D^* shift with σ_in is preserved in
both versions.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import drift_spectrum
from src.aliasing import per_cell_aliased_spectrum
from src.solver import solve_efficient_coding
from src.plotting import (
    setup_style,
    radial_weights,
    band_mask_radial,
    parameter_palette,
)
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, fast_grid

setup_style()


def _solve(f, omega, D, beta, sigma_in, sigma_out, P0, weights_b, mask, aliased):
    F = f[:, None]
    W = omega[None, :]
    base = lambda f_, om: drift_spectrum(f_, om, D=D, beta=beta)
    if aliased:
        C = per_cell_aliased_spectrum(base, F, W, m_max=6)
    else:
        C = base(F, W)
    _, _, I = solve_efficient_coding(
        C, sigma_in, sigma_out, P0, weights_b, band_mask=mask,
    )
    return I


def fig5():
    f, omega = fast_grid()

    sigma_out = 1.0
    P0 = 50.0
    beta = 2.0

    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    weights_b = weights * mask

    D_grid = np.geomspace(0.01, 200.0, 28)
    sigma_in_levels = [0.05, 0.15, 0.4, 1.0, 2.5]

    I_direct = np.zeros((len(sigma_in_levels), len(D_grid)))
    I_alias = np.zeros((len(sigma_in_levels), len(D_grid)))
    for i, sin in enumerate(sigma_in_levels):
        for j, D in enumerate(D_grid):
            I_direct[i, j] = _solve(f, omega, D, beta, sin, sigma_out, P0,
                                     weights_b, mask, aliased=False)
            I_alias[i, j] = _solve(f, omega, D, beta, sin, sigma_out, P0,
                                    weights_b, mask, aliased=True)
        print(f"  σ_in = {sin}: done")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0),
                             gridspec_kw={"wspace": 0.30, "left": 0.08,
                                          "right": 0.97, "top": 0.86,
                                          "bottom": 0.13})
    ax_lin, ax_log = axes

    palette = parameter_palette(len(sigma_in_levels), cmap="plasma")

    for i, (sin, color) in enumerate(zip(sigma_in_levels, palette)):
        for ax in (ax_lin, ax_log):
            ax.plot(D_grid, I_direct[i], color=color, lw=1.5, ls="-",
                    label=rf"$\sigma_\mathrm{{in}} = {sin:g}$" if ax is ax_lin else None)
            ax.plot(D_grid, I_alias[i], color=color, lw=1.5, ls="--", alpha=0.85)
            i_peak_d = int(np.argmax(I_direct[i]))
            i_peak_a = int(np.argmax(I_alias[i]))
            ax.scatter([D_grid[i_peak_d]], [I_direct[i, i_peak_d]],
                       color=color, s=22, zorder=5, edgecolor="white", lw=0.7)
            ax.scatter([D_grid[i_peak_a]], [I_alias[i, i_peak_a]],
                       color=color, s=22, zorder=5, edgecolor="white", lw=0.7,
                       marker="D")

    for ax in (ax_lin, ax_log):
        ax.set_xscale("log")
        ax.set_xlabel(r"drift coefficient  $D$")
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
        ax.grid(True, which="minor", lw=0.2, alpha=0.2)
    ax_lin.set_ylabel(r"$I^*(D)$  (nats)")
    ax_log.set_ylabel(r"$I^*(D)$  (nats, log)")
    ax_log.set_yscale("log")
    ax_log.set_ylim(1e-3, ax_log.get_ylim()[1])

    ax_lin.set_title("linear")
    ax_log.set_title("log")

    # Build legend with the σ_in entries plus a separator entry for ls/markers
    handles, labels = ax_lin.get_legend_handles_labels()
    # Add explanatory entries for line styles
    from matplotlib.lines import Line2D
    handles.append(Line2D([], [], color="0.3", ls="-", label="direct"))
    handles.append(Line2D([], [], color="0.3", ls="--", label="aliased"))
    ax_lin.legend(handles=handles, loc="upper right", handlelength=1.6,
                  frameon=False, fontsize=8)

    fig.suptitle(
        rf"$I^\star(D)$ for varying input noise; "
        rf"$P_0={P0:g}$, $\sigma_\mathrm{{out}}={sigma_out:g}$, $\beta={beta:g}$. "
        rf"Solid: direct.  Dashed: aliased.",
        y=0.97, fontsize=10.5,
    )

    out = "outputs/fig5_information_vs_D.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    print("\nPeak D* (direct -> aliased):")
    for sin, row_d, row_a in zip(sigma_in_levels, I_direct, I_alias):
        i_d = int(np.argmax(row_d))
        i_a = int(np.argmax(row_a))
        print(f"  σ_in = {sin:g}: "
              f"D*={D_grid[i_d]:.3g}, I*={row_d[i_d]:.3f}  ->  "
              f"D*={D_grid[i_a]:.3g}, I*={row_a[i_a]:.3f}")


if __name__ == "__main__":
    fig5()
