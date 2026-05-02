"""Figure 4: maximum mutual information I^*(D) for varying input noise σ_in.

Direct (no aliasing) input spectrum. The inverted-U shape arises because
too-little drift leaves signal at temporal DC where the band excludes it,
and too-much drift pushes signal beyond the upper temporal cutoff.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import SolveConfig, run_many
from src.plotting import (
    setup_style,
    parameter_palette,
)
from src.power_spectrum_library import drift_spectrum_specs

setup_style()


def fig4():
    sigma_out = 2.0
    P0 = 20.0
    beta = 2.0

    D_grid = np.geomspace(0.01, 200.0, 40)
    sigma_in_levels = np.geomspace(0.03, 3.0, 8)
    specs = drift_spectrum_specs(D_grid)

    I_table = np.zeros((len(sigma_in_levels), len(D_grid)))
    for i, sin in enumerate(sigma_in_levels):
        results = run_many(
            specs,
            SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0, grid="fast"),
        )
        I_table[i] = [r.I for r in results]
        print(f"  sigma_in = {sin:.3g}: done")

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0),
                             gridspec_kw={"wspace": 0.30, "left": 0.08,
                                          "right": 0.97, "top": 0.86,
                                          "bottom": 0.13})
    ax_lin, ax_log = axes

    palette = parameter_palette(len(sigma_in_levels), cmap="plasma")

    for i, (sin, color) in enumerate(zip(sigma_in_levels, palette)):
        I_curve = I_table[i]
        for jj, ax in enumerate((ax_log, ax_lin)):
            if jj == 1:
                I_curve = I_curve / max(I_curve)

            ax.plot(D_grid, I_curve, color=color, lw=1.5,
                    label=rf"$\sigma_\mathrm{{in}}={sin:.2g}$" if ax is ax_lin else None)
            i_peak = int(np.argmax(I_curve))
            ax.scatter([D_grid[i_peak]], [I_curve[i_peak]],
                       color=color, s=22, zorder=5, edgecolor="white", lw=0.7)

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

    ax_lin.legend(loc="upper right", handlelength=1.4, frameon=False,
                  fontsize=7.5, ncol=2)

    fig.suptitle(
        rf"$I^\star(D)$ for varying input noise; "
        rf"$P_0={P0:g}$, $\sigma_\mathrm{{out}}={sigma_out:g}$, $\beta={beta:g}$",
        y=0.97, fontsize=10.5,
    )

    out = "./outputs/fig4_information_vs_D.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    print("\nPeak D*:")
    for sin, row in zip(sigma_in_levels, I_table):
        i_peak = int(np.argmax(row))
        print(f"  sigma_in = {sin:.3g}: D* ~ {D_grid[i_peak]:.3g}, "
              f"I* = {row[i_peak]:.3f} nats")


if __name__ == "__main__":
    fig4()
