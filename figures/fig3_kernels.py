"""Figure 3: spatial and temporal kernels of the optimal filter.

Two summaries of the spatiotemporal optimal filter |v*(f, ω)|^2:

  - Spatial kernel v_s(r): radial cross-section of the 2D inverse Fourier
    transform of v_s(f) = sqrt((1/2π) ∫ |v*(f,ω)|^2 dω). Symmetric around
    r = 0.

  - Temporal kernel v_t(t): minimum-phase causal IFT of v_t(ω) =
    |v*(f*, ω)|, where f* maximises ∫|v*|^2 dω. A soft Tukey taper at
    the band edges keeps the cepstral reconstruction stable.

Two parameter sweeps: vary D and vary σ_in.
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


def fig3():
    sigma_out = 2.0
    P0 = 50.0

    sigma_in_fixed = 0.2
    D_sweep = np.geomspace(0.05, 50.0, 8)
    D_fixed = 5.0
    sigma_in_sweep = np.geomspace(0.03, 2.0, 8)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.30,
                                          "left": 0.08, "right": 0.97,
                                          "top": 0.92, "bottom": 0.10})
    ax_sp_D, ax_sp_S = axes[0]
    ax_t_D, ax_t_S = axes[1]

    palette_D = parameter_palette(len(D_sweep), cmap="viridis")
    palette_S = parameter_palette(len(sigma_in_sweep), cmap="plasma")

    D_specs = drift_spectrum_specs(D_sweep)
    D_results = run_many(
        D_specs,
        SolveConfig(sigma_in=sigma_in_fixed, sigma_out=sigma_out, P0=P0, grid="hi_res"),
        kernels=True,
    )
    for D, color, result in zip(D_sweep, palette_D, D_results):
        r, v_r = result.spatial_r, result.spatial_v
        v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
        ax_sp_D.plot(r, v_r_n, color=color, lw=1.3,
                     label=rf"$D={D:.2g}$")
        t, h_t = result.temporal_t, result.temporal_v
        h_t_n = h_t / max(np.max(np.abs(h_t)), 1e-30)
        ax_t_D.plot(t, h_t_n, color=color, lw=1.3,
                    label=rf"$D={D:.2g}$")

    sigma_specs = drift_spectrum_specs([D_fixed] * len(sigma_in_sweep))
    sigma_results = [
        run_many(
            [spec],
            SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0, grid="hi_res"),
            kernels=True,
        )[0]
        for spec, sin in zip(sigma_specs, sigma_in_sweep)
    ]
    for sin, color, result in zip(sigma_in_sweep, palette_S, sigma_results):
        r, v_r = result.spatial_r, result.spatial_v
        v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
        ax_sp_S.plot(r, v_r_n, color=color, lw=1.3,
                     label=rf"$\sigma_\mathrm{{in}}={sin:.2g}$")
        t, h_t = result.temporal_t, result.temporal_v
        h_t_n = h_t / max(np.max(np.abs(h_t)), 1e-30)
        ax_t_S.plot(t, h_t_n, color=color, lw=1.3,
                    label=rf"$\sigma_\mathrm{{in}}={sin:.2g}$")

    for ax in [ax_sp_D, ax_sp_S]:
        ax.set_xlim(-3.0, 3.0)
        ax.axhline(0.0, color="0.7", lw=0.4)
        ax.axvline(0.0, color="0.85", lw=0.3)
        ax.set_xlabel(r"$r$ (units)")
        ax.set_ylabel(r"$v_s(r) / \max v_s$")

    for ax in [ax_t_D, ax_t_S]:
        ax.set_xlim(0.0, 0.6)
        ax.axhline(0.0, color="0.7", lw=0.4)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$v_t(t) / \max v_t$")

    ax_sp_D.set_title(rf"Spatial kernel — vary $D$  ($\sigma_\mathrm{{in}}={sigma_in_fixed}$)")
    ax_sp_S.set_title(rf"Spatial kernel — vary $\sigma_\mathrm{{in}}$  ($D={D_fixed}$)")
    ax_t_D.set_title(rf"Temporal kernel — vary $D$  ($\sigma_\mathrm{{in}}={sigma_in_fixed}$)")
    ax_t_S.set_title(rf"Temporal kernel — vary $\sigma_\mathrm{{in}}$  ($D={D_fixed}$)")

    for ax in axes.flat:
        ax.legend(loc="best", handlelength=1.2, fontsize=6.5,
                  frameon=False, ncol=2)

    fig.suptitle(
        "Spatial and temporal kernels of the optimal filter",
        y=0.98, fontsize=10.5,
    )

    out = "./outputs/fig3_kernels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig3()
