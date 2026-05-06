"""Figure 6: optimal kernels under saccades and Brownian drift.

The two regimes are plain core spectrum objects:

    saccade: C = I(f) Q_saccade(A)
    drift:   C = I(f) Q_drift(D)
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import run, extract_kernels
from src.plotting import setup_style, parameter_palette
from src.spectra import SaccadeSpectrum, DriftSpectrum

setup_style()


def _plot_result_pair(ax_sp, ax_t, result, color, label, spatial_xlim, temporal_xlim):
    v_norm = result.spatial_v / max(np.max(np.abs(result.spatial_v)), 1e-30)
    h_norm = result.temporal_v / max(np.max(np.abs(result.temporal_v)), 1e-30)
    ax_sp.plot(result.spatial_r, v_norm, color=color, lw=1.3, label=label)
    ax_t.plot(result.temporal_t, h_norm, color=color, lw=1.3, label=label)
    ax_sp.set_xlim(*spatial_xlim)
    ax_t.set_xlim(*temporal_xlim)


def fig6():
    saccade_spec = SaccadeSpectrum(A=4.4)
    drift_spec = DriftSpectrum(D=0.0375)

    sigma_out = 1.0
    P0 = 50.0
    sigma_in_sweep = np.geomspace(0.03, 2.0, 8)

    fig, axes = plt.subplots(
        2, 2, figsize=(9.0, 6.0),
        gridspec_kw={"hspace": 0.40, "wspace": 0.30,
                     "left": 0.08, "right": 0.97,
                     "top": 0.90, "bottom": 0.10},
    )
    ax_sp_saccade, ax_sp_drift = axes[0]
    ax_t_saccade, ax_t_drift = axes[1]

    palette = parameter_palette(len(sigma_in_sweep), cmap="plasma")

    print("Solving analytic saccade and drift kernels across sigma_in...")
    for sin, color in zip(sigma_in_sweep, palette):
        r_saccade = run(
            saccade_spec, sigma_in=sin, sigma_out=sigma_out, P0=P0,
            grid="hi_res",
        )
        r_drift = run(
            drift_spec, sigma_in=sin, sigma_out=sigma_out, P0=P0,
            grid="hi_res",
        )
        extract_kernels(r_saccade)
        extract_kernels(r_drift)
        label = rf"$\sigma_\mathrm{{in}}={sin:.2g}$"
        _plot_result_pair(
            ax_sp_saccade, ax_t_saccade, r_saccade, color, label,
            spatial_xlim=(-4.0, 4.0), temporal_xlim=(0.0, 0.30),
        )
        _plot_result_pair(
            ax_sp_drift, ax_t_drift, r_drift, color, label,
            spatial_xlim=(-4.0, 4.0), temporal_xlim=(0.0, 0.30),
        )
        print(
            f"  sigma_in={sin:.3g}: saccade I*={r_saccade.I:.3f}, "
            f"f_peak={r_saccade.f_peak:.3f}; drift I*={r_drift.I:.3f}, "
            f"f_peak={r_drift.f_peak:.3f}"
        )

    for ax in axes[0]:
        ax.axhline(0.0, color="0.7", lw=0.4)
        ax.axvline(0.0, color="0.85", lw=0.3)
        ax.set_xlabel(r"$r$ (deg)")
        ax.set_ylabel(r"$v_s(r) / \max v_s$")
        ax.legend(loc="best", handlelength=1.2, fontsize=6.5,
                  frameon=False, ncol=2)

    for ax in axes[1]:
        ax.axhline(0.0, color="0.7", lw=0.4)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$v_t(t) / \max v_t$")
        ax.legend(loc="best", handlelength=1.2, fontsize=6.5,
                  frameon=False, ncol=2)

    ax_sp_saccade.set_title("Saccade: spatial kernel")
    ax_sp_drift.set_title("Drift: spatial kernel")
    ax_t_saccade.set_title("Saccade: temporal kernel")
    ax_t_drift.set_title("Drift: temporal kernel")

    fig.suptitle(
        "Optimal filters under analytic saccade and Brownian drift spectra "
        r"($C_\mathrm{saccade}=I Q_\mathrm{saccade}$, "
        r"$C_\mathrm{drift}=I Q_\mathrm{drift}$)",
        y=0.98, fontsize=10.5,
    )

    out = "./outputs/fig6_saccade_kernels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig6()
