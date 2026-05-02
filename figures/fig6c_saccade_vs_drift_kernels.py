"""Figure 6c: saccade-vs-drift kernel comparison.

The fixation-cycle model is a selector: early fixation uses the Mostofi
analytic saccade transient and late fixation uses Brownian drift.  This figure
compares their resulting kernels without invoking the old equivalent-drift
Poisson-saccade construction.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import SolveConfig, run_many
from src.plotting import setup_style
from src.power_spectrum_library import drift_spectrum_specs, saccade_spectrum_specs

setup_style()


def fig6c():
    sigma_in, sigma_out, P0 = 0.3, 1.0, 50.0
    cases = [
        (0.5, 0.035, "small saccade / slow drift"),
        (2.5, 0.075, "medium saccade / canonical drift"),
        (7.0, 0.30, "large saccade / fast drift"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.30,
                                          "left": 0.06, "right": 0.98,
                                          "top": 0.92, "bottom": 0.10})

    color_sac = "#1f6fb4"
    color_drift = "#d8540e"

    for col, (A, D, label) in enumerate(cases):
        sac_spec = saccade_spectrum_specs([A])[0]
        drift_spec = drift_spectrum_specs([D])[0]
        sac_result, drift_result = run_many(
            [sac_spec, drift_spec],
            SolveConfig(sigma_in=sigma_in, sigma_out=sigma_out, P0=P0, grid="hi_res"),
            kernels=True,
        )
        r_s, vr_s = sac_result.spatial_r, sac_result.spatial_v
        t_s, ht_s = sac_result.temporal_t, sac_result.temporal_v
        r_d, vr_d = drift_result.spatial_r, drift_result.spatial_v
        t_d, ht_d = drift_result.temporal_t, drift_result.temporal_v
        vr_s_n = vr_s / np.max(np.abs(vr_s))
        vr_d_n = vr_d / np.max(np.abs(vr_d))
        ht_s_n = ht_s / np.max(np.abs(ht_s))
        ht_d_n = ht_d / np.max(np.abs(ht_d))

        ax_sp = axes[0, col]
        ax_sp.plot(r_s, vr_s_n, color=color_sac, lw=1.6,
                   label=rf"saccade $A={A:g}^\circ$ ($I^*={sac_result.I:.2f}$)")
        ax_sp.plot(r_d, vr_d_n, color=color_drift, lw=1.4, ls="--",
                   label=rf"drift $D={D:g}$ ($I^*={drift_result.I:.2f}$)")
        ax_sp.set_xlim(-3.0, 3.0)
        ax_sp.axhline(0.0, color="0.7", lw=0.4)
        ax_sp.axvline(0.0, color="0.85", lw=0.3)
        ax_sp.set_xlabel(r"$r$ (units)")
        ax_sp.set_ylabel(r"$v_s(r) / \max v_s$")
        ax_sp.set_title(rf"{label}: spatial kernel  ($A={A:g}^\circ$)",
                        fontsize=9, pad=2)
        ax_sp.legend(loc="best", fontsize=7, frameon=False)

        ax_t = axes[1, col]
        ax_t.plot(t_s, ht_s_n, color=color_sac, lw=1.6,
                  label="saccade")
        ax_t.plot(t_d, ht_d_n, color=color_drift, lw=1.4, ls="--",
                  label=rf"drift $D={D:g}$")
        ax_t.set_xlim(0.0, 0.4)
        ax_t.axhline(0.0, color="0.7", lw=0.4)
        ax_t.set_xlabel(r"$t$ (s)")
        ax_t.set_ylabel(r"$v_t(t) / \max v_t$")
        ax_t.set_title(f"{label}: temporal kernel", fontsize=9, pad=2)
        ax_t.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(
        "Mostofi saccade transients versus Brownian drift selector states",
        y=0.97, fontsize=10,
    )

    out = "./outputs/fig6c_saccade_vs_drift_kernels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig6c()
