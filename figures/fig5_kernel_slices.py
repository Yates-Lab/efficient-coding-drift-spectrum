"""Figure 5: spatial kernel slices at fixed temporal-frequency ω₀ and
temporal kernel slices at fixed spatial-frequency f₀.

For two σ_in levels (low and moderate), we slice the optimal filter at
fixed ω₀ to get a spatial profile v_s(r; ω₀), and at fixed f₀ to get a
temporal profile v_t(t; f₀).
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import (
    SolveConfig,
    run_many,
    spatial_kernel_slice,
    temporal_kernel_slice,
)
from src.plotting import setup_style
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX
from src.power_spectrum_library import drift_spectrum_specs

setup_style()


def fig5():
    D = 5.0
    beta = 2.0
    sigma_out = 1.0
    P0 = 50.0

    sigma_in_levels = [0.1, 0.5]

    omega_slices = np.geomspace(1.0, 300.0, 8)
    f_slices = np.geomspace(0.08, 3.0, 8)

    cmap_omega = plt.get_cmap("viridis")
    cmap_f = plt.get_cmap("plasma")
    log_om_norm = (np.log(omega_slices) - np.log(omega_slices[0])) / (
        np.log(omega_slices[-1]) - np.log(omega_slices[0])
    )
    log_om_norm = 0.10 + 0.85 * log_om_norm
    omega_colors = [cmap_omega(x) for x in log_om_norm]

    log_f_norm = (np.log(f_slices) - np.log(f_slices[0])) / (
        np.log(f_slices[-1]) - np.log(f_slices[0])
    )
    log_f_norm = 0.10 + 0.85 * log_f_norm
    f_colors = [cmap_f(x) for x in log_f_norm]

    fig, axes = plt.subplots(
        2, 2, figsize=(9.4, 6.4),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30,
                     "left": 0.08, "right": 0.97,
                     "top": 0.93, "bottom": 0.08},
    )

    specs = drift_spectrum_specs([D] * len(sigma_in_levels))
    results = [
        run_many(
            [spec],
            SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0, grid="hi_res"),
        )[0]
        for spec, sin in zip(specs, sigma_in_levels)
    ]

    for row, sin in enumerate(sigma_in_levels):
        result = results[row]

        ax_sp = axes[row, 0]
        ax_t = axes[row, 1]

        for w0, color in zip(omega_slices, omega_colors):
            r, v_r = spatial_kernel_slice(result, w0)
            v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
            f_hz = w0 / (2 * np.pi)
            ax_sp.plot(r, v_r_n, color=color, lw=1.3,
                       label=rf"$\omega_0={w0:.2g}$ ({f_hz:.2g}Hz)")
        ax_sp.set_xlim(-3.0, 3.0)
        ax_sp.axhline(0.0, color="0.7", lw=0.4)
        ax_sp.axvline(0.0, color="0.85", lw=0.3)
        ax_sp.set_xlabel(r"$r$ (units)")
        ax_sp.set_ylabel(r"$v_s(r;\,\omega_0) / \max$")
        ax_sp.set_title(
            rf"Spatial slices, $\sigma_\mathrm{{in}}={sin}$;  "
            rf"$I^* = {result.I:.2f}$ nats",
            pad=2,
        )
        ax_sp.legend(loc="upper right", handlelength=1.2, fontsize=6.5,
                     frameon=False, ncol=2)

        for f0, color in zip(f_slices, f_colors):
            t, h_t = temporal_kernel_slice(result, f0)
            h_t_n = h_t / max(np.max(np.abs(h_t)), 1e-30)
            ax_t.plot(t, h_t_n, color=color, lw=1.3,
                      label=rf"$f_0={f0:.2g}$")
        ax_t.set_xlim(0.0, 0.6)
        ax_t.axhline(0.0, color="0.7", lw=0.4)
        ax_t.set_xlabel(r"$t$ (s)")
        ax_t.set_ylabel(r"$v_t(t;\,f_0) / \max$")
        ax_t.set_title(
            rf"Temporal slices, $\sigma_\mathrm{{in}}={sin}$",
            pad=2,
        )
        ax_t.legend(loc="upper right", handlelength=1.2, fontsize=6.5,
                    frameon=False, ncol=2)

    fig.suptitle(
        rf"Spatial kernels at fixed $\omega_0$ and temporal kernels at "
        rf"fixed $f_0$;  $D={D:g}$.  Band: "
        rf"$f \leq {F_MAX:g}$ cyc/u, "
        rf"${OMEGA_MIN:g} \leq |\omega| \leq {OMEGA_MAX:g}$ rad/s "
        rf"(~ {OMEGA_MIN/(2*np.pi):.2g}-{OMEGA_MAX/(2*np.pi):.2g} Hz)",
        y=0.985, fontsize=10.5,
    )

    out = "./outputs/fig5_kernel_slices.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig5()
