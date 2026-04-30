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

from src.spectra import drift_spectrum
from src.solver import solve_efficient_coding
from src.kernels import (
    spatial_kernel_2d,
    radial_cross_section,
    minimum_phase_temporal_filter,
    soft_band_taper,
)
from src.plotting import (
    setup_style,
    radial_weights,
    band_mask_radial,
)
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, hi_res_grid

setup_style()


def _solve(f, omega, D, beta, sigma_in, sigma_out, P0):
    F = f[:, None]
    W = omega[None, :]
    C = drift_spectrum(F, W, D=D, beta=beta)
    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    weights_b = weights * mask
    v_sq, lam, I = solve_efficient_coding(
        C, sigma_in, sigma_out, P0, weights_b, band_mask=mask,
    )
    return v_sq, I


def _spatial_slice(v_sq, f, i_omega0):
    v_mag_at_w0 = np.sqrt(np.maximum(v_sq[:, i_omega0], 0.0))
    f_fine = np.linspace(0.0, 6.0, 1024)
    v_interp = np.interp(f_fine, f, v_mag_at_w0, left=v_mag_at_w0[0], right=0.0)
    def vmag(k):
        return np.interp(k, f_fine, v_interp, left=v_interp[0], right=0.0)
    rx, ry, v_xy = spatial_kernel_2d(vmag, k_max=8.0, n_k=512)
    r_radial, v_radial = radial_cross_section(v_xy, rx, ry)
    return r_radial, v_radial


def _temporal_slice(v_sq, omega, i_f0):
    v_mag_at_f0 = np.sqrt(np.maximum(v_sq[i_f0, :], 0.0))
    taper = soft_band_taper(omega, OMEGA_MIN, OMEGA_MAX, alpha=0.25)
    v_t_smooth = v_mag_at_f0 * taper
    floor = 1e-3 * max(v_t_smooth.max(), 1e-30)
    v_t_smooth = np.maximum(v_t_smooth, floor)
    t, h_t, _ = minimum_phase_temporal_filter(v_t_smooth, omega)
    return t, h_t


def fig5():
    f, omega = hi_res_grid()

    D = 5.0
    beta = 2.0
    sigma_out = 1.0
    P0 = 50.0

    sigma_in_levels = [0.1, 0.5]

    omega_slices = np.geomspace(1.0, 300.0, 8)
    f_slices = np.geomspace(0.08, 3.0, 8)

    omega_indices = [int(np.argmin(np.abs(omega - w))) for w in omega_slices]
    f_indices = [int(np.argmin(np.abs(f - fv))) for fv in f_slices]

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

    for row, sin in enumerate(sigma_in_levels):
        v_sq, I_star = _solve(f, omega, D, beta, sin, sigma_out, P0)

        ax_sp = axes[row, 0]
        ax_t = axes[row, 1]

        for w0, i_w, color in zip(omega_slices, omega_indices, omega_colors):
            r, v_r = _spatial_slice(v_sq, f, i_w)
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
            rf"$I^* = {I_star:.2f}$ nats",
            pad=2,
        )
        ax_sp.legend(loc="upper right", handlelength=1.2, fontsize=6.5,
                     frameon=False, ncol=2)

        for f0, i_f, color in zip(f_slices, f_indices, f_colors):
            t, h_t = _temporal_slice(v_sq, omega, i_f)
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
