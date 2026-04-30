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
    parameter_palette,
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


def _spatial_kernel_radial(v_sq, f, omega):
    domega = np.gradient(omega)
    v_s_sq = np.sum(v_sq * np.abs(domega)[None, :], axis=1) / (2 * np.pi)
    v_s = np.sqrt(np.maximum(v_s_sq, 0.0))
    f_fine = np.linspace(0.0, 6.0, 1024)
    v_s_interp = np.interp(f_fine, f, v_s, left=v_s[0], right=0.0)
    def vmag(k):
        return np.interp(k, f_fine, v_s_interp, left=v_s_interp[0], right=0.0)
    rx, ry, v_xy = spatial_kernel_2d(vmag, k_max=8.0, n_k=512)
    r_radial, v_radial = radial_cross_section(v_xy, rx, ry)
    return r_radial, v_radial


def _temporal_kernel(v_sq, f, omega):
    domega = np.gradient(omega)
    energy_per_f = np.sum(v_sq * np.abs(domega)[None, :], axis=1)
    i_peak_f = int(np.argmax(energy_per_f))
    v_t_mag = np.sqrt(np.maximum(v_sq[i_peak_f, :], 0.0))
    taper = soft_band_taper(omega, OMEGA_MIN, OMEGA_MAX, alpha=0.25)
    v_t_smooth = v_t_mag * taper
    floor = 1e-3 * max(v_t_smooth.max(), 1e-30)
    v_t_smooth = np.maximum(v_t_smooth, floor)
    t, h_t, _ = minimum_phase_temporal_filter(v_t_smooth, omega)
    return t, h_t


def fig3():
    f, omega = hi_res_grid()

    sigma_out = 1.0
    P0 = 50.0
    beta = 2.0

    sigma_in_fixed = 0.3
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

    for D, color in zip(D_sweep, palette_D):
        v_sq, _ = _solve(f, omega, D, beta, sigma_in_fixed, sigma_out, P0)
        r, v_r = _spatial_kernel_radial(v_sq, f, omega)
        v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
        ax_sp_D.plot(r, v_r_n, color=color, lw=1.3,
                     label=rf"$D={D:.2g}$")
        t, h_t = _temporal_kernel(v_sq, f, omega)
        h_t_n = h_t / max(np.max(np.abs(h_t)), 1e-30)
        ax_t_D.plot(t, h_t_n, color=color, lw=1.3,
                    label=rf"$D={D:.2g}$")

    for sin, color in zip(sigma_in_sweep, palette_S):
        v_sq, _ = _solve(f, omega, D_fixed, beta, sin, sigma_out, P0)
        r, v_r = _spatial_kernel_radial(v_sq, f, omega)
        v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
        ax_sp_S.plot(r, v_r_n, color=color, lw=1.3,
                     label=rf"$\sigma_\mathrm{{in}}={sin:.2g}$")
        t, h_t = _temporal_kernel(v_sq, f, omega)
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

    out = "outputs/fig3_kernels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig3()
