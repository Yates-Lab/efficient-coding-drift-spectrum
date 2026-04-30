"""Figure 6: spatial and temporal kernels of the optimal filter under saccades.

Mirror of figure 3, but with the input spectrum given by saccades of
amplitude A acting on a power-law image:

    C_sac(f, ω; A) = C_I(f) · Q_sac(f, ω; A),

where Q_sac is the angle-averaged redistribution kernel of section
src.spectra.saccade_redistribution. The damped-harmonic-oscillator
template (zeta = 0.6, peak at 40 ms) reproduces the saccade traces
in Mostofi et al. (2020), figure 5A.

Two parameter sweeps:
  vary A from 0.5 to 7 degrees (a microsaccade up to a typical natural
    saccade), at fixed σ_in.
  vary σ_in from 0.03 to 2 at a fixed A = 2.5° (the median amplitude in
    Mostofi et al.'s data).

Spatial kernel: radial cross-section of the 2D inverse Fourier transform
of v_s(f) = sqrt((1/2π) ∫ |v*(f, ω)|² dω). Symmetric around r = 0.
Temporal kernel: minimum-phase causal IFT of v_t(ω) = |v*(f*, ω)|, where
f* maximises ∫ |v*|² dω, with a soft Tukey taper at the band edges.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import saccade_spectrum
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


def _solve(f, omega, A, sigma_in, sigma_out, P0):
    C = saccade_spectrum(f, omega, A=A)
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


def fig6():
    f, omega = hi_res_grid()

    sigma_out = 1.0
    P0 = 50.0

    sigma_in_fixed = 0.3
    A_sweep = np.geomspace(0.3, 7.0, 8)  # microsaccade to large saccade
    A_fixed = 2.5
    sigma_in_sweep = np.geomspace(0.03, 2.0, 8)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.30,
                                          "left": 0.08, "right": 0.97,
                                          "top": 0.92, "bottom": 0.10})
    ax_sp_A, ax_sp_S = axes[0]
    ax_t_A, ax_t_S = axes[1]

    palette_A = parameter_palette(len(A_sweep), cmap="viridis")
    palette_S = parameter_palette(len(sigma_in_sweep), cmap="plasma")

    print("Sweeping A (8 values)...")
    for i, (A, color) in enumerate(zip(A_sweep, palette_A)):
        v_sq, I_star = _solve(f, omega, A, sigma_in_fixed, sigma_out, P0)
        r, v_r = _spatial_kernel_radial(v_sq, f, omega)
        v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
        ax_sp_A.plot(r, v_r_n, color=color, lw=1.3,
                     label=rf"$A={A:.2g}^\circ$")
        t, h_t = _temporal_kernel(v_sq, f, omega)
        h_t_n = h_t / max(np.max(np.abs(h_t)), 1e-30)
        ax_t_A.plot(t, h_t_n, color=color, lw=1.3,
                    label=rf"$A={A:.2g}^\circ$")
        print(f"  A = {A:.3g} deg, I* = {I_star:.3f} nats")

    print("Sweeping sigma_in (8 values)...")
    for i, (sin, color) in enumerate(zip(sigma_in_sweep, palette_S)):
        v_sq, I_star = _solve(f, omega, A_fixed, sin, sigma_out, P0)
        r, v_r = _spatial_kernel_radial(v_sq, f, omega)
        v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
        ax_sp_S.plot(r, v_r_n, color=color, lw=1.3,
                     label=rf"$\sigma_\mathrm{{in}}={sin:.2g}$")
        t, h_t = _temporal_kernel(v_sq, f, omega)
        h_t_n = h_t / max(np.max(np.abs(h_t)), 1e-30)
        ax_t_S.plot(t, h_t_n, color=color, lw=1.3,
                    label=rf"$\sigma_\mathrm{{in}}={sin:.2g}$")
        print(f"  sigma_in = {sin:.3g}, I* = {I_star:.3f} nats")

    for ax in [ax_sp_A, ax_sp_S]:
        ax.set_xlim(-3.0, 3.0)
        ax.axhline(0.0, color="0.7", lw=0.4)
        ax.axvline(0.0, color="0.85", lw=0.3)
        ax.set_xlabel(r"$r$ (units)")
        ax.set_ylabel(r"$v_s(r) / \max v_s$")

    for ax in [ax_t_A, ax_t_S]:
        ax.set_xlim(0.0, 0.6)
        ax.axhline(0.0, color="0.7", lw=0.4)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$v_t(t) / \max v_t$")

    ax_sp_A.set_title(rf"Spatial kernel — vary $A$  ($\sigma_\mathrm{{in}}={sigma_in_fixed}$)")
    ax_sp_S.set_title(rf"Spatial kernel — vary $\sigma_\mathrm{{in}}$  ($A={A_fixed}^\circ$)")
    ax_t_A.set_title(rf"Temporal kernel — vary $A$  ($\sigma_\mathrm{{in}}={sigma_in_fixed}$)")
    ax_t_S.set_title(rf"Temporal kernel — vary $\sigma_\mathrm{{in}}$  ($A={A_fixed}^\circ$)")

    for ax in axes.flat:
        ax.legend(loc="best", handlelength=1.2, fontsize=6.5,
                  frameon=False, ncol=2)

    fig.suptitle(
        "Optimal filter under saccades: spatial and temporal kernels",
        y=0.98, fontsize=10.5,
    )

    out = "./outputs/fig6_saccade_kernels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig6()
