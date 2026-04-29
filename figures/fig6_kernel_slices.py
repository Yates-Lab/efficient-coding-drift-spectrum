"""Figure 6: spatial kernel at fixed temporal-frequency slices,
and temporal kernel at fixed spatial-frequency slices.

For each condition (σ_in × {direct, aliased}), we slice the optimal
filter at fixed ω₀ to get a spatial profile v_s(r; ω₀), and at fixed f₀
to get a temporal profile v_t(t; f₀).

  Spatial slice:  v_s(r; ω₀) = 2D inverse FT of |v*(f, ω₀)|.
                  Plotted as the radial cross-section (symmetric around r=0).
  Temporal slice: v_t(t; f₀) = min-phase IFT of |v*(f₀, ω)|, after a soft
                  Tukey taper at the band edges.

Layout: 4 rows (σ_in low/high × direct/aliased) × 2 columns (spatial, temporal).
Within each panel, slice values are colored by frequency.

D = 5 throughout. σ_in levels are 0.1 (low) and 0.5 (medium). Slices:
  ω₀ ∈ {1, 5, 25, 100, 300} rad/s ≈ {0.16, 0.8, 4, 16, 48} Hz
  f₀ ∈ {0.1, 0.3, 1.0, 2.5} cycles/unit
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/claude/efficient_coding")

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import drift_spectrum
from src.aliasing import per_cell_aliased_spectrum
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


def _solve(f, omega, D, beta, sigma_in, sigma_out, P0, aliased):
    F = f[:, None]
    W = omega[None, :]
    base = lambda f_, om: drift_spectrum(f_, om, D=D, beta=beta)
    if aliased:
        C = per_cell_aliased_spectrum(base, F, W, m_max=12)
    else:
        C = base(F, W)
    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    weights_b = weights * mask
    v_sq, lam, I = solve_efficient_coding(
        C, sigma_in, sigma_out, P0, weights_b, band_mask=mask,
    )
    return v_sq, I, mask


def _spatial_slice(v_sq, f, i_omega0):
    """v_s(r; ω₀): 2D-IFT of |v*(f, ω₀)| then radial cross-section."""
    v_mag_at_w0 = np.sqrt(np.maximum(v_sq[:, i_omega0], 0.0))
    f_fine = np.linspace(0.0, 6.0, 1024)
    v_interp = np.interp(f_fine, f, v_mag_at_w0, left=v_mag_at_w0[0], right=0.0)

    def vmag(k):
        return np.interp(k, f_fine, v_interp, left=v_interp[0], right=0.0)

    rx, ry, v_xy = spatial_kernel_2d(vmag, k_max=8.0, n_k=512)
    r_radial, v_radial = radial_cross_section(v_xy, rx, ry)
    return r_radial, v_radial


def _temporal_slice(v_sq, omega, i_f0):
    """v_t(t; f₀): min-phase IFT of |v*(f₀, ω)| with a soft Tukey taper."""
    v_mag_at_f0 = np.sqrt(np.maximum(v_sq[i_f0, :], 0.0))
    taper = soft_band_taper(omega, OMEGA_MIN, OMEGA_MAX, alpha=0.25)
    v_t_smooth = v_mag_at_f0 * taper
    floor = 1e-3 * max(v_t_smooth.max(), 1e-30)
    v_t_smooth = np.maximum(v_t_smooth, floor)
    t, h_t, _ = minimum_phase_temporal_filter(v_t_smooth, omega)
    return t, h_t


def fig6():
    f, omega = hi_res_grid()

    D = 5.0
    beta = 2.0
    sigma_out = 1.0
    P0 = 50.0

    sigma_in_levels = [0.1, 0.5]
    aliased_flags = [False, True]

    # Slice values, kept inside the band
    omega_slices = [1.0, 5.0, 25.0, 100.0, 300.0]   # rad/s
    f_slices = [0.1, 0.3, 1.0, 2.5]                 # cycles/unit

    # Map each slice value to a grid index
    omega_indices = [int(np.argmin(np.abs(omega - w))) for w in omega_slices]
    f_indices = [int(np.argmin(np.abs(f - fv))) for fv in f_slices]

    # Color palettes encoding slice value on a log scale
    cmap_omega = plt.get_cmap("viridis")
    cmap_f = plt.get_cmap("plasma")
    log_omega_norm = (np.log(omega_slices) - np.log(omega_slices[0])) / (
        np.log(omega_slices[-1]) - np.log(omega_slices[0])
    )
    log_omega_norm = 0.10 + 0.85 * log_omega_norm
    omega_colors = [cmap_omega(x) for x in log_omega_norm]
    log_f_norm = (np.log(f_slices) - np.log(f_slices[0])) / (
        np.log(f_slices[-1]) - np.log(f_slices[0])
    )
    log_f_norm = 0.10 + 0.85 * log_f_norm
    f_colors = [cmap_f(x) for x in log_f_norm]

    fig, axes = plt.subplots(
        4, 2, figsize=(9.4, 11.0),
        gridspec_kw={"hspace": 0.55, "wspace": 0.28,
                     "left": 0.08, "right": 0.97,
                     "top": 0.95, "bottom": 0.05},
    )

    row = 0
    for sin in sigma_in_levels:
        for aliased in aliased_flags:
            v_sq, I_star, _ = _solve(
                f, omega, D, beta, sin, sigma_out, P0, aliased=aliased,
            )

            ax_sp = axes[row, 0]
            ax_t = axes[row, 1]

            # Spatial slices
            for w0, i_w, color in zip(omega_slices, omega_indices, omega_colors):
                r, v_r = _spatial_slice(v_sq, f, i_w)
                # normalize by peak of this slice
                v_r_n = v_r / max(np.max(np.abs(v_r)), 1e-30)
                f_hz = w0 / (2 * np.pi)
                ax_sp.plot(r, v_r_n, color=color, lw=1.4,
                           label=rf"$\omega_0 = {w0:g}$ ({f_hz:.2g} Hz)")
            ax_sp.set_xlim(-3.0, 3.0)
            ax_sp.axhline(0.0, color="0.7", lw=0.4)
            ax_sp.axvline(0.0, color="0.85", lw=0.4)
            ax_sp.set_xlabel(r"$r$ (units)")
            ax_sp.set_ylabel(r"$v_s(r;\,\omega_0) / \max$")
            label = "direct" if not aliased else "aliased"
            ax_sp.set_title(
                rf"Spatial slices, $\sigma_\mathrm{{in}}={sin:g}$ ({label}); "
                rf"$I^* = {I_star:.2f}$ nats",
                pad=2,
            )

            # Temporal slices
            for f0, i_f, color in zip(f_slices, f_indices, f_colors):
                t, h_t = _temporal_slice(v_sq, omega, i_f)
                h_t_n = h_t / max(np.max(np.abs(h_t)), 1e-30)
                ax_t.plot(t, h_t_n, color=color, lw=1.4,
                          label=rf"$f_0 = {f0:g}$ cyc/u")
            ax_t.set_xlim(0.0, 0.6)
            ax_t.axhline(0.0, color="0.7", lw=0.4)
            ax_t.set_xlabel(r"$t$ (s)")
            ax_t.set_ylabel(r"$v_t(t;\,f_0) / \max$")
            ax_t.set_title(
                rf"Temporal slices, $\sigma_\mathrm{{in}}={sin:g}$ ({label})",
                pad=2,
            )

            ax_sp.legend(loc="upper right", handlelength=1.3, fontsize=7,
                         frameon=False)
            ax_t.legend(loc="upper right", handlelength=1.3, fontsize=7,
                        frameon=False)
            row += 1

    fig.suptitle(
        rf"Spatial kernels at fixed $\omega_0$ and temporal kernels at "
        rf"fixed $f_0$;  $D = {D:g}$.  Band: "
        rf"$f \leq {F_MAX:g}$ cyc/u, "
        rf"${OMEGA_MIN:g} \leq |\omega| \leq {OMEGA_MAX:g}$ rad/s "
        rf"(≈ {OMEGA_MIN/(2*np.pi):.2g}–{OMEGA_MAX/(2*np.pi):.2g} Hz)",
        y=0.985, fontsize=10.5,
    )

    out = "/home/claude/efficient_coding/outputs/fig6_kernel_slices.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig6()
