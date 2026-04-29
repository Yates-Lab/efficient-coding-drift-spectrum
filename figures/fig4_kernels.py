"""Figure 4: spatial and temporal kernels of the optimal filter.

Two summaries of the 2+1D optimal filter |v*(f, ω)|^2:

  - Spatial kernel v_s(r): the radial cross-section of the 2D inverse
    Fourier transform of v_s(f) = sqrt((1/2π) ∫ |v*(f,ω)|^2 dω).
    Plotted symmetric around r = 0 to make the rotational symmetry explicit.

  - Temporal kernel v_t(t): the min-phase causal IFT of v_t(ω), where
    v_t(ω) = |v*(f*, ω)| at the f* that maximises ∫|v*|^2 dω.
    Before the cepstral reconstruction we apply a soft Tukey taper at the
    band edges; without it, the hard band cutoff produces large negative
    log-magnitude spikes that contaminate the cepstrum and shift the
    apparent impulse-response peak by ~0.5 s.

Two parameter sweeps (vary D, vary σ_in), each shown twice — once with the
direct input spectrum C(f, ω), once with the per-cell-Nyquist aliased
spectrum C^sample(f, ω). Comparing the two reveals how aliasing reshapes
both the spatial and temporal kernels.
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


def _make_grid():
    return hi_res_grid()


def _solve(f, omega, D, beta, sigma_in, sigma_out, P0, aliased=False):
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


def _spatial_kernel_radial(v_sq, f, omega):
    """Build v_s(f) = sqrt((1/2π) ∫ |v*|^2 dω) and 2D-IFT to get v_s(x, y).
    Returns rx, ry, v_xy (full 2D field, symmetric around origin)."""
    domega = np.gradient(omega)
    v_s_sq = np.sum(v_sq * np.abs(domega)[None, :], axis=1) / (2 * np.pi)
    v_s = np.sqrt(np.maximum(v_s_sq, 0.0))
    f_fine = np.linspace(0.0, 6.0, 1024)
    v_s_interp = np.interp(f_fine, f, v_s, left=v_s[0], right=0.0)
    def vmag(k):
        return np.interp(k, f_fine, v_s_interp, left=v_s_interp[0], right=0.0)
    rx, ry, v_xy = spatial_kernel_2d(vmag, k_max=8.0, n_k=512)
    return rx, ry, v_xy


def _temporal_kernel(v_sq, f, omega):
    """v_t(ω) = |v*(f*, ω)| with a soft Tukey taper at the band edges, then
    min-phase IFT. The taper is essential — without it, log|v_t| has large
    negative spikes outside the band that contaminate the cepstrum.
    """
    domega = np.gradient(omega)
    energy_per_f = np.sum(v_sq * np.abs(domega)[None, :], axis=1)
    i_peak_f = int(np.argmax(energy_per_f))
    v_t_mag = np.sqrt(np.maximum(v_sq[i_peak_f, :], 0.0))
    # Apply soft Tukey taper at the band edges
    taper = soft_band_taper(omega, OMEGA_MIN, OMEGA_MAX, alpha=0.25)
    v_t_smooth = v_t_mag * taper
    # Use a finite floor (1% of peak) so log is well behaved
    floor = 1e-3 * max(v_t_smooth.max(), 1e-30)
    v_t_smooth = np.maximum(v_t_smooth, floor)
    t, h_t, _ = minimum_phase_temporal_filter(v_t_smooth, omega)
    return t, h_t


def fig4():
    f, omega = _make_grid()

    sigma_out = 1.0
    P0 = 50.0
    beta = 2.0

    sigma_in_fixed = 0.3
    D_sweep = [0.1, 1.0, 5.0, 30.0]
    D_fixed = 5.0
    sigma_in_sweep = [0.05, 0.2, 0.5, 1.5]

    fig, axes = plt.subplots(4, 2, figsize=(8.6, 9.0),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.30,
                                          "left": 0.10, "right": 0.97,
                                          "top": 0.94, "bottom": 0.06})
    # Row 0: spatial vs D, direct & aliased
    # Row 1: spatial vs σ_in, direct & aliased
    # Row 2: temporal vs D, direct & aliased
    # Row 3: temporal vs σ_in, direct & aliased

    palette_D = parameter_palette(len(D_sweep), cmap="viridis")
    palette_S = parameter_palette(len(sigma_in_sweep), cmap="plasma")

    def _plot_spatial_sweep(ax, sweep, sigma_in, D, palette, label_fmt, vary):
        for val, color in zip(sweep, palette):
            if vary == "D":
                D_use, sin_use = val, sigma_in
            else:
                D_use, sin_use = D, val
            v_sq, _, _ = _solve(f, omega, D_use, beta, sin_use, sigma_out,
                                 P0, aliased=ax_is_aliased)
            rx, ry, v_xy = _spatial_kernel_radial(v_sq, f, omega)
            r_radial, v_radial = radial_cross_section(v_xy, rx, ry)
            v_radial_n = v_radial / np.max(np.abs(v_radial))
            ax.plot(r_radial, v_radial_n, color=color, lw=1.4,
                    label=label_fmt.format(val))
        ax.set_xlim(-3.0, 3.0)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.axvline(0.0, color="0.85", lw=0.4)
        ax.set_xlabel(r"$r$ (units)")
        ax.set_ylabel(r"$v_s(r) / \max v_s$")

    def _plot_temporal_sweep(ax, sweep, sigma_in, D, palette, label_fmt, vary):
        for val, color in zip(sweep, palette):
            if vary == "D":
                D_use, sin_use = val, sigma_in
            else:
                D_use, sin_use = D, val
            v_sq, _, _ = _solve(f, omega, D_use, beta, sin_use, sigma_out,
                                 P0, aliased=ax_is_aliased)
            t, h_t = _temporal_kernel(v_sq, f, omega)
            h_t_n = h_t / np.max(np.abs(h_t))
            ax.plot(t, h_t_n, color=color, lw=1.4, label=label_fmt.format(val))
        ax.set_xlim(0.0, 1.0)
        ax.axhline(0.0, color="0.7", lw=0.5)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$v_t(t) / \max v_t$")

    # Spatial — row 0 + 1
    for col, aliased_flag in enumerate([False, True]):
        ax_is_aliased = aliased_flag
        col_label = "aliased input" if aliased_flag else "direct input"
        _plot_spatial_sweep(
            axes[0, col], D_sweep, sigma_in_fixed, None,
            palette_D, r"$D = {:g}$", "D",
        )
        axes[0, col].set_title(
            rf"Spatial kernel — vary $D$ ({col_label})"
        )
        _plot_spatial_sweep(
            axes[1, col], sigma_in_sweep, None, D_fixed,
            palette_S, r"$\sigma_\mathrm{{in}} = {:g}$", "sigma_in",
        )
        axes[1, col].set_title(
            rf"Spatial kernel — vary $\sigma_\mathrm{{in}}$ ({col_label})"
        )
        _plot_temporal_sweep(
            axes[2, col], D_sweep, sigma_in_fixed, None,
            palette_D, r"$D = {:g}$", "D",
        )
        axes[2, col].set_title(
            rf"Temporal kernel — vary $D$ ({col_label})"
        )
        _plot_temporal_sweep(
            axes[3, col], sigma_in_sweep, None, D_fixed,
            palette_S, r"$\sigma_\mathrm{{in}} = {:g}$", "sigma_in",
        )
        axes[3, col].set_title(
            rf"Temporal kernel — vary $\sigma_\mathrm{{in}}$ ({col_label})"
        )

    for ax in axes.flat:
        ax.legend(loc="best", handlelength=1.3, fontsize=7.5, frameon=False)

    fig.suptitle(
        "Spatial and temporal kernels: direct vs per-cell-Nyquist aliased input",
        y=0.985, fontsize=10.5,
    )

    out = "outputs/fig4_kernels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig4()
