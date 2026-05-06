"""Figure 6b: diagnostic plot of the saccade redistribution kernel and
spectrum.

Sanity-check that Q_sac(f, ω; A) and C_sac = C_I · Q_sac behave as expected:
  - At low f, Q ∝ f^2 (whitening regime)
  - At high f, Q saturates (saturation regime)
  - Crossover frequency ~ 1/(2A), i.e. shifts left as A increases
  - Q's omega-falloff is approximately 1/ω^2

Layout:
  Top row: 2D log-contour of Q(f, ω; A) at three amplitudes
           (small, medium, large saccade), with the band overlay and the
           predicted crossover f_c = 1/(2A) marked.
  Middle row, left: spatial profile Q(f, ω₀=8 Hz) for several A. The
           predicted f² rise and 1/A crossover should be visible.
  Middle row, right: temporal profile Q(f₀=0.5 cyc/u, ω) for several A.
  Bottom row, left: full input spectrum C_sac(f, ω; A=2.5°) on the
           same scale as a comparable drift spectrum.
  Bottom row, right: same drift spectrum at D=5 for comparison.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import (
    saccade_redistribution,
    saccade_spectrum,
    drift_spectrum,
)
from src.plotting import (
    add_band_edges,
    add_log_colorbar,
    panel_loglog,
    parameter_palette,
    setup_style,
    shared_lims,
)
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, hi_res_grid

setup_style()


def fig6b():
    f, omega = hi_res_grid()
    omega_pos = omega[omega > 0]
    i_pos = omega > 0

    A_panels = [0.5, 2.5, 7.0]
    A_lines = np.geomspace(0.3, 7.0, 6)

    fig = plt.figure(figsize=(11.0, 9.5))
    gs = fig.add_gridspec(
        3, 3,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.50, wspace=0.30,
        left=0.07, right=0.94, top=0.93, bottom=0.07,
    )

    # ---------- Row 1: 2D Q for three amplitudes ----------
    Q_panels = [saccade_redistribution(f, omega, A=A) for A in A_panels]
    vmin_Q, vmax_Q = shared_lims(Q_panels, floor=1e-5)

    cmap = "viridis"

    for col, (A, Q) in enumerate(zip(A_panels, Q_panels)):
        ax = fig.add_subplot(gs[0, col])
        panel_loglog(
            ax,
            f,
            omega,
            Q,
            vmin_Q,
            vmax_Q,
            n_levels=24,
            cmap=cmap,
            f_min=f.min(),
            f_max=f.max(),
            omega_min=omega_pos.min(),
            omega_max=omega_pos.max(),
            positive_only=True,
        )
        # Predicted crossover f_c = 1/(2A)
        f_c = 1.0 / (2.0 * A)
        ax.axvline(f_c, color="white", lw=0.8, ls="--", alpha=0.7)
        add_band_edges(ax, f_max=F_MAX, omega_min=OMEGA_MIN, omega_max=OMEGA_MAX)
        ax.set_title(
            rf"$A = {A:g}^\circ$,  predicted $f_c = 1/(2A) = {f_c:.2g}$",
            pad=2, fontsize=8.5,
        )
        ax.set_xlabel(r"$f$ (cyc/u)")
        if col == 0:
            ax.set_ylabel(r"$\omega$ (rad/s)")

    # Add colorbar at the right of row 1
    cb = add_log_colorbar(
        fig,
        [0.945, 0.69, 0.012, 0.22],
        cmap=cmap,
        vmin=vmin_Q,
        vmax=vmax_Q,
        label=r"$Q_\mathrm{sac}(f,\omega;A)$  [s$^2$]",
        tick_labelsize=6,
    )
    cb.ax.yaxis.label.set_size(8)

    # ---------- Row 2 left: spatial profile at fixed omega ----------
    omega_target_Hz = 8.0
    omega_target = 2 * np.pi * omega_target_Hz
    i_omega = int(np.argmin(np.abs(omega - omega_target)))

    ax = fig.add_subplot(gs[1, 0:2])
    palette = parameter_palette(len(A_lines), cmap="viridis")
    for A, color in zip(A_lines, palette):
        Q = saccade_redistribution(f, omega, A=A)
        Q_slice = Q[:, i_omega]
        ax.loglog(f, Q_slice, color=color, lw=1.4, label=rf"$A={A:.2g}^\circ$")

    # Reference f^2 line through one of the curves at low f
    f_ref = np.geomspace(0.05, 0.5, 50)
    Q_ref_anchor_idx = 5  # ~f=0.07
    ref_A = A_lines[len(A_lines)//2]
    Q_ref = saccade_redistribution(f, omega, A=ref_A)[Q_ref_anchor_idx, i_omega]
    f_anchor = f[Q_ref_anchor_idx]
    ax.plot(f_ref, Q_ref * (f_ref / f_anchor)**2, "k--", lw=0.8, alpha=0.6,
            label=r"$\propto f^2$")

    # Mark predicted 1/(2A) crossovers
    for A, color in zip(A_lines, palette):
        ax.axvline(1.0 / (2.0 * A), color=color, lw=0.4, ls=":", alpha=0.7)

    ax.set_xlim(f.min(), f.max())
    ax.set_xlabel(r"$f$ (cyc/u)")
    ax.set_ylabel(rf"$Q(f, \omega_0)$  at  $\omega_0 = {omega_target_Hz:g}$ Hz")
    ax.set_title(
        rf"Spatial profile at fixed $\omega$.  Vertical dotted lines: predicted "
        rf"$f_c = 1/(2A)$",
        fontsize=9, pad=2,
    )
    ax.legend(loc="lower right", fontsize=7, ncol=2, frameon=False)
    ax.grid(True, which="major", alpha=0.3, lw=0.3)

    # ---------- Row 2 right: temporal profile at fixed f ----------
    f_target = 0.5
    i_f = int(np.argmin(np.abs(f - f_target)))

    ax = fig.add_subplot(gs[1, 2])
    for A, color in zip(A_lines, palette):
        Q = saccade_redistribution(f, omega, A=A)
        Q_slice = Q[i_f, i_pos]
        ax.loglog(omega_pos, Q_slice, color=color, lw=1.4,
                  label=rf"$A={A:.2g}^\circ$")

    # Reference 1/omega^2
    omega_ref = np.geomspace(2.0, 200.0, 50)
    Q_ref_omega = Q_slice[np.argmin(np.abs(omega_pos - 5.0))]
    ax.plot(omega_ref, Q_ref_omega * (5.0 / omega_ref)**2, "k--", lw=0.8,
            alpha=0.6, label=r"$\propto 1/\omega^2$")
    ax.set_xlabel(r"$\omega$ (rad/s)")
    ax.set_ylabel(rf"$Q(f_0, \omega)$  at  $f_0 = {f_target:g}$ cyc/u")
    ax.set_title("Temporal profile", fontsize=9, pad=2)
    ax.legend(loc="lower left", fontsize=6.5, frameon=False)
    ax.grid(True, which="major", alpha=0.3, lw=0.3)

    # ---------- Row 3: full input spectrum, saccade vs drift ----------
    A_compare = 2.5
    D_compare = 5.0

    F_grid = f[:, None]
    W_grid = omega[None, :]
    C_sac = saccade_spectrum(f, omega, A=A_compare)
    C_drift = drift_spectrum(F_grid, W_grid, D=D_compare, beta=2.0)

    vmin_C, vmax_C = shared_lims([C_sac, C_drift], floor=1e-5)

    cmap_C = "magma"

    titles = [
        rf"$C_\mathrm{{sac}}(f,\omega;\, A = {A_compare:g}^\circ)$",
        rf"$C_\mathrm{{drift}}(f,\omega;\, D = {D_compare:g})$",
    ]
    spectra = [C_sac, C_drift]

    for col, (C, title) in enumerate(zip(spectra, titles)):
        ax = fig.add_subplot(gs[2, col])
        panel_loglog(
            ax,
            f,
            omega,
            C,
            vmin_C,
            vmax_C,
            n_levels=24,
            cmap=cmap_C,
            f_min=f.min(),
            f_max=f.max(),
            omega_min=omega_pos.min(),
            omega_max=omega_pos.max(),
            positive_only=True,
        )
        add_band_edges(ax, f_max=F_MAX, omega_min=OMEGA_MIN, omega_max=OMEGA_MAX)
        ax.set_xlabel(r"$f$ (cyc/u)")
        ax.set_title(title, pad=2, fontsize=9)
        if col == 0:
            ax.set_ylabel(r"$\omega$ (rad/s)")

    # Row 3 right column: Q integrated over the temporal band, comparing
    # saccades at multiple amplitudes to drift at multiple D.  This is
    # the test of "do saccades behave like drift but with whitening
    # restricted to lower spatial frequencies?"
    ax = fig.add_subplot(gs[2, 2])
    domega_grid = omega[1] - omega[0]
    band_mask_omega = (omega_pos >= OMEGA_MIN) & (omega_pos <= OMEGA_MAX)

    A_list = [0.5, 2.0, 7.0]
    sacc_palette = parameter_palette(len(A_list), cmap="viridis")
    for A_val, color in zip(A_list, sacc_palette):
        Q = saccade_redistribution(f, omega, A=A_val)
        Q_pos = Q[:, i_pos]
        Q_int = (
            np.sum(Q_pos[:, band_mask_omega], axis=1)
            * domega_grid
            / (2 * np.pi)
        )
        Q_int_dB = 10 * np.log10(np.maximum(Q_int, 1e-30))
        ax.plot(f, Q_int_dB, color=color, lw=1.5,
                label=rf"saccade $A={A_val:g}^\circ$")

    F_grid_d = f[:, None]
    W_grid_d = omega[None, :]
    from src.spectra import drift_lorentzian
    D_list = [0.5, 5.0, 50.0]
    drift_palette = parameter_palette(len(D_list), cmap="Reds")
    for D_val, color in zip(D_list, drift_palette):
        Q_d = drift_lorentzian(F_grid_d, W_grid_d, D=D_val)
        Q_d_pos = Q_d[:, i_pos]
        Q_d_int = (
            np.sum(Q_d_pos[:, band_mask_omega], axis=1)
            * domega_grid
            / (2 * np.pi)
        )
        Q_d_int_dB = 10 * np.log10(np.maximum(Q_d_int, 1e-30))
        ax.plot(f, Q_d_int_dB, color=color, lw=1.0, ls="--",
                label=rf"drift $D={D_val:g}$")

    ax.set_xscale("log")
    ax.set_xlim(f.min(), f.max())
    ax.set_ylim(-45, 5)
    ax.set_xlabel(r"$f$ (cyc/u)")
    ax.set_ylabel(r"$\int_\mathrm{band} Q\, d\omega / 2\pi$  (dB)")
    ax.set_title(
        "Saccades vs drift: same shape,\nnarrower whitening band as A grows",
        fontsize=8, pad=2,
    )
    ax.legend(fontsize=6.5, loc="lower right", frameon=False, ncol=2)
    ax.grid(True, which="major", alpha=0.3, lw=0.3)

    # Colorbar for row 3
    cb = add_log_colorbar(
        fig,
        [0.945, 0.07, 0.012, 0.22],
        cmap=cmap_C,
        vmin=vmin_C,
        vmax=vmax_C,
        label=r"$C(f,\omega)$ (shared scale)",
        tick_labelsize=6,
    )
    cb.ax.yaxis.label.set_size(8)

    fig.suptitle(
        "Saccade redistribution kernel and spectrum: diagnostics",
        y=0.97, fontsize=11,
    )

    out = "./outputs/fig6b_saccade_diagnostics.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")

    # Print numeric checks
    print()
    print("Sanity checks:")
    # 1. Q at low f follows f^2
    Q_low_A = saccade_redistribution(np.geomspace(0.01, 0.05, 5), omega, A=2.5)
    f_check = np.geomspace(0.01, 0.05, 5)
    log_slope = np.polyfit(np.log(f_check),
                            np.log(Q_low_A[:, i_omega]), 1)[0]
    print(f"  Low-f log-log slope (predicted 2.0): {log_slope:.3f}")

    # 2. Crossover scales as 1/A. Detect crossover as the f where the
    # f^2 extrapolation from low-f overshoots Q by 3 dB.
    def crossover_f(A):
        f_grid = np.geomspace(0.005, 5.0, 200)
        Q_arr = saccade_redistribution(f_grid, omega, A=A)[:, i_omega]
        # Fit f^2 line through low-f points
        i_low = np.argmin(np.abs(f_grid - 0.01))
        ref_level = Q_arr[i_low]
        ref_f = f_grid[i_low]
        f_sq_extrap = ref_level * (f_grid / ref_f) ** 2
        # Find first f where extrapolation exceeds Q by 3 dB
        ratio = f_sq_extrap / np.maximum(Q_arr, 1e-30)
        idx = np.where(ratio > 10**0.3)[0]
        return f_grid[idx[0]] if len(idx) > 0 else f_grid[-1]

    fc_1 = crossover_f(1.0)
    fc_4 = crossover_f(4.0)
    print(f"  Crossover at A=1.0: f_c = {fc_1:.3f}  (predicted 1/(2*1) = 0.5)")
    print(f"  Crossover at A=4.0: f_c = {fc_4:.3f}  (predicted 1/(2*4) = 0.125)")
    print(f"  Ratio f_c(1)/f_c(4) = {fc_1/fc_4:.2f}  (predicted 4.0)")

    # 3. Q^integrated vs C_I check
    Q_check = saccade_redistribution(f, omega, A=2.5)
    domega_grid = np.gradient(omega)
    int_Q = np.sum(Q_check * np.abs(domega_grid)[None, :], axis=1) / (2 * np.pi)
    print(f"  ∫ Q dω/(2π) — should be flat in f (window-determined level):")
    print(f"    f=0.05:  {int_Q[0]:.3f}")
    print(f"    f=0.5:   {int_Q[80]:.3f}")
    print(f"    f=2.0:   {int_Q[160]:.3f}")
    print(f"    f=5.0:   {int_Q[-1]:.3f}")


if __name__ == "__main__":
    fig6b()
