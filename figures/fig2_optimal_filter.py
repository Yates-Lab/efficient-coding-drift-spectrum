"""Figure 2: full optimal filter |v*(f, ω)|^2, direct vs per-cell-aliased input.

For a fixed retinal budget P_0, fixed band B, and a power-law image, we
solve the efficient-coding problem at four drift conditions and at four
input-noise levels — twice. The "direct" version uses C(f, ω) as the input
spectrum; the "aliased" version uses C^sample(f, ω), the per-cell-Nyquist
folded version that is what each cell actually sees in a Nyquist-matched
mosaic. Aliasing folds copies of the low-spatial-frequency power into
every cell's band, which raises the apparent SNR; this is reflected in the
larger I^* values in the aliased rows.

Each panel reports the achieved I^* in nats; white dotted lines mark the
band edges. The optimal filter is taken on the same (f, ω) grid in all
panels.
"""

from __future__ import annotations

import sys
sys.path.insert(0, "/home/claude/efficient_coding")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.spectra import drift_spectrum
from src.aliasing import per_cell_aliased_spectrum
from src.solver import solve_efficient_coding
from src.plotting import setup_style, radial_weights, band_mask_radial
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, hi_res_grid

setup_style()


def _solve_grid(f, omega, D, beta, sigma_in, sigma_out, P0, aliased=False):
    F = f[:, None]
    W = omega[None, :]
    base = lambda f_, om: drift_spectrum(f_, om, D=D, beta=beta)
    if aliased:
        C = per_cell_aliased_spectrum(base, F, W, m_max=12)
    else:
        C = base(F, W)
    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    weights = weights * mask
    v_sq, lam, I = solve_efficient_coding(C, sigma_in, sigma_out, P0,
                                          weights, band_mask=mask)
    return C, v_sq, lam, I, mask


def fig2():
    f, omega = hi_res_grid()
    omega_pos = omega[omega > 0]

    sigma_out = 1.0
    P0 = 50.0

    sigma_in_fixed = 0.3
    D_sweep = [0.05, 0.5, 5.0, 50.0]
    D_fixed = 5.0
    sigma_in_sweep = [0.05, 0.2, 0.6, 2.0]

    fig, axes = plt.subplots(4, 4, figsize=(9.6, 9.4), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.50, "wspace": 0.18,
                                          "left": 0.10, "right": 0.91,
                                          "top": 0.94, "bottom": 0.06})

    # Compute everything
    rows = []
    for aliased_flag in [False, True]:
        for D in D_sweep:
            r = _solve_grid(f, omega, D, beta=2.0, sigma_in=sigma_in_fixed,
                            sigma_out=sigma_out, P0=P0,
                            aliased=aliased_flag)
            rows.append(("D", aliased_flag, D, r))
    for aliased_flag in [False, True]:
        for sin in sigma_in_sweep:
            r = _solve_grid(f, omega, D_fixed, beta=2.0, sigma_in=sin,
                            sigma_out=sigma_out, P0=P0,
                            aliased=aliased_flag)
            rows.append(("sigma_in", aliased_flag, sin, r))

    cmap = "viridis"

    # Determine row-shared color limits
    def lims(triplets, floor=1e-5):
        v = np.concatenate([t[3][1].flatten() for t in triplets])
        v = v[np.isfinite(v) & (v > 0)]
        return floor * v.max(), v.max()

    rows_D_direct  = [r for r in rows if r[0] == "D" and r[1] is False]
    rows_D_alias   = [r for r in rows if r[0] == "D" and r[1] is True]
    rows_S_direct  = [r for r in rows if r[0] == "sigma_in" and r[1] is False]
    rows_S_alias   = [r for r in rows if r[0] == "sigma_in" and r[1] is True]

    vmin_D_direct, vmax_D_direct = lims(rows_D_direct)
    vmin_D_alias, vmax_D_alias = lims(rows_D_alias)
    vmin_S_direct, vmax_S_direct = lims(rows_S_direct)
    vmin_S_alias, vmax_S_alias = lims(rows_S_alias)

    def _panel(ax, v_sq, vmin, vmax, mask, title):
        v_disp = np.where(mask, v_sq, vmin)
        v_disp = np.maximum(v_disp, vmin)
        i_pos = omega > 0
        levels = np.geomspace(vmin, vmax, 24)
        ax.contourf(
            f, omega[i_pos], v_disp[:, i_pos].T,
            levels=levels,
            norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
            cmap=cmap, extend="both",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(f.min(), f.max())
        ax.set_ylim(omega_pos.min(), omega.max())
        ax.axvline(F_MAX, color="white", lw=0.5, ls=":")
        ax.axhline(OMEGA_MIN, color="white", lw=0.5, ls=":")
        ax.axhline(OMEGA_MAX, color="white", lw=0.5, ls=":")
        ax.set_title(title, pad=2)

    # Row 0: D-sweep direct
    for ax, (_, _, D, (_, v_sq, _, I, mask)) in zip(axes[0], rows_D_direct):
        _panel(ax, v_sq, vmin_D_direct, vmax_D_direct, mask,
               rf"$D = {D:g}$,  $I^* = {I:.2f}$")
    # Row 1: D-sweep aliased
    for ax, (_, _, D, (_, v_sq, _, I, mask)) in zip(axes[1], rows_D_alias):
        _panel(ax, v_sq, vmin_D_alias, vmax_D_alias, mask,
               rf"$D = {D:g}$,  $I^* = {I:.2f}$")
    # Row 2: σ_in-sweep direct
    for ax, (_, _, sin, (_, v_sq, _, I, mask)) in zip(axes[2], rows_S_direct):
        _panel(ax, v_sq, vmin_S_direct, vmax_S_direct, mask,
               rf"$\sigma_\mathrm{{in}} = {sin:g}$,  $I^* = {I:.2f}$")
    # Row 3: σ_in-sweep aliased
    for ax, (_, _, sin, (_, v_sq, _, I, mask)) in zip(axes[3], rows_S_alias):
        _panel(ax, v_sq, vmin_S_alias, vmax_S_alias, mask,
               rf"$\sigma_\mathrm{{in}} = {sin:g}$,  $I^* = {I:.2f}$")

    for ax in axes[-1]:
        ax.set_xlabel(r"$f$ (cycles/unit)")
    for row in axes:
        row[0].set_ylabel(r"$\omega$ (rad/s)")

    # Banner labels
    fig.text(0.02, 0.85, "vary $D$\n(direct)", rotation=90,
             fontsize=9, va="center", ha="center")
    fig.text(0.02, 0.62, "vary $D$\n(aliased)", rotation=90,
             fontsize=9, va="center", ha="center")
    fig.text(0.02, 0.385, "vary $\\sigma_\\mathrm{in}$\n(direct)",
             rotation=90, fontsize=9, va="center", ha="center")
    fig.text(0.02, 0.155, "vary $\\sigma_\\mathrm{in}$\n(aliased)",
             rotation=90, fontsize=9, va="center", ha="center")

    cbar_ax = fig.add_axes([0.93, 0.10, 0.012, 0.78])
    cb = mpl.colorbar.ColorbarBase(
        cbar_ax,
        cmap=plt.get_cmap(cmap),
        norm=mpl.colors.LogNorm(vmin=1e-5, vmax=1.0),
        orientation="vertical",
        extend="min",
    )
    cb.set_label(r"$|v^*(f,\omega)|^2 / \max |v^*|^2$  (per-row)")
    cb.ax.tick_params(direction="out", labelsize=7)

    fig.suptitle(
        r"Optimal filter $|v^\star(f,\omega)|^2$: direct vs per-cell-Nyquist aliased",
        y=0.985, fontsize=10.5,
    )

    out = "/home/claude/efficient_coding/outputs/fig2_optimal_filter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig2()
