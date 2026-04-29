"""Figure 3: aliasing under the per-cell-Nyquist mosaic assumption.

Assumption: cells tuned to spatial frequency f0 form a mosaic whose Nyquist
exactly equals f0 (i.e. sampling spacing Δ = π/f0, sampling vector
k_s = 2 f0). The aliased input spectrum at the cell's preferred frequency is

    C^sample(f0, ω) = Σ_m C(|f0 + 2m f0|, ω)
                     = C(f0, ω) + 2 Σ_{q≥1} C((2q+1) f0, ω).

Top row:  direct vs aliased input spectrum, plus aliased/direct ratio.
Bottom row: optimal filter |v*|^2 with and without aliasing, and the
ω-integrated gain per spatial frequency.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.spectra import drift_spectrum
from src.aliasing import per_cell_aliased_spectrum
from src.solver import solve_efficient_coding
from src.plotting import setup_style, radial_weights, band_mask_radial
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, hi_res_grid

setup_style()


def fig3():
    f, omega = hi_res_grid()
    F = f[:, None]
    W = omega[None, :]

    D = 5.0
    beta = 2.0
    sigma_in = 0.3
    sigma_out = 1.0
    P0 = 50.0

    def C_func(f_, om):
        return drift_spectrum(f_, om, D=D, beta=beta)

    C_direct = C_func(F, W)
    C_aliased = per_cell_aliased_spectrum(C_func, F, W, m_max=12)

    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    weights_b = weights * mask

    v_direct, lam_d, I_direct = solve_efficient_coding(
        C_direct, sigma_in, sigma_out, P0, weights_b, band_mask=mask,
    )
    v_aliased, lam_a, I_aliased = solve_efficient_coding(
        C_aliased, sigma_in, sigma_out, P0, weights_b, band_mask=mask,
    )

    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.85,
                                          "left": 0.06, "right": 0.97,
                                          "top": 0.90, "bottom": 0.10})
    ax_direct_C, ax_alias_C, ax_ratio = axes[0]
    ax_direct_v, ax_alias_v, ax_gain = axes[1]

    cmap_C = "magma"
    cmap_v = "viridis"
    omega_pos = omega > 0

    C_max = max(np.nanmax(C_direct), np.nanmax(C_aliased))
    v_floor = 1e-5
    v_max = max(np.nanmax(v_direct), np.nanmax(v_aliased))
    # Round limits to decades so colorbar ticks look clean.
    C_lo = 10 ** (np.floor(np.log10(C_max)) - 5)
    C_hi = 10 ** np.ceil(np.log10(C_max))
    v_lo = 10 ** np.floor(np.log10(v_max * v_floor))
    v_hi = 10 ** np.ceil(np.log10(v_max))
    levels_C = np.geomspace(C_lo, C_hi, 24)
    levels_v = np.geomspace(v_lo, v_hi, 24)

    def plot_log(ax, M, levels, cmap, vmin, vmax):
        Mc = np.maximum(M, vmin)
        cf = ax.contourf(f, omega[omega_pos], Mc[:, omega_pos].T,
                          levels=levels,
                          norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
                          cmap=cmap, extend="min")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(f.min(), f.max())
        ax.set_ylim(omega[omega_pos].min(), omega.max())
        return cf

    cf_d = plot_log(ax_direct_C, C_direct, levels_C, cmap_C,
                    levels_C[0], levels_C[-1])
    plot_log(ax_alias_C, C_aliased, levels_C, cmap_C,
             levels_C[0], levels_C[-1])

    ax_direct_C.set_title("Direct $C(f_0,\\omega)$", pad=2)
    ax_alias_C.set_title("Aliased $C^{\\mathrm{sample}}(f_0,\\omega)$", pad=2)

    # Use make_axes_locatable to attach colorbar inside the host axis envelope
    div = make_axes_locatable(ax_alias_C)
    cax_C = div.append_axes("right", size="4%", pad="3%")
    cb_C = fig.colorbar(cf_d, cax=cax_C)
    cb_C.set_label(r"$C$")
    cb_C.ax.tick_params(direction="out", labelsize=7)

    # Ratio
    ratio = C_aliased / np.maximum(C_direct, 1e-30)
    ratio = np.clip(ratio, 1.0, 1e3)
    levels_r = np.geomspace(1.0, 1e3, 18)
    cf_r = ax_ratio.contourf(
        f, omega[omega_pos], ratio[:, omega_pos].T,
        levels=levels_r,
        norm=mpl.colors.LogNorm(vmin=1.0, vmax=1e3),
        cmap="cividis", extend="max",
    )
    ax_ratio.set_xscale("log")
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlim(f.min(), f.max())
    ax_ratio.set_ylim(omega[omega_pos].min(), omega.max())
    ax_ratio.set_title("Aliased / direct ratio", pad=2)
    div_r = make_axes_locatable(ax_ratio)
    cax_r = div_r.append_axes("right", size="4%", pad="3%")
    cb_r = fig.colorbar(cf_r, cax=cax_r)
    cb_r.set_label("ratio")
    cb_r.ax.tick_params(direction="out", labelsize=7)

    cf_v = plot_log(ax_direct_v, np.where(mask, v_direct, levels_v[0]),
                     levels_v, cmap_v, levels_v[0], levels_v[-1])
    plot_log(ax_alias_v, np.where(mask, v_aliased, levels_v[0]),
             levels_v, cmap_v, levels_v[0], levels_v[-1])

    ax_direct_v.set_title(f"Filter (direct);  $I^* = {I_direct:.2f}$ nats", pad=2)
    ax_alias_v.set_title(f"Filter (aliased); $I^* = {I_aliased:.2f}$ nats", pad=2)
    for ax in [ax_direct_v, ax_alias_v]:
        ax.axvline(F_MAX, color="white", lw=0.5, ls=":")
        ax.axhline(OMEGA_MIN, color="white", lw=0.5, ls=":")
        ax.axhline(OMEGA_MAX, color="white", lw=0.5, ls=":")

    div_v = make_axes_locatable(ax_alias_v)
    cax_v = div_v.append_axes("right", size="4%", pad="3%")
    cb_v = fig.colorbar(cf_v, cax=cax_v)
    cb_v.set_label(r"$|v^\star|^2$")
    cb_v.ax.tick_params(direction="out", labelsize=7)

    domega_arr = np.gradient(omega)
    gain_direct = np.sum(v_direct * np.abs(domega_arr) / (2 * np.pi), axis=1)
    gain_aliased = np.sum(v_aliased * np.abs(domega_arr) / (2 * np.pi), axis=1)
    ax_gain.plot(f, gain_direct, color="#0a6cb1", lw=1.5, label="direct")
    ax_gain.plot(f, gain_aliased, color="#cf3a4f", lw=1.5, label="aliased")
    ax_gain.set_xscale("log")
    ax_gain.set_yscale("log")
    ax_gain.set_xlabel(r"$f$ (cycles/unit)")
    ax_gain.set_ylabel(r"$\int |v^\star|^2 \,\mathrm{d}\omega/(2\pi)$")
    ax_gain.set_title("Gain per spatial freq.", pad=2)
    ax_gain.axvline(F_MAX, color="0.5", lw=0.5, ls=":")
    ax_gain.legend(loc="lower left", handlelength=1.2)

    for ax in [ax_direct_C, ax_alias_C, ax_ratio]:
        ax.set_xlabel(r"$f_0$ (cycles/unit)")
    ax_direct_C.set_ylabel(r"$\omega$ (rad/s)")
    for ax in [ax_direct_v, ax_alias_v]:
        ax.set_xlabel(r"$f$ (cycles/unit)")
    ax_direct_v.set_ylabel(r"$\omega$ (rad/s)")

    fig.suptitle(
        r"Per-cell-tuning aliasing: input mosaic Nyquist matched to $f_0$",
        y=0.97, fontsize=10.5,
    )

    out = "outputs/fig3_aliasing.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig3()
