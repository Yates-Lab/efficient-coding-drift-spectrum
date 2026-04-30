"""Q3: Does modeling the saccade-fixation cycle's non-stationarity
yield magno- and parvo-like cell classes?

Two regimes in the cycle (Boi et al. 2017):
  Late fixation: drift-dominated. Spectrum is whitened (slope ~0 in f).
                 Optimal kernel: high spatial freq, sustained temporal —
                 parvo-like.
  Early fixation: saccade-transient. Spectrum preserves natural-image 1/f^2
                  shape. Optimal kernel: low spatial freq, fast temporal —
                  magno-like.

Layout:
  Row 1 (late = drift): spatial and temporal kernels as D varies.
  Row 2 (early = saccade transient): spatial and temporal kernels as A varies.
  Row 3 (summary):
    (left)  peak spatial frequency f* across regimes and parameters.
    (right) temporal centroid (Hz) — "speed" of the kernel — across the same.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import BoiCycleEarlySpectrum, BoiCycleLateSpectrum
from src.pipeline import run, extract_kernels
from src.plotting import setup_style, parameter_palette
from src.params import OMEGA_MIN, OMEGA_MAX

setup_style()


def _temporal_centroid(result):
    """Energy-weighted centroid of |v|^2 along the positive-omega axis,
    in Hz. Restricted to the band to avoid edge effects."""
    v_sq = result.v_sq
    f = result.f
    omega = result.omega
    # Pick the f at peak; look at temporal-frequency content there
    domega = np.gradient(omega)
    energy_per_f = np.sum(v_sq * np.abs(domega)[None, :], axis=1)
    i_peak = int(np.argmax(energy_per_f))

    om_pos_mask = omega > 0
    om_pos = omega[om_pos_mask]
    v_t = v_sq[i_peak, om_pos_mask]
    in_band = (om_pos >= OMEGA_MIN) & (om_pos <= OMEGA_MAX)
    if not in_band.any():
        return 0.0
    dom = np.gradient(om_pos)
    norm = np.sum(v_t[in_band] * np.abs(dom)[in_band])
    if norm <= 0:
        return 0.0
    omega_centroid = np.sum(om_pos[in_band] * v_t[in_band] *
                            np.abs(dom)[in_band]) / norm
    return omega_centroid / (2.0 * np.pi)  # Hz


def fig_q3():
    sigma_in, sigma_out, P0 = 0.3, 1.0, 50.0

    D_sweep = [0.5, 2.0, 8.0, 30.0]                 # late-fixation drift
    A_sweep = [1.0, 2.5, 4.4, 7.0]                  # early-fixation amplitude
    T_win_fixed = 0.150                             # 150 ms early-fixation window

    palette_late = parameter_palette(len(D_sweep), cmap="Purples", lo=0.35, hi=0.95)
    palette_early = parameter_palette(len(A_sweep), cmap="Reds", lo=0.35, hi=0.95)

    # -------------------------------------------------------------------
    # Run all conditions
    # -------------------------------------------------------------------
    results_late = []
    print("Late regime (drift):")
    for D, color in zip(D_sweep, palette_late):
        r = run(BoiCycleLateSpectrum(D=D),
                sigma_in=sigma_in, sigma_out=sigma_out, P0=P0, grid="hi_res")
        extract_kernels(r)
        omega_c = _temporal_centroid(r)
        results_late.append((D, color, r, omega_c))
        print(f"  D = {D:5.2f}: I*={r.I:.3f}  f_peak={r.f_peak:.3f}  "
              f"f_temporal={omega_c:.1f} Hz")

    results_early = []
    print("Early regime (saccade transient):")
    for A, color in zip(A_sweep, palette_early):
        r = run(BoiCycleEarlySpectrum(A=A, T_win=T_win_fixed),
                sigma_in=sigma_in, sigma_out=sigma_out, P0=P0, grid="hi_res")
        extract_kernels(r)
        omega_c = _temporal_centroid(r)
        results_early.append((A, color, r, omega_c))
        print(f"  A = {A:5.2f}: I*={r.I:.3f}  f_peak={r.f_peak:.3f}  "
              f"f_temporal={omega_c:.1f} Hz")

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(8.5, 8.5),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.30,
                                          "left": 0.10, "right": 0.97,
                                          "top": 0.94, "bottom": 0.07})

    # Row 1: late regime
    ax_sp_late = axes[0, 0]
    ax_t_late = axes[0, 1]
    for D, color, r, _ in results_late:
        v_norm = r.spatial_v / max(np.max(np.abs(r.spatial_v)), 1e-30)
        ax_sp_late.plot(r.spatial_r, v_norm, color=color, lw=1.3,
                        label=rf"$D={D}$")
        h_norm = r.temporal_v / max(np.max(np.abs(r.temporal_v)), 1e-30)
        ax_t_late.plot(r.temporal_t, h_norm, color=color, lw=1.3,
                       label=rf"$D={D}$")
    ax_sp_late.set_xlim(-3, 3)
    ax_sp_late.axhline(0, color="0.7", lw=0.4)
    ax_sp_late.set_xlabel(r"$r$ (deg)")
    ax_sp_late.set_ylabel(r"$v_s(r)/\max v_s$")
    ax_sp_late.set_title("Late fixation: spatial kernel (parvo-like?)")
    ax_sp_late.legend(fontsize=7)

    ax_t_late.set_xlim(0, 0.30)
    ax_t_late.axhline(0, color="0.7", lw=0.4)
    ax_t_late.set_xlabel(r"$t$ (s)")
    ax_t_late.set_ylabel(r"$v_t(t)/\max v_t$")
    ax_t_late.set_title("Late fixation: temporal kernel (sustained)")
    ax_t_late.legend(fontsize=7)

    # Row 2: early regime
    ax_sp_early = axes[1, 0]
    ax_t_early = axes[1, 1]
    for A, color, r, _ in results_early:
        v_norm = r.spatial_v / max(np.max(np.abs(r.spatial_v)), 1e-30)
        ax_sp_early.plot(r.spatial_r, v_norm, color=color, lw=1.3,
                         label=rf"$A={A}^\circ$")
        h_norm = r.temporal_v / max(np.max(np.abs(r.temporal_v)), 1e-30)
        ax_t_early.plot(r.temporal_t, h_norm, color=color, lw=1.3,
                        label=rf"$A={A}^\circ$")
    ax_sp_early.set_xlim(-15, 15)
    ax_sp_early.axhline(0, color="0.7", lw=0.4)
    ax_sp_early.set_xlabel(r"$r$ (deg)")
    ax_sp_early.set_ylabel(r"$v_s(r)/\max v_s$")
    ax_sp_early.set_title("Early fixation: spatial kernel (magno-like?)")
    ax_sp_early.legend(fontsize=7)

    ax_t_early.set_xlim(0, 0.10)
    ax_t_early.axhline(0, color="0.7", lw=0.4)
    ax_t_early.set_xlabel(r"$t$ (s)")
    ax_t_early.set_ylabel(r"$v_t(t)/\max v_t$")
    ax_t_early.set_title("Early fixation: temporal kernel (transient/biphasic)")
    ax_t_early.legend(fontsize=7)

    # Row 3: scatter summary, and centroid sweep
    ax_scatter = axes[2, 0]
    f_late = [r.f_peak for _, _, r, _ in results_late]
    f_early = [r.f_peak for _, _, r, _ in results_early]
    fT_late = [c for _, _, _, c in results_late]
    fT_early = [c for _, _, _, c in results_early]

    for D, color, r, c in results_late:
        ax_scatter.scatter(r.f_peak, c, s=80, color=color, zorder=3,
                           edgecolor="black", linewidth=0.5)
        ax_scatter.annotate(rf"$D={D}$", (r.f_peak, c),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color="0.2")
    for A, color, r, c in results_early:
        ax_scatter.scatter(r.f_peak, c, s=80, color=color, zorder=3,
                           edgecolor="black", linewidth=0.5, marker="s")
        ax_scatter.annotate(rf"$A={A}^\circ$", (r.f_peak, c),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color="0.2")
    # Annotate magno / parvo regions
    ax_scatter.text(0.1, 30, "magno-like\n(low f, fast)", fontsize=8,
                    color="darkred", ha="left", va="bottom")
    ax_scatter.text(2.0, 3, "parvo-like\n(high f, slow)", fontsize=8,
                    color="indigo", ha="left", va="bottom")

    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(r"Peak spatial frequency $f^*$ (cyc/deg)")
    ax_scatter.set_ylabel("Temporal centroid (Hz)")
    ax_scatter.set_title("(c) Magno-vs-parvo scatter: cycle regimes")
    ax_scatter.grid(True, alpha=0.3, which="both")
    # Marker legend
    ax_scatter.scatter([], [], s=60, marker="o", color="0.3",
                       edgecolor="black", linewidth=0.5,
                       label="late (drift, parvo)")
    ax_scatter.scatter([], [], s=60, marker="s", color="0.3",
                       edgecolor="black", linewidth=0.5,
                       label="early (saccade, magno)")
    ax_scatter.legend(loc="lower left", fontsize=7)

    # T_win sweep for early-fixation
    ax_Tw = axes[2, 1]
    T_win_vals = np.array([0.05, 0.080, 0.120, 0.150, 0.20, 0.30, 0.512])
    A_for_sweep = 4.4
    f_peaks = []
    f_centroids = []
    print("Early-regime T_win sweep:")
    for Tw in T_win_vals:
        r = run(BoiCycleEarlySpectrum(A=A_for_sweep, T_win=Tw),
                sigma_in=sigma_in, sigma_out=sigma_out, P0=P0, grid="hi_res")
        extract_kernels(r)
        c = _temporal_centroid(r)
        f_peaks.append(r.f_peak)
        f_centroids.append(c)
        print(f"  T_win = {Tw*1000:5.0f} ms: f_peak={r.f_peak:.3f}, "
              f"f_temporal={c:.1f} Hz, I*={r.I:.3f}")
    ax_Tw.semilogx(T_win_vals * 1000, f_peaks, "s-", color="tab:red",
                   ms=5, lw=1.0, label="spatial $f^*$ (cyc/deg)")
    ax_Tw_b = ax_Tw.twinx()
    ax_Tw_b.semilogx(T_win_vals * 1000, f_centroids, "^-", color="darkblue",
                     ms=5, lw=1.0, label="temporal centroid (Hz)")
    ax_Tw.set_xlabel(r"$T_\mathrm{win}$ (ms)")
    ax_Tw.set_ylabel(r"$f^*$ (cyc/deg)", color="tab:red")
    ax_Tw_b.set_ylabel("temporal centroid (Hz)", color="darkblue")
    ax_Tw.tick_params(axis="y", labelcolor="tab:red")
    ax_Tw_b.tick_params(axis="y", labelcolor="darkblue")
    ax_Tw.set_title(rf"(d) Early regime vs $T_\mathrm{{win}}$ ($A={A_for_sweep}^\circ$)")
    # Single combined legend
    h1, l1 = ax_Tw.get_legend_handles_labels()
    h2, l2 = ax_Tw_b.get_legend_handles_labels()
    ax_Tw.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7)
    ax_Tw.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        rf"Q3: Magno- and parvo-like classes from the saccade-fixation cycle  "
        rf"($\sigma_\mathrm{{in}}={sigma_in}$, $T_\mathrm{{win}}={T_win_fixed*1000:.0f}$ ms)",
        y=0.985, fontsize=10.5,
    )

    out = "./outputs/figQ3_magno_parvo.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_q3()
