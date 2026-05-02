"""Q3: Magno/parvo-like kernels from analytic cycle spectra.

This figure uses the exact same canonical analytic spectra as Figure 7:

    early fixation: C_early = I(f) Q_saccade
    late fixation:  C_late  = I(f) Q_drift_total

The sweep varies encoder input noise only.  The movement spectrum itself is
not regenerated here; both regimes are read from the Figure 7 source spectrum.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import run, extract_kernels
from src.plotting import setup_style, parameter_palette
from src.params import OMEGA_MIN, OMEGA_MAX
from src.power_spectrum_library import (
    cycle_solver_spectra,
)

setup_style()


def _temporal_centroid(result):
    v_sq = result.v_sq
    omega = result.omega
    domega = np.gradient(omega)
    energy_per_f = np.sum(v_sq * np.abs(domega)[None, :], axis=1)
    i_peak = int(np.argmax(energy_per_f))

    pos = omega > 0
    om_pos = omega[pos]
    v_t = v_sq[i_peak, pos]
    in_band = (om_pos >= OMEGA_MIN) & (om_pos <= OMEGA_MAX)
    dom = np.gradient(om_pos)
    norm = np.sum(v_t[in_band] * np.abs(dom)[in_band])
    if norm <= 0:
        return 0.0
    omega_centroid = np.sum(
        om_pos[in_band] * v_t[in_band] * np.abs(dom)[in_band]
    ) / norm
    return omega_centroid / (2.0 * np.pi)


def _run_regime(early, late, *, regime, sigma_in=0.3, sigma_out=1.0, P0=50.0):
    spec = early if regime == "early" else late
    result = run(
        spec, sigma_in=sigma_in, sigma_out=sigma_out, P0=P0,
        grid="hi_res",
    )
    extract_kernels(result)
    return result, _temporal_centroid(result)


def fig_q3():
    early, late = cycle_solver_spectra(use_modulated_early=True)
    sigma_out, P0 = 1.0, 50.0

    sigma_in_sweep = np.geomspace(0.05, 2.0, 7)

    palette_late = parameter_palette(len(sigma_in_sweep), cmap="Purples",
                                     lo=0.35, hi=0.95)
    palette_early = parameter_palette(len(sigma_in_sweep), cmap="Reds",
                                      lo=0.35, hi=0.95)

    results_late = []
    print("Late regime (canonical Figure 7 drift spectrum):")
    for sigma_in, color in zip(sigma_in_sweep, palette_late):
        r, c = _run_regime(
            early, late, regime="late",
            sigma_in=sigma_in, sigma_out=sigma_out, P0=P0,
        )
        results_late.append((sigma_in, color, r, c))
        print(f"  sigma_in={sigma_in:.3g}: I*={r.I:.3f}  "
              f"f_peak={r.f_peak:.3f}  f_temporal={c:.1f} Hz")

    results_early = []
    print("Early regime (canonical Figure 7 Mostofi saccade spectrum):")
    for sigma_in, color in zip(sigma_in_sweep, palette_early):
        r, c = _run_regime(
            early, late, regime="early",
            sigma_in=sigma_in, sigma_out=sigma_out, P0=P0,
        )
        results_early.append((sigma_in, color, r, c))
        print(f"  sigma_in={sigma_in:.3g}: I*={r.I:.3f}  "
              f"f_peak={r.f_peak:.3f}  f_temporal={c:.1f} Hz")

    fig, axes = plt.subplots(
        3, 2, figsize=(8.5, 8.5),
        gridspec_kw={"hspace": 0.45, "wspace": 0.30,
                     "left": 0.10, "right": 0.97,
                     "top": 0.94, "bottom": 0.07},
    )

    ax_sp_late, ax_t_late = axes[0]
    for sigma_in, color, r, _ in results_late:
        v_norm = r.spatial_v / max(np.max(np.abs(r.spatial_v)), 1e-30)
        h_norm = r.temporal_v / max(np.max(np.abs(r.temporal_v)), 1e-30)
        ax_sp_late.plot(r.spatial_r, v_norm, color=color, lw=1.3,
                        label=rf"$\sigma_\mathrm{{in}}={sigma_in:.2g}$")
        ax_t_late.plot(r.temporal_t, h_norm, color=color, lw=1.3,
                       label=rf"$\sigma_\mathrm{{in}}={sigma_in:.2g}$")

    ax_sp_early, ax_t_early = axes[1]
    for sigma_in, color, r, _ in results_early:
        v_norm = r.spatial_v / max(np.max(np.abs(r.spatial_v)), 1e-30)
        h_norm = r.temporal_v / max(np.max(np.abs(r.temporal_v)), 1e-30)
        ax_sp_early.plot(r.spatial_r, v_norm, color=color, lw=1.3,
                         label=rf"$\sigma_\mathrm{{in}}={sigma_in:.2g}$")
        ax_t_early.plot(r.temporal_t, h_norm, color=color, lw=1.3,
                        label=rf"$\sigma_\mathrm{{in}}={sigma_in:.2g}$")

    for ax in [ax_sp_late, ax_sp_early]:
        ax.axhline(0, color="0.7", lw=0.4)
        ax.axvline(0, color="0.85", lw=0.3)
        ax.set_xlabel(r"$r$ (deg)")
        ax.set_ylabel(r"$v_s(r)/\max v_s$")
        ax.legend(fontsize=7)
    ax_sp_late.set_xlim(-4, 4)
    ax_sp_early.set_xlim(-4, 4)
    ax_sp_late.set_title("Late fixation: spatial kernel")
    ax_sp_early.set_title("Early fixation: spatial kernel")

    for ax in [ax_t_late, ax_t_early]:
        ax.axhline(0, color="0.7", lw=0.4)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$v_t(t)/\max v_t$")
        ax.legend(fontsize=7)
    ax_t_late.set_xlim(0, 0.30)
    ax_t_early.set_xlim(0, 0.30)
    ax_t_late.set_title("Late fixation: temporal kernel")
    ax_t_early.set_title("Early fixation: temporal kernel")

    ax_scatter = axes[2, 0]
    for sigma_in, color, r, c in results_late:
        ax_scatter.scatter(r.f_peak, c, s=80, color=color, zorder=3,
                           edgecolor="black", linewidth=0.5)
        ax_scatter.annotate(rf"$\sigma={sigma_in:.2g}$", (r.f_peak, c),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color="0.2")
    for sigma_in, color, r, c in results_early:
        ax_scatter.scatter(r.f_peak, c, s=80, color=color, zorder=3,
                           edgecolor="black", linewidth=0.5, marker="s")
        ax_scatter.annotate(rf"$\sigma={sigma_in:.2g}$", (r.f_peak, c),
                            xytext=(5, 4), textcoords="offset points",
                            fontsize=7, color="0.2")
    ax_scatter.set_xscale("log")
    ax_scatter.set_yscale("log")
    ax_scatter.set_xlabel(r"Peak spatial frequency $f^*$ (cpd)")
    ax_scatter.set_ylabel("Temporal centroid (Hz)")
    ax_scatter.set_title("Cycle-regime kernel summary")
    ax_scatter.grid(True, alpha=0.3, which="both")
    ax_scatter.scatter([], [], s=60, marker="o", color="0.3",
                       edgecolor="black", linewidth=0.5, label="late")
    ax_scatter.scatter([], [], s=60, marker="s", color="0.3",
                       edgecolor="black", linewidth=0.5, label="early")
    ax_scatter.legend(loc="lower left", fontsize=7)

    ax_info = axes[2, 1]
    labels = ["early", "late"]
    med_f = [
        np.median([r.f_peak for _, _, r, _ in results_early]),
        np.median([r.f_peak for _, _, r, _ in results_late]),
    ]
    med_t = [
        np.median([c for _, _, _, c in results_early]),
        np.median([c for _, _, _, c in results_late]),
    ]
    x = np.arange(2)
    ax_info.bar(x - 0.18, med_f, width=0.36, color=["tab:red", "indigo"],
                alpha=0.75, label=r"median $f^*$")
    ax_info_b = ax_info.twinx()
    ax_info_b.bar(x + 0.18, med_t, width=0.36, color=["salmon", "mediumpurple"],
                  alpha=0.75, label="median temporal centroid")
    ax_info.set_xticks(x)
    ax_info.set_xticklabels(labels)
    ax_info.set_yscale("log")
    ax_info_b.set_yscale("log")
    ax_info.set_ylabel(r"median $f^*$ (cpd)")
    ax_info_b.set_ylabel("median temporal centroid (Hz)")
    ax_info.set_title("Early vs late summary")

    fig.suptitle(
        "Q3: Magno/parvo-like kernels from the canonical Figure 7 cycle spectra",
        y=0.985, fontsize=10.5,
    )

    out = "./outputs/figQ3_magno_parvo.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_q3()
