"""Q1: How does the power spectrum change with movement, and what does
that mean for the optimal filter?

Compares the on-retina spectrum and the optimal filter / kernels across
the full spectrum library:
  - drift only (Kuang)
  - saccade only (Mostofi)
  - drift + saccade unified stationary (this work)
  - Rucci/Boi cycle: late (= drift) and early (saccade transient)

For a fair visual comparison we keep image, noise, and budget fixed and
sweep the movement model. Three rows: spatial input spectrum |C(f, ω)|
slices at low and high ω; spatial kernel; temporal kernel.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import run, extract_kernels
from src.plotting import setup_style
from src.params import F_MAX
from src.power_spectrum_library import (
    spectrum_comparison_specs,
)

setup_style()


def fig_q1():
    sigma_in, sigma_out, P0 = 0.3, 1.0, 50.0
    spectra_to_compare = spectrum_comparison_specs(include_controls=True)

    results = []
    for label, spec, color in spectra_to_compare:
        r = run(spec, sigma_in=sigma_in, sigma_out=sigma_out, P0=P0,
                grid="hi_res")
        extract_kernels(r)
        results.append((label, spec, color, r))
        print(f"  {label:30s} I* = {r.I:.3f} nats   f_peak = {r.f_peak:.2f}")

    fig, axes = plt.subplots(3, 2, figsize=(8.5, 8.0),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.30,
                                          "left": 0.10, "right": 0.97,
                                          "top": 0.94, "bottom": 0.07})

    # Row 1: Cθ(f, ω) at two slices in ω
    ax_cf_lo, ax_cf_hi = axes[0]
    f = results[0][3].f
    omega = results[0][3].omega
    omega_lo_target = 2 * np.pi * 5  # 5 Hz
    omega_hi_target = 2 * np.pi * 30  # 30 Hz
    i_lo = int(np.argmin(np.abs(omega - omega_lo_target)))
    i_hi = int(np.argmin(np.abs(omega - omega_hi_target)))
    for label, _, color, r in results:
        # Avoid numerical zeros for log plot
        ax_cf_lo.loglog(f, np.maximum(r.C[:, i_lo], 1e-30),
                        color=color, lw=1.2, label=label)
        ax_cf_hi.loglog(f, np.maximum(r.C[:, i_hi], 1e-30),
                        color=color, lw=1.2)
    for ax, om in [(ax_cf_lo, 5), (ax_cf_hi, 30)]:
        ax.set_xlabel(r"$f$ (cycles/deg)")
        ax.set_ylabel(r"$C_\theta(f, \omega)$")
        ax.set_title(rf"Spectrum slice at $\omega/(2\pi) = {om}$ Hz")
        ax.set_xlim(0.1, F_MAX)
        ax.grid(True, alpha=0.3, which="both")
    ax_cf_lo.legend(loc="lower left", fontsize=7)

    # Row 2: spatial kernel
    ax_sp = axes[1, 0]
    ax_sp_zoom = axes[1, 1]
    for label, _, color, r in results:
        v_norm = r.spatial_v / max(np.max(np.abs(r.spatial_v)), 1e-30)
        ax_sp.plot(r.spatial_r, v_norm, color=color, lw=1.2, label=label)
        ax_sp_zoom.plot(r.spatial_r, v_norm, color=color, lw=1.2)
    for ax in [ax_sp, ax_sp_zoom]:
        ax.axhline(0, color="0.7", lw=0.4)
        ax.set_xlabel(r"$r$ (deg)")
        ax.set_ylabel(r"$v_s(r) / \max v_s$")
    ax_sp.set_xlim(-15, 15)
    ax_sp.set_title("Spatial kernel (full)")
    ax_sp_zoom.set_xlim(-3, 3)
    ax_sp_zoom.set_title("Spatial kernel (zoom on center)")

    # Row 3: temporal kernel
    ax_t = axes[2, 0]
    ax_t_zoom = axes[2, 1]
    for label, _, color, r in results:
        h_norm = r.temporal_v / max(np.max(np.abs(r.temporal_v)), 1e-30)
        ax_t.plot(r.temporal_t, h_norm, color=color, lw=1.2, label=label)
        ax_t_zoom.plot(r.temporal_t, h_norm, color=color, lw=1.2)
    for ax in [ax_t, ax_t_zoom]:
        ax.axhline(0, color="0.7", lw=0.4)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$v_t(t) / \max v_t$")
    ax_t.set_xlim(0, 0.6)
    ax_t.set_title("Temporal kernel")
    ax_t_zoom.set_xlim(0, 0.15)
    ax_t_zoom.set_title("Temporal kernel (early time zoom)")

    fig.suptitle(
        rf"Q1: Optimal filter across movement models  "
        rf"($\sigma_\mathrm{{in}}={sigma_in}$, $\sigma_\mathrm{{out}}={sigma_out}$, $P_0={P0}$)",
        y=0.985, fontsize=10.5,
    )

    out = "./outputs/figQ1_spectrum_library.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")
    return results


if __name__ == "__main__":
    fig_q1()
