"""Q2: How does I* change with movement parameters, and how does that
depend on input noise?

Three panels:
  (a) I*(D) for drift only — shows the inverted-U from Atick: at small D
      drift fails to whiten so power piles up at low f and noise wins;
      at large D the cell loses high-f content because power moves out
      of band. There's an optimal D.
  (b) I*(s) for the Dong-Atick linear-motion approximation.
  (c) I*(A) for the Mostofi analytic saccade approximation.
  (d) A head-to-head comparison of the four supported spectrum families.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.pipeline import SolveConfig, run_many
from src.plotting import setup_style, parameter_palette
from src.power_spectrum_library import (
    drift_spectrum_specs,
    linear_motion_spectrum_specs,
    saccade_spectrum_specs,
    separable_movie_spectrum_specs,
)

setup_style()


def fig_q2():
    sigma_out, P0 = 1.0, 50.0

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.32,
                                          "left": 0.10, "right": 0.97,
                                          "top": 0.93, "bottom": 0.10})

    # --------------------------------------------------------------------
    # (a) I*(D) for drift only, multiple sigma_in
    # --------------------------------------------------------------------
    D_vals = np.geomspace(0.01, 200.0, 25)
    sigma_in_vals = [0.05, 0.1, 0.3, 1.0]
    palette = parameter_palette(len(sigma_in_vals), cmap="viridis")

    ax = axes[0, 0]
    print("Panel (a): I*(D) for drift only")
    drift_specs = drift_spectrum_specs(D_vals)
    for sin, color in zip(sigma_in_vals, palette):
        results = run_many(
            drift_specs,
            SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0),
        )
        I_vals = [r.I for r in results]
        ax.semilogx(D_vals, I_vals, "o-", color=color, ms=3, lw=1.0,
                    label=rf"$\sigma_\mathrm{{in}}={sin}$")
        i_max = int(np.argmax(I_vals))
        print(f"  sigma_in={sin}: max I* = {I_vals[i_max]:.3f} at D = {D_vals[i_max]:.2g}")
    ax.set_xlabel(r"$D$ (deg$^2$/s)")
    ax.set_ylabel(r"$I^*$ (nats)")
    ax.set_title("(a) Drift only: $I^*(D)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # --------------------------------------------------------------------
    # (b) I*(s) for Dong-Atick linear motion.
    # --------------------------------------------------------------------
    ax = axes[0, 1]
    print("Panel (b): I*(s) Dong-Atick linear motion")
    s_vals = np.geomspace(0.03, 50.0, 24)
    linear_specs = linear_motion_spectrum_specs(s_vals)
    for sin, color in zip(sigma_in_vals, palette):
        results = run_many(
            linear_specs,
            SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0),
        )
        I_vals = [r.I for r in results]
        ax.semilogx(s_vals, I_vals, "s-", color=color, ms=3, lw=1.0,
                    label=rf"$\sigma_\mathrm{{in}}={sin}$")
        i_max = int(np.argmax(I_vals))
        print(f"  sigma_in={sin}: max I* = {I_vals[i_max]:.3f} at s = {s_vals[i_max]:.2g}")
    ax.set_xlabel(r"$s$ (deg/s)")
    ax.set_ylabel(r"$I^*$ (nats)")
    ax.set_title(r"(b) Dong-Atick linear motion: $I^*(s)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # --------------------------------------------------------------------
    # (c) I*(A) for the Mostofi saccade approximation.
    # --------------------------------------------------------------------
    ax = axes[1, 0]
    print("Panel (c): I*(A) saccade only")
    A_vals = np.geomspace(0.05, 10.0, 18)
    saccade_specs = saccade_spectrum_specs(A_vals)
    for sin, color in zip(sigma_in_vals, palette):
        results = run_many(
            saccade_specs,
            SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0),
        )
        I_vals = [r.I for r in results]
        ax.semilogx(A_vals, I_vals, "^-", color=color, ms=3, lw=1.0,
                    label=rf"$\sigma_\mathrm{{in}}={sin}$")
        print(f"  sigma_in={sin}: I* range [{min(I_vals):.3f}, {max(I_vals):.3f}]")
    ax.set_xlabel(r"$A$ (deg)")
    ax.set_ylabel(r"$I^*$ (nats)")
    ax.set_title(r"(c) Mostofi saccade: $I^*(A)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # --------------------------------------------------------------------
    # (d) Head-to-head comparison of the supported spectrum families.
    # --------------------------------------------------------------------
    ax = axes[1, 1]
    print("Panel (d): supported spectrum families")
    sin = 0.3
    config = SolveConfig(sigma_in=sin, sigma_out=sigma_out, P0=P0)
    family_specs = [
        drift_spectrum_specs([2.0], color="tab:blue")[0],
        saccade_spectrum_specs([2.5], color="tab:orange")[0],
        linear_motion_spectrum_specs([1.0], color="tab:green")[0],
        separable_movie_spectrum_specs([0.05], color="tab:gray")[0],
    ]
    family_results = run_many(family_specs, config)
    labels = [spec.title or spec.label for spec in family_specs]
    colors = [spec.color for spec in family_specs]
    I_vals = [r.I for r in family_results]
    ax.bar(np.arange(len(labels)), I_vals, color=colors)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(["drift", "saccade", "linear", "separable"], rotation=20, ha="right")
    ax.set_ylabel(r"$I^*$ (nats)")
    ax.set_title(rf"(d) Comparison at $\sigma_\mathrm{{in}}={sin}$")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        rf"Q2: Information vs movement parameters and noise  ($\sigma_\mathrm{{out}}={sigma_out}$, $P_0={P0}$)",
        y=0.985, fontsize=10.5,
    )

    out = "./outputs/figQ2_information_sweeps.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig_q2()
