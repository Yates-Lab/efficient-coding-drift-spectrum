"""Q2: How does I* change with movement parameters, and how does that
depend on input noise?

Three panels:
  (a) I*(D) for drift only — shows the inverted-U from Atick: at small D
      drift fails to whiten so power piles up at low f and noise wins;
      at large D the cell loses high-f content because power moves out
      of band. There's an optimal D.
  (b) I*(D) for drift + saccades at fixed lambda, A — does the saccade
      contribution shift the optimum or modify the curve shape?
  (c) I*(A) at fixed lambda, varying sigma_in — flat in the handoff
      because saccades just redistribute, but show the noise dependence.
  (d) I*(D) sweeps for a few sigma_in values — when does drift help, when
      does it hurt?
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import DriftSpectrum, DriftPlusSaccadeSpectrum, SaccadeSpectrum
from src.pipeline import run
from src.plotting import setup_style, parameter_palette

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
    for sin, color in zip(sigma_in_vals, palette):
        I_vals = []
        for D in D_vals:
            r = run(DriftSpectrum(D=D), sigma_in=sin, sigma_out=sigma_out, P0=P0)
            I_vals.append(r.I)
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
    # (b) I*(D) drift + saccade at fixed lambda, A. How do saccades
    #     change the optimal D?
    # --------------------------------------------------------------------
    ax = axes[0, 1]
    print("Panel (b): I*(D) drift + saccade")
    A_fixed = 2.5
    lam_fixed = 3.0
    for sin, color in zip(sigma_in_vals, palette):
        I_combined = []
        for D in D_vals:
            r = run(DriftPlusSaccadeSpectrum(D=D, A=A_fixed, lam=lam_fixed),
                    sigma_in=sin, sigma_out=sigma_out, P0=P0)
            I_combined.append(r.I)
        ax.semilogx(D_vals, I_combined, "s-", color=color, ms=3, lw=1.0,
                    label=rf"$\sigma_\mathrm{{in}}={sin}$")
        i_max = int(np.argmax(I_combined))
        print(f"  sigma_in={sin}: max I* = {I_combined[i_max]:.3f} at D = {D_vals[i_max]:.2g}")
    ax.set_xlabel(r"$D$ (deg$^2$/s)")
    ax.set_ylabel(r"$I^*$ (nats)")
    ax.set_title(rf"(b) Drift + saccade ($A={A_fixed}^\circ$, $\lambda={lam_fixed}$): $I^*(D)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # --------------------------------------------------------------------
    # (c) I*(A) at fixed lambda, varying sigma_in: how does saccade
    #     amplitude alone shape information?
    # --------------------------------------------------------------------
    ax = axes[1, 0]
    print("Panel (c): I*(A) saccade only")
    A_vals = np.geomspace(0.05, 10.0, 18)
    for sin, color in zip(sigma_in_vals, palette):
        I_vals = []
        for A in A_vals:
            r = run(SaccadeSpectrum(A=A, lam=lam_fixed),
                    sigma_in=sin, sigma_out=sigma_out, P0=P0)
            I_vals.append(r.I)
        ax.semilogx(A_vals, I_vals, "^-", color=color, ms=3, lw=1.0,
                    label=rf"$\sigma_\mathrm{{in}}={sin}$")
        print(f"  sigma_in={sin}: I* range [{min(I_vals):.3f}, {max(I_vals):.3f}]")
    ax.set_xlabel(r"$A$ (deg)")
    ax.set_ylabel(r"$I^*$ (nats)")
    ax.set_title(rf"(c) Saccade only ($\lambda={lam_fixed}$): $I^*(A)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

    # --------------------------------------------------------------------
    # (d) I*(D) drift only vs I*(D) drift + saccade, head-to-head at
    #     sigma_in = 0.3. Does adding saccade lift the curve?
    # --------------------------------------------------------------------
    ax = axes[1, 1]
    print("Panel (d): drift vs drift+saccade head-to-head")
    sin = 0.3
    I_drift = []
    I_combined = []
    for D in D_vals:
        r1 = run(DriftSpectrum(D=D),
                 sigma_in=sin, sigma_out=sigma_out, P0=P0)
        r2 = run(DriftPlusSaccadeSpectrum(D=D, A=A_fixed, lam=lam_fixed),
                 sigma_in=sin, sigma_out=sigma_out, P0=P0)
        I_drift.append(r1.I)
        I_combined.append(r2.I)
    ax.semilogx(D_vals, I_drift, "o-", color="tab:blue", ms=3, lw=1.0,
                label="drift only")
    ax.semilogx(D_vals, I_combined, "s-", color="tab:green", ms=3, lw=1.0,
                label=rf"drift + saccade ($A={A_fixed}^\circ$)")
    # mark the saccade-only baseline
    r_sac = run(SaccadeSpectrum(A=A_fixed, lam=lam_fixed),
                sigma_in=sin, sigma_out=sigma_out, P0=P0)
    ax.axhline(r_sac.I, color="tab:orange", ls="--", lw=0.9,
               label=f"saccade only: $I^*$={r_sac.I:.2f}")
    ax.set_xlabel(r"$D$ (deg$^2$/s)")
    ax.set_ylabel(r"$I^*$ (nats)")
    ax.set_title(rf"(d) Comparison at $\sigma_\mathrm{{in}}={sin}$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, which="both")

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
