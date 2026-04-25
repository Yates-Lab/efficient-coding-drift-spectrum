"""
Fig 3: Mutual information as a function of diffusion constant D.

Two things we're asking (per user's framing):
  (a) How does drift change the optimal encoding?
      -> covered in figs 1 and 2.
  (b) Given a fixed encoding, how does drift change MI?
      -> this figure has two sweeps:
           (i)  I_optimal(D): retrain the optimal filter at each D.
                Upper envelope of (ii).
           (ii) I_fixed_filter(D; D_fit): build the optimal filter at
                D=D_fit, then evaluate MI for the same filter as D varies.
                Each curve has a peak at D = D_fit.
      Together (i) and (ii) show how mismatched encoding and drift lose MI.

Finite-bandwidth effects are essential: infinite bandwidth would give
scale invariance (Section 3.4), but our finite kmax and wmax break it
and produce an information-maximizing D*.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from grids import build_radial_weights, omega_grid, radial_grid
from information import info_density
from optimizer import solve_lambda, v_star_sq
from spectrum import C_D
from style import set_defaults


def run():
    set_defaults()

    # ----- Parameters -----
    A, beta, k0 = 1.0, 2.0, 0.02
    sigma_in_sq = 1e-3
    sigma_out_sq = 1e-3
    P_target = 0.05

    # Grid. Use fixed finite bandwidth; this is what creates the MI peak.
    kmax, Nk = 3.0, 400
    wmax, Nw = 100.0, 801
    k, wk = radial_grid(kmax, Nk, kmin=k0)
    omega, dw_over_2pi = omega_grid(wmax, Nw)
    weights = build_radial_weights(wk, dw_over_2pi)

    # D sweep (log-spaced).
    D_sweep = np.logspace(-2, 3, 41)

    # Filters trained at these specific D values (shown as separate curves).
    D_fixed_at = [0.1, 1.0, 10.0, 100.0]

    # ----- (i) Optimal MI at each D -----
    I_opt = np.zeros_like(D_sweep)
    lam_opt = np.zeros_like(D_sweep)
    for i, D in enumerate(D_sweep):
        Cx = C_D(k[:, None], omega[None, :], A=A, beta=beta, D=D, k0=k0)
        lam = solve_lambda(Cx, sigma_in_sq, sigma_out_sq, P_target, weights)
        v2 = v_star_sq(Cx, sigma_in_sq, sigma_out_sq, lam)
        I_opt[i] = np.sum(
            info_density(v2, Cx, sigma_in_sq, sigma_out_sq) * weights
        )
        lam_opt[i] = lam

    # Find D* on a finer log grid interpolation.
    i_star = int(np.argmax(I_opt))
    D_star = D_sweep[i_star]
    I_star = I_opt[i_star]
    print(f"I maximizer on sweep: D* ≈ {D_star:.3g}, I* = {I_star:.3g}")

    # ----- (ii) Fixed-filter MI as D varies, rescaled to meet constraint -----
    # For each D_fit, build optimal |v|^2 there, then evaluate information
    # with the SAME FILTER SHAPE rescaled (by a scalar multiplier) so that
    # total power equals P_target at each test D.
    #
    # The rationale: the filter's spatiotemporal SHAPE represents fixed
    # retinal hardware; the scalar gain represents adaptation. This isolates
    # the effect of mismatched filter shape on MI.
    #
    # Power is linear in the scalar: P(alpha * v^2) = alpha * P(v^2). So
    # the required scalar is alpha = P_target / P_current.
    from optimizer import total_power
    fixed_curves = {}
    for D_fit in D_fixed_at:
        Cx_fit = C_D(k[:, None], omega[None, :], A=A, beta=beta, D=D_fit, k0=k0)
        lam_fit = solve_lambda(
            Cx_fit, sigma_in_sq, sigma_out_sq, P_target, weights
        )
        v2_fit = v_star_sq(Cx_fit, sigma_in_sq, sigma_out_sq, lam_fit)

        I_curve = np.zeros_like(D_sweep)
        for i, D in enumerate(D_sweep):
            Cx = C_D(k[:, None], omega[None, :], A=A, beta=beta, D=D, k0=k0)
            P_current = total_power(Cx + sigma_in_sq, v2_fit, weights)
            alpha = P_target / P_current
            v2_rescaled = alpha * v2_fit
            I_curve[i] = np.sum(
                info_density(v2_rescaled, Cx, sigma_in_sq, sigma_out_sq) * weights
            )
        fixed_curves[D_fit] = I_curve
        print(f"D_fit={D_fit}: peak I "
              f"= {I_curve.max():.3g} at D={D_sweep[I_curve.argmax()]:.3g}")

    # ----- Plot -----
    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 3.2), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.4, 1.0]},
    )

    ax = axes[0]
    # Fixed-filter curves first (lighter).
    cmap = plt.get_cmap("viridis")
    for idx, D_fit in enumerate(D_fixed_at):
        color = cmap(0.15 + 0.7 * idx / max(len(D_fixed_at) - 1, 1))
        ax.plot(
            D_sweep, fixed_curves[D_fit], color=color, lw=1.2,
            label=rf"$v^\star$ at $D={D_fit:g}$",
        )
        # Mark the training D with a dot on the curve.
        i_fit = np.argmin(np.abs(D_sweep - D_fit))
        ax.plot(D_sweep[i_fit], fixed_curves[D_fit][i_fit],
                "o", color=color, ms=5, mec="white", mew=0.7)

    # Optimal envelope on top.
    ax.plot(D_sweep, I_opt, color="black", lw=1.8, label=r"$v^\star$ re-optimized")
    ax.plot(D_star, I_star, "*", color="black", ms=11,
            mec="white", mew=0.8, zorder=10,
            label=rf"$D^\star \approx {D_star:.2g}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"diffusion constant $D$")
    ax.set_ylabel(r"mutual information $I(D)$")
    ax.legend(loc="lower center", frameon=False, ncol=1, fontsize=7)

    # Right: lambda(D).
    ax2 = axes[1]
    ax2.loglog(D_sweep, lam_opt, color="C3", lw=1.4)
    ax2.axvline(D_star, color="0.6", lw=0.6, ls="--")
    ax2.set_xlabel(r"diffusion constant $D$")
    ax2.set_ylabel(r"$\lambda(D)$")
    ax2.set_title(r"Lagrange multiplier")

    os.makedirs("figures", exist_ok=True)
    out = "figures/fig3_info_vs_drift.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
