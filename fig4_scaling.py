"""
Fig 4 (companion): Empirical scaling of the filter's active-set extent with D.

From Section 3.3 of the notes, holding C_th fixed gives
    k_max(D) ~ D^{-1/4},  omega_max(D) ~ D^{1/2}.
In our sweep lambda(D), and hence C_th(lambda), vary with D, so these are
only approximate scalings. We extract empirical k_max and omega_max from
the active-set support of |v*_D|^2 and compare to the scaling predictions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from grids import build_radial_weights, omega_grid, radial_grid
from optimizer import solve_lambda, v_star_sq
from spectrum import C_D
from style import set_defaults


def active_extents(v2, k, omega, quantile=0.99):
    """
    Return (k_max, omega_max) as the quantile of k, omega weighted by v^2.
    Using a quantile (rather than an absolute threshold) is more robust
    than reading off the hard rectification edge, which is noisy.
    """
    v2k = v2.sum(axis=1)
    v2w = v2.sum(axis=0)
    if v2k.sum() == 0 or v2w.sum() == 0:
        return np.nan, np.nan
    ck = np.cumsum(v2k) / v2k.sum()
    cw = np.cumsum(v2w) / v2w.sum()
    i_k = np.searchsorted(ck, quantile)
    i_w = np.searchsorted(cw, quantile)
    return k[min(i_k, len(k)-1)], omega[min(i_w, len(omega)-1)]


def run():
    set_defaults()

    A, beta, k0 = 1.0, 2.0, 0.02
    # Use higher noise so the active set is bounded by the rectification
    # threshold rather than by the grid edge. The Section 3.3 scaling holds
    # only in that regime.
    sigma_in_sq = 1e-2
    sigma_out_sq = 1e-2
    P_target = 0.01

    kmax, Nk = 3.0, 600
    wmax, Nw = 200.0, 1201
    k, wk = radial_grid(kmax, Nk, kmin=k0)
    # Use only positive omega for the extent measurement.
    omega_full, dw = omega_grid(wmax, Nw)
    weights = build_radial_weights(wk, dw)

    D_sweep = np.logspace(-1, 2.5, 15)

    kmaxes = np.zeros_like(D_sweep)
    wmaxes = np.zeros_like(D_sweep)
    for i, D in enumerate(D_sweep):
        Cx = C_D(k[:, None], omega_full[None, :], A=A, beta=beta, D=D, k0=k0)
        lam = solve_lambda(Cx, sigma_in_sq, sigma_out_sq, P_target, weights)
        v2 = v_star_sq(Cx, sigma_in_sq, sigma_out_sq, lam)
        # Restrict to omega > 0 half.
        half = omega_full.size // 2
        v2_pos = v2[:, half:]
        omega_pos = omega_full[half:]
        km, wm = active_extents(v2_pos, k, omega_pos, quantile=0.95)
        kmaxes[i] = km
        wmaxes[i] = wm
        print(f"D={D:>8.3g}  k95={km:.3g}  w95={wm:.3g}")

    # Fit a power law to log k vs log D over the central region.
    fit_mask = (D_sweep > 1.0) & (D_sweep < 30.0)
    p_k = np.polyfit(np.log(D_sweep[fit_mask]), np.log(kmaxes[fit_mask]), 1)
    p_w = np.polyfit(np.log(D_sweep[fit_mask]), np.log(wmaxes[fit_mask]), 1)
    print(f"Empirical exponents on central region: "
          f"k ~ D^{p_k[0]:.3f} (predicted -1/4), "
          f"omega ~ D^{p_w[0]:.3f} (predicted +1/2)")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)

    ax = axes[0]
    ax.loglog(D_sweep, kmaxes, "o-", color="C0", ms=4, lw=1.2, label="measured")
    # Predicted scaling normalized to match the first point.
    norm = kmaxes[fit_mask][0] / D_sweep[fit_mask][0] ** (-0.25)
    ax.loglog(D_sweep, norm * D_sweep ** (-0.25),
              "--", color="0.4", lw=1.0, label=r"$\propto D^{-1/4}$")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$k_{\max}$ (95th pct)")
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    ax.loglog(D_sweep, wmaxes, "o-", color="C3", ms=4, lw=1.2, label="measured")
    norm = wmaxes[fit_mask][0] / D_sweep[fit_mask][0] ** 0.5
    ax.loglog(D_sweep, norm * D_sweep ** 0.5,
              "--", color="0.4", lw=1.0, label=r"$\propto D^{1/2}$")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"$\omega_{\max}$ (95th pct)")
    ax.legend(frameon=False, fontsize=7)

    os.makedirs("figures", exist_ok=True)
    out = "figures/fig4_scaling.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
