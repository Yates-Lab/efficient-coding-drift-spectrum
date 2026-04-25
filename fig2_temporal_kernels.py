"""
Fig 2: Causal temporal kernels v*_D(k, t) at representative spatial
frequencies, for several D.

Pipeline (per (D, k) pair):
  1. Compute |v*_D(k, omega)|^2 along the omega axis.
  2. Apply minimum-phase reconstruction along omega to get v*_D(k, omega).
  3. Inverse FT along omega to get v*_D(k, t).

No 2D spatial FFT is needed: the derivation in Section 2.6 of the notes
(and Jun et al. Appendix A.3) treats k as a fixed label and reconstructs
the temporal filter at each k independently.

Layout: rows = representative spatial frequencies k, columns = D values.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from grids import build_radial_weights, omega_grid, radial_grid
from optimizer import solve_lambda, v_star_sq
from phase import ifft_to_time, minimum_phase
from spectrum import C_D
from style import set_defaults


def run():
    set_defaults()

    # ----- Parameters -----
    A, beta, k0 = 1.0, 2.0, 0.02
    sigma_in_sq = 5e-3
    sigma_out_sq = 1e-3
    P_target = 0.05  # matches fig 1 for consistent lambda(D)

    D_values = [0.08, 0.8, 8.0, 80.0]
    k_display = [0.1, 0.5, 1.5]  # representative spatial frequencies

    # ----- Per-D grids. dt = pi/wmax must resolve the kernel, whose
    # timescale is ~ 1/(D * k^2). We scale wmax with D so dt*D is roughly
    # constant. For each D we also keep Nw modest.
    kmax, Nk = 3.0, 400
    k, wk = radial_grid(kmax, Nk, kmin=k0)

    # ----- Compute -----
    kernels = {}
    for D in D_values:
        # Target dt ~ 0.2 / (D * k_min_used^2). With k ~ 0.1..1.5 the
        # minimum kernel timescale is 1/(D * 1.5^2) = 0.44/D. Want dt
        # well below this: pi/wmax < 0.05/D => wmax > 60*D.
        wmax = max(200.0, 120.0 * D)
        Nw = 2001 if D < 20 else 4001
        omega, dw = omega_grid(wmax, Nw)
        weights = build_radial_weights(wk, dw)

        Cx = C_D(k[:, None], omega[None, :], A=A, beta=beta, D=D, k0=k0)
        lam = solve_lambda(Cx, sigma_in_sq, sigma_out_sq, P_target, weights)
        v2 = v_star_sq(Cx, sigma_in_sq, sigma_out_sq, lam)

        v_complex = minimum_phase(v2, floor_rel=1e-10, pad_factor=2)

        for kd in k_display:
            idx = np.argmin(np.abs(k - kd))
            v_om = v_complex[idx, :]
            t, v_t = ifft_to_time(v_om, omega)
            kernels[(D, kd)] = (t, v_t.real, k[idx])

        print(f"D={D:<8g}  wmax={wmax:.0f}, Nw={Nw}, dt={np.pi/wmax:.4g}, "
              f"lambda={lam:.4g}")

    # ----- Plot: one row per k, one column per D. Shared x axis so the
    # shift in kernel timescale with D is visually obvious: fast kernels
    # (large D) appear as thin pulses near t=0, slow kernels (small D)
    # fill more of the window.
    nrows = len(k_display)
    ncols = len(D_values)
    t_half_shared = 0.4

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9.0, 2.1 * nrows),
        sharex=True, constrained_layout=True,
    )

    for i, kd in enumerate(k_display):
        for j, D in enumerate(D_values):
            ax = axes[i, j] if nrows > 1 else axes[j]
            t, vt, k_actual = kernels[(D, kd)]
            m = (t > -0.15 * t_half_shared) & (t < t_half_shared)
            ts, vts = t[m], vt[m]
            ax.plot(ts, vts, color="C0", lw=1.3)
            ax.axhline(0, color="0.5", lw=0.4)
            ax.axvline(0, color="0.5", lw=0.4)
            ylim = np.max(np.abs(vts)) * 1.2
            ax.set_ylim(-ylim, ylim)
            ax.set_xlim(-0.15 * t_half_shared, t_half_shared)

            if i == 0:
                ax.set_title(rf"$D={D:g}$")
            if j == 0:
                ax.set_ylabel(rf"$k={k_actual:.2f}$" "\n" r"$v^\star(k,t)$")
            if i == nrows - 1:
                ax.set_xlabel(r"$t$")

    os.makedirs("figures", exist_ok=True)
    out = "figures/fig2_temporal_kernels.pdf"
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    run()
