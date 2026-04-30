"""Figure 6c: saccade-vs-equivalent-drift kernel comparison.

For three saccade amplitudes A, compute the optimal kernel under the
saccade model and under a drift model with the matched effective
diffusion coefficient D_eff = π² λ A². At small kA (microsaccades),
the saccade Q reduces to a drift Lorentzian with this D_eff, so the
kernels should agree. At large kA (medium and large saccades), the
saccade Q saturates while the drift Q keeps growing as k², so the
kernels diverge: drift produces broad spatial kernels and saccades
produce narrow ones.

This is the answer to the question "if saccades behave like drift,
why don't large saccades give broad spatial kernels?" Large saccades
have high D_eff, but the saturation regime caps the high-spatial-
frequency amplification, so the optimal filter doesn't extend to
high f the way a true high-D drift would.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import saccade_spectrum, drift_spectrum
from src.solver import solve_efficient_coding
from src.kernels import (
    spatial_kernel_2d, radial_cross_section,
    minimum_phase_temporal_filter, soft_band_taper,
)
from src.plotting import setup_style, radial_weights, band_mask_radial
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, hi_res_grid

setup_style()


def _kernels(C, f, omega, sigma_in, sigma_out, P0, weights_b, mask):
    v_sq, _, I_star = solve_efficient_coding(
        C, sigma_in, sigma_out, P0, weights_b, band_mask=mask,
    )
    domega = np.gradient(omega)
    v_s_sq = np.sum(v_sq * np.abs(domega)[None, :], axis=1) / (2 * np.pi)
    v_s = np.sqrt(np.maximum(v_s_sq, 0.0))
    f_fine = np.linspace(0.0, 6.0, 1024)
    v_s_interp = np.interp(f_fine, f, v_s, left=v_s[0], right=0.0)
    rx, ry, v_xy = spatial_kernel_2d(
        lambda k: np.interp(k, f_fine, v_s_interp,
                             left=v_s_interp[0], right=0.0),
        k_max=8.0, n_k=512,
    )
    r, v_r = radial_cross_section(v_xy, rx, ry)

    energy_per_f = np.sum(v_sq * np.abs(domega)[None, :], axis=1)
    i_peak_f = int(np.argmax(energy_per_f))
    v_t_mag = np.sqrt(np.maximum(v_sq[i_peak_f, :], 0.0))
    taper = soft_band_taper(omega, OMEGA_MIN, OMEGA_MAX, alpha=0.25)
    v_t_smooth = np.maximum(v_t_mag * taper, 1e-3 * (v_t_mag * taper).max())
    t, h_t, _ = minimum_phase_temporal_filter(v_t_smooth, omega)
    return r, v_r, t, h_t, I_star


def fig6c():
    f, omega = hi_res_grid()
    F = f[:, None]
    W = omega[None, :]
    sigma_in, sigma_out, P0 = 0.3, 1.0, 50.0
    weights = radial_weights(f, omega)
    mask = band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    weights_b = weights * mask
    lam = 3.0

    cases = [
        (0.3, "small (microsaccade)"),
        (2.5, "medium (typical natural)"),
        (7.0, "large"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6),
                             gridspec_kw={"hspace": 0.40, "wspace": 0.30,
                                          "left": 0.06, "right": 0.98,
                                          "top": 0.92, "bottom": 0.10})

    color_sac = "#1f6fb4"
    color_drift = "#d8540e"

    for col, (A, label) in enumerate(cases):
        D_eff = np.pi ** 2 * lam * A ** 2
        C_sac = saccade_spectrum(f, omega, A=A, lam=lam)
        C_drift = drift_spectrum(F, W, D=D_eff, beta=2.0)

        r_s, vr_s, t_s, ht_s, I_s = _kernels(
            C_sac, f, omega, sigma_in, sigma_out, P0, weights_b, mask,
        )
        r_d, vr_d, t_d, ht_d, I_d = _kernels(
            C_drift, f, omega, sigma_in, sigma_out, P0, weights_b, mask,
        )
        vr_s_n = vr_s / np.max(np.abs(vr_s))
        vr_d_n = vr_d / np.max(np.abs(vr_d))
        ht_s_n = ht_s / np.max(np.abs(ht_s))
        ht_d_n = ht_d / np.max(np.abs(ht_d))

        ax_sp = axes[0, col]
        ax_sp.plot(r_s, vr_s_n, color=color_sac, lw=1.6,
                   label=rf"saccade $A={A:g}^\circ$ ($I^*={I_s:.2f}$)")
        ax_sp.plot(r_d, vr_d_n, color=color_drift, lw=1.4, ls="--",
                   label=rf"drift $D_\mathrm{{eff}}={D_eff:.1f}$ ($I^*={I_d:.2f}$)")
        ax_sp.set_xlim(-3.0, 3.0)
        ax_sp.axhline(0.0, color="0.7", lw=0.4)
        ax_sp.axvline(0.0, color="0.85", lw=0.3)
        ax_sp.set_xlabel(r"$r$ (units)")
        ax_sp.set_ylabel(r"$v_s(r) / \max v_s$")
        ax_sp.set_title(rf"{label}: spatial kernel  ($A={A:g}^\circ$)",
                        fontsize=9, pad=2)
        ax_sp.legend(loc="best", fontsize=7, frameon=False)

        ax_t = axes[1, col]
        ax_t.plot(t_s, ht_s_n, color=color_sac, lw=1.6,
                  label="saccade")
        ax_t.plot(t_d, ht_d_n, color=color_drift, lw=1.4, ls="--",
                  label=rf"drift $D_\mathrm{{eff}}={D_eff:.1f}$")
        ax_t.set_xlim(0.0, 0.4)
        ax_t.axhline(0.0, color="0.7", lw=0.4)
        ax_t.set_xlabel(r"$t$ (s)")
        ax_t.set_ylabel(r"$v_t(t) / \max v_t$")
        ax_t.set_title(f"{label}: temporal kernel", fontsize=9, pad=2)
        ax_t.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(
        r"Saccades vs equivalent drift ($D_\mathrm{eff} = \pi^2 \lambda A^2$).  "
        r"Match at small $A$ (small $kA$ regime) but diverge at large $A$  "
        r"due to saccade saturation cap at $f_c = 1/(2A)$.",
        y=0.97, fontsize=10,
    )

    out = "./outputs/fig6c_saccade_vs_drift_kernels.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig6c()
