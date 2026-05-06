"""Figure 8: Mostofi-style analytic saccade-transient approximation.

Reproduces the qualitative structure of Mostofi et al. Figure 4 using the
same cumulative-Gaussian-smoothed step approximation now used by the shared
pipeline:

    Q_sac(f, omega; A) =
        2 [1 - J_0(2 pi f A)] exp[-(omega sigma(A))^2] / omega^2.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt

from src.spectra import (
    DriftSpectrum,
    saccade_amplitude_average,
)
from src.plotting import setup_style

setup_style()


def natural_image_power(f_cpd, beta=2.0):
    return np.asarray(f_cpd, dtype=float) ** (-float(beta))


def kelly_temporal_weight(f_hz, k_cpd):
    f = np.asarray(f_hz, dtype=float)[:, None]
    k = np.asarray(k_cpd, dtype=float)[None, :]
    G = (6.1 + 7.3 * np.abs(np.log(np.maximum(k, 1e-6) / 3.0)) ** 3) * f * k
    G *= np.exp(-(f + 2.0 * k) / 22.95)
    return G / np.maximum(np.sum(G, axis=0, keepdims=True), 1e-30)


def amplitude_bin_Q(k_cpd, f_hz, A_low, A_high, n_amp=31):
    amps = np.linspace(A_low, A_high, n_amp) if A_high > A_low else np.array([A_low])
    omega = 2.0 * np.pi * np.asarray(f_hz, dtype=float)
    return saccade_amplitude_average(k_cpd, omega, amps).T


def fig8():
    k = np.logspace(np.log10(0.03), np.log10(30.0), 220)
    f_map = np.linspace(1.0, 35.0, 180)
    f_int = np.linspace(0.5, 80.0, 260)
    I_k = natural_image_power(k)

    Q_2_3 = amplitude_bin_Q(k, f_map, 2.0, 3.0, n_amp=31)
    S_2_3 = Q_2_3 * I_k[None, :]
    idx_5 = int(np.argmin(np.abs(f_map - 5.0)))
    low_k_mask = (k > 0.045) & (k < 0.09)
    offset_A = -35.0 - np.median(10.0 * np.log10(np.maximum(S_2_3[idx_5, low_k_mask], 1e-300)))
    S_2_3_db = 10.0 * np.log10(np.maximum(S_2_3, 1e-300)) + offset_A

    W = kelly_temporal_weight(f_int, k)
    amp_bins = [
        ("drift", None),
        ("<0.5 deg", (0.10, 0.50)),
        ("0.5-1 deg", (0.50, 1.00)),
        ("1-2 deg", (1.00, 2.00)),
        ("2-3 deg", (2.00, 3.00)),
        ("3-4 deg", (3.00, 4.00)),
        ("4-5 deg", (4.00, 5.00)),
        ("5-6 deg", (5.00, 6.00)),
        ("6-7 deg", (6.00, 7.00)),
        (">7 deg", (7.00, 9.00)),
    ]

    integrated = {}
    omega_int = 2.0 * np.pi * f_int
    for label, bounds in amp_bins:
        if bounds is None:
            Q = DriftSpectrum(D=0.035).redistribution(k, omega_int).T
            S = 0.055 * Q * I_k[None, :]
        else:
            Q = amplitude_bin_Q(k, f_int, bounds[0], bounds[1], n_amp=31)
            S = Q * I_k[None, :]
        integrated[label] = np.sum(W * S, axis=0)

    ref = np.median(integrated[">7 deg"][low_k_mask])
    panelC_db = {
        label: 10.0 * np.log10(np.maximum(vals / ref, 1e-300)) - 20.0
        for label, vals in integrated.items()
    }

    fig = plt.figure(figsize=(12.5, 4.8))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.05, 1.05, 1.35],
        height_ratios=[1, 1],
        wspace=0.58,
        hspace=0.45,
    )
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[:, 1])
    axC = fig.add_subplot(gs[:, 2])

    mesh = axA.pcolormesh(k, f_map, S_2_3_db, shading="auto", vmin=-90, vmax=-10)
    axA.set_xscale("log")
    axA.set_xlim(0.03, 30.0)
    axA.set_ylim(1.0, 35.0)
    axA.set_xlabel("spatial frequency (cycles/deg)")
    axA.set_ylabel("temporal frequency (Hz)")
    axA.set_title("A  2-3 deg saccades over natural scenes")
    for yy in [5, 9, 16, 30]:
        axA.axhline(yy, linestyle=":", linewidth=0.9)
    cb = fig.colorbar(mesh, ax=axA, fraction=0.046, pad=0.06)
    cb.set_label("spectral density (dB)")

    for ft in [5, 9, 16, 30]:
        idx = int(np.argmin(np.abs(f_map - ft)))
        axB.semilogx(k, S_2_3_db[idx], label=f"{ft} Hz")
    ref_k = np.array([0.7, 20.0])
    anchor_k = 1.0
    anchor_idx = int(np.argmin(np.abs(k - anchor_k)))
    anchor_y = S_2_3_db[idx_5, anchor_idx]
    axB.semilogx(
        ref_k,
        anchor_y - 20.0 * np.log10(ref_k / anchor_k),
        linestyle="--",
        linewidth=1.2,
        label=r"$k^{-2}$ ref.",
    )
    axB.set_xlim(0.03, 30.0)
    axB.set_ylim(-90, -25)
    axB.set_xlabel("spatial frequency (cycles/deg)")
    axB.set_ylabel("spectral density (dB)", labelpad=8)
    axB.set_title("B  sections")
    axB.legend(frameon=False, fontsize=8)

    for label, _ in amp_bins:
        axC.semilogx(k, panelC_db[label], label=label)
    axC.set_xlim(0.03, 30.0)
    axC.set_ylim(-75, -15)
    axC.set_xlabel("spatial frequency (cpd)")
    axC.set_ylabel("spectral density (dB)")
    axC.set_title("C  temporal-sensitivity-weighted power")
    axC.legend(frameon=False, fontsize=7, ncol=1)

    fig.suptitle(
        "Mostofi et al. Figure 4 reproduction using the analytic saccade transient",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    out = "outputs/fig8_mostofi_saccade_approximation.png"
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig8()
