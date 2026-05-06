"""Audit movement spectra before interpreting optimal filters or cell classes.

This script is intentionally diagnostic.  It checks the unit conventions and
basic invariants that must hold before the cell-class story is meaningful:

1. Brownian drift half-width: omega_1/2(f) = D (2*pi*f)^2.
2. Brownian drift temporal power conservation: ∫ Q dω/(2π) ≈ 1.
3. Stationary separable control has near-perfect log-additive separability.
4. Movement-generated spectra are nonseparable in log-additive coordinates.

Run from the repository root:

    python scripts/run_spectrum_audit.py --outdir outputs/spectrum_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")

from src.cell_class_figures import log_additive_separability_r2, normalize_for_plot
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX, fast_grid
from src.plotting import radial_weights, band_mask_radial, log_contourf, setup_style
from src.power_spectrum_library import stationary_vs_active_story_specs
from src.spectra import DriftSpectrum
from src.spectrum_diagnostics import temporal_centroid_log_slope

TWOPI = 2.0 * np.pi


def measured_halfwidth(omega: np.ndarray, Q_row: np.ndarray) -> float:
    """Return the first positive omega where Q falls to half its DC value."""
    omega = np.asarray(omega, dtype=float)
    q = np.asarray(Q_row, dtype=float)
    i0 = int(np.argmin(np.abs(omega)))
    q0 = float(q[i0])
    if q0 <= 0 or not np.isfinite(q0):
        return np.nan
    pos = omega >= 0
    w = omega[pos]
    qp = q[pos]
    target = 0.5 * q0
    below = np.flatnonzero(qp <= target)
    if below.size == 0:
        return np.nan
    j = int(below[0])
    if j == 0:
        return float(w[0])
    x0, x1 = qp[j - 1], qp[j]
    w0, w1 = w[j - 1], w[j]
    if x1 == x0:
        return float(w1)
    frac = (target - x0) / (x1 - x0)
    return float(w0 + frac * (w1 - w0))


def audit_brownian_drift(D: float, outdir: Path) -> dict:
    # Restrict to frequencies whose Lorentzian half-width is well resolved on
    # the numerical omega grid. The very lowest f values have sub-bin widths and
    # should be audited analytically, not by a rectangular grid.
    f = np.geomspace(0.5, 5.0, 50)
    omega_max = 2500.0
    omega = np.linspace(-omega_max, omega_max, 200001)
    Q = DriftSpectrum(D=D).redistribution(f, omega)
    integral = np.trapz(Q, omega, axis=1) / TWOPI

    hw = np.array([measured_halfwidth(omega, Q[i]) for i in range(f.size)])
    theory = D * (TWOPI * f) ** 2
    valid = np.isfinite(hw) & (theory > 0)
    ratio = hw[valid] / theory[valid]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), constrained_layout=True)
    axes[0].loglog(f, theory, label=r"theory $D(2\pi f)^2$")
    axes[0].loglog(f[valid], hw[valid], "o", ms=3, label="measured")
    axes[0].set_xlabel("spatial frequency f (cpd)")
    axes[0].set_ylabel(r"half-width $\omega_{1/2}$ (rad/s)")
    axes[0].set_title("Brownian drift unit check")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].semilogx(f, integral)
    axes[1].axhline(1.0, color="0.5", lw=1, ls="--")
    axes[1].set_xlabel("spatial frequency f (cpd)")
    axes[1].set_ylabel(r"$\int Q d\omega/(2\pi)$")
    axes[1].set_title("Temporal power conservation")
    axes[1].grid(True, alpha=0.25)
    fig.savefig(outdir / "audit_brownian_drift_units.png", dpi=200)
    fig.savefig(outdir / "audit_brownian_drift_units.pdf")
    plt.close(fig)

    return {
        "D": float(D),
        "halfwidth_ratio_median": float(np.nanmedian(ratio)),
        "halfwidth_ratio_iqr": [float(np.nanpercentile(ratio, 25)), float(np.nanpercentile(ratio, 75))],
        "temporal_power_integral_median": float(np.nanmedian(integral)),
        "temporal_power_integral_max_abs_error": float(np.nanmax(np.abs(integral - 1.0))),
    }


def audit_story_spectra(outdir: Path) -> list[dict]:
    f, omega = fast_grid()
    weights = radial_weights(f, omega) * band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    specs = stationary_vs_active_story_specs()
    rows = []
    C_list = []

    for spec in specs:
        C = spec.spectrum.C(f, omega)
        C_list.append(C)
        rows.append({
            "key": spec.key,
            "label": spec.label,
            "family": spec.family,
            "log_additive_separability_R2": float(log_additive_separability_r2(C, weights)),
            "temporal_centroid_log_slope": float(temporal_centroid_log_slope(
                C, f, omega, f_lo=0.1, f_hi=4.0, omega_min=OMEGA_MIN
            )),
            "reference": spec.reference,
        })

    omega_pos = omega > 0
    fig, axes = plt.subplots(2, len(specs), figsize=(2.4 * len(specs), 4.7), constrained_layout=True)
    if len(specs) == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for j, (spec, C, row) in enumerate(zip(specs, C_list, rows)):
        C_plot = normalize_for_plot(C[:, omega_pos])
        cf = log_contourf(axes[0, j], f, omega[omega_pos], C_plot.T, n_levels=18, cmap="magma", vmin_floor=1e-6)
        axes[0, j].set_title(f"{spec.title}\n$R^2_{{sep}}$={row['log_additive_separability_R2']:.2f}")
        axes[0, j].set_xlabel("f (cpd)")
        if j == 0:
            axes[0, j].set_ylabel(r"$\omega$ (rad/s)")
        else:
            axes[0, j].set_yticklabels([])

    x = np.arange(len(specs))
    width = 0.38
    axes[1, 0].bar(x - width/2, [r["log_additive_separability_R2"] for r in rows], width, label=r"$R^2_{sep}$")
    ax2 = axes[1, 0].twinx()
    ax2.bar(x + width/2, [r["temporal_centroid_log_slope"] for r in rows], width, color="tab:orange", label=r"slope $\bar\omega(f)$")
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].set_ylabel(r"log-additive separability $R^2$")
    ax2.set_ylabel(r"$d\log \bar\omega / d\log f$")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([s.key for s in specs], rotation=35, ha="right")
    axes[1, 0].set_title("Separability and f--omega coupling")
    axes[1, 0].grid(True, axis="y", alpha=0.25)
    axes[1, 0].legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    for ax in axes[1, 1:]:
        ax.axis("off")
    fig.colorbar(cf, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.015, label="normalized power")
    fig.savefig(outdir / "audit_stationary_vs_active_spectra.png", dpi=200)
    fig.savefig(outdir / "audit_stationary_vs_active_spectra.pdf")
    plt.close(fig)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/spectrum_audit")
    parser.add_argument("--D", type=float, default=0.0375, help="Brownian drift D in deg^2/s")
    args = parser.parse_args()

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    payload = {
        "brownian_drift": audit_brownian_drift(args.D, outdir),
        "story_spectra": audit_story_spectra(outdir),
        "notes": [
            "Brownian drift should satisfy omega_1/2 = D (2*pi*f)^2 for f in cycles/deg.",
            "The separable stationary control should have R2_sep near 1 by construction.",
            "Log-additive R2 is a conservative diagnostic; smooth movement spectra can still have high R2.",
            "The temporal-centroid slope is the complementary coupling check: separable controls should have slope near zero.",
        ],
    }
    with open(outdir / "spectrum_audit_summary.json", "w") as fobj:
        json.dump(payload, fobj, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
