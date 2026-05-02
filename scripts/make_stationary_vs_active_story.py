"""Make narrative figures contrasting old stationary controls with active sensing.

The figure uses the shared spectrum library and the shared efficient-coding
pipeline.  It adds the missing separable stationary control and places it next
to: (i) the Dong--Atick linear-motion control, and (ii) the Rucci/Boi early and
late cycle spectra.

Run from the repository root:

    python scripts/make_stationary_vs_active_story.py --grid hi_res

For a quick version:

    python scripts/make_stationary_vs_active_story.py --grid fast --no-kernels
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
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX
from src.pipeline import run, extract_kernels
from src.plotting import radial_weights, band_mask_radial, log_contourf, setup_style
from src.power_spectrum_library import stationary_vs_active_story_specs
from src.spectrum_diagnostics import temporal_centroid_log_slope


def _positive(Z, omega):
    m = omega > 0
    return Z[:, m], omega[m]


def _safe_norm(y):
    y = np.asarray(y, dtype=float)
    m = np.nanmax(np.abs(y))
    if not np.isfinite(m) or m <= 0:
        return y
    return y / m


def run_story(config):
    specs = stationary_vs_active_story_specs()
    results = []
    for spec in specs:
        r = run(
            spec.spectrum,
            sigma_in=config.sigma_in,
            sigma_out=config.sigma_out,
            P0=config.P0,
            grid=config.grid,
        )
        if not config.no_kernels:
            extract_kernels(r)
        results.append((spec, r))
        print(f"{spec.key:24s} I*={r.I:.5g}")
    return results


def plot_spectra_and_oracles(results, outdir: Path):
    n = len(results)
    f = results[0][1].f
    omega = results[0][1].omega
    weights = radial_weights(f, omega) * band_mask_radial(f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX)
    omega_pos = omega > 0

    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 4.6), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    rows = []
    for j, (spec, r) in enumerate(results):
        r2_C = log_additive_separability_r2(r.C, weights)
        r2_G = log_additive_separability_r2(r.v_sq, weights)
        slope_C = temporal_centroid_log_slope(
            r.C, f, omega, f_lo=0.1, f_hi=4.0, omega_min=OMEGA_MIN
        )
        rows.append({
            "key": spec.key,
            "label": spec.label,
            "I_star": float(r.I),
            "C_R2_sep": float(r2_C),
            "G_R2_sep": float(r2_G),
            "C_temporal_centroid_log_slope": float(slope_C),
            "reference": spec.reference,
        })

        C_plot = normalize_for_plot(r.C[:, omega_pos])
        G_plot = normalize_for_plot(r.v_sq[:, omega_pos])
        cf0 = log_contourf(axes[0, j], f, omega[omega_pos], C_plot.T, n_levels=18, cmap="magma", vmin_floor=1e-6)
        cf1 = log_contourf(axes[1, j], f, omega[omega_pos], G_plot.T, n_levels=18, cmap="viridis", vmin_floor=1e-6)
        axes[0, j].set_title(f"{spec.title}\n$R^2_{{sep}}(C)$={r2_C:.2f}, slope={slope_C:.2f}")
        axes[1, j].set_title(f"oracle $G^*$\n$R^2_{{sep}}(G)$={r2_G:.2f}")
        axes[1, j].set_xlabel("spatial frequency f (cpd)")
        if j == 0:
            axes[0, j].set_ylabel(r"input $C_q$: $\omega$ (rad/s)")
            axes[1, j].set_ylabel(r"oracle $G_q^*$: $\omega$ (rad/s)")
        else:
            axes[0, j].set_yticklabels([])
            axes[1, j].set_yticklabels([])

    fig.colorbar(cf0, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.012, label="norm. input power")
    fig.colorbar(cf1, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.012, label="norm. filter power")
    fig.suptitle("Stationary approximations versus movement-conditioned retinal input", y=1.03)
    fig.savefig(outdir / "fig_story_01_spectra_and_oracles.png", dpi=220)
    fig.savefig(outdir / "fig_story_01_spectra_and_oracles.pdf")
    plt.close(fig)

    with open(outdir / "stationary_vs_active_summary.json", "w") as fobj:
        json.dump(rows, fobj, indent=2)


def plot_kernel_summary(results, outdir: Path):
    if getattr(results[0][1], "spatial_v", None) is None:
        return
    fig, axes = plt.subplots(2, 1, figsize=(6.6, 5.2), constrained_layout=True)
    for spec, r in results:
        axes[0].plot(r.spatial_r, _safe_norm(r.spatial_v), lw=1.4, label=spec.label)
        axes[1].plot(r.temporal_t, _safe_norm(r.temporal_v), lw=1.4, label=spec.label)
    axes[0].axhline(0, color="0.7", lw=0.7)
    axes[1].axhline(0, color="0.7", lw=0.7)
    axes[0].set_xlim(-5, 5)
    axes[1].set_xlim(0, 0.5)
    axes[0].set_ylabel("normalized spatial kernel")
    axes[1].set_ylabel("normalized temporal kernel")
    axes[1].set_xlabel("time (s)")
    axes[0].set_xlabel("space (deg or arbitrary unit)")
    axes[0].set_title("Oracle kernels under each spectrum")
    axes[1].legend(ncol=2, fontsize=8)
    fig.savefig(outdir / "fig_story_02_oracle_kernels.png", dpi=220)
    fig.savefig(outdir / "fig_story_02_oracle_kernels.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/stationary_vs_active_story")
    parser.add_argument("--grid", type=str, default="hi_res", choices=("fast", "hi_res"))
    parser.add_argument("--sigma-in", type=float, default=0.2)
    parser.add_argument("--sigma-out", type=float, default=2.0)
    parser.add_argument("--P0", type=float, default=50.0)
    parser.add_argument("--no-kernels", action="store_true")
    args = parser.parse_args()

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = run_story(args)
    plot_spectra_and_oracles(results, outdir)
    plot_kernel_summary(results, outdir)
    print(f"wrote {outdir}")


if __name__ == "__main__":
    main()
