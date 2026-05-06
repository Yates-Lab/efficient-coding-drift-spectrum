"""Sweep oracle and localized-class performance over movement strategies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from src.cell_class_localized import (  # noqa: E402
    build_strategy_conditions,
    fit_cell_classes_localized,
    refit_modulation_for_fixed_classes,
)
from src.cell_class_learning import solve_oracle_stack  # noqa: E402
from src.plotting import setup_style  # noqa: E402


def _parse_values(text: str, *, default):
    if not text:
        return np.asarray(default, dtype=float)
    return np.asarray([float(x) for x in text.split(",") if x.strip()], dtype=float)


def _heatmap(values, D_values, A_values, outpath: Path, title: str, cbar_label: str):
    fig, ax = plt.subplots(figsize=(4.6, 3.5), constrained_layout=True)
    im = ax.imshow(values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(A_values)))
    ax.set_xticklabels([f"{v:g}" for v in A_values], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(D_values)))
    ax.set_yticklabels([f"{v:g}" for v in D_values])
    ax.set_xlabel("saccade amplitude A (deg)")
    ax.set_ylabel(r"drift coefficient D (deg$^2$/s)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=cbar_label)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/strategy_sweep")
    parser.add_argument("--grid", type=str, default="fast", choices=("fast", "hi_res"))
    parser.add_argument("--D-values", type=str, default="")
    parser.add_argument("--A-values", type=str, default="")
    parser.add_argument("--ref-D", type=float, default=0.15)
    parser.add_argument("--ref-A", type=float, default=2.0)
    parser.add_argument("--sigma-in", type=float, default=0.3)
    parser.add_argument("--sigma-out", type=float, default=1.0)
    parser.add_argument("--P0", type=float, default=50.0)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--loc-weight", type=float, default=0.5)
    parser.add_argument("--delta-max", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--refit-steps", type=int, default=300)
    parser.add_argument("--retune-adapt-weight", type=float, default=0.0,
                        help="if positive, also refit bounded per-condition retuning")
    parser.add_argument("--retune-smooth-weight", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    args = parser.parse_args()

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    D_values = _parse_values(args.D_values, default=np.geomspace(0.05, 2.0, 8))
    A_values = _parse_values(args.A_values, default=np.geomspace(0.5, 8.0, 8))

    print("Fitting reference localized classes...")
    ref_conditions, ref_pi = build_strategy_conditions(D=args.ref_D, A=args.ref_A, grid=args.grid)
    ref_oracle = solve_oracle_stack(
        ref_conditions,
        sigma_in=args.sigma_in,
        sigma_out=args.sigma_out,
        P0=args.P0,
        grid=args.grid,
        condition_weights=ref_pi,
    )
    ref_fit = fit_cell_classes_localized(
        ref_oracle.C_stack,
        ref_oracle.weights,
        ref_oracle.f,
        sigma_in=args.sigma_in,
        sigma_out=args.sigma_out,
        P0=args.P0,
        K=args.K,
        condition_weights=ref_oracle.condition_weights,
        G_star=ref_oracle.G_star,
        loc_weight=args.loc_weight,
        delta_max=args.delta_max,
        n_steps=args.steps,
        n_restarts=2,
        device=args.device,
        dtype=args.dtype,
    )

    J_star = np.zeros((len(D_values), len(A_values)), dtype=float)
    J_K_loc = np.zeros_like(J_star)
    J_K_loc_plus = np.zeros_like(J_star)
    rho_qc = np.zeros((len(D_values), len(A_values), 2, args.K), dtype=float)
    delta_qc = np.zeros_like(rho_qc)

    for i, D in enumerate(D_values):
        for j, A in enumerate(A_values):
            print(f"Strategy D={D:g}, A={A:g}")
            conditions, pi = build_strategy_conditions(D=float(D), A=float(A), grid=args.grid)
            oracle = solve_oracle_stack(
                conditions,
                sigma_in=args.sigma_in,
                sigma_out=args.sigma_out,
                P0=args.P0,
                grid=args.grid,
                condition_weights=pi,
            )
            J_star[i, j] = oracle.J_star
            fit = refit_modulation_for_fixed_classes(
                oracle.C_stack,
                oracle.weights,
                oracle.f,
                ref_fit.H,
                ref_fit.rho0,
                sigma_in=args.sigma_in,
                sigma_out=args.sigma_out,
                P0=args.P0,
                condition_weights=oracle.condition_weights,
                delta_max=args.delta_max,
                n_steps=args.refit_steps,
                device=args.device,
                dtype=args.dtype,
            )
            J_K_loc[i, j] = fit.J
            rho_qc[i, j] = fit.rho_qc
            delta_qc[i, j] = fit.delta_qc
            if args.retune_adapt_weight > 0:
                fit_plus = refit_modulation_for_fixed_classes(
                    oracle.C_stack,
                    oracle.weights,
                    oracle.f,
                    ref_fit.H,
                    ref_fit.rho0,
                    sigma_in=args.sigma_in,
                    sigma_out=args.sigma_out,
                    P0=args.P0,
                    condition_weights=oracle.condition_weights,
                    delta_max=args.delta_max,
                    retune=True,
                    adapt_weight=args.retune_adapt_weight,
                    adapt_smooth_weight=args.retune_smooth_weight,
                    n_steps=args.refit_steps,
                    device=args.device,
                    dtype=args.dtype,
                )
                J_K_loc_plus[i, j] = fit_plus.J
            else:
                J_K_loc_plus[i, j] = fit.J

    regret = (J_star - J_K_loc) / np.maximum(np.abs(J_star), 1e-300)
    retune_gain = (J_K_loc_plus - J_K_loc) / np.maximum(np.abs(J_star), 1e-300)
    np.savez_compressed(
        outdir / "strategy_sweep.npz",
        D_values=D_values,
        A_values=A_values,
        J_star=J_star,
        J_K_loc=J_K_loc,
        J_K_loc_plus=J_K_loc_plus,
        rho_qc_per_strategy=rho_qc,
        delta_qc_per_strategy=delta_qc,
        ref_H=ref_fit.H,
        ref_rho0=ref_fit.rho0,
    )
    with open(outdir / "strategy_sweep_summary.json", "w") as fobj:
        json.dump(
            {
                "args": vars(args),
                "ref_J_star": float(ref_oracle.J_star),
                "ref_fit_J": float(ref_fit.J),
                "ref_rho0": ref_fit.rho0.tolist(),
                "mean_regret": float(np.mean(regret)),
                "mean_retune_gain": float(np.mean(retune_gain)),
            },
            fobj,
            indent=2,
        )
    _heatmap(J_star, D_values, A_values, outdir / "fig_strategy_J_star.png", "Oracle landscape", r"$J^*$")
    _heatmap(regret, D_values, A_values, outdir / "fig_strategy_regret.png", "Localized-class regret", r"$(J^*-J_K)/J^*$")
    _heatmap(retune_gain, D_values, A_values, outdir / "fig_strategy_retune_gain.png", "Bounded-retuning gain", r"$(J_K^+-J_K)/J^*$")
    _heatmap(rho_qc[:, :, 0, 0], D_values, A_values, outdir / "fig_strategy_share_modulation.png", "Saccade condition class 0 share", r"$\rho_{saccade,0}$")

    j_ref = int(np.argmin(np.abs(D_values - args.ref_D)))
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    for c in range(args.K):
        ax.plot(A_values, delta_qc[j_ref, :, 0, c], marker="o", label=f"class {c}")
    ax.set_xscale("log")
    ax.set_xlabel("saccade amplitude A (deg)")
    ax.set_ylabel(r"saccade log-gain $\delta_{qc}$")
    ax.set_title(f"Prediction slice at D={D_values[j_ref]:g}")
    ax.legend()
    fig.savefig(outdir / "fig_strategy_delta_slice.png", dpi=180)
    plt.close(fig)
    print("Wrote outputs to", outdir)


if __name__ == "__main__":
    main()
