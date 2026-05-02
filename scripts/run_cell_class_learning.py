"""Run information-aware cell-class learning on early/late movement spectra.

Run from the repository root:

    pip install torch
    python scripts/run_cell_class_learning.py --grid fast --kmax 4

The script defaults to the canonical Figure 7 Rucci/Boi early/late spectra,
solves the existing one-filter oracle for each, then fits K=1..Kmax reusable
class spectra.
The production default uses the fast optimizer from ``src.cell_class_learning_fast``;
pass ``--optimizer reference`` for the original slower float64 optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")

from src.cell_class_learning import (  # noqa: E402
    build_cell_learning_conditions,
    build_named_cell_learning_conditions,
    class_budget_shares,
    class_centroids,
    class_summary_table,
    condition_table,
    effective_class_gains,
    per_condition_regret,
    response_power_budget,
    solve_oracle_stack,
    sweep_cell_classes,
)
from src.cell_class_learning_fast import sweep_cell_classes_fast  # noqa: E402
from src.plotting import setup_style, log_contourf  # noqa: E402

def save_regret_plot(sweep, outdir: Path):
    Ks = sorted(sweep.fits)
    regrets = [sweep.regret[K] for K in Ks]
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    ax.plot(Ks, regrets, marker="o")
    ax.set_xlabel("number of cell classes K")
    ax.set_ylabel("information regret")
    ax.set_xticks(Ks)
    ax.set_title("Class-count selection")
    ax.grid(True, alpha=0.25)
    fig.savefig(outdir / "cell_class_regret.png")
    plt.close(fig)


def save_class_spectra_plot(f, omega, fit, outdir: Path):
    K = fit.K
    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3.0), constrained_layout=True)
    if K == 1:
        axes = [axes]
    omega_pos = omega > 0
    for c, ax in enumerate(axes):
        H = fit.H[c][:, omega_pos]
        H = H / max(float(np.nanmax(H)), 1e-300)
        cf = log_contourf(
            ax,
            f,
            omega[omega_pos],
            H.T,
            n_levels=18,
            cmap="viridis",
            vmin_floor=1e-5,
        )
        ax.set_xlabel("f (cyc/unit)")
        ax.set_ylabel(r"$\omega$ (rad/s)")
        ax.set_title(f"class {c}")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(outdir / f"cell_class_K{K}_spectra.png")
    plt.close(fig)


def save_alpha_plot(oracle, fit, outdir: Path):
    labels = [c.name for c in oracle.conditions]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 3.0), constrained_layout=True)
    for c in range(fit.K):
        ax.plot(x, fit.alpha[:, c], marker="o", label=f"class {c}")
    ax.set_ylabel(r"mixture weight $\alpha_{qc}$")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(ncol=min(fit.K, 4))
    ax.set_title(f"Condition-dependent class weights, K={fit.K}")
    fig.savefig(outdir / f"cell_class_K{fit.K}_alpha.png")
    plt.close(fig)


def save_budget_share_plot(oracle, fit, sigma_in: float, outdir: Path):
    rho, _ = class_budget_shares(
        oracle.C_stack,
        fit.H,
        fit.alpha,
        fit.scale,
        oracle.weights,
        sigma_in,
    )
    labels = [c.name for c in oracle.conditions]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8.0, 3.0), constrained_layout=True)
    for c in range(fit.K):
        ax.plot(x, rho[:, c], marker="o", label=f"class {c}")
    ax.set_ylabel("response-budget share")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(ncol=min(fit.K, 4))
    ax.set_title(f"Condition-dependent budget share, K={fit.K}")
    fig.savefig(outdir / f"cell_class_K{fit.K}_budget_share.png")
    plt.close(fig)


def _parse_float_list(text: str):
    return tuple(float(x) for x in str(text).split(",") if str(x).strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/cell_classes")
    parser.add_argument("--grid", type=str, default="fast", choices=("fast", "hi_res"))
    parser.add_argument("--sigma-in", type=float, default=0.3)
    parser.add_argument("--sigma-out", type=float, default=1.0)
    parser.add_argument("--P0", type=float, default=50.0)
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--condition-set", type=str, default="cycle_pair",
                        choices=("cycle_pair", "movement_sweep"),
                        help="cycle_pair is the canonical early/late pair; movement_sweep is 5 saccades + 5 drifts")
    parser.add_argument("--early-A-values", type=str, default="1,2,4,6,8",
                        help="comma-separated saccade amplitudes for movement_sweep")
    parser.add_argument("--late-D-values", type=str, default="0.0375,0.075,0.15,0.3,0.6",
                        help="comma-separated cycles-aware drift D values for movement_sweep")
    parser.add_argument("--early-weight", type=float, default=0.5)
    parser.add_argument("--late-weight", type=float, default=0.5)
    parser.add_argument("--n-saccades", type=int, default=32)
    parser.add_argument("--n-orientations", type=int, default=12)
    parser.add_argument("--saccade-window", type=float, default=0.150)
    parser.add_argument("--optimizer", type=str, default="fast",
                        choices=("fast", "reference"),
                        help="fast is the production default; reference is the original optimizer")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--smooth", type=float, default=None)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None,
                        help="fast: auto/cpu/mps/cuda; reference: usually cpu")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"),
                        help="fast optimizer dtype")
    parser.add_argument("--patience", type=int, default=20,
                        help="fast optimizer early-stopping patience")
    parser.add_argument("--check-every", type=int, default=25,
                        help="fast optimizer early-stopping check interval")
    parser.add_argument("--torch-threads", type=int, default=1,
                        help="set PyTorch CPU thread count for the fast optimizer; 1 is usually fastest/stablest on laptops; 0 leaves default")
    args = parser.parse_args()

    if args.optimizer == "fast":
        if args.steps is None:
            args.steps = 1600
        if args.restarts is None:
            args.restarts = 2
        if args.lr is None:
            args.lr = 5e-2
        if args.smooth is None:
            args.smooth = 0.0
        if args.device is None:
            args.device = "auto"
    else:
        if args.steps is None:
            args.steps = 4000
        if args.restarts is None:
            args.restarts = 8
        if args.lr is None:
            args.lr = 3e-2
        if args.smooth is None:
            args.smooth = 1e-4
        if args.device is None:
            args.device = "cpu"

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Building movement conditions ({args.condition_set})...")
    if args.condition_set == "movement_sweep":
        conditions, pi = build_named_cell_learning_conditions(
            "movement_sweep",
            grid=args.grid,
            early_A_values=_parse_float_list(args.early_A_values),
            late_D_values=_parse_float_list(args.late_D_values),
            early_weight=args.early_weight,
            late_weight=args.late_weight,
            saccade_n_saccades=args.n_saccades,
            saccade_n_orientations=args.n_orientations,
            saccade_T_win_s=args.saccade_window,
        )
    else:
        conditions, pi = build_named_cell_learning_conditions(
            "cycle_pair",
            early_weight=args.early_weight,
            late_weight=args.late_weight,
            use_modulated_early=True,
        )

    print("Solving one-filter oracle stack...")
    oracle = solve_oracle_stack(
        conditions,
        sigma_in=args.sigma_in,
        sigma_out=args.sigma_out,
        P0=args.P0,
        grid=args.grid,
        condition_weights=pi,
    )

    budgets = response_power_budget(oracle.C_stack, oracle.G_star, oracle.weights, args.sigma_in)
    print("Oracle condition table:")
    for row in condition_table(oracle):
        print(
            f"  {row['index']:02d} {row['name']:>14s}  "
            f"weight={row['weight']:.3f} I*={row['I_star']:.5g}"
        )
    print("Oracle weighted J*:", oracle.J_star)
    print("Oracle budget check: min/mean/max =", budgets.min(), budgets.mean(), budgets.max())

    print(f"Fitting class models with {args.optimizer} optimizer...")
    if args.optimizer == "fast":
        sweep = sweep_cell_classes_fast(
            oracle,
            K_values=tuple(range(1, args.kmax + 1)),
            n_steps=args.steps,
            n_restarts=args.restarts,
            lr=args.lr,
            smooth_weight=args.smooth,
            entropy_weight=args.entropy,
            device=args.device,
            dtype=args.dtype,
            patience=args.patience,
            check_every=args.check_every,
            torch_threads=args.torch_threads if args.torch_threads > 0 else None,
        )
    else:
        sweep = sweep_cell_classes(
            oracle,
            K_values=tuple(range(1, args.kmax + 1)),
            n_steps=args.steps,
            n_restarts=args.restarts,
            lr=args.lr,
            smooth_weight=args.smooth,
            entropy_weight=args.entropy,
            device=args.device,
        )

    for K, fit in sweep.fits.items():
        diag = getattr(fit, "fast_diagnostics", None)
        diag_msg = ""
        if diag is not None:
            diag_msg = (
                f" [{diag.device}, {diag.dtype}, active F="
                f"{diag.n_active_freqs}/{diag.n_total_freqs}, steps={diag.steps_run}]"
            )
        print(f"K={K}: J={fit.J:.6g}, regret={sweep.regret[K]:.3%}{diag_msg}")
        print("alpha:")
        print(np.round(fit.alpha, 3))
        print("centroids:", class_centroids(fit.H, oracle.f, oracle.omega, oracle.weights))

    # Save numeric results. Store H/alpha/G for each K with explicit keys.
    payload = {
        "f": oracle.f,
        "omega": oracle.omega,
        "weights": oracle.weights,
        "C_stack": oracle.C_stack,
        "G_star": oracle.G_star,
        "condition_weights": oracle.condition_weights,
        "I_star_q": oracle.I_star_q,
        "J_star": np.array(oracle.J_star),
        "budget_star_q": budgets,
        "K_values": np.array(sorted(sweep.fits)),
        "J_values": np.array([sweep.fits[K].J for K in sorted(sweep.fits)]),
        "regret_values": np.array([sweep.regret[K] for K in sorted(sweep.fits)]),
        "sigma_in": np.array(args.sigma_in),
        "sigma_out": np.array(args.sigma_out),
        "P0": np.array(args.P0),
    }
    summary_payload = {
        "args": vars(args),
        "J_star": float(oracle.J_star),
        "fits": {},
    }
    for K, fit in sweep.fits.items():
        rho, spend = class_budget_shares(
            oracle.C_stack, fit.H, fit.alpha, fit.scale, oracle.weights, args.sigma_in
        )
        payload[f"H_K{K}"] = fit.H
        payload[f"alpha_K{K}"] = fit.alpha
        payload[f"scale_K{K}"] = fit.scale
        payload[f"G_K{K}"] = fit.G
        payload[f"I_q_K{K}"] = fit.I_q
        payload[f"regret_q_K{K}"] = per_condition_regret(oracle.I_star_q, fit.I_q)
        payload[f"budget_share_K{K}"] = rho
        payload[f"budget_spend_K{K}"] = spend
        payload[f"effective_gain_K{K}"] = effective_class_gains(fit)
        summary_payload["fits"][str(K)] = {
            "J": float(fit.J),
            "regret": float(sweep.regret[K]),
            "centroids": class_centroids(fit.H, oracle.f, oracle.omega, oracle.weights),
            "summary": class_summary_table(fit.H, oracle.f, oracle.omega, oracle.weights),
        }
    np.savez_compressed(outdir / "cell_class_fit_results.npz", **payload)

    with open(outdir / "condition_table.json", "w") as fobj:
        json.dump(condition_table(oracle), fobj, indent=2)
    with open(outdir / "cell_class_fit_summary.json", "w") as fobj:
        json.dump(summary_payload, fobj, indent=2)

    save_regret_plot(sweep, outdir)
    for K in sorted(sweep.fits):
        save_class_spectra_plot(oracle.f, oracle.omega, sweep.fits[K], outdir)
        save_alpha_plot(oracle, sweep.fits[K], outdir)
        save_budget_share_plot(oracle, sweep.fits[K], args.sigma_in, outdir)

    print("Wrote outputs to", outdir)


if __name__ == "__main__":
    main()
