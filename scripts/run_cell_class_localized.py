"""Run localized, bounded-share cell-class learning.

Example:

    python scripts/run_cell_class_localized.py --condition-set movement_sweep --kmax 4
    python scripts/run_cell_class_localized.py --sweep loc_weight=0,0.05,0.2,0.5,1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")

from src.cell_class_learning import (  # noqa: E402
    build_named_cell_learning_conditions,
    condition_table,
    per_condition_regret,
    response_power_budget,
    solve_oracle_stack,
)
from src.cell_class_localized import fit_cell_classes_localized  # noqa: E402
from src.plotting import log_contourf, setup_style  # noqa: E402


def _parse_float_list(text: str):
    return tuple(float(x) for x in str(text).split(",") if str(x).strip())


def _parse_sweep(text: str):
    if not text:
        return "loc_weight", None
    name, values = text.split("=", 1)
    if name != "loc_weight":
        raise ValueError("only --sweep loc_weight=... is supported")
    return name, _parse_float_list(values)


def save_spectra_plot(f, omega, fit, outdir: Path, tag: str):
    fig, axes = plt.subplots(1, fit.K, figsize=(3.2 * fit.K, 3.0), constrained_layout=True)
    if fit.K == 1:
        axes = [axes]
    omega_pos = omega > 0
    for c, ax in enumerate(axes):
        H = fit.H[c][:, omega_pos]
        H = H / max(float(np.nanmax(H)), 1e-300)
        cf = log_contourf(ax, f, omega[omega_pos], H.T, n_levels=18, cmap="viridis", vmin_floor=1e-5)
        ax.set_xlabel("f (cyc/deg)")
        ax.set_ylabel(r"$\omega$ (rad/s)")
        ax.set_title(f"class {c}")
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(outdir / f"localized_{tag}_spectra.png", dpi=180)
    plt.close(fig)


def save_rho_plot(labels, fit, outdir: Path, tag: str):
    fig, ax = plt.subplots(figsize=(max(4.0, 0.55 * len(labels)), 2.8), constrained_layout=True)
    im = ax.imshow(fit.rho_qc, aspect="auto", vmin=0.0, vmax=1.0, cmap="magma")
    ax.set_xticks(np.arange(fit.K))
    ax.set_xticklabels([f"class {c}" for c in range(fit.K)])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(r"bounded shares $\rho_{qc}$")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(outdir / f"localized_{tag}_rho.png", dpi=180)
    plt.close(fig)


def save_kernel_plot(fit, outdir: Path, tag: str):
    x = np.arange(fit.K)
    width = 0.35
    fig, ax1 = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    ax1.bar(x - width / 2, fit.f_centroid_cpd, width, label="centroid cpd")
    ax1.set_ylabel("spatial centroid (cyc/deg)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"class {c}" for c in range(fit.K)])
    ax2 = ax1.twinx()
    ax2.bar(x + width / 2, fit.spatial_rf_width_deg, width, color="0.35", label="RF width")
    ax2.set_ylabel("RF width (deg)")
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    fig.savefig(outdir / f"localized_{tag}_rf_width.png", dpi=180)
    plt.close(fig)


def save_regret_path(rows, outdir: Path):
    rows = sorted(rows, key=lambda r: (r["K"], r["loc_weight"]))
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    for K in sorted({r["K"] for r in rows}):
        sub = [r for r in rows if r["K"] == K]
        ax.plot([r["loc_weight"] for r in sub], [r["regret"] for r in sub], marker="o", label=f"K={K}")
    ax.set_xscale("symlog", linthresh=0.05)
    ax.set_xlabel(r"localization weight $\lambda_{loc}$")
    ax.set_ylabel("information regret")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "regret_path.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/cell_classes_localized")
    parser.add_argument("--grid", type=str, default="fast", choices=("fast", "hi_res"))
    parser.add_argument("--condition-set", type=str, default="movement_sweep", choices=("cycle_pair", "movement_sweep"))
    parser.add_argument("--early-A-values", type=str, default="1,2,4,6,8")
    parser.add_argument("--late-D-values", type=str, default="0.0375,0.075,0.15,0.3,0.6")
    parser.add_argument("--sigma-in", type=float, default=0.3)
    parser.add_argument("--sigma-out", type=float, default=1.0)
    parser.add_argument("--P0", type=float, default=50.0)
    parser.add_argument("--kmax", type=int, default=4)
    parser.add_argument("--loc-weight", type=float, default=0.5)
    parser.add_argument("--delta-max", type=float, default=0.5)
    parser.add_argument("--sweep", type=str, default="")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--check-every", type=int, default=25)
    args = parser.parse_args()

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.condition_set == "movement_sweep":
        conditions, pi = build_named_cell_learning_conditions(
            "movement_sweep",
            grid=args.grid,
            early_A_values=_parse_float_list(args.early_A_values),
            late_D_values=_parse_float_list(args.late_D_values),
        )
    else:
        conditions, pi = build_named_cell_learning_conditions("cycle_pair")

    oracle = solve_oracle_stack(
        conditions,
        sigma_in=args.sigma_in,
        sigma_out=args.sigma_out,
        P0=args.P0,
        grid=args.grid,
        condition_weights=pi,
    )
    budgets = response_power_budget(oracle.C_stack, oracle.G_star, oracle.weights, args.sigma_in)
    print(f"Oracle J*: {oracle.J_star:.6g}; budget min/mean/max: {budgets.min():.6g}/{budgets.mean():.6g}/{budgets.max():.6g}")

    _, sweep_values = _parse_sweep(args.sweep)
    loc_values = sweep_values if sweep_values is not None else (args.loc_weight,)
    rows = []
    labels = [c.name for c in oracle.conditions]
    payload = {
        "f": oracle.f,
        "omega": oracle.omega,
        "weights": oracle.weights,
        "C_stack": oracle.C_stack,
        "G_star": oracle.G_star,
        "condition_weights": oracle.condition_weights,
        "I_star_q": oracle.I_star_q,
        "J_star": np.array(oracle.J_star),
    }
    summary = {"args": vars(args), "J_star": float(oracle.J_star), "fits": []}

    for loc in loc_values:
        for K in range(1, args.kmax + 1):
            print(f"Fitting K={K}, loc_weight={loc:g}")
            fit = fit_cell_classes_localized(
                oracle.C_stack,
                oracle.weights,
                oracle.f,
                sigma_in=args.sigma_in,
                sigma_out=args.sigma_out,
                P0=args.P0,
                K=K,
                condition_weights=oracle.condition_weights,
                G_star=oracle.G_star,
                loc_weight=loc,
                delta_max=args.delta_max,
                n_steps=args.steps,
                n_restarts=args.restarts,
                lr=args.lr,
                smooth_weight=args.smooth,
                device=args.device,
                dtype=args.dtype,
                patience=args.patience,
                check_every=args.check_every,
            )
            regret = (oracle.J_star - fit.J) / max(abs(oracle.J_star), 1e-300)
            tag = f"K{K}_loc{loc:g}".replace(".", "p")
            rows.append({"K": K, "loc_weight": float(loc), "J": float(fit.J), "regret": float(regret)})
            summary["fits"].append(
                {
                    "K": K,
                    "loc_weight": float(loc),
                    "J": float(fit.J),
                    "regret": float(regret),
                    "rho0": fit.rho0.tolist(),
                    "f_centroid_cpd": fit.f_centroid_cpd.tolist(),
                    "f_log_std": fit.f_log_std.tolist(),
                    "spatial_rf_width_deg": fit.spatial_rf_width_deg.tolist(),
                    "R_loc": float(fit.R_loc),
                }
            )
            np.savez_compressed(
                outdir / f"localized_fit_{tag}.npz",
                H=fit.H,
                G=fit.G,
                G_class=fit.G_class,
                rho0=fit.rho0,
                rho_qc=fit.rho_qc,
                delta_qc=fit.delta_qc,
                I_q=fit.I_q,
                regret_q=per_condition_regret(oracle.I_star_q, fit.I_q),
                f_centroid_cpd=fit.f_centroid_cpd,
                f_log_std=fit.f_log_std,
                spatial_rf_width_deg=fit.spatial_rf_width_deg,
                J=np.array(fit.J),
                R_loc=np.array(fit.R_loc),
            )
            payload[f"H_{tag}"] = fit.H
            payload[f"G_{tag}"] = fit.G
            payload[f"rho_qc_{tag}"] = fit.rho_qc
            save_spectra_plot(oracle.f, oracle.omega, fit, outdir, tag)
            save_rho_plot(labels, fit, outdir, tag)
            save_kernel_plot(fit, outdir, tag)

    np.savez_compressed(outdir / "localized_fit_results.npz", **payload)
    with open(outdir / "localized_summary.json", "w") as fobj:
        json.dump(summary, fobj, indent=2)
    with open(outdir / "condition_table.json", "w") as fobj:
        json.dump(condition_table(oracle), fobj, indent=2)
    save_regret_path(rows, outdir)
    print("Wrote outputs to", outdir)


if __name__ == "__main__":
    main()

