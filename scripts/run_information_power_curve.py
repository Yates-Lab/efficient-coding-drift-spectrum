"""Run information--power curves for oracle and reusable cell-class models.

Example:

    python scripts/run_information_power_curve.py \
      --condition-set movement_sweep \
      --fit-mode fixed-H \
      --alpha-mode bounded_log_gain \
      --gain-delta-max 0.5 \
      --grid fast \
      --k-values 1,2,3 \
      --outdir outputs/information_power_curve
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, ".")

from src.cell_class_learning import (  # noqa: E402
    build_named_cell_learning_conditions,
    condition_table,
    information_from_filter_power,
    response_power_budget,
    solve_oracle_stack,
)
from src.cell_class_learning_fast import (  # noqa: E402
    fit_cell_classes_fast,
    refit_alpha_for_fixed_H_fast,
)
from src.plotting import setup_style  # noqa: E402


def _parse_float_list(text: str):
    return tuple(float(x) for x in str(text).split(",") if str(x).strip())


def _parse_int_list(text: str):
    return tuple(int(x) for x in str(text).split(",") if str(x).strip())


def _build_conditions(args):
    if args.condition_set == "movement_sweep":
        return build_named_cell_learning_conditions(
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
    return build_named_cell_learning_conditions(
        "cycle_pair",
        early_weight=args.early_weight,
        late_weight=args.late_weight,
        use_modulated_early=True,
    )


def _fit_kwargs(args):
    return dict(
        n_steps=args.steps_H,
        n_restarts=args.restarts,
        lr=args.lr,
        smooth_weight=args.smooth,
        entropy_weight=args.entropy,
        device=args.device,
        dtype=args.dtype,
        patience=args.patience,
        check_every=args.check_every,
        torch_threads=args.torch_threads if args.torch_threads > 0 else None,
        alpha_mode=args.alpha_mode,
        alpha_floor=args.alpha_floor,
        gain_delta_max=args.gain_delta_max,
        learn_baseline_mix=args.learn_baseline_mix,
        baseline_mix_weight=args.baseline_mix_weight,
        kl_to_baseline_weight=args.kl_to_baseline_weight,
        share_floor=args.share_floor,
        share_floor_weight=args.share_floor_weight,
    )


def _alpha_kwargs(args):
    return dict(
        n_steps=args.steps_alpha,
        lr=args.lr,
        device=args.device,
        dtype=args.dtype,
        patience=args.patience,
        check_every=args.check_every,
        torch_threads=args.torch_threads if args.torch_threads > 0 else None,
        alpha_mode=args.alpha_mode,
        alpha_floor=args.alpha_floor,
        gain_delta_max=args.gain_delta_max,
        learn_baseline_mix=args.learn_baseline_mix,
        baseline_mix_weight=args.baseline_mix_weight,
        kl_to_baseline_weight=args.kl_to_baseline_weight,
        share_floor=args.share_floor,
        share_floor_weight=args.share_floor_weight,
    )


def _plot_curves(power_fracs, oracle_retention, retention_by_K, regret_by_K, outdir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    ax.plot(power_fracs, oracle_retention, color="0.2", marker="o", label="oracle")
    for K, y in retention_by_K.items():
        ax.plot(power_fracs, y, marker="o", label=f"K={K}")
    ax.set_xlabel(r"response power $P/P_{\rm ref}$")
    ax.set_ylabel(r"information retention $J(P)/J_\star(P_{\rm ref})$")
    ax.set_title("Information--power retention")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "information_power_retention.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for K, y in regret_by_K.items():
        ax.plot(power_fracs, y, marker="o", label=f"K={K}")
    ax.set_xlabel(r"response power $P/P_{\rm ref}$")
    ax.set_ylabel(r"matched-power regret $(J_\star-J_K)/J_\star$")
    ax.set_title("Cost of fixed reusable classes")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "information_power_matched_regret.png", dpi=220)
    plt.close(fig)


def _plot_budget_share_summary(power_fracs, budget_share_by_K, outdir: Path):
    fig, ax = plt.subplots(figsize=(5.2, 3.4), constrained_layout=True)
    for K, shares in budget_share_by_K.items():
        max_share = np.nanmax(shares, axis=(1, 2))
        ax.plot(power_fracs, max_share, marker="o", label=f"K={K}")
    ax.set_xlabel(r"response power $P/P_{\rm ref}$")
    ax.set_ylabel(r"max response-budget share $\max_{q,c}\rho_{qc}$")
    ax.set_ylim(0, 1.03)
    ax.set_title("Gain-switching diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(outdir / "information_power_max_budget_share.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/information_power_curve")
    parser.add_argument("--condition-set", type=str, default="movement_sweep",
                        choices=("cycle_pair", "movement_sweep"))
    parser.add_argument("--grid", type=str, default="fast", choices=("fast", "hi_res"))
    parser.add_argument("--k-values", type=str, default="1,2,3")
    parser.add_argument("--power-fracs", type=str, default="0.05,0.075,0.1,0.15,0.25,0.4,0.6,0.8,1.0")
    parser.add_argument("--fit-mode", type=str, default="fixed-H",
                        choices=("shrink", "fixed-H", "full-refit"))
    parser.add_argument("--P-ref", type=float, default=50.0)
    parser.add_argument("--sigma-in", type=float, default=0.3)
    parser.add_argument("--sigma-out", type=float, default=1.0)
    parser.add_argument("--early-A-values", type=str, default="1,2,4,6,8")
    parser.add_argument("--late-D-values", type=str, default="0.0375,0.075,0.15,0.3,0.6")
    parser.add_argument("--early-weight", type=float, default=0.5)
    parser.add_argument("--late-weight", type=float, default=0.5)
    parser.add_argument("--n-saccades", type=int, default=32)
    parser.add_argument("--n-orientations", type=int, default=12)
    parser.add_argument("--saccade-window", type=float, default=0.150)
    parser.add_argument("--steps-H", type=int, default=1500)
    parser.add_argument("--steps-alpha", type=int, default=400)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--smooth", type=float, default=0.0)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--alpha-mode", type=str, default="bounded_log_gain",
                        choices=("softmax", "floor", "bounded_log_gain"))
    parser.add_argument("--alpha-floor", type=float, default=0.0)
    parser.add_argument("--gain-delta-max", type=float, default=0.5)
    parser.add_argument("--learn-baseline-mix", action="store_true")
    parser.add_argument("--baseline-mix-weight", type=float, default=0.0)
    parser.add_argument("--kl-to-baseline-weight", type=float, default=0.0)
    parser.add_argument("--share-floor", type=float, default=0.0)
    parser.add_argument("--share-floor-weight", type=float, default=0.0)
    args = parser.parse_args()

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    K_values = _parse_int_list(args.k_values)
    power_fracs = np.asarray(_parse_float_list(args.power_fracs), dtype=float)
    P_values = power_fracs * float(args.P_ref)

    print(f"Building conditions ({args.condition_set})...")
    conditions, pi = _build_conditions(args)

    print("Solving reference oracle stack...")
    oracle_ref = solve_oracle_stack(
        conditions,
        sigma_in=args.sigma_in,
        sigma_out=args.sigma_out,
        P0=args.P_ref,
        grid=args.grid,
        condition_weights=pi,
    )

    J_oracle = np.zeros(P_values.size, dtype=float)
    I_oracle_qP = np.zeros((P_values.size, len(conditions)), dtype=float)
    for i, P in enumerate(P_values):
        print(f"Oracle P/P_ref={power_fracs[i]:.3g}...")
        oracle_P = solve_oracle_stack(
            conditions,
            sigma_in=args.sigma_in,
            sigma_out=args.sigma_out,
            P0=float(P),
            grid=args.grid,
            condition_weights=pi,
        )
        J_oracle[i] = oracle_P.J_star
        I_oracle_qP[i] = oracle_P.I_star_q

    retention_by_K = {}
    regret_by_K = {}
    budget_share_by_K = {}
    J_by_K = {}
    I_by_K = {}
    base_fits = {}
    fits_by_KP = {}

    for K in K_values:
        print(f"Fitting K={K} at P_ref...")
        base_fit = fit_cell_classes_fast(
            oracle_ref.C_stack,
            oracle_ref.weights,
            sigma_in=args.sigma_in,
            sigma_out=args.sigma_out,
            P0=args.P_ref,
            K=K,
            condition_weights=oracle_ref.condition_weights,
            G_star=oracle_ref.G_star,
            seed=100 * K,
            **_fit_kwargs(args),
        )
        base_fits[K] = base_fit

        J_K = np.zeros(P_values.size, dtype=float)
        I_qP = np.zeros((P_values.size, len(conditions)), dtype=float)
        alpha_P = np.zeros((P_values.size, len(conditions), K), dtype=float)
        share_P = np.zeros((P_values.size, len(conditions), K), dtype=float)
        scale_P = np.zeros((P_values.size, len(conditions)), dtype=float)
        fits_by_KP[K] = []

        for i, P in enumerate(P_values):
            print(f"K={K} {args.fit_mode} P/P_ref={power_fracs[i]:.3g}...")
            if args.fit_mode == "shrink":
                G = base_fit.G * (float(P) / float(args.P_ref))
                I_q = information_from_filter_power(
                    oracle_ref.C_stack, G, oracle_ref.weights, args.sigma_in, args.sigma_out
                )
                J = float(np.sum(oracle_ref.condition_weights * I_q))
                fit = base_fit
                alpha = base_fit.alpha
                share = getattr(base_fit, "budget_share", np.full_like(base_fit.alpha, np.nan))
                scale = base_fit.scale * (float(P) / float(args.P_ref))
            elif args.fit_mode == "fixed-H":
                fit = refit_alpha_for_fixed_H_fast(
                    oracle_ref.C_stack,
                    oracle_ref.weights,
                    base_fit.H,
                    sigma_in=args.sigma_in,
                    sigma_out=args.sigma_out,
                    P0=float(P),
                    condition_weights=oracle_ref.condition_weights,
                    seed=1000 * K + i,
                    **_alpha_kwargs(args),
                )
                I_q = fit.I_q
                J = fit.J
                alpha = fit.alpha
                share = fit.budget_share
                scale = fit.scale
            else:
                fit = fit_cell_classes_fast(
                    oracle_ref.C_stack,
                    oracle_ref.weights,
                    sigma_in=args.sigma_in,
                    sigma_out=args.sigma_out,
                    P0=float(P),
                    K=K,
                    condition_weights=oracle_ref.condition_weights,
                    G_star=oracle_ref.G_star,
                    seed=1000 * K + i,
                    **_fit_kwargs(args),
                )
                I_q = fit.I_q
                J = fit.J
                alpha = fit.alpha
                share = fit.budget_share
                scale = fit.scale
            budgets = response_power_budget(oracle_ref.C_stack, fit.G if args.fit_mode != "shrink" else G,
                                            oracle_ref.weights, args.sigma_in)
            if not np.allclose(budgets, float(P), rtol=3e-3, atol=3e-3):
                print(f"warning: budget check for K={K}, P={P:g}: {budgets}")
            J_K[i] = J
            I_qP[i] = I_q
            alpha_P[i] = alpha
            share_P[i] = share
            scale_P[i] = scale
            fits_by_KP[K].append(fit)

        J_by_K[K] = J_K
        I_by_K[K] = I_qP
        retention_by_K[K] = J_K / max(float(oracle_ref.J_star), 1e-300)
        regret_by_K[K] = (J_oracle - J_K) / np.maximum(np.abs(J_oracle), 1e-300)
        budget_share_by_K[K] = share_P

        np.savez_compressed(
            outdir / f"information_power_K{K}.npz",
            P_values=P_values,
            power_fracs=power_fracs,
            J_oracle_P=J_oracle,
            I_oracle_qP=I_oracle_qP,
            J_K=J_K,
            I_qP=I_qP,
            retention=retention_by_K[K],
            matched_regret=regret_by_K[K],
            alpha=alpha_P,
            budget_share=share_P,
            scale=scale_P,
            H_ref=base_fit.H,
            alpha_ref=base_fit.alpha,
            budget_share_ref=getattr(base_fit, "budget_share", np.full_like(base_fit.alpha, np.nan)),
            baseline_mix_ref=getattr(base_fit, "baseline_mix", np.full(K, np.nan)),
        )

    payload = {
        "args": vars(args),
        "J_oracle_ref": float(oracle_ref.J_star),
        "P_values": P_values.tolist(),
        "power_fracs": power_fracs.tolist(),
        "K_values": list(K_values),
        "J_oracle_P": J_oracle.tolist(),
        "J_KP": {str(K): J_by_K[K].tolist() for K in K_values},
        "retention_KP": {str(K): retention_by_K[K].tolist() for K in K_values},
        "matched_regret_KP": {str(K): regret_by_K[K].tolist() for K in K_values},
        "max_budget_share_KP": {
            str(K): np.nanmax(budget_share_by_K[K], axis=(1, 2)).tolist()
            for K in K_values
        },
    }
    with open(outdir / "run_config.json", "w") as fobj:
        json.dump(vars(args), fobj, indent=2)
    with open(outdir / "condition_table.json", "w") as fobj:
        json.dump(condition_table(oracle_ref), fobj, indent=2)
    with open(outdir / "information_power_summary.json", "w") as fobj:
        json.dump(payload, fobj, indent=2)

    oracle_retention = J_oracle / max(float(oracle_ref.J_star), 1e-300)
    _plot_curves(power_fracs, oracle_retention, retention_by_K, regret_by_K, outdir)
    _plot_budget_share_summary(power_fracs, budget_share_by_K, outdir)
    print("Wrote outputs to", outdir)


if __name__ == "__main__":
    main()
