"""Input-noise sweep for the information-aware cell-class model.

This script asks whether the number of useful reusable classes changes with the
input-noise floor. It reuses the same movement-condition stack and repeats the
oracle + K-class fits for each sigma_in. The default condition stack is the
direct saccade / Brownian drift pair.

Example
-------
python scripts/run_cell_class_noise_sweep.py \
    --grid fast \
    --sigma-in-values 0.1,0.3,0.6,1.0 \
    --kmax 3 \
    --steps 1200 \
    --restarts 2

The production default uses the fast optimizer from
``src.cell_class_learning_fast``. Pass ``--optimizer reference`` for the
original slower float64 optimizer.
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
    condition_table,
    solve_oracle_stack,
    sweep_cell_classes,
)
from src.cell_class_learning_fast import sweep_cell_classes_fast  # noqa: E402
from src.plotting import setup_style  # noqa: E402


def _parse_float_list(s: str):
    return tuple(float(x) for x in s.split(",") if x.strip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs/cell_classes_noise_sweep")
    parser.add_argument("--grid", type=str, default="fast", choices=("fast", "hi_res"))
    parser.add_argument("--sigma-in-values", type=str, default="0.1,0.3,0.6,1.0")
    parser.add_argument("--sigma-out", type=float, default=1.0)
    parser.add_argument("--P0", type=float, default=50.0)
    parser.add_argument("--kmax", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="fast",
                        choices=("fast", "reference"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--smooth", type=float, default=None)
    parser.add_argument("--entropy", type=float, default=0.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dtype", type=str, default="float32", choices=("float32", "float64"))
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--check-every", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()

    if args.optimizer == "fast":
        if args.steps is None:
            args.steps = 1200
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
            args.steps = 2500
        if args.restarts is None:
            args.restarts = 4
        if args.lr is None:
            args.lr = 3e-2
        if args.smooth is None:
            args.smooth = 1e-4
        if args.device is None:
            args.device = "cpu"

    setup_style()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sigma_values = _parse_float_list(args.sigma_in_values)
    K_values = tuple(range(1, args.kmax + 1))

    conditions, pi = build_cell_learning_conditions()

    regrets = np.zeros((len(sigma_values), len(K_values)))
    J_star = np.zeros(len(sigma_values))
    J_fit = np.zeros_like(regrets)

    for i, sigma_in in enumerate(sigma_values):
        print(f"sigma_in={sigma_in:g}")
        oracle = solve_oracle_stack(
            conditions,
            sigma_in=sigma_in,
            sigma_out=args.sigma_out,
            P0=args.P0,
            grid=args.grid,
            condition_weights=pi,
        )
        if args.optimizer == "fast":
            sweep = sweep_cell_classes_fast(
                oracle,
                K_values=K_values,
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
                K_values=K_values,
                n_steps=args.steps,
                n_restarts=args.restarts,
                lr=args.lr,
                smooth_weight=args.smooth,
                entropy_weight=args.entropy,
                device=args.device,
            )
        J_star[i] = oracle.J_star
        for j, K in enumerate(K_values):
            regrets[i, j] = sweep.regret[K]
            J_fit[i, j] = sweep.fits[K].J
            print(f"  K={K}: regret={regrets[i,j]:.3%}")

    np.savez_compressed(
        outdir / "noise_sweep_results.npz",
        sigma_in_values=np.asarray(sigma_values),
        K_values=np.asarray(K_values),
        regret=regrets,
        J_star=J_star,
        J_fit=J_fit,
        condition_weights=pi,
        optimizer=np.array(args.optimizer),
    )
    with open(outdir / "condition_table.json", "w") as fobj:
        # Use the last oracle if it exists; otherwise just serialize conditions.
        json.dump(condition_table(oracle), fobj, indent=2)
    with open(outdir / "noise_sweep_args.json", "w") as fobj:
        json.dump(vars(args), fobj, indent=2)

    fig, ax = plt.subplots(figsize=(4.8, 3.2), constrained_layout=True)
    for j, K in enumerate(K_values):
        ax.plot(sigma_values, regrets[:, j], marker="o", label=f"K={K}")
    ax.set_xscale("log")
    ax.set_xlabel(r"input noise $\sigma_{\mathrm{in}}$")
    ax.set_ylabel("information regret")
    ax.set_title("Class-count benefit across input noise")
    ax.grid(True, alpha=0.25)
    ax.legend()
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_cellclass_noise_sweep.{ext}")
    plt.close(fig)
    print("Wrote", outdir)


if __name__ == "__main__":
    main()
