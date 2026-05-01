"""Make publication-level figures for the cell-class analysis.

This script expects the richer output saved by ``scripts/run_cell_class_learning.py``
from the production patch. Run the learning script first, then:

    python scripts/make_cell_class_story_figures.py \
        --indir outputs/cell_classes \
        --outdir outputs/cell_classes_story \
        --K 2

Figures generated
-----------------
fig_cellclass_01_spectra_and_oracles
    Representative early/late spectra and the fully adaptive oracle filters.
fig_cellclass_02_log_separability
    Nonseparability metric for input spectra and oracle filters.
fig_cellclass_03_regret_and_allocation
    Information regret, per-condition regret, budget shares, and effective gains.
fig_cellclass_04_learned_classes
    Learned class spectra plus reconstructed spatial and temporal kernels.
fig_cellclass_05_oracle_vs_class
    Oracle filters compared with constrained K-class filters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, ".")

from src.plotting import setup_style, log_contourf  # noqa: E402
from src.cell_class_learning import (  # noqa: E402
    class_budget_shares,
    effective_class_gains,
    log_separability_residual,
    per_condition_regret,
    spatial_kernel_from_filter_power,
    temporal_kernel_from_filter_power,
)


def load_results(indir: Path):
    npz_path = indir / "cell_class_fit_results.npz"
    json_path = indir / "condition_table.json"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}; run scripts/run_cell_class_learning.py first")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}; run scripts/run_cell_class_learning.py first")
    data = np.load(npz_path)
    with open(json_path) as fobj:
        rows = json.load(fobj)
    required = {"C_stack", "G_star", "f", "omega", "weights", "I_star_q"}
    missing = [k for k in required if k not in data.files]
    if missing:
        raise KeyError(
            "The results file was written by an older script and is missing "
            f"{missing}. Re-run scripts/run_cell_class_learning.py from the production patch."
        )
    return data, rows


def savefig(fig, outdir: Path, stem: str):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{stem}.{ext}")
    plt.close(fig)


def condition_labels(rows):
    return [r["name"].replace("early_", "early\n").replace("late_", "late\n") for r in rows]


def condition_title(row):
    name = row["name"]
    if name == "early_cycle":
        return "early: C = I(f) Q_saccade"
    if name == "late_cycle":
        return "late: C = I(f) Q_drift"
    return name.replace("_", " ")


def representative_indices(rows):
    early = [i for i, r in enumerate(rows) if r["epoch"] == "early"]
    late = [i for i, r in enumerate(rows) if r["epoch"] == "late"]
    idx = []
    if early:
        idx.extend([early[0], early[-1]])
    if late:
        idx.extend([late[0], late[-1]])
    # Preserve order while removing duplicates.
    out = []
    for i in idx:
        if i not in out:
            out.append(i)
    return out


def positive_band_mask(omega, weights):
    return (np.asarray(omega) > 0) & (np.any(np.asarray(weights) > 0, axis=0))


def normalize_for_display(Z, mask=None):
    Z = np.asarray(Z, dtype=float).copy()
    if mask is not None and Z.ndim == 2:
        denom = np.nanmax(Z[:, mask])
    else:
        denom = np.nanmax(Z)
    denom = max(float(denom), 1e-300)
    return Z / denom


def contour_panel(
    ax,
    f,
    omega,
    Z,
    *,
    title=None,
    ylabel=None,
    cmap="viridis",
    floor=1e-5,
    temporal_hz=True,
):
    omega_mask = omega > 0
    y = omega[omega_mask] / (2.0 * np.pi) if temporal_hz else omega[omega_mask]
    Zp = normalize_for_display(Z, omega_mask)[:, omega_mask]
    cf = log_contourf(
        ax,
        f,
        y,
        Zp.T,
        n_levels=18,
        cmap=cmap,
        vmin_floor=floor,
    )
    if title is not None:
        ax.set_title(title)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("spatial frequency f (cpd)")
    return cf


def make_spectra_and_oracle_figure(data, rows, outdir: Path):
    f = data["f"]
    omega = data["omega"]
    C = data["C_stack"]
    G = data["G_star"]
    idx = representative_indices(rows)
    labels = [condition_title(rows[i]) for i in idx]

    fig, axes = plt.subplots(2, len(idx), figsize=(2.45 * len(idx), 4.1), constrained_layout=True)
    if len(idx) == 1:
        axes = axes[:, None]
    for j, q in enumerate(idx):
        cf0 = contour_panel(axes[0, j], f, omega, C[q], title=labels[j], ylabel="input spectrum\n$C_q(f,\\nu)$" if j == 0 else None, cmap="magma")
        cf1 = contour_panel(axes[1, j], f, omega, G[q], ylabel="oracle filter\n$G_q^\\star=|v_q^\\star|^2$" if j == 0 else None, cmap="viridis")
    fig.colorbar(cf0, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.01, label="normalized power")
    fig.colorbar(cf1, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.01, label="normalized filter power")
    fig.suptitle("Movement spectra and fully adaptive efficient-coding optima")
    savefig(fig, outdir, "fig_cellclass_01_spectra_and_oracles")


def make_log_separability_figure(data, rows, outdir: Path):
    omega = data["omega"]
    weights = data["weights"]
    omega_mask = positive_band_mask(omega, weights)
    mask2d = (weights > 0) & omega_mask[None, :]
    C = data["C_stack"]
    G = data["G_star"]
    labels = condition_labels(rows)
    sep_C = [log_separability_residual(C[q], mask=mask2d) for q in range(C.shape[0])]
    sep_G = [log_separability_residual(G[q], mask=mask2d) for q in range(G.shape[0])]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 3.0), constrained_layout=True)
    ax.bar(x - width / 2, sep_C, width, label="input spectrum")
    ax.bar(x + width / 2, sep_G, width, label="oracle filter")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("log-separability residual")
    ax.set_title("Space-time inseparability across movement conditions")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.25)
    savefig(fig, outdir, "fig_cellclass_02_log_separability")


def make_regret_allocation_figure(data, rows, outdir: Path, K: int):
    labels = condition_labels(rows)
    x = np.arange(len(labels))
    K_values = data["K_values"].astype(int)
    regrets = data["regret_values"]
    I_star = data["I_star_q"]

    if f"I_q_K{K}" not in data.files:
        raise KeyError(f"K={K} not found in results")
    alpha = data[f"alpha_K{K}"]
    scale = data[f"scale_K{K}"]
    H = data[f"H_K{K}"]
    sigma_in = float(data["sigma_in"]) if "sigma_in" in data.files else 0.3
    rho, _ = class_budget_shares(data["C_stack"], H, alpha, scale, data["weights"], sigma_in)
    eff = scale[:, None] * alpha

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.0), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(K_values, regrets, marker="o")
    ax.set_xlabel("number of classes K")
    ax.set_ylabel("information regret")
    ax.set_title("Class-count selection")
    ax.set_xticks(K_values)
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    for kk in K_values:
        key = f"I_q_K{kk}"
        if key in data.files:
            r_q = per_condition_regret(I_star, data[key])
            ax.plot(x, r_q, marker="o", label=f"K={kk}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("per-condition regret")
    ax.set_title("Where information is lost")
    ax.legend(ncol=min(len(K_values), 4))
    ax.grid(True, alpha=0.25)

    ax = axes[1, 0]
    for c in range(K):
        ax.plot(x, rho[:, c], marker="o", label=f"class {c}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("response-budget share")
    ax.set_title(f"Budget allocation, K={K}")
    ax.legend(ncol=min(K, 4))

    ax = axes[1, 1]
    eff_norm = eff / np.maximum(np.nanmax(eff, axis=0, keepdims=True), 1e-300)
    for c in range(K):
        ax.plot(x, eff_norm[:, c], marker="o", label=f"class {c}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("normalized effective gain")
    ax.set_title(f"Gain modulation, K={K}")
    ax.legend(ncol=min(K, 4))

    fig.suptitle("Information-aware cell-class factorization")
    savefig(fig, outdir, "fig_cellclass_03_regret_and_allocation")


def make_learned_class_figure(data, outdir: Path, K: int):
    f = data["f"]
    omega = data["omega"]
    H = data[f"H_K{K}"]
    fig, axes = plt.subplots(3, K, figsize=(2.75 * K, 7.0), constrained_layout=True)
    if K == 1:
        axes = axes[:, None]
    for c in range(K):
        contour_panel(axes[0, c], f, omega, H[c], title=f"class {c}", ylabel="class spectrum\n$H_c(f,\\omega)$" if c == 0 else None, cmap="viridis")
        r, v_r = spatial_kernel_from_filter_power(H[c], f, omega, n_k=256)
        if np.max(np.abs(v_r)) > 0:
            v_r = v_r / np.max(np.abs(v_r))
        axes[1, c].plot(r, v_r)
        axes[1, c].axhline(0, lw=0.6, alpha=0.5)
        axes[1, c].set_xlim(-3, 3)
        axes[1, c].set_xlabel("space")
        if c == 0:
            axes[1, c].set_ylabel("normalized\nspatial kernel")

        t, h_t, f_peak = temporal_kernel_from_filter_power(H[c], f, omega)
        if np.max(np.abs(h_t)) > 0:
            h_t = h_t / np.max(np.abs(h_t))
        axes[2, c].plot(t, h_t)
        axes[2, c].axhline(0, lw=0.6, alpha=0.5)
        axes[2, c].set_xlim(0, 0.6)
        axes[2, c].set_xlabel("time (s)")
        axes[2, c].set_title(f"peak f={f_peak:.2g}")
        if c == 0:
            axes[2, c].set_ylabel("normalized\ntemporal kernel")
    fig.suptitle(f"Learned reusable class filters, K={K}")
    savefig(fig, outdir, "fig_cellclass_04_learned_classes")


def make_oracle_vs_class_figure(data, rows, outdir: Path, K: int):
    f = data["f"]
    omega = data["omega"]
    G_star = data["G_star"]
    G_fit = data[f"G_K{K}"]
    idx = representative_indices(rows)
    labels = [rows[i]["name"].replace("_", " ") for i in idx]
    fig, axes = plt.subplots(2, len(idx), figsize=(2.45 * len(idx), 4.1), constrained_layout=True)
    if len(idx) == 1:
        axes = axes[:, None]
    for j, q in enumerate(idx):
        contour_panel(axes[0, j], f, omega, G_star[q], title=labels[j], ylabel="oracle\n$G_q^\\star$" if j == 0 else None, cmap="viridis")
        contour_panel(axes[1, j], f, omega, G_fit[q], ylabel=f"K={K} class\n$G_q^{{(K)}}$" if j == 0 else None, cmap="viridis")
    fig.suptitle("Fully adaptive optima versus class-constrained filters")
    savefig(fig, outdir, "fig_cellclass_05_oracle_vs_class")


def make_temporal_oracle_class_figure(data, rows, outdir: Path, K: int):
    f = data["f"]
    omega = data["omega"]
    G_star = data["G_star"]
    G_fit = data[f"G_K{K}"]
    idx = representative_indices(rows)

    fig, axes = plt.subplots(1, len(idx), figsize=(2.65 * len(idx), 2.4), constrained_layout=True)
    if len(idx) == 1:
        axes = [axes]
    for ax, q in zip(axes, idx):
        t0, h0, fp0 = temporal_kernel_from_filter_power(G_star[q], f, omega)
        t1, h1, fp1 = temporal_kernel_from_filter_power(G_fit[q], f, omega)
        if np.max(np.abs(h0)) > 0:
            h0 = h0 / np.max(np.abs(h0))
        if np.max(np.abs(h1)) > 0:
            h1 = h1 / np.max(np.abs(h1))
        ax.plot(t0, h0, label="oracle")
        ax.plot(t1, h1, ls="--", label=f"K={K}")
        ax.axhline(0, lw=0.6, alpha=0.5)
        ax.set_xlim(0, 0.45)
        ax.set_title(rows[q]["name"].replace("_", " "))
        ax.set_xlabel("time (s)")
        ax.text(0.02, 0.05, f"f* {fp0:.2g}/{fp1:.2g}", transform=ax.transAxes, fontsize=7)
    axes[0].set_ylabel("normalized temporal kernel")
    axes[-1].legend()
    fig.suptitle("Temporal kernels: oracle versus class-constrained filters")
    savefig(fig, outdir, "fig_cellclass_06_temporal_oracle_vs_class")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default="outputs/cell_classes")
    parser.add_argument("--outdir", type=str, default="outputs/cell_classes_story")
    parser.add_argument("--K", type=int, default=2)
    args = parser.parse_args()

    setup_style()
    indir = Path(args.indir)
    outdir = Path(args.outdir)
    data, rows = load_results(indir)

    make_spectra_and_oracle_figure(data, rows, outdir)
    make_log_separability_figure(data, rows, outdir)
    make_regret_allocation_figure(data, rows, outdir, args.K)
    make_learned_class_figure(data, outdir, args.K)
    make_oracle_vs_class_figure(data, rows, outdir, args.K)
    make_temporal_oracle_class_figure(data, rows, outdir, args.K)
    print(f"Wrote story figures to {outdir}")


if __name__ == "__main__":
    main()
