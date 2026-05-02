"""Production plotting and diagnostics for information-aware cell classes.

This module contains small, reusable functions for turning the learned
cell-class fits into publication-style figures.  It assumes the numerical
objects saved by ``scripts/run_cell_class_learning.py``:

    C_stack[q, f, omega]       movement-conditioned input spectra
    G_star[q, f, omega]        unconstrained oracle filters |v_q^*|^2
    H_K{K}[c, f, omega]        learned class spectra
    alpha_K{K}[q, c]           condition-dependent mixture weights
    scale_K{K}[q]              per-condition budget normalization
    G_K{K}[q, f, omega]        class-constrained filter powers

The diagnostics here are intentionally model-facing.  They quantify what the
figures are meant to show: nonstationarity across movement conditions,
nonseparability within spectra, information regret relative to the oracle, and
how the response-power budget is allocated across learned classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import json
import numpy as np
import matplotlib.pyplot as plt

from src.plotting import log_contourf, setup_style

Array = np.ndarray


# ---------------------------------------------------------------------------
# Numeric diagnostics
# ---------------------------------------------------------------------------


def positive_omega_view(Z: Array, omega: Array) -> Tuple[Array, Array]:
    """Return Z restricted to omega > 0.

    Z can have shape (..., Nf, Nomega).  The returned array has the same leading
    axes and only the positive temporal-frequency columns.
    """
    omega = np.asarray(omega, dtype=float).ravel()
    m = omega > 0
    return np.asarray(Z)[..., m], omega[m]


def normalize_for_plot(Z: Array, floor: float = 1e-12) -> Array:
    """Per-panel normalization for log contour plots."""
    Z = np.asarray(Z, dtype=float)
    zmax = np.nanmax(np.where(Z > 0, Z, np.nan))
    if not np.isfinite(zmax) or zmax <= 0:
        return np.zeros_like(Z)
    return np.maximum(Z / zmax, floor)


def _weighted_mean(x: Array, w: Array, axis=None, keepdims: bool = False) -> Array:
    wsum = np.sum(w, axis=axis, keepdims=keepdims)
    return np.sum(x * w, axis=axis, keepdims=keepdims) / np.maximum(wsum, 1e-300)


def log_additive_separability_r2(
    Z: Array,
    weights: Array,
    *,
    floor_rel: float = 1e-8,
) -> float:
    """Weighted R^2 for the best log-additive separable approximation.

    A positive array is multiplicatively separable when

        Z(f, omega) = A(f) B(omega).

    Taking logs turns this into an additive model,

        log Z(f, omega) = a(f) + b(omega).

    This function computes the weighted two-way additive fit and returns the
    fraction of weighted log-variance explained.  Values near one mean that the
    spectrum is close to separable.  Values far below one mean strong space-time
    coupling.
    """
    Z = np.asarray(Z, dtype=float)
    W = np.asarray(weights, dtype=float)
    if Z.shape != W.shape:
        raise ValueError(f"Z shape {Z.shape} must match weights shape {W.shape}")

    support = (W > 0) & np.isfinite(Z) & (Z > 0)
    if support.sum() < 4:
        return np.nan

    W0 = np.where(support, W, 0.0)
    L = np.zeros_like(Z, dtype=float)
    L[support] = np.log(Z[support])

    grand = _weighted_mean(L, W0)
    row = _weighted_mean(L, W0, axis=1, keepdims=True)
    col = _weighted_mean(L, W0, axis=0, keepdims=True)
    pred = row + col - grand

    sst = np.sum(W0 * (L - grand) ** 2)
    sse = np.sum(W0 * (L - pred) ** 2)
    if sst <= 0:
        return np.nan
    return float(1.0 - sse / sst)


def condition_distance_matrix(G_stack: Array, weights: Array, *, log_space: bool = True) -> Array:
    """Pairwise weighted distances between condition-specific filters.

    Used to visualize nonstationarity: if movement conditions required the same
    filter shape, this matrix would be nearly zero after normalization.
    """
    G = np.asarray(G_stack, dtype=float)
    Q = G.shape[0]
    W = np.asarray(weights, dtype=float).reshape(-1)
    X = G.reshape(Q, -1)

    Xn = np.zeros_like(X)
    for q in range(Q):
        x = X[q]
        if log_space:
            xmax = np.nanmax(np.where(x > 0, x, np.nan))
            floor = max(1e-8 * xmax, 1e-300)
            x = np.log(np.maximum(x / max(xmax, 1e-300), floor))
        else:
            denom = np.sum(x * W) + 1e-300
            x = x / denom
        # Center each condition so the distance reflects shape, not scale.
        x = x - np.sum(x * W) / np.maximum(np.sum(W), 1e-300)
        Xn[q] = x

    D = np.zeros((Q, Q), dtype=float)
    for i in range(Q):
        for j in range(Q):
            diff = Xn[i] - Xn[j]
            D[i, j] = np.sqrt(np.sum(W * diff * diff) / np.maximum(np.sum(W), 1e-300))
    return D


def class_budget_share(
    C_stack: Array,
    H: Array,
    alpha: Array,
    scale: Array,
    weights: Array,
    sigma_in: float,
) -> Array:
    """Fraction of the response-power budget carried by each class.

    The mixture weight alpha_{qc} tells us how the class spectra are combined
    before budget normalization.  The response-power budget share accounts for
    the condition spectrum C_q and is therefore closer to the biological reading:
    what fraction of the available response power is spent through each class?
    """
    C = np.asarray(C_stack, dtype=float)
    H = np.asarray(H, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    scale = np.asarray(scale, dtype=float).ravel()
    W = np.asarray(weights, dtype=float)

    Q = C.shape[0]
    K = H.shape[0]
    C_flat = C.reshape(Q, -1)
    H_flat = H.reshape(K, -1)
    W_flat = W.reshape(-1)
    s_in2 = float(sigma_in) ** 2

    P = np.zeros((Q, K), dtype=float)
    for q in range(Q):
        for c in range(K):
            P[q, c] = (
                scale[q]
                * alpha[q, c]
                * np.sum(H_flat[c] * (C_flat[q] + s_in2) * W_flat)
            )
    return P / np.maximum(P.sum(axis=1, keepdims=True), 1e-300)


def per_condition_regret(I_star_q: Array, I_q: Array) -> Array:
    """Condition-wise information regret relative to the oracle."""
    I_star_q = np.asarray(I_star_q, dtype=float)
    I_q = np.asarray(I_q, dtype=float)
    return (I_star_q - I_q) / np.maximum(np.abs(I_star_q), 1e-300)


def class_summaries(
    H: Array,
    f: Array,
    omega: Array,
    weights: Array,
    *,
    f_cut: float = 1.0,
    omega_cut: float = 50.0,
) -> List[Dict[str, float]]:
    """Centroids and mass fractions for learned class spectra."""
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    W = np.asarray(weights, dtype=float)
    F = f[:, None]
    Om = np.abs(omega)[None, :]

    rows: List[Dict[str, float]] = []
    for c in range(H.shape[0]):
        mass = np.sum(H[c] * W) + 1e-300
        rows.append(
            {
                "class": int(c),
                "f_centroid": float(np.sum(H[c] * F * W) / mass),
                "omega_centroid": float(np.sum(H[c] * Om * W) / mass),
                "high_f_fraction": float(np.sum(H[c] * W * (F >= f_cut)) / mass),
                "high_omega_fraction": float(np.sum(H[c] * W * (Om >= omega_cut)) / mass),
                "separability_log_additive_R2": float(log_additive_separability_r2(H[c], W)),
            }
        )
    return rows


def condition_summaries(
    C_stack: Array,
    G_star: Array,
    weights: Array,
    labels: Sequence[str],
) -> List[Dict[str, float | str]]:
    """Summaries of nonseparability for each condition and its oracle filter."""
    rows: List[Dict[str, float | str]] = []
    for q, label in enumerate(labels):
        rows.append(
            {
                "index": int(q),
                "name": str(label),
                "C_log_additive_R2": float(log_additive_separability_r2(C_stack[q], weights)),
                "Gstar_log_additive_R2": float(log_additive_separability_r2(G_star[q], weights)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def _condition_labels(condition_rows: List[Dict]) -> List[str]:
    return [str(row.get("name", f"q={i}")) for i, row in enumerate(condition_rows)]


def load_condition_rows(path: Path) -> List[Dict]:
    with open(path, "r") as fobj:
        return json.load(fobj)


def plot_oracle_condition_panel(
    f: Array,
    omega: Array,
    C_stack: Array,
    G_star: Array,
    weights: Array,
    condition_rows: List[Dict],
    outpath: Path,
    *,
    condition_indices: Optional[Sequence[int]] = None,
    vmin_floor: float = 1e-5,
):
    """Panel: movement spectra and their oracle optimal filters.

    Top row: normalized input spectra C_q(f, omega).
    Bottom row: normalized unconstrained optima G_q^*(f, omega)=|v_q^*|^2.

    The title of each column includes the log-additive separability R^2 for the
    spectrum and filter, so the figure makes the nonseparability point without
    requiring a separate diagnostic figure.
    """
    setup_style()
    labels = _condition_labels(condition_rows)
    Q = C_stack.shape[0]
    if condition_indices is None:
        # Default: representative early, low/mid/high late if present.
        condition_indices = list(range(min(Q, 6)))
    condition_indices = list(condition_indices)
    n = len(condition_indices)

    omega_pos = omega > 0
    fig, axes = plt.subplots(2, n, figsize=(2.05 * n, 3.7), constrained_layout=True)
    if n == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, q in enumerate(condition_indices):
        Cq = normalize_for_plot(C_stack[q][:, omega_pos])
        Gq = normalize_for_plot(G_star[q][:, omega_pos])
        r2_C = log_additive_separability_r2(C_stack[q], weights)
        r2_G = log_additive_separability_r2(G_star[q], weights)

        cf0 = log_contourf(
            axes[0, col],
            f,
            omega[omega_pos],
            Cq.T,
            n_levels=18,
            cmap="magma",
            vmin_floor=vmin_floor,
        )
        cf1 = log_contourf(
            axes[1, col],
            f,
            omega[omega_pos],
            Gq.T,
            n_levels=18,
            cmap="viridis",
            vmin_floor=vmin_floor,
        )
        axes[0, col].set_title(f"{labels[q]}\n$R^2_{{sep}}(C)$={r2_C:.2f}")
        axes[1, col].set_title(f"$R^2_{{sep}}(G^*)$={r2_G:.2f}")
        axes[1, col].set_xlabel("spatial freq. f")
        if col == 0:
            axes[0, col].set_ylabel(r"$C_q$: $\omega$ (rad/s)")
            axes[1, col].set_ylabel(r"$G_q^*$: $\omega$ (rad/s)")
        else:
            axes[0, col].set_yticklabels([])
            axes[1, col].set_yticklabels([])

    fig.colorbar(cf0, ax=axes[0, :].ravel().tolist(), fraction=0.025, pad=0.015, label="norm. power")
    fig.colorbar(cf1, ax=axes[1, :].ravel().tolist(), fraction=0.025, pad=0.015, label="norm. filter")
    fig.suptitle("Movement-conditioned spectra and unconstrained optimal filters", y=1.02)
    fig.savefig(outpath)
    plt.close(fig)


def plot_regret_and_condition_losses(
    K_values: Array,
    regret_values: Array,
    I_star_q: Array,
    I_by_K: Dict[int, Array],
    condition_rows: List[Dict],
    outpath: Path,
):
    """Panel: average regret and per-condition regret."""
    setup_style()
    labels = _condition_labels(condition_rows)
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)

    axes[0].plot(K_values, regret_values, marker="o")
    axes[0].set_xlabel("number of classes K")
    axes[0].set_ylabel("weighted information regret")
    axes[0].set_xticks(K_values)
    axes[0].grid(True, alpha=0.25)
    axes[0].set_title("Class-count selection")

    for K in sorted(I_by_K):
        r_q = per_condition_regret(I_star_q, I_by_K[K])
        axes[1].plot(x, r_q, marker="o", label=f"K={K}")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_ylabel("per-condition regret")
    axes[1].set_title("Where the approximation loses information")
    axes[1].legend(ncol=2)
    axes[1].grid(True, alpha=0.25)

    fig.savefig(outpath)
    plt.close(fig)


def plot_learned_classes_and_gains(
    f: Array,
    omega: Array,
    H: Array,
    alpha: Array,
    budget_share: Array,
    condition_rows: List[Dict],
    outpath: Path,
    *,
    vmin_floor: float = 1e-5,
):
    """Panel: learned class spectra, mixture weights, and budget shares."""
    setup_style()
    labels = _condition_labels(condition_rows)
    K = H.shape[0]
    x = np.arange(len(labels))
    omega_pos = omega > 0

    fig = plt.figure(figsize=(max(6.0, 2.5 * K), 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, K, height_ratios=[1.25, 1.0])

    for c in range(K):
        ax = fig.add_subplot(gs[0, c])
        Hc = normalize_for_plot(H[c][:, omega_pos])
        cf = log_contourf(
            ax,
            f,
            omega[omega_pos],
            Hc.T,
            n_levels=18,
            cmap="viridis",
            vmin_floor=vmin_floor,
        )
        ax.set_title(f"class {c}: fixed $H_c(f,\\omega)$")
        ax.set_xlabel("f")
        if c == 0:
            ax.set_ylabel(r"$\omega$")
        else:
            ax.set_yticklabels([])
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.02)

    ax_alpha = fig.add_subplot(gs[1, :])
    for c in range(K):
        ax_alpha.plot(x, alpha[:, c], marker="o", linestyle="--", label=fr"$\alpha_{{q{c}}}$")
        ax_alpha.plot(x, budget_share[:, c], marker="s", label=fr"budget class {c}")
    ax_alpha.set_xticks(x)
    ax_alpha.set_xticklabels(labels, rotation=45, ha="right")
    ax_alpha.set_ylim(-0.03, 1.03)
    ax_alpha.set_ylabel("fraction")
    ax_alpha.set_title("Condition-dependent class use: mixture weights and response-budget shares")
    ax_alpha.legend(ncol=min(4, 2 * K))
    ax_alpha.grid(True, alpha=0.25)

    fig.savefig(outpath)
    plt.close(fig)

def plot_nonstationarity_diagnostics(
    C_stack: Array,
    G_star: Array,
    weights: Array,
    condition_rows: List[Dict],
    outpath: Path,
):
    """Diagnostics: separability scores and pairwise oracle-filter distances."""
    setup_style()
    labels = _condition_labels(condition_rows)
    x = np.arange(len(labels))
    C_r2 = [log_additive_separability_r2(C_stack[q], weights) for q in range(C_stack.shape[0])]
    G_r2 = [log_additive_separability_r2(G_star[q], weights) for q in range(G_star.shape[0])]
    D = condition_distance_matrix(G_star, weights)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), constrained_layout=True)
    axes[0].plot(x, C_r2, marker="o", label="input spectrum")
    axes[0].plot(x, G_r2, marker="s", label="oracle filter")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].set_ylabel(r"log-additive separability $R^2$")
    axes[0].set_title("Space-time separability diagnostic")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    im = axes[1].imshow(D, origin="lower", aspect="auto")
    axes[1].set_xticks(x)
    axes[1].set_yticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    axes[1].set_yticklabels(labels)
    axes[1].set_title("Shape distance between oracle filters")
    cb = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cb.set_label("weighted log-shape distance")

    fig.savefig(outpath)
    plt.close(fig)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fobj:
        json.dump(obj, fobj, indent=2)

# ---------------------------------------------------------------------------
# Kernel summaries for supplements
# ---------------------------------------------------------------------------


def temporal_kernel_from_filter_power(
    G: Array,
    f: Array,
    omega: Array,
    *,
    omega_min: float = 0.5,
    omega_max: float = 400.0,
    taper_alpha: float = 0.25,
    floor_rel: float = 1e-3,
) -> Tuple[float, Array, Array]:
    """Minimum-phase temporal kernel at the spatial frequency of peak energy.

    This mirrors ``src.pipeline.extract_temporal_kernel`` but works directly on
    saved arrays.  It is intended for summaries of how the oracle and
    class-constrained effective filters change across movement conditions.
    """
    from src.kernels import minimum_phase_temporal_filter, soft_band_taper

    G = np.asarray(G, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    domega = np.gradient(omega)
    energy_per_f = np.sum(G * np.abs(domega)[None, :], axis=1)
    i_peak = int(np.argmax(energy_per_f))
    f_peak = float(f[i_peak])

    mag = np.sqrt(np.maximum(G[i_peak], 0.0))
    taper = soft_band_taper(omega, omega_min, omega_max, alpha=taper_alpha)
    mag = mag * taper
    floor = floor_rel * max(float(np.nanmax(mag)), 1e-30)
    mag = np.maximum(mag, floor)
    t, h_t, _ = minimum_phase_temporal_filter(mag, omega)
    return f_peak, t, h_t


def plot_temporal_kernel_supplement(
    f: Array,
    omega: Array,
    G_star: Array,
    G_fit: Array,
    condition_rows: List[Dict],
    outpath: Path,
    *,
    condition_indices: Optional[Sequence[int]] = None,
    t_max: float = 0.35,
):
    """Supplement: oracle versus class-constrained temporal kernels.

    For each selected movement condition, the oracle can change its full
    temporal selectivity.  The class-constrained effective filter can still
    change by switching the mixture weights between fixed classes.  This plot
    makes the adaptation program visible in the time domain.
    """
    setup_style()
    labels = _condition_labels(condition_rows)
    Q = G_star.shape[0]
    if condition_indices is None:
        condition_indices = list(range(Q))
    condition_indices = list(condition_indices)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    for ax, G_all, title in zip(axes, [G_star, G_fit], ["oracle $G_q^*$", "class-constrained $G_q^{(K)}$"]):
        for q in condition_indices:
            f_peak, t, h = temporal_kernel_from_filter_power(G_all[q], f, omega)
            m = (t >= 0) & (t <= t_max)
            h_plot = h[m]
            denom = np.max(np.abs(h_plot)) if h_plot.size else 1.0
            denom = max(float(denom), 1e-12)
            ax.plot(t[m], h_plot / denom, label=f"{labels[q]} (f={f_peak:.2g})")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("time after input (s)")
        ax.set_ylabel("normalized min-phase kernel")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[1].legend(fontsize=6, ncol=1, loc="upper right")
    fig.savefig(outpath)
    plt.close(fig)
