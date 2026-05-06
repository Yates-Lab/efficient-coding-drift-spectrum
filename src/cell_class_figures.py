"""Small diagnostics shared by cell-class scripts."""

from __future__ import annotations

import numpy as np

Array = np.ndarray


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

    This computes the weighted two-way additive fit and returns the fraction of
    weighted log-variance explained. Values near one mean that the spectrum is
    close to separable; lower values mean stronger space-time coupling.
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
