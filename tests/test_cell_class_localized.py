"""Tests for localized bounded-share cell-class learning."""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

import numpy as np
import pytest

pytest.importorskip("torch")

from src.cell_class_learning import response_power_budget, solve_oracle_stack
from src.cell_class_learning_fast import fit_cell_classes_fast
from src.cell_class_localized import (
    build_strategy_conditions,
    fit_cell_classes_localized,
    refit_modulation_for_fixed_classes,
    spatial_rf_width_from_H,
)


def _toy_stack():
    f = np.geomspace(0.15, 6.0, 8)
    t = np.arange(7)
    weights = np.ones((f.size, t.size), dtype=float)
    weights /= weights.sum()
    low = np.exp(-0.5 * ((np.log(f) - np.log(0.35)) / 0.35) ** 2)[:, None]
    high = np.exp(-0.5 * ((np.log(f) - np.log(3.0)) / 0.35) ** 2)[:, None]
    temporal = 0.4 + np.exp(-0.5 * ((t - 3.0) / 1.6) ** 2)[None, :]
    C_stack = np.stack(
        [
            0.05 + 3.0 * low * temporal + 0.2 * high,
            0.05 + 0.2 * low + 3.0 * high * temporal,
        ],
        axis=0,
    )
    return C_stack, weights, f


def test_localized_fit_preserves_total_and_per_class_budget():
    C_stack, weights, f = _toy_stack()
    P0 = 4.0
    sigma_in = 0.25
    fit = fit_cell_classes_localized(
        C_stack,
        weights,
        f,
        sigma_in=sigma_in,
        sigma_out=1.0,
        P0=P0,
        K=2,
        loc_weight=0.2,
        delta_max=0.4,
        n_steps=8,
        n_restarts=1,
        check_every=2,
        patience=0,
        device="cpu",
        seed=2,
    )

    budgets = response_power_budget(C_stack, fit.G, weights, sigma_in=sigma_in)
    np.testing.assert_allclose(budgets, P0, rtol=1e-5, atol=1e-5)
    class_spend = np.sum(
        fit.G_class * (C_stack[:, None, :, :] + sigma_in**2) * weights[None, None, :, :],
        axis=(2, 3),
    )
    np.testing.assert_allclose(class_spend, fit.rho_qc * P0, rtol=1e-5, atol=1e-5)


def test_relaxed_single_class_matches_fast_optimizer():
    C_stack, weights, f = _toy_stack()
    kwargs = dict(
        sigma_in=0.3,
        sigma_out=1.0,
        P0=3.0,
        K=1,
        n_steps=15,
        n_restarts=1,
        check_every=5,
        patience=0,
        device="cpu",
        seed=7,
    )
    localized = fit_cell_classes_localized(
        C_stack,
        weights,
        f,
        loc_weight=0.0,
        delta_max=10.0,
        learn_baseline_share=True,
        **kwargs,
    )
    fast = fit_cell_classes_fast(C_stack, weights, alpha_mode="softmax", **kwargs)
    np.testing.assert_allclose(localized.J, fast.J, rtol=1e-2)


def test_bounded_delta_prevents_class_starvation():
    C_stack, weights, f = _toy_stack()
    fit = fit_cell_classes_localized(
        C_stack,
        weights,
        f,
        sigma_in=0.3,
        sigma_out=1.0,
        P0=3.0,
        K=2,
        loc_weight=0.5,
        delta_max=0.3,
        n_steps=10,
        n_restarts=1,
        check_every=2,
        patience=0,
        device="cpu",
        seed=4,
    )
    assert np.min(fit.rho_qc) >= 0.15


def test_localization_and_rf_width_diagnostics_separate_hand_built_classes():
    f = np.geomspace(0.1, 8.0, 32)
    omega = np.linspace(-50.0, 50.0, 17)
    weights = np.ones((f.size, omega.size), dtype=float)
    low = np.exp(-0.5 * ((np.log(f) - np.log(0.35)) / 0.18) ** 2)
    high = np.exp(-0.5 * ((np.log(f) - np.log(4.0)) / 0.18) ** 2)
    temporal = np.exp(-0.5 * (omega / 20.0) ** 2)
    H = np.stack([low[:, None] * temporal[None, :], high[:, None] * temporal[None, :]], axis=0)
    H = H / np.sum(H * weights[None, :, :], axis=(1, 2), keepdims=True)

    widths = spatial_rf_width_from_H(H, f, omega, weights, n_k=128)
    assert widths[0] > 1.5 * widths[1]


def test_fixed_class_modulation_refit_preserves_budgets():
    C_stack, weights, f = _toy_stack()
    base = fit_cell_classes_localized(
        C_stack,
        weights,
        f,
        sigma_in=0.3,
        sigma_out=1.0,
        P0=3.0,
        K=2,
        n_steps=6,
        n_restarts=1,
        check_every=2,
        patience=0,
        device="cpu",
        seed=5,
    )
    fit = refit_modulation_for_fixed_classes(
        C_stack,
        weights,
        f,
        base.H,
        base.rho0,
        sigma_in=0.3,
        sigma_out=1.0,
        P0=3.0,
        n_steps=6,
        check_every=2,
        patience=0,
        device="cpu",
    )
    budgets = response_power_budget(C_stack, fit.G, weights, sigma_in=0.3)
    np.testing.assert_allclose(budgets, 3.0, rtol=1e-5, atol=1e-5)


def test_strategy_sweep_builder_matches_direct_oracle_reproducibly():
    conditions, pi = build_strategy_conditions(D=0.15, A=2.0, grid="fast")
    a = solve_oracle_stack(conditions, sigma_in=0.3, sigma_out=1.0, P0=50.0, grid="fast", condition_weights=pi)
    b = solve_oracle_stack(conditions, sigma_in=0.3, sigma_out=1.0, P0=50.0, grid="fast", condition_weights=pi)
    np.testing.assert_allclose(a.J_star, b.J_star, rtol=0.0, atol=1e-6)

