"""Tests for constrained class-gain parameterizations and fixed-H refits."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.cell_class_learning import response_power_budget
from src.cell_class_learning_fast import make_alpha, refit_alpha_for_fixed_H_fast


def test_alpha_floor_bounds_and_normalization():
    Z = torch.tensor([[8.0, -2.0, 0.5], [-5.0, 2.0, 7.0]])
    alpha_floor = 0.10
    alpha = make_alpha(Z, mode="floor", alpha_floor=alpha_floor)
    a = alpha.detach().numpy()

    assert np.all(a >= alpha_floor - 1e-7)
    assert np.all(a <= 1.0 - (Z.shape[1] - 1) * alpha_floor + 1e-7)
    np.testing.assert_allclose(a.sum(axis=1), 1.0, atol=1e-7)


def test_bounded_log_gain_limits_balanced_two_class_alpha():
    Z = torch.tensor([[100.0, -100.0], [-100.0, 100.0], [0.0, 0.0]])
    delta = 0.5
    alpha = make_alpha(Z, mode="bounded_log_gain", gain_delta_max=delta)
    a = alpha.detach().numpy()
    alpha_max = 1.0 / (1.0 + np.exp(-2.0 * delta))
    alpha_min = 1.0 - alpha_max

    assert np.all(a >= alpha_min - 1e-6)
    assert np.all(a <= alpha_max + 1e-6)
    np.testing.assert_allclose(a.sum(axis=1), 1.0, atol=1e-7)


def test_fixed_H_alpha_refit_preserves_H_and_power_budget():
    rng = np.random.default_rng(4)
    C_stack = rng.uniform(0.05, 2.0, size=(2, 4, 5))
    weights = rng.uniform(0.01, 0.2, size=(4, 5))
    H_fixed = rng.uniform(0.01, 1.0, size=(2, 4, 5))
    P0 = 3.0

    fit = refit_alpha_for_fixed_H_fast(
        C_stack,
        weights,
        H_fixed,
        sigma_in=0.3,
        sigma_out=1.0,
        P0=P0,
        n_steps=3,
        check_every=1,
        patience=0,
        device="cpu",
    )

    np.testing.assert_allclose(fit.H, H_fixed)
    np.testing.assert_allclose(fit.budget_share.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(fit.budget_share >= 0.0)
    budgets = response_power_budget(C_stack, fit.G, weights, sigma_in=0.3)
    np.testing.assert_allclose(budgets, P0, rtol=1e-5, atol=1e-5)
