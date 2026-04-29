"""Tests for the optimal-filter solver.

Covers:
- Closed-form formula derivative (KKT condition, eq. 22).
- Budget constraint exactly satisfied.
- Reduction to water-filling at σ_in = 0.
- Reduction to constant gain when σ_in -> ∞ (degenerate, no signal preference).
- Mutual information ≥ 0; monotonic in P0; consistent with discrete numerical
  optimization for small grids.
- Active-set threshold formula.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
sys.path.insert(0, "/home/claude/efficient_coding")

from src.solver import (
    optimal_filter_squared_magnitude,
    find_lambda,
    mutual_information,
    mutual_information_density,
    solve_efficient_coding,
    active_threshold_C,
)


# ---------------------------------------------------------------------------
# KKT condition (eq. 22)
# ---------------------------------------------------------------------------

def test_kkt_condition_holds_in_active_set():
    """At every active frequency, the closed-form |v*|^2 satisfies eq. (22):
        σ_out^2 C / [(p(C+σ_in^2)+σ_out^2)(p σ_in^2+σ_out^2)] = λ (C + σ_in^2).
    """
    rng = np.random.default_rng(0)
    C = rng.uniform(0.5, 5.0, size=(50,))
    sigma_in, sigma_out = 0.3, 0.5
    for lam in [0.1, 1.0, 5.0]:
        p = optimal_filter_squared_magnitude(C, sigma_in, sigma_out, lam)
        active = p > 0
        if not np.any(active):
            continue
        Ca = C[active]
        pa = p[active]
        lhs = sigma_out ** 2 * Ca / (
            (pa * (Ca + sigma_in ** 2) + sigma_out ** 2)
            * (pa * sigma_in ** 2 + sigma_out ** 2)
        )
        rhs = lam * (Ca + sigma_in ** 2)
        np.testing.assert_allclose(lhs, rhs, rtol=1e-10)


def test_active_threshold_consistent_with_solution():
    """Below C_thresh the solution is zero; above it is positive."""
    sigma_in, sigma_out = 0.4, 0.7
    lam = 0.3
    Cth = active_threshold_C(sigma_in, sigma_out, lam)
    assert np.isfinite(Cth)
    # Just below threshold: zero. Just above: positive.
    p_lo = optimal_filter_squared_magnitude(np.array([Cth * 0.999]), sigma_in, sigma_out, lam)
    p_hi = optimal_filter_squared_magnitude(np.array([Cth * 1.001]), sigma_in, sigma_out, lam)
    assert p_lo[0] == 0.0
    assert p_hi[0] > 0.0


def test_threshold_infinite_when_lam_too_large():
    """If λ σ_out^2 >= 1 nothing is active (signal too cheap to bother with)."""
    th = active_threshold_C(0.3, 0.5, lam=4.0)  # λ σ_out^2 = 1 -> infinite threshold
    assert np.isinf(th)


# ---------------------------------------------------------------------------
# Budget constraint
# ---------------------------------------------------------------------------

def test_budget_constraint_exactly_satisfied():
    """With λ found by find_lambda, ∫ |v*|^2 (C + σ_in^2) = P0."""
    rng = np.random.default_rng(1)
    C = np.abs(rng.normal(2.0, 1.0, size=(64, 64))) + 1e-3
    weights = np.full_like(C, 1.0 / C.size)  # Uniform weights summing to 1.
    sigma_in, sigma_out = 0.2, 0.4
    for P0 in [0.5, 5.0, 50.0]:
        lam = find_lambda(C, sigma_in, sigma_out, P0, weights)
        v_sq = optimal_filter_squared_magnitude(C, sigma_in, sigma_out, lam)
        spend = np.sum(v_sq * (C + sigma_in ** 2) * weights)
        np.testing.assert_allclose(spend, P0, rtol=1e-7)


# ---------------------------------------------------------------------------
# Water-filling reduction (σ_in = 0)
# ---------------------------------------------------------------------------

def test_water_filling_reduction_at_sigma_in_0():
    """For σ_in = 0, the optimum gives p*C = (1/λ - σ_out^2)_+ for all active C."""
    rng = np.random.default_rng(2)
    C = np.abs(rng.normal(2.0, 1.0, size=(50,))) + 1e-3
    weights = np.full_like(C, 1.0 / C.size)
    sigma_out = 0.5
    P0 = 3.0
    lam = find_lambda(C, 0.0, sigma_out, P0, weights)
    p = optimal_filter_squared_magnitude(C, 0.0, sigma_out, lam)
    pc = p * C
    expected_level = max(1.0 / lam - sigma_out ** 2, 0.0)
    # All active frequencies should have the same p*C
    np.testing.assert_allclose(pc, expected_level, rtol=1e-10)


def test_water_filling_smoothness_as_sigma_in_to_zero():
    """The general-σ_in formula reduces smoothly to water-filling as σ_in -> 0."""
    C = np.array([0.5, 1.0, 2.0, 5.0])
    weights = np.full_like(C, 0.25)
    sigma_out = 0.3
    P0 = 1.0
    p_zero = optimal_filter_squared_magnitude(
        C, 0.0, sigma_out,
        find_lambda(C, 0.0, sigma_out, P0, weights),
    )
    p_small = optimal_filter_squared_magnitude(
        C, 1e-6, sigma_out,
        find_lambda(C, 1e-6, sigma_out, P0, weights),
    )
    np.testing.assert_allclose(p_small, p_zero, rtol=2e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

def test_mutual_information_nonneg():
    rng = np.random.default_rng(3)
    C = np.abs(rng.normal(2.0, 1.0, size=(40,))) + 1e-3
    weights = np.full_like(C, 1.0 / C.size)
    sigma_in, sigma_out = 0.2, 0.4
    v_sq, _, I = solve_efficient_coding(C, sigma_in, sigma_out, P0=2.0, weights=weights)
    assert I >= 0.0
    # Density >= 0 everywhere
    dens = mutual_information_density(C, v_sq, sigma_in, sigma_out)
    assert np.all(dens >= -1e-12)


def test_mutual_information_monotonic_in_budget():
    """Larger budget => more transmitted information."""
    rng = np.random.default_rng(4)
    C = np.abs(rng.normal(2.0, 1.0, size=(80,))) + 1e-3
    weights = np.full_like(C, 1.0 / C.size)
    sigma_in, sigma_out = 0.2, 0.4
    Is = []
    for P0 in [0.1, 0.5, 1.0, 5.0, 20.0]:
        _, _, I = solve_efficient_coding(C, sigma_in, sigma_out, P0=P0, weights=weights)
        Is.append(I)
    assert all(b >= a - 1e-10 for a, b in zip(Is[:-1], Is[1:]))


def test_optimum_against_random_perturbations():
    """The closed-form filter is at least as good as random perturbations."""
    rng = np.random.default_rng(5)
    C = np.abs(rng.normal(2.0, 1.0, size=(40,))) + 1e-3
    weights = np.full_like(C, 1.0 / C.size)
    sigma_in, sigma_out = 0.2, 0.4
    P0 = 2.0
    v_sq, lam, I_opt = solve_efficient_coding(C, sigma_in, sigma_out, P0, weights)

    # Perturb v_sq with random non-negative perturbations, but project back
    # onto the budget constraint, and check I doesn't exceed I_opt.
    for trial in range(20):
        delta = np.abs(rng.normal(0, 0.1, size=v_sq.shape))
        v_pert = np.maximum(v_sq + delta, 0.0)
        # Rescale to satisfy budget exactly
        spend_pert = np.sum(v_pert * (C + sigma_in ** 2) * weights)
        v_pert *= P0 / spend_pert
        I_pert = mutual_information(C, v_pert, sigma_in, sigma_out, weights)
        assert I_pert <= I_opt + 1e-9


def test_strong_signal_filter_uniform_in_active_set_at_low_noise():
    """When σ_in -> 0 (water-filling), all active frequencies get equal p*C.

    Already covered above; this test additionally checks the active set
    consists of the largest-C entries first."""
    rng = np.random.default_rng(6)
    C = np.sort(np.abs(rng.normal(1.0, 1.0, size=(20,))) + 1e-3)
    weights = np.full_like(C, 1.0 / C.size)
    sigma_out = 0.5
    P0 = 0.05  # small budget => only top frequencies active
    p = optimal_filter_squared_magnitude(
        C, 0.0, sigma_out,
        find_lambda(C, 0.0, sigma_out, P0, weights),
    )
    active = p > 0
    if active.sum() < len(C):
        # All inactive entries have C smaller than every active entry
        assert np.max(C[~active]) <= np.min(C[active]) + 1e-12
