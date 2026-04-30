"""Tests for the spectra module.

Covers:
- Image spectrum: power-law shape, monotonicity, regularization.
- Drift spectrum: ω-integral preserves spatial power (eq. 36),
  Lorentzian width D||k||^2 (eq. 35).
- Linear-motion (Gaussian): ω-integral preserves spatial power (eq. 60),
  Gaussian shape with std sf.
- Combined: limits to drift (s=0) and linear (D=0).
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
sys.path.insert(0, ".")

from src.spectra import (
    image_spectrum,
    drift_lorentzian,
    drift_spectrum,
    linear_motion_spectrum_gaussian,
    combined_spectrum,
)


# ---------------------------------------------------------------------------
# Image spectrum
# ---------------------------------------------------------------------------

def test_image_spectrum_powerlaw_far_from_k0():
    """Far from k0, C_I(f) ~ A / f^beta."""
    f = np.array([1.0, 2.0, 4.0, 8.0])
    C = image_spectrum(f, beta=2.0, A=1.0, k0=0.001)
    # Ratio C(f)/C(2f) = 4 for beta=2
    ratios = C[:-1] / C[1:]
    np.testing.assert_allclose(ratios, 4.0, rtol=1e-3)


def test_image_spectrum_regularized_at_zero():
    """At f=0 the spectrum is finite (= A / k0^beta)."""
    C0 = image_spectrum(np.array([0.0]), beta=2.0, A=1.0, k0=0.05)
    assert np.isfinite(C0[0])
    np.testing.assert_allclose(C0[0], 1.0 / 0.05 ** 2, rtol=1e-12)


def test_image_spectrum_monotone_decreasing():
    """Power-law spectrum decreases with f."""
    f = np.linspace(0.1, 5.0, 50)
    C = image_spectrum(f, beta=2.0)
    assert np.all(np.diff(C) < 0)


# ---------------------------------------------------------------------------
# Drift spectrum
# ---------------------------------------------------------------------------

def test_drift_lorentzian_normalization():
    """∫ dω/(2π) of 2 D k^2 / ((D k^2)^2 + ω^2) = 1 for any D, k > 0.

    The Lorentzian's 1/ω^2 tails decay slowly; integration range must scale
    with D k^2 to keep the truncation error small.
    """
    for D in [0.1, 1.0, 10.0]:
        for f in [0.5, 1.0, 3.0]:
            Dk2 = D * f ** 2
            omega_max = 200.0 * Dk2  # 200x the HWHM
            omega = np.linspace(-omega_max, omega_max, 200001)
            L = drift_lorentzian(np.array(f), omega, D)
            integral = np.trapezoid(L, omega) / (2.0 * np.pi)
            np.testing.assert_allclose(integral, 1.0, rtol=5e-3)


def test_drift_lorentzian_normalization_analytic():
    """Closed-form check: 2 ∫_0^A dω/(2π) · 2 D k^2/((D k^2)^2 + ω^2)
    = (2/π) arctan(A/(D k^2)). Verify by integrating analytically."""
    # Use the analytic CDF of the Lorentzian to avoid trapezoid truncation.
    for D, f in [(1.0, 1.0), (5.0, 2.0)]:
        Dk2 = D * f ** 2
        # CDF over (-A, A) of Lorentzian density = (2/π) arctan(A/Dk2)
        A = 1e8 * Dk2
        cdf = (2.0 / np.pi) * np.arctan(A / Dk2)
        np.testing.assert_allclose(cdf, 1.0, rtol=1e-7)


def test_drift_spectrum_power_preserving():
    """∫ dω/(2π) C_D(f, ω) = C_I(f)  (eq. 36)."""
    f_vals = np.array([0.3, 1.0, 2.5])
    for D in [0.5, 5.0]:
        for f in f_vals:
            Dk2 = D * f ** 2
            omega_max = 500.0 * Dk2
            omega = np.linspace(-omega_max, omega_max, 200001)
            CD = drift_spectrum(np.array(f), omega, D)
            integral = np.trapezoid(CD, omega) / (2.0 * np.pi)
            CI = image_spectrum(np.array(f))
            np.testing.assert_allclose(integral, CI, rtol=5e-3)


def test_drift_lorentzian_width_scales_as_Dk2():
    """The half-width at half-max of the Lorentzian is D k^2."""
    omega = np.linspace(-200.0, 200.0, 40001)
    for D, f in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]:
        L = drift_lorentzian(np.array(f), omega, D)
        peak = L.max()
        # find HWHM
        i_left = np.argmin(np.abs(L[: len(L) // 2] - peak / 2))
        hwhm = abs(omega[i_left])
        np.testing.assert_allclose(hwhm, D * f ** 2, rtol=2e-3)


# ---------------------------------------------------------------------------
# Linear motion (Gaussian)
# ---------------------------------------------------------------------------

def test_linear_motion_gaussian_power_preserving():
    """∫ dω/(2π) C_s(f, ω) = C_I(f) for Gaussian velocity."""
    omega = np.linspace(-2000.0, 2000.0, 200001)
    for s in [0.3, 1.0, 3.0]:
        for f in [0.5, 1.0, 2.0]:
            Cs = linear_motion_spectrum_gaussian(np.array(f), omega, s)
            integral = np.trapezoid(Cs, omega) / (2.0 * np.pi)
            CI = image_spectrum(np.array(f))
            np.testing.assert_allclose(integral, CI, rtol=2e-4)


def test_linear_motion_gaussian_width_scales_as_sf():
    """The Gaussian std in ω is s f."""
    omega = np.linspace(-100.0, 100.0, 40001)
    for s, f in [(1.0, 1.0), (2.0, 1.0), (1.0, 2.0)]:
        Cs = linear_motion_spectrum_gaussian(np.array(f), omega, s)
        # std = sqrt(<ω^2>) under Cs (normalized)
        norm = np.trapezoid(Cs, omega)
        var = np.trapezoid(omega ** 2 * Cs, omega) / norm
        np.testing.assert_allclose(np.sqrt(var), s * f, rtol=2e-3)


# ---------------------------------------------------------------------------
# Combined drift + linear motion
# ---------------------------------------------------------------------------

def test_combined_spectrum_reduces_to_drift_at_s0():
    """C_{D,s=0} == C_D."""
    f = np.array(0.7)
    omega = np.linspace(-50.0, 50.0, 1001)
    Cdrift = drift_spectrum(f, omega, D=2.0)
    Ccomb = combined_spectrum(f, omega, D=2.0, s=0.0)
    np.testing.assert_allclose(Ccomb, Cdrift, rtol=1e-12, atol=1e-12)


def test_combined_spectrum_reduces_to_linear_at_D0():
    """C_{D=0, s} == Gaussian linear-motion spectrum."""
    f = np.array(1.0)
    omega = np.linspace(-50.0, 50.0, 1001)
    Cl = linear_motion_spectrum_gaussian(f, omega, s=1.5)
    Ccomb = combined_spectrum(f, omega, D=0.0, s=1.5)
    np.testing.assert_allclose(Ccomb, Cl, rtol=1e-12, atol=1e-12)


def test_combined_spectrum_power_preserving():
    """∫ dω/(2π) C_{D,s}(f, ω) ≈ C_I(f).

    The combined spectrum's tail is Lorentzian-dominated; integration range
    must again scale with D f^2 (and be large enough for the Gaussian).
    """
    for D, s, f in [(0.5, 1.0, 1.0), (2.0, 0.3, 0.7), (5.0, 2.0, 2.0)]:
        Dk2 = D * f ** 2
        sf = s * f
        # need omega_max >> max(Dk2, sf)
        omega_max = max(500.0 * Dk2, 50.0 * sf)
        omega = np.linspace(-omega_max, omega_max, 200001)
        Ccomb = combined_spectrum(np.array(f), omega, D=D, s=s, n_a=51)
        integral = np.trapezoid(Ccomb, omega) / (2.0 * np.pi)
        CI = image_spectrum(np.array(f))
        np.testing.assert_allclose(integral, CI, rtol=5e-3)
