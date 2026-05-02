"""Tests for the class-based Spectrum API and the new unified
drift-plus-saccade and Boi-cycle spectra.

Covers:
- Class-API equivalence to free functions (drift, Mostofi saccade, linear motion)
- BoiCycleLateSpectrum equivalence to DriftSpectrum
- BoiCycleEarlySpectrum sanity (non-negative, finite, saturated at high f)
- Spectrum.describe() returns a sensible parameter summary
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
sys.path.insert(0, ".")

from src.spectra import (
    ImageParams,
    DEFAULT_IMAGE,
    DriftSpectrum,
    SaccadeSpectrum,
    LinearMotionSpectrum,
    BoiCycleEarlySpectrum,
    BoiCycleLateSpectrum,
    image_spectrum,
    drift_spectrum,
    saccade_spectrum,
    saccade_redistribution,
    linear_motion_spectrum_gaussian,
)


# ---------------------------------------------------------------------------
# Class-API equivalence to free functions
# ---------------------------------------------------------------------------

def test_drift_class_matches_free_function():
    f = np.geomspace(0.05, 5.0, 50)
    omega = np.linspace(-100, 100, 401)
    s = DriftSpectrum(D=2.0)
    np.testing.assert_allclose(
        s.C(f, omega),
        drift_spectrum(f[:, None], omega[None, :], D=2.0),
        rtol=1e-12,
    )


def test_saccade_class_matches_free_function():
    f = np.geomspace(0.05, 5.0, 50)
    omega = np.linspace(-100, 100, 401)
    s = SaccadeSpectrum(A=2.5, lam=3.0)
    np.testing.assert_allclose(
        s.C(f, omega), saccade_spectrum(f, omega, A=2.5, lam=3.0), rtol=1e-12
    )


def test_linear_motion_class_matches_free_function():
    f = np.geomspace(0.05, 5.0, 50)
    omega = np.linspace(-100, 100, 401)
    s = LinearMotionSpectrum(s=1.5)
    np.testing.assert_allclose(
        s.C(f, omega),
        linear_motion_spectrum_gaussian(f[:, None], omega[None, :], s=1.5),
        rtol=1e-12,
    )


def test_class_carries_provenance():
    s = SaccadeSpectrum(A=2.5)
    assert s.name == "saccade"
    assert "Mostofi" in s.reference
    desc = s.describe()
    assert "saccade" in desc
    assert "A=2.5" in desc


def test_image_params_default_matches_free_function():
    f = np.geomspace(0.05, 5.0, 30)
    np.testing.assert_allclose(DEFAULT_IMAGE.C(f), image_spectrum(f), rtol=1e-12)


def test_custom_image_params_threaded_through():
    f = np.geomspace(0.05, 5.0, 30)
    omega = np.linspace(-100, 100, 201)
    custom = ImageParams(beta=2.5, A_image=2.0, k0=0.1)
    s = DriftSpectrum(D=1.0, image=custom)
    expected = drift_spectrum(f[:, None], omega[None, :],
                              D=1.0, beta=2.5, A=2.0, k0=0.1)
    np.testing.assert_allclose(s.C(f, omega), expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# BoiCycleLateSpectrum: equivalence to drift
# ---------------------------------------------------------------------------

def test_boi_late_equals_drift():
    """The late-fixation regime is drift-only by construction."""
    f = np.geomspace(0.05, 5.0, 50)
    omega = np.linspace(-100, 100, 401)
    s_late = BoiCycleLateSpectrum(D=2.0)
    s_drift = DriftSpectrum(D=2.0)
    np.testing.assert_allclose(
        s_late.C(f, omega), s_drift.C(f, omega), rtol=1e-12
    )


# ---------------------------------------------------------------------------
# BoiCycleEarlySpectrum: sanity tests
# ---------------------------------------------------------------------------

def test_boi_early_nonneg_finite():
    """Q_early >= 0 and finite everywhere."""
    f = np.geomspace(0.05, 5.0, 60)
    omega = np.linspace(-200, 200, 1024)
    s_early = BoiCycleEarlySpectrum(A=4.4)
    Q = s_early.redistribution(f, omega)
    assert Q.shape == (60, 1024)
    assert np.all(Q >= 0)
    assert np.all(np.isfinite(Q))


def test_boi_early_C_factors_image():
    """C_early(f, ω) = C_I(f) * Q_early(f, ω)."""
    f = np.geomspace(0.05, 5.0, 30)
    omega = np.linspace(-200, 200, 512)
    s_early = BoiCycleEarlySpectrum(A=4.4)
    C = s_early.C(f, omega)
    Q = s_early.redistribution(f, omega)
    CI = image_spectrum(f)
    np.testing.assert_allclose(C, CI[:, None] * Q, rtol=1e-12)


def test_boi_early_zero_at_zero_amplitude():
    """A = 0 means no saccade transient — Q is identically 0 (j0(0)=1
    constant in time, mean-subtracted to zero)."""
    f = np.geomspace(0.05, 5.0, 20)
    omega = np.linspace(-200, 200, 512)
    s_early = BoiCycleEarlySpectrum(A=0.0)
    Q = s_early.redistribution(f, omega)
    np.testing.assert_allclose(Q, 0.0, atol=1e-20)


def test_boi_early_low_freq_emphasis():
    """Boi et al. report that saccades follow the 1/k^2 image spectrum
    at all relevant temporal frequencies (Fig 2F). After multiplying by
    C_I ~ 1/k^2, the on-retina saccade spectrum should fall off
    significantly with f. Concretely: power at f=2 should be much less
    than power at f=0.5 in the early-fixation regime."""
    f = np.array([0.5, 1.0, 2.0])
    omega = np.linspace(-200, 200, 1024)
    s_early = BoiCycleEarlySpectrum(A=4.4)
    C = s_early.C(f, omega)
    # Total in-band power (use a representative ω range to avoid edge effects):
    in_band = (np.abs(omega) > 5) & (np.abs(omega) < 100)
    P = C[:, in_band].sum(axis=1)
    # Assert monotonic decrease and significant drop.
    assert P[0] > P[1] > P[2]
    assert P[0] / P[2] > 5.0  # at least 5x more power at low f


# ---------------------------------------------------------------------------
# Spectrum.describe()
# ---------------------------------------------------------------------------

def test_describe_excludes_image_params():
    """describe() should report movement params but not the (verbose)
    image params block."""
    s = DriftSpectrum(D=2.0)
    desc = s.describe()
    assert "D=2.0" in desc
    assert "ImageParams" not in desc
    assert "beta" not in desc


# ---------------------------------------------------------------------------
# BoiEarlyCleanApprox / BoiLateDriftApprox (corrected clean approximations)
# ---------------------------------------------------------------------------

def test_boi_early_clean_approx_shape_and_factorization():
    """C = I(f) * Q for the clean approximation."""
    from src.spectra import BoiEarlyCleanApprox
    f = np.geomspace(0.05, 5.0, 30)
    omega = np.linspace(-200, 200, 256)
    s = BoiEarlyCleanApprox()
    C = s.C(f, omega)
    Q = s.redistribution(f, omega)
    CI = image_spectrum(f)
    assert C.shape == (30, 256)
    np.testing.assert_allclose(C, CI[:, None] * Q, rtol=1e-12)


def test_boi_early_clean_approx_low_f_whitening_high_f_saturation():
    """At low f (well below 1/(2*A_mean)) S should be ~ flat in f; at high
    f (in the saturation regime) S should follow the f^-2 image slope."""
    from src.spectra import BoiEarlyCleanApprox
    s = BoiEarlyCleanApprox()
    f = np.geomspace(0.05, 4.0, 60)
    omega = np.linspace(-200, 200, 512)
    S = s.C(f, omega)
    in_b = (np.abs(omega) > 5) & (np.abs(omega) < 100)
    Pf = S[:, in_b].sum(axis=1)
    slope_lo = np.polyfit(np.log(f[:10]), np.log(Pf[:10]), 1)[0]
    slope_hi = np.polyfit(np.log(f[-10:]), np.log(Pf[-10:]), 1)[0]
    assert abs(slope_lo) < 0.5, f"low-f slope should be ~0, got {slope_lo}"
    assert -2.2 < slope_hi < -1.6, f"high-f slope should be ~-2, got {slope_hi}"


def test_boi_late_drift_approx_matches_drift_lorentzian_with_2pi():
    """BoiLateDriftApprox(f_is_cycles=True) is drift_lorentzian with
    a = D * (2*pi*f)^2 (cycles-to-radians conversion)."""
    from src.spectra import BoiLateDriftApprox, drift_lorentzian
    f = np.geomspace(0.1, 4.0, 30)
    omega = np.geomspace(0.5, 400, 30)
    D = 0.05
    Q = BoiLateDriftApprox(D=D, f_is_cycles=True).redistribution(f, omega)
    D_eff = D * (2.0 * np.pi) ** 2
    ref = drift_lorentzian(f[:, None], omega[None, :], D_eff)
    np.testing.assert_allclose(Q, ref, atol=1e-14)


def test_corrected_orientation_estimator_differs_from_old_pattern():
    """The corrected <|FT|^2>_theta result should NOT equal the old
    |FT[<.>_theta]|^2 = |FT[J_0(...)]|^2 result. Sanity guard against
    accidentally restoring the broken averaging order."""
    from src.spectra import _windowed_saccade_redistribution, saccade_template
    from scipy.special import j0
    f = np.geomspace(0.1, 4.0, 12)
    omega = np.linspace(-200, 200, 41)
    A = 4.4
    T_win = 0.512
    n_t = 4096
    new_Q = _windowed_saccade_redistribution(f, omega, A=A, T_win=T_win, n_t=n_t)
    dt = T_win / n_t
    t = (np.arange(n_t) - n_t // 2) * dt
    u = saccade_template(t)
    j0_factor = j0(2.0 * np.pi * f[:, None] * A * u[None, :])
    j0_factor = j0_factor - j0_factor.mean(axis=1, keepdims=True)
    g = np.fft.ifftshift(j0_factor, axes=1)
    G = np.fft.fft(g, axis=1) * dt
    G = np.fft.fftshift(G, axes=1)
    old_P = (np.abs(G) ** 2) / T_win
    domega = 2.0 * np.pi / (n_t * dt)
    om_native = (np.arange(n_t) - n_t // 2) * domega
    old_Q = np.empty((f.size, omega.size))
    for i in range(f.size):
        old_Q[i] = np.interp(omega, om_native, old_P[i], left=0.0, right=0.0)
    assert not np.allclose(new_Q, old_Q), "Corrected estimator must differ from |FT[<·>_θ]|^2"
    assert np.max(np.abs(new_Q - old_Q)) > 1e-3
