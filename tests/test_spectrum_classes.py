"""Tests for the class-based Spectrum API and unit conventions."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

from src.spectra import (
    DEFAULT_IMAGE,
    DriftSpectrum,
    ImageParams,
    LinearMotionSpectrum,
    SaccadeSpectrum,
    angular_spatial_frequency,
    drift_lorentzian,
    drift_spectrum,
    image_spectrum,
    linear_motion_spectrum_gaussian,
    saccade_redistribution,
    saccade_spectrum,
)


def test_drift_class_matches_free_function():
    k = np.geomspace(0.05, 5.0, 50)
    omega = np.linspace(-100, 100, 401)
    s = DriftSpectrum(D=2.0)
    np.testing.assert_allclose(
        s.C(k, omega),
        drift_spectrum(k[:, None], omega[None, :], D=2.0),
        rtol=1e-12,
    )


def test_drift_lorentzian_uses_cycles_per_degree():
    k = np.array([0.5, 1.0, 2.0])
    omega = np.linspace(-1000.0, 1000.0, 20001)
    D = 0.0375
    Q = drift_lorentzian(k[:, None], omega[None, :], D)
    a = D * angular_spatial_frequency(k) ** 2
    for i, ai in enumerate(a):
        q0 = np.interp(0.0, omega, Q[i])
        qai = np.interp(ai, omega, Q[i])
        assert abs(qai / q0 - 0.5) < 0.01


def test_drift_spectrum_is_flat_in_k_at_low_omega():
    spec = DriftSpectrum(D=0.0375)
    k = np.array([0.01, 0.05, 0.1, 0.5])
    omega = np.array([2.0 * np.pi])
    C = spec.C(k, omega)[:, 0]
    assert np.ptp(C) / np.mean(C) < 0.05


def test_drift_spectrum_no_low_k_upturn():
    spec = DriftSpectrum(D=0.0375)
    k = np.array([0.01, 0.05])
    omega = np.array([2.0 * np.pi])
    C = spec.C(k, omega)[:, 0]
    assert abs(C[0] - C[1]) / C[1] < 0.1


def test_saccade_class_matches_free_function():
    k = np.geomspace(0.05, 5.0, 50)
    omega = np.linspace(-100, 100, 401)
    s = SaccadeSpectrum(A=2.5)
    np.testing.assert_allclose(
        s.C(k, omega), saccade_spectrum(k, omega, A=2.5), rtol=1e-12
    )


def test_saccade_zero_at_zero_amplitude():
    k = np.geomspace(0.05, 5.0, 20)
    omega = np.linspace(-200, 200, 512)
    Q = SaccadeSpectrum(A=0.0).redistribution(k, omega)
    np.testing.assert_allclose(Q, 0.0, atol=1e-20)


def test_saccade_spectrum_factors_image():
    k = np.geomspace(0.05, 5.0, 30)
    omega = np.linspace(-200, 200, 512)
    C = SaccadeSpectrum(A=4.4).C(k, omega)
    Q = saccade_redistribution(k, omega, A=4.4)
    np.testing.assert_allclose(C, image_spectrum(k)[:, None] * Q, rtol=1e-12)


def test_linear_motion_class_matches_free_function():
    k = np.geomspace(0.05, 5.0, 50)
    omega = np.linspace(-100, 100, 401)
    s = LinearMotionSpectrum(s=1.5)
    np.testing.assert_allclose(
        s.C(k, omega),
        linear_motion_spectrum_gaussian(k[:, None], omega[None, :], s=1.5),
        rtol=1e-12,
    )


def test_class_carries_provenance():
    s = SaccadeSpectrum(A=2.5)
    assert s.name == "saccade"
    assert "saccade" in s.reference
    desc = s.describe()
    assert "saccade" in desc
    assert "A=2.5" in desc


def test_image_params_default_matches_free_function():
    k = np.geomspace(0.05, 5.0, 30)
    np.testing.assert_allclose(DEFAULT_IMAGE.C(k), image_spectrum(k), rtol=1e-12)


def test_custom_image_params_threaded_through():
    k = np.geomspace(0.05, 5.0, 30)
    omega = np.linspace(-100, 100, 201)
    custom = ImageParams(beta=2.5, A_image=2.0, k0=0.1)
    s = DriftSpectrum(D=1.0, image=custom)
    expected = drift_spectrum(
        k[:, None],
        omega[None, :],
        D=1.0,
        beta=2.5,
        A=2.0,
        k0=0.1,
    )
    np.testing.assert_allclose(s.C(k, omega), expected, rtol=1e-12)


def test_describe_excludes_image_params():
    s = DriftSpectrum(D=2.0)
    desc = s.describe()
    assert "D=2.0" in desc
    assert "ImageParams" not in desc
    assert "beta" not in desc
