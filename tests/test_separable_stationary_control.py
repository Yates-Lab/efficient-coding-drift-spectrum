"""Tests for the stationary separable control and cycles-aware drift audit."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

from src.cell_class_figures import log_additive_separability_r2
from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX
from src.plotting import band_mask_radial, radial_weights
from src.spectra import (
    DEFAULT_IMAGE,
    DriftSpectrum,
    SeparableMovieSpectrum,
    separable_movie_spectrum,
    temporal_lorentzian,
)
from src.power_spectrum_library import get_spectrum_set, stationary_vs_active_story_specs


def test_temporal_lorentzian_power_conservation_on_wide_grid():
    omega0 = 10.0
    omega = np.linspace(-5000.0, 5000.0, 200001)
    Q = temporal_lorentzian(omega, omega0=omega0)
    integral = np.trapz(Q, omega) / (2.0 * np.pi)
    assert abs(integral - 1.0) < 0.003


def test_separable_movie_class_matches_free_function():
    f = np.geomspace(0.05, 5.0, 20)
    omega = np.linspace(-100.0, 100.0, 101)
    s = SeparableMovieSpectrum(omega0=7.5)
    expected = separable_movie_spectrum(
        f[:, None],
        omega[None, :],
        omega0=7.5,
        beta=DEFAULT_IMAGE.beta,
        A=DEFAULT_IMAGE.A_image,
        k0=DEFAULT_IMAGE.k0,
    )
    np.testing.assert_allclose(s.C(f, omega), expected, rtol=1e-12)


def test_drift_spectrum_uses_cycles_aware_width():
    f = np.array([0.5, 1.0, 2.0])
    D = 0.0375
    omega = np.linspace(-1000.0, 1000.0, 20001)
    Q = DriftSpectrum(D=D).redistribution(f, omega)
    # At omega=a, Lorentzian falls to exactly half of its value at zero.
    a = D * (2.0 * np.pi * f) ** 2
    for i, ai in enumerate(a):
        q0 = np.interp(0.0, omega, Q[i])
        qai = np.interp(ai, omega, Q[i])
        assert abs(qai / q0 - 0.5) < 0.01


def test_story_spectrum_set_includes_separable_control():
    specs = stationary_vs_active_story_specs()
    keys = [s.key for s in specs]
    assert keys[0] == "separable_stationary"
    assert "dong_atick_linear" in keys
    assert "saccade_A_4.4" in keys
    assert "drift_D_0.0375" in keys

    named = get_spectrum_set("stationary_vs_active_story")
    assert [s.key for s in named] == keys


def test_power_law_separable_control_has_exact_log_additive_score():
    f = np.geomspace(0.05, 5.0, 120)
    omega = np.linspace(-500.0, 500.0, 1024)
    C = SeparableMovieSpectrum(omega0=0.05).C(f, omega)
    weights = radial_weights(f, omega) * band_mask_radial(
        f, omega, F_MAX, OMEGA_MIN, OMEGA_MAX
    )
    assert log_additive_separability_r2(C, weights) > 0.999999
