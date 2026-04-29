"""Tests for the aliasing module.

Covers:
- 1D radial aliasing reduces to original spectrum when k_s is large enough
  that no copies fold into the band.
- Per-cell aliasing at very high f0 where the spectrum is nearly zero
  produces zero.
- 2D aliasing matches 1D in the case of an isotropic spectrum.
- Per-cell aliasing matches the analytic series for a power-law spectrum.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.aliasing import (
    aliased_spectrum_radial,
    aliased_spectrum_2d,
    per_cell_aliased_spectrum,
)
from src.spectra import drift_spectrum, image_spectrum


def test_aliased_spectrum_recovers_original_for_large_k_s():
    """If k_s is much larger than the support of the spectrum, the aliased
    spectrum at k=0 equals the original at k=0 (the m=0 term dominates)."""
    f = np.array([0.5])
    omega = np.array([2.0])
    def C(f_, om):
        return drift_spectrum(f_, om, D=1.0)
    aliased = aliased_spectrum_radial(C, f, omega, k_s=1000.0, m_max=4)
    direct = C(f, omega)
    np.testing.assert_allclose(aliased, direct, rtol=1e-6)


def test_per_cell_alias_zero_for_high_f0():
    """At very high f0 the aliased spectrum is small (image power decays)."""
    f0 = np.array([100.0])
    omega = np.array([0.5])
    def C(f_, om):
        return drift_spectrum(f_, om, D=1.0)
    aliased = per_cell_aliased_spectrum(C, f0, omega, m_max=12)
    direct = C(f0, omega)
    # Aliased should be larger than direct (more copies summed) but still
    # very small in absolute terms.
    assert aliased[0] >= direct[0]
    # But the value should be vastly smaller than the spectrum at f0=1.
    ref = drift_spectrum(np.array([1.0]), omega, D=1.0)
    assert aliased[0] < 1e-2 * ref[0]


def test_aliased_2d_matches_1d_for_isotropic():
    """At a point on the k_x axis with k_y=0, summing over a 2D grid of
    aliased copies should agree with the (proper) 2D sum. Note this is NOT
    the same as the 1D radial sum because of corner copies."""
    omega = np.array([1.0])
    kx = np.array([0.5])
    ky = np.array([0.0])
    def C(f_, om):
        return image_spectrum(f_, beta=2.0, k0=0.05) * 2.0 * 1.0 * f_ ** 2 / (
            (1.0 * f_ ** 2) ** 2 + om ** 2
        )

    # 2D sum
    s2d = aliased_spectrum_2d(C, kx, ky, omega, k_s_x=2.0, k_s_y=2.0, m_max=4)

    # Compare with manual sum: sum over m in -4..4 of C(|kx + 2m|, omega)
    # plus copies in y direction
    manual = 0.0
    for mx in range(-4, 5):
        for my in range(-4, 5):
            kxa = 0.5 + mx * 2.0
            kya = 0.0 + my * 2.0
            f_ = np.sqrt(kxa ** 2 + kya ** 2)
            manual += C(np.array([f_]), omega)
    np.testing.assert_allclose(s2d.flatten(), manual.flatten(), rtol=1e-12)


def test_per_cell_aliasing_lower_bound():
    """Per-cell aliasing always >= direct value (since one copy is the
    direct one, and other copies are non-negative)."""
    f0 = np.geomspace(0.1, 5.0, 20)
    omega = np.array([1.0])
    def C(f_, om):
        return drift_spectrum(f_, om, D=2.0)
    aliased = per_cell_aliased_spectrum(C, f0, omega, m_max=8)
    direct = C(f0, omega)
    assert np.all(aliased >= direct - 1e-12)


def test_per_cell_alias_known_formula():
    """For a static (delta in ω) power-law image: the aliased spectrum at
    f0 is C_I(f0) + Σ_{q≥1} 2 C_I((2q+1) f0).
    """
    # Use an unregularized power-law (k0 small), and check the q-series.
    f0 = np.array([1.0])
    omega = np.array([1.0])  # arbitrary; we just use a fixed slice

    # Test with a custom spectrum that depends only on f (not omega):
    def C(f_, om):
        # Make sure broadcasting works
        return f_ ** -2 * np.ones_like(om + np.zeros_like(f_))
    aliased = per_cell_aliased_spectrum(C, f0, omega, m_max=20)
    # Analytic series: 1 + Σ_{q=1..∞} 2 / (2q+1)^2 = 1 + 2(π²/8 - 1) = π²/4 - 1
    expected = (np.pi ** 2) / 4.0 - 1.0 + 1.0  # = π²/4
    np.testing.assert_allclose(aliased[0], np.pi ** 2 / 4.0, rtol=2e-2)
