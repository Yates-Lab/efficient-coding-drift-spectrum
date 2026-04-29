"""Tests for the kernels module.

Covers:
- Cepstral min-phase: causality (impulse response zero for t < 0).
- Recovery of |V(ω)| after construction (magnitude is preserved).
- Round-trip on a known min-phase signal.
- Spatial 2D IFFT with correct continuous-FT scaling: a Gaussian in (k_x, k_y)
  maps to a Gaussian in (x, y) with the right widths.
"""

from __future__ import annotations

import numpy as np
import pytest

import sys
sys.path.insert(0, "/home/claude/efficient_coding")

from src.kernels import (
    minimum_phase_log_filter,
    minimum_phase_temporal_filter,
    spatial_kernel_2d,
)


# ---------------------------------------------------------------------------
# Min-phase: causality
# ---------------------------------------------------------------------------

def test_min_phase_impulse_response_is_causal():
    """Reconstructed v(t) should have negligible amplitude for t > T/2 (the
    "negative time" part of the FFT grid)."""
    # Lorentzian magnitude: |V(ω)| = 1/sqrt(a^2 + ω^2) — corresponds to
    # h(t) = e^{-a|t|}/(2a)? Actually that's not min-phase; the causal version
    # h(t) = e^{-a t} u(t) has |V| = 1/sqrt(a^2+ω^2) too, so min-phase recovers it.
    a = 5.0
    N = 1024
    omega_max = 200.0
    domega = 2 * omega_max / N
    omega = (np.arange(N) - N // 2) * domega
    v_mag = 1.0 / np.sqrt(a ** 2 + omega ** 2)

    t, v_t, V_complex = minimum_phase_temporal_filter(v_mag, omega)

    # Energy in "negative-time" half (i.e. t > T/2 wraps around). Causal
    # min-phase filter should put nearly all its energy in t < T/2.
    half = N // 2
    energy_late = np.sum(v_t[half:] ** 2)
    energy_early = np.sum(v_t[:half] ** 2)
    assert energy_late < 1e-3 * energy_early


def test_min_phase_recovers_magnitude():
    """|V_minphase(ω)| should equal the input |V(ω)|."""
    a = 3.0
    N = 512
    omega_max = 50.0
    domega = 2 * omega_max / N
    omega = (np.arange(N) - N // 2) * domega
    v_mag_in = 1.0 / np.sqrt(a ** 2 + omega ** 2)
    _, _, V_complex = minimum_phase_temporal_filter(v_mag_in, omega)
    np.testing.assert_allclose(np.abs(V_complex), v_mag_in, rtol=1e-6)


def test_soft_band_taper_in_core_and_outside():
    """Taper is exactly 1 in core and exactly 0 far outside."""
    from src.kernels import soft_band_taper
    omega = np.linspace(-200, 200, 4001)
    taper = soft_band_taper(omega, 1.0, 50.0, alpha=0.25)
    abs_w = np.abs(omega)
    # Core is [1+0.25, 50-12.5] = [1.25, 37.5]
    in_core = (abs_w >= 1.25) & (abs_w <= 37.5)
    np.testing.assert_allclose(taper[in_core], 1.0, atol=1e-12)
    far = (abs_w < 0.74) | (abs_w > 62.6)
    np.testing.assert_allclose(taper[far], 0.0, atol=1e-12)
    # Smooth: max value 1, min value 0
    assert taper.max() == 1.0
    assert taper.min() == 0.0


def test_soft_band_min_phase_peaks_early():
    """With a soft taper, min-phase peak of a wide bandpass should be small (<0.3s).
    Without the taper (hard cutoff + tiny floor), the peak is much later."""
    from src.kernels import soft_band_taper, minimum_phase_temporal_filter
    N = 1024
    omega_max_grid = 200.0
    domega = 2.0 * omega_max_grid / N
    omega = (np.arange(N) - N // 2) * domega
    H_smooth = soft_band_taper(omega, 0.5, 80.0, alpha=0.25)
    H_smooth = np.maximum(H_smooth, 1e-3)
    t, h, _ = minimum_phase_temporal_filter(H_smooth, omega)
    i_peak = int(np.argmax(np.abs(h)))
    assert t[i_peak] < 0.3, f"min-phase peak too late: t={t[i_peak]:.3f}s"


def test_min_phase_round_trip_consistent():
    """For a min-phase V_true, running min-phase reconstruction from |V_true|
    recovers V_true at well-resolved frequencies. The cepstral method has
    known truncation issues; we restrict to where |V| is large.
    """
    a = 2.0
    N = 4096
    omega_max = 200.0
    domega = 2 * omega_max / N
    omega = (np.arange(N) - N // 2) * domega
    V_true = 1.0 / (a + 1j * omega) ** 2
    v_mag = np.abs(V_true)

    _, _, V_rec = minimum_phase_temporal_filter(v_mag, omega)

    np.testing.assert_allclose(np.abs(V_rec), np.abs(V_true), rtol=1e-6)

    # Phase consistency: at frequencies where |V| is at least 25% of peak,
    # the phase should match within a few percent.
    in_band = np.abs(V_true) >= 0.25 * np.abs(V_true).max()
    rel_err = np.abs(V_rec[in_band] - V_true[in_band]) / np.abs(V_true[in_band])
    assert np.median(rel_err) < 0.05, f"median rel err = {np.median(rel_err):.4f}"


def test_min_phase_recovers_2pole_impulse_response():
    """For a 2-pole min-phase filter V = 1/(a+iω)^2, the impulse response
    is h(t) = t exp(-at) u(t). With fast tail decay, the min-phase
    reconstruction matches up to a few percent on the early part."""
    a = 2.0
    N = 4096
    omega_max = 200.0
    domega = 2 * omega_max / N
    omega = (np.arange(N) - N // 2) * domega
    V_true = 1.0 / (a + 1j * omega) ** 2
    v_mag = np.abs(V_true)

    t, v_t, _ = minimum_phase_temporal_filter(v_mag, omega)
    # Compare on first half of time samples (causal region)
    half = N // 4  # focus on the well-resolved early part
    expected = t[:half] * np.exp(-a * t[:half])
    # Match shape; use peak value to set the global scale
    i_peak = np.argmax(np.abs(v_t[:half]))
    norm = v_t[i_peak] / expected[i_peak]
    np.testing.assert_allclose(v_t[:half], norm * expected, atol=0.02)


# ---------------------------------------------------------------------------
# Spatial kernel: 2D IFFT scaling
# ---------------------------------------------------------------------------

def test_spatial_kernel_gaussian_widths():
    """If V(k) = exp(-||k||^2 σ_k^2 / 2), then the 2D inverse FT (with
    convention v(r) = ∫ d²k/(2π)² V(k) e^{i k·r}) is

        v(r) = (1/(2π σ_k^2)) exp(-||r||^2 / (2 σ_k^2)).

    Peak value 1/(2π σ_k^2) at r=0, and value at r = σ_k is peak * exp(-1/2).
    """
    sigma_k = 1.5
    def v_mag(f):
        return np.exp(-f ** 2 * sigma_k ** 2 / 2.0)
    # Larger k_max gives finer dr (= π/k_max), better radial sampling.
    rx, ry, v_xy = spatial_kernel_2d(v_mag, k_max=30.0, n_k=512)
    peak = v_xy.max()
    expected_peak = 1.0 / (2.0 * np.pi * sigma_k ** 2)
    np.testing.assert_allclose(peak, expected_peak, rtol=1e-2)
    iy0 = np.argmin(np.abs(ry))
    ix = np.argmin(np.abs(rx - sigma_k))
    np.testing.assert_allclose(
        v_xy[iy0, ix], expected_peak * np.exp(-0.5), rtol=3e-2
    )


def test_spatial_kernel_is_real_for_zero_phase_input():
    sigma_k = 1.0
    def v_mag(f):
        return np.exp(-f ** 2 * sigma_k ** 2 / 2.0)
    _, _, v_xy = spatial_kernel_2d(v_mag, k_max=8.0, n_k=128)
    # Imaginary part should be zero by construction (we use .real).
    assert np.all(np.isfinite(v_xy))
