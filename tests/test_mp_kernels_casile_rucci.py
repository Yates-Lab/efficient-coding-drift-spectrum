"""Tests for the Casile-Rucci M/P kernel helpers."""

from __future__ import annotations

import numpy as np
import pytest

import sys
sys.path.insert(0, ".")

from src.mp_kernels_casile_rucci import (
    mp_kernel_curves,
    spatial_frequency_kernel,
    spatial_kernel_profile,
    spatiotemporal_filter_power,
    temporal_frequency_kernel,
    temporal_impulse_response,
)


def test_mp_frequency_responses_are_finite():
    sf = np.logspace(np.log10(0.2), np.log10(60.0), 64)
    tf = np.logspace(np.log10(0.2), np.log10(120.0), 72)

    for cell in ("M", "P"):
        K = spatial_frequency_kernel(sf, cell)
        H = temporal_frequency_kernel(tf, cell)
        power = spatiotemporal_filter_power(sf, tf, cell)

        assert K.shape == sf.shape
        assert H.shape == tf.shape
        assert power.shape == (sf.size, tf.size)
        assert np.all(np.isfinite(K))
        assert np.all(np.isfinite(H))
        assert np.all(np.isfinite(power))
        assert np.nanmax(power) > 0.0


def test_mp_space_and_time_kernel_curves_are_plottable():
    curves = mp_kernel_curves(n_temporal_samples=512)

    assert set(curves) == {"M", "P"}
    for curve in curves.values():
        assert curve["r"].shape == curve["spatial"].shape
        assert curve["t"].shape == curve["temporal"].shape
        assert np.all(np.isfinite(curve["spatial"]))
        assert np.all(np.isfinite(curve["temporal"]))
        np.testing.assert_allclose(curve["t"][0], 0.0)


def test_spatial_profile_has_center_surround_structure():
    r = np.linspace(-2.0, 2.0, 801)

    for cell in ("M", "P"):
        v = spatial_kernel_profile(r, cell)
        center_idx = int(np.argmin(np.abs(r)))

        assert v[center_idx] > 0.0
        assert np.min(v) < 0.0


def test_temporal_impulse_response_rejects_empty_grid():
    with pytest.raises(ValueError):
        temporal_impulse_response("M", n_samples=0)
