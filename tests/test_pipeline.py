"""Tests for the pipeline module."""

from __future__ import annotations

import numpy as np
import pytest

import sys
sys.path.insert(0, ".")

from src.spectra import (
    DriftSpectrum, SaccadeSpectrum, BoiCycleLateSpectrum,
    BoiCycleEarlySpectrum,
)
from src.pipeline import (
    SolveConfig,
    run,
    run_many,
    extract_kernels,
    extract_spatial_kernel,
    extract_temporal_kernel,
    spatial_kernel_slice,
    temporal_kernel_slice,
)
from src.power_spectrum_library import drift_spectrum_specs


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------

def test_run_returns_populated_result():
    r = run(DriftSpectrum(D=2.0), grid="fast")
    assert r.spectrum.D == 2.0
    assert r.f.ndim == 1
    assert r.omega.ndim == 1
    assert r.C.shape == (r.f.size, r.omega.size)
    assert r.v_sq.shape == r.C.shape
    assert np.isfinite(r.I) and r.I > 0
    assert np.isfinite(r.lam) and r.lam > 0


def test_run_grid_choices():
    r_fast = run(DriftSpectrum(D=2.0), grid="fast")
    r_hi = run(DriftSpectrum(D=2.0), grid="hi_res")
    assert r_fast.f.size < r_hi.f.size
    assert r_fast.omega.size < r_hi.omega.size
    # I* should agree to within fast-grid discretization (~1%)
    np.testing.assert_allclose(r_fast.I, r_hi.I, rtol=2e-2)


def test_run_unknown_grid_raises():
    with pytest.raises(ValueError):
        run(DriftSpectrum(), grid="bogus")


def test_run_boi_late_matches_drift():
    """BoiCycleLate and DriftSpectrum must produce identical I* (same C)."""
    r_drift = run(DriftSpectrum(D=2.0), grid="fast")
    r_late = run(BoiCycleLateSpectrum(D=2.0), grid="fast")
    np.testing.assert_allclose(r_late.I, r_drift.I, rtol=1e-12)
    np.testing.assert_allclose(r_late.v_sq, r_drift.v_sq, rtol=1e-12)


def test_run_saccade_mostofi_spectrum_is_finite():
    """Cross-check the Mostofi saccade approximation across the pipeline boundary."""
    r_sac = run(SaccadeSpectrum(A=2.5), grid="fast")
    assert np.isfinite(r_sac.I) and r_sac.I > 0
    assert np.all(np.isfinite(r_sac.C))


def test_run_many_accepts_spectrum_specs():
    specs = drift_spectrum_specs([0.5, 2.0])
    results = run_many(specs, SolveConfig(grid="fast"))
    assert len(results) == 2
    assert [r.spectrum.D for r in results] == [0.5, 2.0]
    assert all(np.isfinite(r.I) and r.I > 0 for r in results)


# ---------------------------------------------------------------------------
# extract_kernels()
# ---------------------------------------------------------------------------

def test_extract_spatial_kernel_populates_fields():
    r = run(DriftSpectrum(D=2.0), grid="hi_res")
    extract_spatial_kernel(r)
    assert r.spatial_r is not None
    assert r.spatial_v is not None
    assert r.spatial_r.shape == r.spatial_v.shape
    # FFT grid is centered with one extra negative sample (standard for even N).
    # Inner samples (excluding the unmatched leftmost bin) are symmetric.
    n = r.spatial_r.size
    inner_pos = r.spatial_r[1:]                    # n-1 samples
    inner_neg = -r.spatial_r[1:][::-1]             # mirror of the same n-1 samples
    np.testing.assert_allclose(inner_pos, inner_neg, atol=1e-9)


def test_extract_temporal_kernel_populates_fields():
    r = run(DriftSpectrum(D=2.0), grid="hi_res")
    extract_temporal_kernel(r)
    assert r.temporal_t is not None
    assert r.temporal_v is not None
    assert r.temporal_t.shape == r.temporal_v.shape
    # Causal: t starts at 0
    np.testing.assert_allclose(r.temporal_t[0], 0.0, atol=1e-12)
    # f_peak is set and reasonable
    assert 0 < r.f_peak <= 5.0


def test_extract_kernels_does_both():
    r = run(DriftSpectrum(D=2.0), grid="hi_res")
    extract_kernels(r)
    assert r.spatial_r is not None
    assert r.temporal_t is not None


def test_kernel_slice_helpers_return_curves():
    r = run(DriftSpectrum(D=2.0), grid="hi_res")
    spatial_r, spatial_v = spatial_kernel_slice(r, omega0=10.0)
    temporal_t, temporal_v = temporal_kernel_slice(r, f0=0.5)
    assert spatial_r.shape == spatial_v.shape
    assert temporal_t.shape == temporal_v.shape
    assert np.all(np.isfinite(spatial_v))
    assert np.all(np.isfinite(temporal_v))


def test_pipeline_magno_parvo_ordering_in_boi_cycle():
    """The early-fixation regime should have a peak spatial frequency at
    LOWER f than the late-fixation regime — this is the magno/parvo
    qualitative prediction."""
    r_late = run(BoiCycleLateSpectrum(D=2.0), grid="hi_res")
    r_early = run(BoiCycleEarlySpectrum(A=4.4, T_win=0.150), grid="hi_res")
    extract_temporal_kernel(r_late)
    extract_temporal_kernel(r_early)
    assert r_early.f_peak < r_late.f_peak, (
        f"Early regime peak f={r_early.f_peak:.3f} should be < "
        f"late regime peak f={r_late.f_peak:.3f}"
    )
