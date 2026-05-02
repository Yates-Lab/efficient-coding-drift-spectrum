"""Tests for the analytic saccade/fixation-cycle spectra."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

from src.pipeline import run
from src.rucci_cycle_spectra import (
    ImageParams,
    SaccadeTraceParams,
    DriftTraceParams,
    EstimatorParams,
    image_spectrum,
    make_saccade_traces,
    make_drift_traces,
    estimate_Q_from_traces,
    make_rucci_cycle_spectra,
    spectra_as_wrappers,
    temporal_power_integral,
)


def _small_grid():
    f = np.geomspace(0.05, 4.0, 18)
    n_omega = 128
    omega_max = 2.0 * np.pi * 80.0
    domega = 2.0 * omega_max / n_omega
    omega = (np.arange(n_omega) - n_omega // 2) * domega
    return f, omega


def _small_cycle():
    f, omega = _small_grid()
    return make_rucci_cycle_spectra(
        f,
        omega,
        image_params=ImageParams(beta=2.0, f0=0.03, high_cut_cpd=60.0),
        saccade_params=SaccadeTraceParams(
            n_saccades=4,
            amplitude_mode="fixed",
            fixed_A_deg=4.4,
            T_win_s=0.128,
            dt_s=0.002,
            seed=123,
        ),
        drift_params=DriftTraceParams(
            n_traces=6,
            T_win_s=0.256,
            dt_s=0.002,
            speed_rms_deg_s=1.0,
            tau_v_s=0.075,
            seed=456,
        ),
        estimator_params=EstimatorParams(
            n_orientations=4,
            n_fft=256,
            use_fft=True,
            window="rect",
        ),
    )


def test_saccade_and_drift_trace_shapes():
    sac_params = SaccadeTraceParams(
        n_saccades=5,
        amplitude_mode="fixed",
        fixed_A_deg=2.5,
        T_win_s=0.064,
        dt_s=0.002,
        seed=1,
    )
    traces, t, amplitudes = make_saccade_traces(sac_params)
    assert traces.shape == (5, 32, 2)
    assert t.shape == (32,)
    np.testing.assert_allclose(amplitudes, 2.5)
    assert np.all(np.isfinite(traces))

    drift_params = DriftTraceParams(n_traces=7, T_win_s=0.08, dt_s=0.002, seed=2)
    drift, drift_t = make_drift_traces(drift_params)
    assert drift.shape == (7, 40, 2)
    assert drift_t.shape == (40,)
    np.testing.assert_allclose(drift[:, 0, :], 0.0, atol=1e-12)


def test_estimator_total_power_for_static_trace_is_near_one():
    f, omega = _small_grid()
    n_t = 64
    dt = 0.002
    t = np.arange(n_t) * dt
    traces = np.zeros((3, n_t, 2))
    Q = estimate_Q_from_traces(
        f,
        omega,
        traces,
        t,
        params=EstimatorParams(n_orientations=4, n_fft=256),
        subtract_temporal_mean=False,
    )
    integral = temporal_power_integral(Q, omega)
    np.testing.assert_allclose(integral, 1.0, rtol=0.08, atol=0.08)

    Q_mod = estimate_Q_from_traces(
        f,
        omega,
        traces,
        t,
        params=EstimatorParams(n_orientations=4, n_fft=256),
        subtract_temporal_mean=True,
    )
    np.testing.assert_allclose(Q_mod, 0.0, atol=1e-20)


def test_cycle_spectra_shapes_and_image_factorization():
    cycle = _small_cycle()
    assert cycle.Q_saccade_total.shape == (18, 128)
    assert cycle.Q_saccade_mod.shape == (18, 128)
    assert cycle.Q_drift_total.shape == (18, 128)
    assert cycle.C_early_mod.shape == (18, 128)
    assert np.all(cycle.C_early_mod >= 0)
    assert np.all(cycle.C_late_total >= 0)

    image = image_spectrum(
        cycle.f,
        ImageParams(beta=2.0, f0=0.03, high_cut_cpd=60.0),
    )
    np.testing.assert_allclose(cycle.image, image, rtol=1e-12)
    np.testing.assert_allclose(
        cycle.C_early_mod,
        image[:, None] * cycle.Q_saccade_mod,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        cycle.C_late_total,
        image[:, None] * cycle.Q_drift_total,
        rtol=1e-12,
    )


def test_saccade_total_and_mod_match_for_analytic_transient():
    cycle = _small_cycle()
    total_power = temporal_power_integral(cycle.Q_saccade_total, cycle.omega)
    mod_power = temporal_power_integral(cycle.Q_saccade_mod, cycle.omega)
    np.testing.assert_allclose(total_power, mod_power, rtol=1e-12)
    assert np.any(cycle.Q_saccade_mod > 0)


def test_array_wrappers_work_with_pipeline_grid_interpolation():
    cycle = _small_cycle()
    early, late = spectra_as_wrappers(cycle, use_modulated_early=True)
    nonzero = cycle.omega != 0.0
    dc = cycle.omega == 0.0

    C_exact = early.C(cycle.f, cycle.omega)
    np.testing.assert_allclose(C_exact[:, nonzero], cycle.C_early_mod[:, nonzero], rtol=1e-12)

    C_late_exact = late.C(cycle.f, cycle.omega)
    np.testing.assert_allclose(
        C_late_exact[:, nonzero],
        cycle.C_late_total[:, nonzero],
        rtol=1e-12,
    )
    assert np.nanmax(C_late_exact[:, dc]) <= np.nanmax(cycle.C_late_total[:, nonzero])

    r_early = run(early, grid="fast")
    r_late = run(late, grid="fast")
    assert np.isfinite(r_early.I)
    assert np.isfinite(r_late.I)
    assert r_early.C.shape == (r_early.f.size, r_early.omega.size)
    assert r_late.C.shape == (r_late.f.size, r_late.omega.size)
