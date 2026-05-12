import numpy as np

from src.params import Band
from src.spectra import ImageParams, DriftSpectrum, SaccadeSpectrum, LinearMotionSpectrum, SeparableMovieSpectrum
from src.noise import WhiteNoise, TemporalPowerLawNoise, ConeLikeNoise
from src.solver import solve_on_grid, response_power_spend
from src.plotting import radial_weights, band_mask_radial
from src.kernels import temporal_kernel_slice, spatial_kernel_slice, default_slice_frequencies
from src.mp_kernels import mp_filter_power, retinal_input_power_map, response_power_from_input


def test_four_spectra_and_three_noises_are_finite():
    band = Band(f_min=0.01, f_max=50, tf_min_hz=0.05, tf_max_hz=80)
    f, tf = band.log_symmetric_grid(n_f=24, n_tf_pos=32)
    img = ImageParams()
    spectra = [
        DriftSpectrum(D=40/3600, image=img),
        SaccadeSpectrum(A=3.5, image=img),
        LinearMotionSpectrum(s=1.0, image=img),
        SeparableMovieSpectrum(image=img),
    ]
    noises = [
        WhiteNoise.from_sigma(0.01),
        TemporalPowerLawNoise.from_sigma(0.01),
        ConeLikeNoise.from_sigma(0.01),
    ]
    for spec in spectra:
        C = spec.C(f, tf)
        assert C.shape == (f.size, tf.size)
        assert np.all(np.isfinite(C))
        assert np.all(C >= 0)
    for noise in noises:
        S = noise.power(f, tf)
        assert S.shape == (f.size, tf.size)
        assert np.all(np.isfinite(S))
        assert np.all(S >= 0)


def test_solver_spends_budget_and_kernels_reconstruct():
    band = Band(f_min=0.02, f_max=40, tf_min_hz=0.05, tf_max_hz=60)
    f, tf = band.log_symmetric_grid(n_f=32, n_tf_pos=48)
    spec = DriftSpectrum(D=40/3600)
    noise = ConeLikeNoise.from_sigma(0.001)
    out = WhiteNoise.from_sigma(0.005)
    P0 = 5.0
    r = solve_on_grid(spec, f, tf, P0=P0, input_noise=noise, output_noise=out, band=band.edges)
    weights = radial_weights(f, tf) * band_mask_radial(f, tf, *band.edges)
    spend = response_power_spend(r.C, r.v_sq, weights, r.input_noise_power)
    assert np.isfinite(r.I) and r.I >= 0
    np.testing.assert_allclose(spend, P0, rtol=1e-5, atol=1e-8)
    f0, tf0 = default_slice_frequencies(f, tf, r.v_sq)
    x, s, _ = spatial_kernel_slice(f, tf, r.v_sq, tf0, f_max=band.f_max, n=128)
    t, h, _, _ = temporal_kernel_slice(f, tf, r.v_sq, f0, tf_min_hz=band.tf_min_hz, tf_max_hz=band.tf_max_hz, n_uniform=512)
    assert x.shape == s.shape
    assert t.shape == h.shape
    assert np.all(np.isfinite(s))
    assert np.all(np.isfinite(h))


def test_mp_kernels_match_repo_grid_and_response_power_is_finite():
    sf = np.geomspace(0.1, 30.0, 16)
    tf = np.concatenate([-np.geomspace(0.1, 80.0, 20)[::-1], np.geomspace(0.1, 80.0, 20)])

    for cell in ["M", "P"]:
        power = mp_filter_power(sf, tf, cell)
        np.testing.assert_allclose(power, mp_filter_power(sf, tf, cell, gamma=1.0, rho=1.0))
        assert not np.allclose(power, mp_filter_power(sf, tf, cell, gamma=0.5, rho=1.0 / 1.6))
        assert power.shape == (sf.size, tf.size)
        assert np.all(np.isfinite(power))
        assert np.all(power >= 0)
        assert np.max(power) > 0

    tf_fft, P_input = retinal_input_power_map(
        sf[:5],
        duration_s=0.2,
        sample_rate_hz=200.0,
        n_trials=2,
        n_orientations=2,
        n_phases=2,
        ramp_s=0.02,
        seed=3,
    )
    response = response_power_from_input(sf[:5], tf_fft, P_input, "M")
    assert response.shape == P_input.shape
    assert np.all(np.isfinite(response))
    assert np.all(response >= 0)
