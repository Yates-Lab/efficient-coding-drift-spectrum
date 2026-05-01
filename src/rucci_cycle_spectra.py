"""Rucci-style saccade/fixation-cycle spectra for efficient-coding models.

This module implements a parsimonious trace-based generator for the two input
spectra needed for the saccade/fixation-cycle hypothesis:

    early fixation:  C_early(f, omega) = I(f) Q_saccade(f, omega)
    late fixation:   C_late(f, omega)  = I(f) Q_drift(f, omega)

The important design choice is that saccades are NOT modeled as a stationary
Poisson jump process.  Instead, both saccades and drift are represented as
explicit eye-position traces, and Q is estimated by the same Fourier-domain
redistribution estimator used in the Rucci/Boi/Mostofi line of work:

    Q(f, omega) = < | integral exp[-i 2*pi*f*n.x(t)] exp[-i omega t] dt |^2 > / T

where the average is over eye-movement traces and spatial-frequency
orientations n.  Multiplying Q by a natural-image spectrum I(f) gives the
retinal input spectrum seen by the efficient-coding solver.

Conventions
-----------
- f is radial spatial frequency in cycles/degree, or cycles per arbitrary
  length unit if traces are in the same length unit.
- omega is temporal angular frequency in rad/s.
- traces are eye positions in degrees, shape (n_trace, n_time, 2).
- Q integrates to approximately one over d omega / (2*pi) for each f when
  subtract_temporal_mean=False and the temporal grid covers the FFT support.

Recommended use
---------------
1. Generate the spectra once with `make_rucci_cycle_spectra`.
2. Feed `cycle.C_early_mod` or `cycle.C_early_total` and `cycle.C_late_total`
   to the efficient-coding solver.
3. Use the `ArraySpectrum` wrapper if your pipeline expects objects with a
   `.C(f, omega)` method.

For finite isolated saccade windows, `total` includes the static/DC component
from immobile pre/post-saccadic periods.  `mod` subtracts the temporal mean of
exp[-i 2*pi*f*n.x(t)] before Fourier transforming and is often the cleaner
input for a retinal temporal band that excludes DC/adaptation-dominated power.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

TWOPI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Natural-image spectrum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageParams:
    """Regularized natural-image spatial power spectrum.

    I(f) = amplitude / (f^2 + f0^2)^(beta/2)

    beta=2 is the standard natural-image approximation used in the Rucci
    spectral analyses.  f0 prevents a singularity at zero spatial frequency.
    """

    beta: float = 2.0
    amplitude: float = 1.0
    f0: float = 0.03
    high_cut_cpd: Optional[float] = None
    high_cut_order: float = 4.0


def image_spectrum(f, params: ImageParams = ImageParams()) -> np.ndarray:
    """Return I(f) on a 1D spatial-frequency grid."""
    f = np.asarray(f, dtype=float)
    I = params.amplitude / (f * f + params.f0 * params.f0) ** (params.beta / 2.0)
    if params.high_cut_cpd is not None:
        # Smooth optical/stimulus rolloff to make high-frequency extensions safe.
        I = I / (1.0 + (f / float(params.high_cut_cpd)) ** float(params.high_cut_order))
    return I


# ---------------------------------------------------------------------------
# Saccade traces: damped harmonic step with optional weak main-sequence scaling
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SaccadeTraceParams:
    """Parameters for synthetic isolated saccade traces.

    The default amplitude distribution approximates the natural-viewing range
    used for Boi-style early-fixation spectra: a truncated normal over 1--10 deg
    with mean 4.4 deg and SD 1.3 deg.  For Mostofi-style single-amplitude
    checks, set amplitude_mode='fixed' and fixed_A_deg to the desired amplitude.

    `peak_time_s` and `zeta` define the damped-harmonic step response.  `psi`
    rescales time.  The default psi model is deliberately weak, because Mostofi
    Fig. 5 indicates that 1/psi remains close to one across common amplitudes.
    """

    n_saccades: int = 64
    amplitude_mode: str = "truncnorm"  # 'truncnorm', 'uniform', or 'fixed'
    mean_A_deg: float = 4.4
    sd_A_deg: float = 1.3
    min_A_deg: float = 1.0
    max_A_deg: float = 10.0
    fixed_A_deg: float = 2.5
    uniform_min_A_deg: float = 1.0
    uniform_max_A_deg: float = 10.0

    T_win_s: float = 0.512
    dt_s: float = 0.001
    onset_time_s: float = 0.0
    peak_time_s: float = 0.040
    zeta: float = 0.60

    # A weak empirical surrogate for the main-sequence scaling in Mostofi Fig. 5B:
    # 1/psi ~= psi_intercept + psi_slope_per_deg * A.  Set slope to 0 for no scaling.
    psi_intercept: float = 0.88
    psi_slope_per_deg: float = 0.06
    psi_min: float = 0.65
    psi_max: float = 1.35

    seed: int = 1234
    random_directions: bool = True


@dataclass(frozen=True)
class DriftTraceParams:
    """Parameters for synthetic inter-saccadic drift traces.

    Drift is generated as integrated Ornstein-Uhlenbeck velocity.  This gives
    smooth finite-speed traces while approaching Brownian diffusion at time
    scales much longer than tau_v_s.

    speed_rms_deg_s is the RMS total 2D speed, sqrt(E[vx^2 + vy^2]).
    The long-time Brownian-equivalent diffusion coefficient is approximately

        D_eff = speed_rms_deg_s^2 * tau_v_s / 2.
    """

    n_traces: int = 128
    T_win_s: float = 1.024
    dt_s: float = 0.001
    speed_rms_deg_s: float = 1.0
    tau_v_s: float = 0.075
    seed: int = 5678
    remove_start_position: bool = True

    @property
    def D_eff_deg2_s(self) -> float:
        return self.speed_rms_deg_s ** 2 * self.tau_v_s / 2.0


@dataclass(frozen=True)
class EstimatorParams:
    """Parameters for Q(f, omega) estimation from traces."""

    n_orientations: int = 32
    n_fft: Optional[int] = None
    use_fft: bool = True
    window: str = "rect"  # 'rect' or 'hann'
    smooth_sigma_omega_bins: float = 2.5


def _rng(seed: Optional[int]) -> np.random.Generator:
    return np.random.default_rng(seed)


def _sample_truncated_normal(
    rng: np.random.Generator,
    n: int,
    mean: float,
    sd: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    """Simple rejection sampler for a truncated normal."""
    if sd <= 0:
        return np.full(n, float(mean))
    out = []
    # Oversample in chunks to avoid scipy dependency.
    while len(out) < n:
        draw = rng.normal(mean, sd, size=max(4 * (n - len(out)), 64))
        draw = draw[(draw >= lo) & (draw <= hi)]
        out.extend(draw.tolist())
    return np.asarray(out[:n], dtype=float)


def sample_saccade_amplitudes(params: SaccadeTraceParams) -> np.ndarray:
    """Draw saccade amplitudes in degrees according to params."""
    rng = _rng(params.seed)
    n = int(params.n_saccades)
    mode = params.amplitude_mode.lower()
    if mode == "fixed":
        return np.full(n, float(params.fixed_A_deg))
    if mode == "uniform":
        return rng.uniform(params.uniform_min_A_deg, params.uniform_max_A_deg, n)
    if mode == "truncnorm":
        return _sample_truncated_normal(
            rng, n, params.mean_A_deg, params.sd_A_deg,
            params.min_A_deg, params.max_A_deg,
        )
    raise ValueError("amplitude_mode must be 'fixed', 'uniform', or 'truncnorm'")


def damped_harmonic_step(tau, *, peak_time_s: float = 0.040, zeta: float = 0.60) -> np.ndarray:
    """Unit-amplitude underdamped second-order step response.

    tau is time since saccade onset in seconds.  The response is zero for
    tau < 0 and approaches one as tau -> infinity.  The formula is the standard
    underdamped step response.  peak_time_s sets the first overshoot time.
    """
    tau = np.asarray(tau, dtype=float)
    if not (0.0 < zeta < 1.0):
        raise ValueError("zeta must lie in (0, 1) for an underdamped response")
    omega0 = np.pi / (float(peak_time_s) * np.sqrt(1.0 - zeta * zeta))
    omegad = omega0 * np.sqrt(1.0 - zeta * zeta)
    out = np.zeros_like(tau, dtype=float)
    active = tau >= 0.0
    if np.any(active):
        tt = tau[active]
        decay = np.exp(-zeta * omega0 * tt)
        out[active] = 1.0 - decay * (
            np.cos(omegad * tt)
            + (zeta / np.sqrt(1.0 - zeta * zeta)) * np.sin(omegad * tt)
        )
    return out


def psi_from_amplitude(A_deg, params: SaccadeTraceParams) -> np.ndarray:
    """Return the temporal scale psi(A) used in u((t-onset)/psi)."""
    A = np.asarray(A_deg, dtype=float)
    inv_psi = params.psi_intercept + params.psi_slope_per_deg * A
    inv_psi = np.maximum(inv_psi, 1e-6)
    psi = 1.0 / inv_psi
    return np.clip(psi, params.psi_min, params.psi_max)


def make_saccade_traces(params: SaccadeTraceParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate isolated 2D saccade traces.

    Returns
    -------
    traces : ndarray, shape (n_saccades, n_time, 2)
        Eye position in degrees.
    t : ndarray, shape (n_time,)
        Time in seconds.  The window is centered around zero by default.
    amplitudes : ndarray, shape (n_saccades,)
        Saccade amplitudes in degrees.
    """
    rng = _rng(params.seed)
    n_t = int(round(params.T_win_s / params.dt_s))
    if n_t < 8:
        raise ValueError("T_win_s/dt_s is too small")
    t = (np.arange(n_t, dtype=float) - n_t // 2) * params.dt_s
    A = sample_saccade_amplitudes(params)
    psi = psi_from_amplitude(A, params)

    if params.random_directions:
        theta = rng.uniform(0.0, TWOPI, size=A.size)
    else:
        theta = np.zeros(A.size)
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)

    traces = np.zeros((A.size, n_t, 2), dtype=float)
    for i, (amp, scale, d) in enumerate(zip(A, psi, dirs)):
        u = damped_harmonic_step(
            (t - params.onset_time_s) / scale,
            peak_time_s=params.peak_time_s,
            zeta=params.zeta,
        )
        traces[i] = (amp * u)[:, None] * d[None, :]
    return traces, t, A


# ---------------------------------------------------------------------------
# Drift traces
# ---------------------------------------------------------------------------

def make_drift_traces(params: DriftTraceParams) -> Tuple[np.ndarray, np.ndarray]:
    """Generate smooth drift traces by integrating OU velocity."""
    rng = _rng(params.seed)
    n_t = int(round(params.T_win_s / params.dt_s))
    if n_t < 8:
        raise ValueError("T_win_s/dt_s is too small")
    t = np.arange(n_t, dtype=float) * params.dt_s

    a = np.exp(-params.dt_s / params.tau_v_s)
    sigma_coord = params.speed_rms_deg_s / np.sqrt(2.0)
    noise_sd = sigma_coord * np.sqrt(1.0 - a * a)

    v = np.zeros((params.n_traces, n_t, 2), dtype=float)
    v[:, 0, :] = rng.normal(0.0, sigma_coord, size=(params.n_traces, 2))
    for i in range(1, n_t):
        v[:, i, :] = a * v[:, i - 1, :] + rng.normal(0.0, noise_sd, size=(params.n_traces, 2))

    # Position is integral of velocity.  The first sample is zero if requested.
    q = np.cumsum(v, axis=1) * params.dt_s
    if params.remove_start_position:
        q = q - q[:, :1, :]
    return q, t


def brownian_drift_Q(f, omega, D_deg2_s: float) -> np.ndarray:
    """Analytic Brownian-drift redistribution with f in cycles/deg.

    Q(f, omega) = 2 a / (a^2 + omega^2), a = D (2*pi*f)^2.
    This integrates to one over d omega/(2*pi).
    """
    f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    omega = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
    a = float(D_deg2_s) * (TWOPI * f[:, None]) ** 2
    w = omega[None, :]
    return 2.0 * a / (a * a + w * w)


# ---------------------------------------------------------------------------
# Rucci-style trace estimator for Q
# ---------------------------------------------------------------------------

def _window_vector(n_t: int, kind: str) -> np.ndarray:
    kind = kind.lower()
    if kind == "rect":
        w = np.ones(n_t, dtype=float)
    elif kind == "hann":
        w = np.hanning(n_t)
    else:
        raise ValueError("window must be 'rect' or 'hann'")
    # Unit RMS keeps Parseval/power scales comparable across windows.
    rms = np.sqrt(np.mean(w * w))
    if rms > 0:
        w = w / rms
    return w


def _gaussian_smooth_axis(a: np.ndarray, sigma_bins: float, axis: int) -> np.ndarray:
    """Gaussian smooth along one array axis without adding a scipy dependency."""
    sigma = float(sigma_bins)
    if sigma <= 0:
        return a
    radius = max(1, int(np.ceil(4.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    arr = np.moveaxis(np.asarray(a, dtype=float), axis, -1)
    pad = [(0, 0)] * arr.ndim
    pad[-1] = (radius, radius)
    padded = np.pad(arr, pad, mode="edge")
    smoothed = np.apply_along_axis(
        lambda v: np.convolve(v, kernel, mode="valid"),
        axis=-1,
        arr=padded,
    )
    return np.moveaxis(smoothed, -1, axis)


def _smooth_and_preserve_temporal_power(
    Q: np.ndarray, omega: np.ndarray, sigma_bins: float
) -> np.ndarray:
    """Suppress periodogram sidelobes while preserving row-wise total power."""
    if sigma_bins <= 0:
        return np.maximum(Q, 0.0)
    Q = np.maximum(np.asarray(Q, dtype=float), 0.0)
    smoothed = np.maximum(_gaussian_smooth_axis(Q, sigma_bins, axis=1), 0.0)
    old_power = np.trapz(Q, omega, axis=1)
    new_power = np.trapz(smoothed, omega, axis=1)
    scale = np.ones_like(old_power)
    ok = (old_power > 0) & (new_power > 0)
    scale[ok] = old_power[ok] / new_power[ok]
    return smoothed * scale[:, None]


def estimate_Q_from_traces(
    f,
    omega,
    traces_xy,
    t,
    *,
    params: EstimatorParams = EstimatorParams(),
    subtract_temporal_mean: bool = False,
    trace_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Estimate the movement redistribution function Q(f, omega).

    This computes the orientation-averaged periodogram of

        g(t) = exp[-i 2*pi*f*n_dot_x(t)]

    for each trace and spatial-frequency orientation.  The orientation average
    is an average of Fourier powers, not the Fourier power of an orientation-
    averaged Bessel amplitude.

    Parameters
    ----------
    f : array_like
        Radial spatial frequencies in cycles/degree.
    omega : array_like
        Temporal angular frequencies in rad/s.  Can include positive and
        negative values.
    traces_xy : ndarray
        Eye-position traces in degrees, shape (n_trace, n_time, 2), (n_time,2),
        (n_trace,n_time), or (n_time,).  One-dimensional traces are treated as x.
    t : ndarray
        Time samples in seconds, shape (n_time,).  Uniform spacing is assumed.
    params : EstimatorParams
        FFT/orientation/window settings.
    subtract_temporal_mean : bool
        If False, returns the literal total spectrum.  If True, removes the DC
        component of g(t) before the Fourier transform, isolating movement-
        induced modulation at nonzero temporal frequencies.
    trace_weights : optional ndarray
        Nonnegative weights for traces.  Defaults to equal weighting.
    """
    f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    omega = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
    t = np.asarray(t, dtype=float).ravel()
    x = np.asarray(traces_xy, dtype=float)

    if x.ndim == 1:
        x = x[None, :, None]
    elif x.ndim == 2:
        if x.shape[-1] == 2:
            x = x[None, :, :]
        else:
            x = x[:, :, None]
    elif x.ndim != 3:
        raise ValueError("traces_xy must be 1D, 2D, or 3D")
    if x.shape[-1] == 1:
        x = np.concatenate([x, np.zeros_like(x)], axis=-1)
    if x.shape[-1] != 2:
        raise ValueError("last trace dimension must be 1 or 2")

    n_trace, n_t, _ = x.shape
    if t.size != n_t:
        raise ValueError("t must have one entry per time sample")
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1e-4, atol=1e-10):
        raise ValueError("t must be uniformly sampled")
    T = n_t * dt

    if trace_weights is None:
        tw = np.full(n_trace, 1.0 / n_trace)
    else:
        tw = np.asarray(trace_weights, dtype=float).ravel()
        if tw.size != n_trace:
            raise ValueError("trace_weights must have one entry per trace")
        if np.any(tw < 0) or tw.sum() <= 0:
            raise ValueError("trace_weights must be nonnegative and sum positive")
        tw = tw / tw.sum()

    n_orient = int(params.n_orientations)
    if n_orient < 1:
        raise ValueError("n_orientations must be >= 1")
    # [0, pi) is sufficient for power because opposite orientations conjugate.
    theta = (np.arange(n_orient, dtype=float) + 0.5) * np.pi / n_orient
    dirs = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    window = _window_vector(n_t, params.window)

    if params.use_fft:
        n_fft = int(params.n_fft or n_t)
        if n_fft < n_t:
            n_fft = n_t
        omega_native = TWOPI * np.fft.fftfreq(n_fft, d=dt)
        order = np.argsort(omega_native)
        omega_sorted = omega_native[order]
        P_accum = np.zeros((f.size, n_fft), dtype=float)

        for tr, wtr in zip(x, tw):
            # proj has shape (n_orient, n_time)
            proj = dirs @ tr.T
            for proj_o in proj:
                phase = np.exp(-1j * TWOPI * f[:, None] * proj_o[None, :])
                if subtract_temporal_mean:
                    phase = phase - phase.mean(axis=1, keepdims=True)
                phase = phase * window[None, :]
                G = np.fft.fft(phase, n=n_fft, axis=1) * dt
                P_accum += (wtr / n_orient) * (np.abs(G) ** 2) / T

        P_sorted = P_accum[:, order]
        Q = np.empty((f.size, omega.size), dtype=float)
        for i in range(f.size):
            Q[i] = np.interp(omega, omega_sorted, P_sorted[i], left=0.0, right=0.0)
        return _smooth_and_preserve_temporal_power(
            Q, omega, params.smooth_sigma_omega_bins
        )

    # Direct nonuniform-frequency estimator.  Slower but exact for requested omega.
    expw = np.exp(-1j * omega[None, :] * t[:, None])
    Q = np.zeros((f.size, omega.size), dtype=float)
    for tr, wtr in zip(x, tw):
        proj = dirs @ tr.T
        for proj_o in proj:
            phase = np.exp(-1j * TWOPI * f[:, None] * proj_o[None, :])
            if subtract_temporal_mean:
                phase = phase - phase.mean(axis=1, keepdims=True)
            ft = (phase * window[None, :]) @ expw * dt
            Q += (wtr / n_orient) * (np.abs(ft) ** 2) / T
    return _smooth_and_preserve_temporal_power(
        Q, omega, params.smooth_sigma_omega_bins
    )


# ---------------------------------------------------------------------------
# Cycle spectra arrays and spectrum wrappers
# ---------------------------------------------------------------------------

@dataclass
class RucciCycleSpectra:
    """Container returned by make_rucci_cycle_spectra."""

    f: np.ndarray
    omega: np.ndarray
    image: np.ndarray
    Q_saccade_total: np.ndarray
    Q_saccade_mod: np.ndarray
    Q_drift_total: np.ndarray
    Q_drift_mod: np.ndarray
    C_early_total: np.ndarray
    C_early_mod: np.ndarray
    C_late_total: np.ndarray
    C_late_mod: np.ndarray
    saccade_amplitudes_deg: np.ndarray
    drift_D_eff_deg2_s: float

    def save_npz(self, path: str) -> None:
        np.savez_compressed(
            path,
            f=self.f,
            omega=self.omega,
            image=self.image,
            Q_saccade_total=self.Q_saccade_total,
            Q_saccade_mod=self.Q_saccade_mod,
            Q_drift_total=self.Q_drift_total,
            Q_drift_mod=self.Q_drift_mod,
            C_early_total=self.C_early_total,
            C_early_mod=self.C_early_mod,
            C_late_total=self.C_late_total,
            C_late_mod=self.C_late_mod,
            saccade_amplitudes_deg=self.saccade_amplitudes_deg,
            drift_D_eff_deg2_s=self.drift_D_eff_deg2_s,
        )


def make_rucci_cycle_spectra(
    f,
    omega,
    *,
    image_params: ImageParams = ImageParams(),
    saccade_params: SaccadeTraceParams = SaccadeTraceParams(),
    drift_params: DriftTraceParams = DriftTraceParams(),
    estimator_params: EstimatorParams = EstimatorParams(),
    drift_mode: str = "analytic_brownian",
) -> RucciCycleSpectra:
    """Generate early/saccade and late/drift spectra for the cycle.

    Both Q_total and Q_mod are returned.  The corresponding C arrays are simply
    I(f)[:, None] multiplied by Q.
    """
    f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    omega = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
    I = image_spectrum(f, image_params)

    sac_traces, sac_t, A = make_saccade_traces(saccade_params)
    drift_mode = drift_mode.lower()
    if drift_mode not in {"analytic_brownian", "trace"}:
        raise ValueError("drift_mode must be 'analytic_brownian' or 'trace'")

    Q_sac_total = estimate_Q_from_traces(
        f, omega, sac_traces, sac_t,
        params=estimator_params,
        subtract_temporal_mean=False,
    )
    Q_sac_mod = estimate_Q_from_traces(
        f, omega, sac_traces, sac_t,
        params=estimator_params,
        subtract_temporal_mean=True,
    )
    if drift_mode == "analytic_brownian":
        # The empirical Rucci/Boi drift estimator is based on displacement
        # probability.  With our OU trace parameters, the long-time Brownian
        # limit gives the smooth population spectrum without finite-window
        # periodogram bands.
        Q_drift_total = brownian_drift_Q(f, omega, drift_params.D_eff_deg2_s)
        Q_drift_mod = Q_drift_total.copy()
    else:
        drift_traces, drift_t = make_drift_traces(drift_params)
        Q_drift_total = estimate_Q_from_traces(
            f, omega, drift_traces, drift_t,
            params=estimator_params,
            subtract_temporal_mean=False,
        )
        Q_drift_mod = estimate_Q_from_traces(
            f, omega, drift_traces, drift_t,
            params=estimator_params,
            subtract_temporal_mean=True,
        )

    return RucciCycleSpectra(
        f=f,
        omega=omega,
        image=I,
        Q_saccade_total=Q_sac_total,
        Q_saccade_mod=Q_sac_mod,
        Q_drift_total=Q_drift_total,
        Q_drift_mod=Q_drift_mod,
        C_early_total=I[:, None] * Q_sac_total,
        C_early_mod=I[:, None] * Q_sac_mod,
        C_late_total=I[:, None] * Q_drift_total,
        C_late_mod=I[:, None] * Q_drift_mod,
        saccade_amplitudes_deg=A,
        drift_D_eff_deg2_s=drift_params.D_eff_deg2_s,
    )


# ---------------------------------------------------------------------------
# Canonical Figure 7 cycle
# ---------------------------------------------------------------------------

def figure7_cycle_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Return the canonical grid used by Figure 7's Rucci-style cycle.

    Other saccade/fixation-cycle figures should consume this spectrum through
    `figure7_cycle_wrappers` rather than regenerating their own trace estimate.
    The wrappers interpolate onto the solver or plotting grid as needed.
    """
    f = np.geomspace(0.03, 20.0, 72)
    n_omega = 512
    omega_max = TWOPI * 120.0
    domega = 2.0 * omega_max / n_omega
    omega = (np.arange(n_omega) - n_omega // 2) * domega
    return f, omega


@lru_cache(maxsize=1)
def make_figure7_rucci_cycle_spectra() -> RucciCycleSpectra:
    """Generate the single canonical Rucci/Boi-style Figure 7 spectrum.

    This is the source-of-truth spectrum for all early/late cycle analyses:

        early: C_early = I(f) Q_saccade_mod
        late:  C_late  = I(f) Q_drift_total
    """
    f, omega = figure7_cycle_grid()
    return make_rucci_cycle_spectra(
        f,
        omega,
        image_params=ImageParams(beta=2.0, f0=0.03, high_cut_cpd=60.0),
        saccade_params=SaccadeTraceParams(
            n_saccades=12,
            amplitude_mode="truncnorm",
            mean_A_deg=4.4,
            sd_A_deg=1.3,
            min_A_deg=1.0,
            max_A_deg=10.0,
            T_win_s=0.512,
            dt_s=0.001,
            seed=1234,
        ),
        drift_params=DriftTraceParams(
            n_traces=24,
            T_win_s=1.024,
            dt_s=0.002,
            speed_rms_deg_s=1.0,
            tau_v_s=0.075,
            seed=5678,
        ),
        estimator_params=EstimatorParams(
            n_orientations=8,
            n_fft=1024,
            use_fft=True,
            window="rect",
            smooth_sigma_omega_bins=5.0,
        ),
    )


def figure7_cycle_wrappers(*, use_modulated_early: bool = True) -> Tuple[ArraySpectrum, ArraySpectrum]:
    """Return wrappers for the canonical Figure 7 early and late spectra."""
    cycle = make_figure7_rucci_cycle_spectra()
    return spectra_as_wrappers(cycle, use_modulated_early=use_modulated_early)


class ArraySpectrum:
    """Small wrapper so precomputed C arrays can be passed to pipeline.run.

    The existing pipeline calls `spectrum.C(f, omega)` and `spectrum.describe()`.
    If queried on exactly the precomputed grid, this returns the stored array.
    If queried on a different grid, it performs separable linear interpolation.
    """

    def __init__(self, f, omega, C, label: str, *, ignore_dc_for_interp: bool = False):
        self.f = np.asarray(f, dtype=float).ravel()
        self.omega = np.asarray(omega, dtype=float).ravel()
        self._C = np.asarray(C, dtype=float)
        if self._C.shape != (self.f.size, self.omega.size):
            raise ValueError("C must have shape (len(f), len(omega))")
        self.label = str(label)
        self.ignore_dc_for_interp = bool(ignore_dc_for_interp)

        order = np.argsort(self.omega)
        self._omega_interp = self.omega[order]
        self._C_interp = self._C[:, order].copy()
        if self.ignore_dc_for_interp:
            # The Rucci/Boi cycle figures intentionally display nonzero temporal
            # frequencies.  The Brownian late-fixation spectrum has a huge DC
            # bin, and including that bin in interpolation leaks artificial
            # near-zero power into solver grids whose first positive frequency
            # lies below the first nonzero Figure 7 sample.
            nonzero = self._omega_interp != 0.0
            if np.count_nonzero(nonzero) < 2:
                raise ValueError("ignore_dc_for_interp requires at least two nonzero omega samples")
            self._omega_interp = self._omega_interp[nonzero]
            self._C_interp = self._C_interp[:, nonzero]

    def describe(self) -> str:
        return self.label

    def C(self, f, omega) -> np.ndarray:
        f_q = np.asarray(f, dtype=float).ravel()
        w_q = np.asarray(omega, dtype=float).ravel()
        if (
            not self.ignore_dc_for_interp
            and np.array_equal(f_q, self.f)
            and np.array_equal(w_q, self.omega)
        ):
            return self._C.copy()

        # Interpolate in omega first, then f.  This avoids requiring scipy.
        tmp = np.empty((self.f.size, w_q.size), dtype=float)
        for i in range(self.f.size):
            tmp[i] = np.interp(
                w_q,
                self._omega_interp,
                self._C_interp[i],
                left=0.0,
                right=0.0,
            )
        out = np.empty((f_q.size, w_q.size), dtype=float)
        for j in range(w_q.size):
            out[:, j] = np.interp(f_q, self.f, tmp[:, j], left=0.0, right=0.0)
        return np.maximum(out, 0.0)


def spectra_as_wrappers(cycle: RucciCycleSpectra, *, use_modulated_early: bool = True) -> Tuple[ArraySpectrum, ArraySpectrum]:
    """Return (early, late) ArraySpectrum wrappers for the existing pipeline."""
    early_C = cycle.C_early_mod if use_modulated_early else cycle.C_early_total
    early_label = "Rucci synthetic early fixation: saccade transient"
    if use_modulated_early:
        early_label += " (temporal mean removed)"
    late_C = cycle.C_late_total
    late_label = f"Rucci synthetic late fixation: OU drift (D_eff={cycle.drift_D_eff_deg2_s:.4g} deg^2/s)"
    return (
        ArraySpectrum(cycle.f, cycle.omega, early_C, early_label, ignore_dc_for_interp=True),
        ArraySpectrum(cycle.f, cycle.omega, late_C, late_label, ignore_dc_for_interp=True),
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def temporal_power_integral(Q, omega) -> np.ndarray:
    """Approximate integral Q(f,omega) d omega/(2*pi) for each f."""
    omega = np.asarray(omega, dtype=float).ravel()
    return np.trapz(Q, omega, axis=1) / TWOPI


def spatial_slope_loglog(f, C_slice, f_lo: float, f_hi: float) -> float:
    """Fit d log C / d log f over [f_lo, f_hi] for one omega slice."""
    f = np.asarray(f, dtype=float).ravel()
    y = np.asarray(C_slice, dtype=float).ravel()
    m = (f >= f_lo) & (f <= f_hi) & (y > 0)
    if m.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(f[m]), np.log(y[m]), 1)[0])


__all__ = [
    "ImageParams",
    "SaccadeTraceParams",
    "DriftTraceParams",
    "EstimatorParams",
    "RucciCycleSpectra",
    "ArraySpectrum",
    "image_spectrum",
    "damped_harmonic_step",
    "psi_from_amplitude",
    "sample_saccade_amplitudes",
    "make_saccade_traces",
    "make_drift_traces",
    "brownian_drift_Q",
    "estimate_Q_from_traces",
    "make_rucci_cycle_spectra",
    "figure7_cycle_grid",
    "make_figure7_rucci_cycle_spectra",
    "figure7_cycle_wrappers",
    "spectra_as_wrappers",
    "temporal_power_integral",
    "spatial_slope_loglog",
]
