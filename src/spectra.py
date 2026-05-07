"""Input power spectra C_θ(k, ω) for moving-sensor efficient coding.

Each spectrum is a frozen dataclass that stores its parameters and exposes a
single ``C(k, omega)`` method returning the spectrum on the (Nk, Nω) grid
spanned by 1D arrays ``k`` and ``omega``.  Spectra that factor as
``C_I(k) * Q(k, ω)`` also expose ``redistribution(k, omega)`` returning ``Q``.

Conventions
-----------
- k : spatial frequency magnitude [cycles/degree]
- omega : temporal angular frequency [rad/sec]
- D : Brownian drift diffusion coefficient [deg^2/sec]
- A : saccade amplitude [deg]
- s : linear-motion velocity standard deviation [deg/sec]
- The spectra are two-sided in omega (defined on (-∞, ∞)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

import numpy as np
from scipy.special import j0


TWOPI = 2.0 * np.pi


# ---------------------------------------------------------------------------
# Image spectrum
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageParams:
    """Regularized natural-image power-law spectrum.

    C_I(k) = A_image / (k^2 + k0^2)^(beta/2)
    """
    beta: float = 2.0
    A_image: float = 1.0
    k0: float = 1e-6

    def C(self, k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        return self.A_image / (k ** 2 + self.k0 ** 2) ** (self.beta / 2.0)


DEFAULT_IMAGE = ImageParams()


# ---------------------------------------------------------------------------
# Saccade main sequence (Mostofi et al. 2020 calibration)
# ---------------------------------------------------------------------------

# Mostofi Fig 1G: peak velocity × duration ≈ 5.5 · A (deg).  For a
# cumulative-Gaussian saccade x(t) = A · Φ(t/σ_A), the peak velocity is
# A / (σ_A √(2π)) and the duration is T_A.  Therefore
#     σ_A = T_A / (5.5 · √(2π)) = T_A / SACCADE_SIGMA_DIVISOR.
SACCADE_SIGMA_DIVISOR = 5.5 * np.sqrt(TWOPI)


def saccade_main_sequence_duration(A, base_s: float = 0.021,
                                   slope_s_per_deg: float = 0.0022) -> np.ndarray:
    """Saccade duration T_A in seconds from amplitude A in degrees.

    Linear approximation T_A = base_s + slope_s_per_deg · A from Bahill et al.
    1975 / Mostofi et al. 2020 main sequence.
    """
    A = np.asarray(A, dtype=float)
    return base_s + slope_s_per_deg * A


def saccade_smoothing_sigma(A, base_s: float = 0.021,
                            slope_s_per_deg: float = 0.0022) -> np.ndarray:
    """Cumulative-Gaussian smoothing scale σ_A in seconds.

    Calibrated from the Mostofi et al. 2020 main sequence so that the model's
    peak velocity matches the v_peak·T_A ≈ 5.5·A relation in their Fig 1G.
    """
    return saccade_main_sequence_duration(A, base_s, slope_s_per_deg) / SACCADE_SIGMA_DIVISOR


# ---------------------------------------------------------------------------
# Spectrum base class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Spectrum(ABC):
    """Base class for on-retina input spectra C_θ(k, ω).

    Subclasses are frozen dataclasses: parameter values are immutable once
    constructed, which keeps results reproducible from the Spectrum instance
    alone.  Each subclass sets ``name`` and ``reference`` in __post_init__.
    """
    name: str = field(init=False)
    reference: str = field(init=False)

    @abstractmethod
    def C(self, k, omega) -> np.ndarray:
        """Spectrum on the (Nk, Nω) grid spanned by 1D arrays k, omega."""

    def describe(self) -> str:
        skip = {"name", "reference"}
        params = []
        for fld in fields(self):
            if fld.name in skip:
                continue
            v = getattr(self, fld.name)
            if isinstance(v, ImageParams):
                continue
            params.append(f"{fld.name}={v}")
        return f"{self.name}({', '.join(params)})"


# ---------------------------------------------------------------------------
# Brownian drift
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DriftSpectrum(Spectrum):
    """Brownian fixational drift (Kuang et al. 2012; Aytekin et al. 2014).

    Q_drift(k, ω) = 2 a / (a² + ω²), with a = D · (2π k)².

    The (2πk)² accounts for k being in cycles/degree while the image phase
    is 2π k x with x in degrees. Q_drift integrates to 1 over dω/(2π).

    A typical free-viewing fixational diffusion is D ≈ 0.005-0.05 deg²/sec
    (Aytekin et al. 2014, Mostofi et al. 2020).
    """
    D: float = 0.0375
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "drift")
        object.__setattr__(self, "reference", "Kuang et al. 2012")

    def redistribution(self, k, omega) -> np.ndarray:
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        if self.D == 0.0:
            return np.zeros((k_arr.size, omega_arr.size), dtype=float)
        a = float(self.D) * (TWOPI * k_arr[:, None]) ** 2
        return 2.0 * a / (a ** 2 + omega_arr[None, :] ** 2)

    def C(self, k, omega) -> np.ndarray:
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        return self.image.C(k_arr)[:, None] * self.redistribution(k_arr, omega)


# ---------------------------------------------------------------------------
# Saccade transients (cumulative-Gaussian approximation)
# ---------------------------------------------------------------------------

def _saccade_redistribution(k_arr: np.ndarray, omega_arr: np.ndarray,
                            A: float, sigma: float) -> np.ndarray:
    """Internal: Q_sac on prepared 1D arrays.  Returns (Nk, Nω) array.

    Q_sac(k, ω; A) = 2 [1 - J_0(2π k A)] · exp[-(σ ω)²] / ω².

    No floor is applied to ω: the analysis band always excludes a
    neighbourhood of ω = 0 (see ``Band.omega_min``).  The spatial factor
    vanishes at k = 0, so the k → 0 limit is well-defined.
    """
    if A == 0.0:
        return np.zeros((k_arr.size, omega_arr.size), dtype=float)
    spatial = 2.0 * (1.0 - j0(TWOPI * k_arr[:, None] * A))
    omega2 = omega_arr[None, :] ** 2
    # The user must avoid omega == 0 in the analysis grid; the band mask
    # enforces this.  We still divide safely to avoid raw division-by-zero
    # warnings if a stray ω = 0 sample slips through.
    with np.errstate(divide="ignore", invalid="ignore"):
        temporal = np.exp(-(sigma * omega_arr[None, :]) ** 2) / omega2
    temporal = np.where(np.isfinite(temporal), temporal, 0.0)
    return np.maximum(spatial * temporal, 0.0)


@dataclass(frozen=True)
class SaccadeSpectrum(Spectrum):
    """Cumulative-Gaussian analytic saccade-transient approximation.

    The trajectory is x(t) = A · Φ(t/σ_A) with Φ the standard normal CDF.
    The σ_A scale is calibrated from Mostofi et al. 2020 Fig 1G so that the
    model's peak velocity matches their main-sequence regression.

    Q_sac(k, ω; A) = 2 [1 - J_0(2π k A)] · exp[-(σ_A ω)²] / ω².

    The spatial factor ``2(1 - J_0(2π k A))`` is the orientation-averaged
    power of an A-degree displacement step: it produces Mostofi's whitening
    regime at k < 1/(2A) and saturates above.  The temporal factor is the
    smoothed-step power spectrum.
    """
    A: float = 2.5
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "saccade")
        object.__setattr__(self, "reference",
                           "Mostofi et al. 2020 cumulative-Gaussian saccade approximation")

    @property
    def sigma(self) -> float:
        """Cumulative-Gaussian smoothing scale σ_A in seconds."""
        return float(saccade_smoothing_sigma(self.A))

    def redistribution(self, k, omega) -> np.ndarray:
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        return _saccade_redistribution(k_arr, omega_arr, float(self.A), self.sigma)

    def C(self, k, omega) -> np.ndarray:
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        return self.image.C(k_arr)[:, None] * self.redistribution(k_arr, omega)


@dataclass(frozen=True)
class SaccadeAmplitudeMixture(Spectrum):
    """Saccade-transient redistribution averaged over an amplitude distribution.

    Q(k, ω) = Σ_i w_i Q_sac(k, ω; A_i),  Σ_i w_i = 1.

    Useful when modelling natural saccade-amplitude statistics rather than
    a single amplitude.  Per-amplitude factor uses the same Mostofi-calibrated
    σ_A as ``SaccadeSpectrum``.
    """
    amplitudes: tuple = (0.5, 1.0, 2.0, 4.0, 8.0)
    weights: tuple = ()  # empty -> uniform
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "saccade_mixture")
        object.__setattr__(self, "reference",
                           "Mostofi et al. 2020 saccade-amplitude mixture")

    def _normalized_weights(self) -> np.ndarray:
        A = np.asarray(self.amplitudes, dtype=float).ravel()
        if A.size == 0:
            raise ValueError("amplitudes must contain at least one value")
        if len(self.weights) == 0:
            return np.full(A.size, 1.0 / A.size, dtype=float)
        w = np.asarray(self.weights, dtype=float).ravel()
        if w.size != A.size:
            raise ValueError("weights must have the same length as amplitudes")
        if np.any(w < 0.0) or w.sum() <= 0.0:
            raise ValueError("weights must be nonnegative and sum positive")
        return w / w.sum()

    def redistribution(self, k, omega) -> np.ndarray:
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        A_arr = np.asarray(self.amplitudes, dtype=float).ravel()
        w = self._normalized_weights()
        Q = np.zeros((k_arr.size, omega_arr.size), dtype=float)
        for A_i, w_i in zip(A_arr, w):
            sigma_i = float(saccade_smoothing_sigma(A_i))
            Q += w_i * _saccade_redistribution(k_arr, omega_arr, float(A_i), sigma_i)
        return Q

    def C(self, k, omega) -> np.ndarray:
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        return self.image.C(k_arr)[:, None] * self.redistribution(k_arr, omega)


# ---------------------------------------------------------------------------
# Linear motion (Dong & Atick 1995)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearMotionSpectrum(Spectrum):
    """Linear motion with Gaussian-distributed isotropic velocity (Dong & Atick 1995).

    Speed s is the standard deviation of velocity in any direction (deg/sec).
    """
    s: float = 1.0
    image: ImageParams = DEFAULT_IMAGE
    k_floor: float = 1e-10  # numerical floor for the 1/k singularity

    def __post_init__(self):
        object.__setattr__(self, "name", "linear_motion_gaussian")
        object.__setattr__(self, "reference", "Dong & Atick 1995")

    def C(self, k, omega) -> np.ndarray:
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        k_safe = np.maximum(np.abs(TWOPI * k_arr[:, None]), self.k_floor)
        sf = float(self.s) * k_safe
        C_I = self.image.C(k_arr)[:, None]
        return np.sqrt(TWOPI) * C_I / sf * np.exp(-omega_arr[None, :] ** 2 / (2.0 * sf ** 2))


# ---------------------------------------------------------------------------
# Stationary separable movie control
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SeparableMovieSpectrum(Spectrum):
    """Stationary separable natural-movie approximation.

    S(k, ω) ∝ 1 / (max(|k|, k0)^β · max(|ω|, ω0)^β_t).

    The explicit control for passive efficient-coding models that factor the
    spectrum into independent spatial and temporal power laws.  k0 and ω0 are
    numerical floors; set them below the analysis band to recover straight
    diagonal log-log contours.
    """
    omega0: float = 0.05
    temporal_beta: float = 2.0
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "separable_movie")
        object.__setattr__(self, "reference", "stationary separable movie control")

    def C(self, k, omega) -> np.ndarray:
        if self.omega0 <= 0:
            raise ValueError("omega0 must be positive")
        if self.temporal_beta <= 0:
            raise ValueError("temporal_beta must be positive")
        k_arr = np.atleast_1d(np.asarray(k, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        spatial = self.image.A_image / np.maximum(np.abs(k_arr), self.image.k0) ** self.image.beta
        temporal = 1.0 / np.maximum(np.abs(omega_arr), self.omega0) ** self.temporal_beta
        return spatial[:, None] * temporal[None, :]


@dataclass(frozen=True)
class StaticImageSpectrum(Spectrum):
    """Static image (no eye movement). All power at ω=0; not representable on
    a discrete ω grid.  Provided for documentation / type-symmetry.
    """
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "static_image")
        object.__setattr__(self, "reference", "Field 1987")

    def C(self, k, omega) -> np.ndarray:
        raise NotImplementedError(
            "Static image has no spatiotemporal spectrum on a finite ω grid; "
            "use a movement model (DriftSpectrum, SaccadeSpectrum, ...)."
        )


# ---------------------------------------------------------------------------
# Back-compatibility shims
# ---------------------------------------------------------------------------
# Existing callers (scripts, tests, the aliasing-check) still import the old
# free functions.  We keep thin wrappers around the class API so existing
# code keeps working without forcing a callsite-by-callsite migration.

def image_spectrum(k, beta: float = 2.0, A: float = 1.0,
                   k0: float = 1e-6) -> np.ndarray:
    return ImageParams(beta=beta, A_image=A, k0=k0).C(k)


def drift_lorentzian(k, omega, D: float) -> np.ndarray:
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    out_shape = np.broadcast_shapes(k_arr.shape, omega_arr.shape)
    if D == 0:
        return np.zeros(out_shape)
    a = float(D) * (TWOPI * k_arr) ** 2
    return 2.0 * a / (a ** 2 + omega_arr ** 2)


def drift_spectrum(k, omega, D: float, beta: float = 2.0,
                   A: float = 1.0, k0: float = 1e-6) -> np.ndarray:
    return image_spectrum(k, beta, A, k0) * drift_lorentzian(k, omega, D)


def linear_motion_spectrum_gaussian(k, omega, s: float, beta: float = 2.0,
                                    A: float = 1.0, k0: float = 1e-6,
                                    k_floor: float = 1e-10) -> np.ndarray:
    k_arr = np.atleast_1d(np.asarray(k, dtype=float))
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float))
    k_safe = np.maximum(np.abs(TWOPI * k_arr), k_floor)
    C_I = image_spectrum(k, beta, A, k0)
    sf = float(s) * k_safe
    return np.sqrt(TWOPI) * C_I / sf * np.exp(-omega_arr ** 2 / (2.0 * sf ** 2))


def temporal_lorentzian(omega, omega0: float = 10.0) -> np.ndarray:
    if omega0 <= 0:
        raise ValueError("omega0 must be positive")
    omega = np.asarray(omega, dtype=float)
    return 2.0 * float(omega0) / (omega0 ** 2 + omega ** 2)


def separable_movie_spectrum(k, omega, omega0: float = 0.05, beta: float = 2.0,
                             A: float = 1.0, k0: float = 1e-6,
                             temporal_beta: float = 2.0) -> np.ndarray:
    return SeparableMovieSpectrum(
        omega0=omega0, temporal_beta=temporal_beta,
        image=ImageParams(beta=beta, A_image=A, k0=k0),
    ).C(k, omega)


def saccade_redistribution(k, omega, A) -> np.ndarray:
    return SaccadeSpectrum(A=float(A)).redistribution(k, omega)


def saccade_amplitude_average(k, omega, amplitudes, *, weights=None) -> np.ndarray:
    spec = SaccadeAmplitudeMixture(
        amplitudes=tuple(float(a) for a in amplitudes),
        weights=tuple(float(w) for w in weights) if weights is not None else (),
    )
    return spec.redistribution(k, omega)


def saccade_spectrum(k, omega, A, beta: float = 2.0, A_image: float = 1.0,
                     k0: float = 1e-6) -> np.ndarray:
    return SaccadeSpectrum(
        A=float(A), image=ImageParams(beta=beta, A_image=A_image, k0=k0)
    ).C(k, omega)


def angular_spatial_frequency(k) -> np.ndarray:
    """Convert spatial frequency from cycles/degree to radians/degree."""
    return TWOPI * np.asarray(k, dtype=float)


__all__ = [
    # Constants
    "TWOPI", "SACCADE_SIGMA_DIVISOR",
    # Image
    "ImageParams", "DEFAULT_IMAGE",
    # Spectra
    "Spectrum",
    "DriftSpectrum",
    "SaccadeSpectrum", "SaccadeAmplitudeMixture",
    "LinearMotionSpectrum",
    "SeparableMovieSpectrum",
    "StaticImageSpectrum",
    # Helpers
    "saccade_main_sequence_duration", "saccade_smoothing_sigma",
    "angular_spatial_frequency",
    # Back-compat free functions
    "image_spectrum", "drift_lorentzian", "drift_spectrum",
    "linear_motion_spectrum_gaussian",
    "temporal_lorentzian", "separable_movie_spectrum",
    "saccade_redistribution", "saccade_amplitude_average", "saccade_spectrum",
]
