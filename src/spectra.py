"""Input power spectra C_theta(f, omega) for moving-sensor efficient coding.

All spectra are written for spatial frequency magnitude f in cycles/degree
(radial form). The 2D spectrum is recovered by rotational symmetry where
applicable.

Two complementary APIs are provided:

1. Free functions. image_spectrum, drift_spectrum,
   linear_motion_spectrum_gaussian, saccade_redistribution, saccade_spectrum.

2. Spectrum classes. Each class stores its parameters and provenance and
   exposes a single C(f, omega) method that returns the on-retina
   spectrum on the given grid. Classes are the recommended API for new
   code. Available:
       DriftSpectrum                  Kuang et al. 2012, Aytekin et al. 2014
       SaccadeSpectrum                cumulative-Gaussian saccade approximation
       SeparableMovieSpectrum         Dong & Atick stationary control
       LinearMotionSpectrum           Dong & Atick 1995

Conventions
-----------
- f: spatial frequency magnitude [cycles/degree]
- omega: temporal angular frequency [rad/sec]
- D: Brownian drift diffusion coefficient [deg^2/sec]
- A: saccade amplitude [deg]
- s: linear velocity standard deviation [deg/sec]
- The spectra are two-sided in omega (defined on (-inf, inf)).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Static image spectrum (free function)
# ---------------------------------------------------------------------------

def image_spectrum(f, beta=2.0, A=1.0, k0=0.05):
    """Regularized power-law image spectrum.

    C_I(f) = A / (f^2 + k0^2)^(beta/2)
    """
    f = np.asarray(f, dtype=float)
    return A / (f ** 2 + k0 ** 2) ** (beta / 2)


# ---------------------------------------------------------------------------
# Brownian drift (free functions)
# ---------------------------------------------------------------------------

TWOPI = 2.0 * np.pi


def angular_spatial_frequency(f):
    """Convert spatial frequency from cycles/degree to radians/degree."""
    return TWOPI * np.asarray(f, dtype=float)


def drift_lorentzian(f, omega, D):
    """Brownian drift redistribution for f in cycles/degree.

    The Brownian phase width is a = D * (2*pi*f)^2 because the image phase is
    2*pi*f*x when f is measured in cycles/degree and x in degrees:

        Q_drift(f, omega) = 2 a / (a^2 + omega^2).

    Integrates to 1 over dω / (2π). For D == 0 returns 0 everywhere.
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if D == 0:
        return np.zeros(np.broadcast_shapes(f.shape, omega.shape))
    a = float(D) * angular_spatial_frequency(f) ** 2
    return 2.0 * a / (a ** 2 + omega ** 2)


def drift_spectrum(f, omega, D, beta=2.0, A=1.0, k0=0.05):
    """Brownian fixational drift spectrum.

    C_D(f, ω) = C_I(f) * Q_drift(f, ω)
    """
    return image_spectrum(f, beta, A, k0) * drift_lorentzian(f, omega, D)


# ---------------------------------------------------------------------------
# Linear motion with Gaussian velocity distribution (free function)
# ---------------------------------------------------------------------------

def linear_motion_spectrum_gaussian(f, omega, s, beta=2.0, A=1.0, k0=0.05,
                                    f_floor=1e-10):
    """Linear motion spectrum with isotropic Gaussian velocity distribution.

    Velocity ``s`` is in degrees/sec and f is in cycles/degree, so the temporal
    angular scale is s * 2*pi*f.
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    k_safe = np.maximum(np.abs(angular_spatial_frequency(f)), f_floor)
    C_I = image_spectrum(f, beta, A, k0)
    sf = float(s) * k_safe
    return np.sqrt(2.0 * np.pi) * C_I / sf * np.exp(-omega ** 2 / (2.0 * sf ** 2))


# ---------------------------------------------------------------------------
# Stationary separable movie approximation
# ---------------------------------------------------------------------------

def temporal_lorentzian(omega, omega0=10.0):
    """Normalized temporal 1/omega^2-tail spectrum.

    Q_T(omega) = 2 omega0 / (omega0^2 + omega^2)

    integrates to one over d omega / (2*pi). Multiplying by the static image
    spectrum gives the deliberately separable stationary control

        C_sep(f, omega) = C_I(f) Q_T(omega).

    This is the clean control for the older approximation that treats natural
    movies as having independent spatial and temporal power-law factors. It is
    not intended as a faithful eye-movement spectrum.
    """
    omega = np.asarray(omega, dtype=float)
    omega0 = float(omega0)
    if omega0 <= 0:
        raise ValueError("omega0 must be positive")
    return 2.0 * omega0 / (omega0 ** 2 + omega ** 2)


def temporal_power_law(omega, omega0=0.05, gamma=2.0):
    """Regularized temporal power law for passive natural-movie controls.

    The paper-style stationary movie approximation is proportional to
    ``1 / |omega|^gamma``.  ``omega0`` is only a low-frequency numerical floor
    to avoid the singularity at zero; choose it below the represented temporal
    band when the plotted/fit spectrum should look like a pure power law.
    """
    omega = np.asarray(omega, dtype=float)
    omega0 = float(omega0)
    gamma = float(gamma)
    if omega0 <= 0:
        raise ValueError("omega0 must be positive")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    return 1.0 / np.maximum(np.abs(omega), omega0) ** gamma


def separable_movie_spectrum(
    f,
    omega,
    omega0=0.05,
    beta=2.0,
    A=1.0,
    k0=0.05,
    temporal_beta=2.0,
):
    """Stationary separable natural-movie power-law spectrum.

    Implements the passive natural-movie approximation

        S(k, omega) proportional to 1 / (|k|^beta |omega|^temporal_beta).

    ``k0`` and ``omega0`` are low-frequency floors for numerical stability,
    not scale parameters of the represented spectrum.  Set them below the
    analysis band to recover straight diagonal log-log contours.
    """
    f = np.asarray(f, dtype=float)
    spatial = A / np.maximum(np.abs(f), float(k0)) ** float(beta)
    temporal = temporal_power_law(omega, omega0=omega0, gamma=temporal_beta)
    return spatial * temporal


# ---------------------------------------------------------------------------
# Saccades (free functions)
# ---------------------------------------------------------------------------

def saccade_main_sequence_duration(A, base_s=0.021, slope_s_per_deg=0.0022):
    """Approximate saccade duration in seconds from amplitude in degrees."""
    A = np.asarray(A, dtype=float)
    return float(base_s) + float(slope_s_per_deg) * A


def saccade_smoothing_sigma(A, duration_divisor=8.0):
    """Gaussian smoothing sigma for the cumulative-Gaussian transient model."""
    return saccade_main_sequence_duration(A) / float(duration_divisor)


def saccade_redistribution(
    f,
    omega,
    A,
    *,
    duration_divisor: float = 8.0,
    omega_floor: float = 1e-15,
) -> np.ndarray:
    """Cumulative-Gaussian saccade-transient redistribution.

    The approximation treats a saccade as a cumulative-Gaussian-smoothed step.
    For a fixed amplitude A,

        Q_sac(f, omega; A) =
            2 [1 - J_0(2 pi f A)] exp[-(omega sigma(A))^2] / omega^2.

    The spatial term is the orientation-averaged power of a displacement step;
    the temporal term is the smoothed-step power spectrum.  This is not a
    stationary Poisson jump model and is not power-normalized like Brownian
    drift.
    """
    from scipy.special import j0

    f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
    A = float(A)
    if A == 0.0:
        return np.zeros((f_arr.size, omega_arr.size), dtype=float)
    spatial = 2.0 * (1.0 - j0(2.0 * np.pi * f_arr[:, None] * A))
    sigma = float(saccade_smoothing_sigma(A, duration_divisor=duration_divisor))
    w = np.abs(omega_arr)[None, :]
    temporal = np.exp(-(w * sigma) ** 2) / np.maximum(w ** 2, float(omega_floor) ** 2)
    return np.maximum(spatial * temporal, 0.0)


def saccade_amplitude_average(
    f,
    omega,
    amplitudes,
    *,
    weights=None,
    duration_divisor: float = 8.0,
    omega_floor: float = 1e-15,
) -> np.ndarray:
    """Average the saccade transient approximation over amplitudes."""
    from scipy.special import j0

    f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
    A = np.atleast_1d(np.asarray(amplitudes, dtype=float)).ravel()
    if A.size == 0:
        raise ValueError("amplitudes must contain at least one value")
    if weights is None:
        w_amp = np.full(A.size, 1.0 / A.size, dtype=float)
    else:
        w_amp = np.asarray(weights, dtype=float).ravel()
        if w_amp.size != A.size:
            raise ValueError("weights must have the same length as amplitudes")
        if np.any(w_amp < 0.0) or w_amp.sum() <= 0.0:
            raise ValueError("weights must be nonnegative and sum positive")
        w_amp = w_amp / w_amp.sum()

    spatial = 2.0 * (1.0 - j0(2.0 * np.pi * f_arr[:, None] * A[None, :]))
    sigma = saccade_smoothing_sigma(A, duration_divisor=duration_divisor)
    om = np.abs(omega_arr)[None, :]
    temporal = np.exp(-(sigma[:, None] * om) ** 2) / np.maximum(
        om ** 2, float(omega_floor) ** 2
    )
    return np.maximum((spatial * w_amp[None, :]) @ temporal, 0.0)


def saccade_spectrum(f, omega, A, beta=2.0, A_image=1.0, k0=0.05):
    """Saccade-induced retinal input spectrum.

        C_sac(f, ω; A) = C_I(f) · Q_sac(f, ω; A)
    """
    f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    C_I = image_spectrum(f_arr, beta=beta, A=A_image, k0=k0)
    Q = saccade_redistribution(f_arr, omega, A)
    return C_I[:, None] * Q


# ---------------------------------------------------------------------------
# Class-based API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageParams:
    """Parameters of the regularized power-law image spectrum.

    C_I(f) = A_image / (f^2 + k0^2)^(beta/2)
    """
    beta: float = 2.0
    A_image: float = 1.0
    k0: float = 0.05

    def C(self, f) -> np.ndarray:
        return image_spectrum(f, beta=self.beta, A=self.A_image, k0=self.k0)


DEFAULT_IMAGE = ImageParams()


@dataclass(frozen=True)
class Spectrum(ABC):
    """Base class for on-retina input spectra C_θ(f, ω).

    Concrete subclasses store their parameters and implement C(f, omega).
    Subclasses also typically expose redistribution(f, omega) returning
    the Q kernel separately from the image factor, when that decomposition
    is meaningful for the model.

    Subclasses are frozen dataclasses: parameter values are immutable
    once constructed, which keeps results reproducible from the
    Spectrum instance alone.
    """
    name: str = field(init=False)
    reference: str = field(init=False)

    @abstractmethod
    def C(self, f, omega) -> np.ndarray:
        """Input spectrum on the (Nf, Nω) grid spanned by 1D arrays f, omega."""

    def describe(self) -> str:
        """Human-readable parameter summary."""
        from dataclasses import fields
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


@dataclass(frozen=True)
class StaticImageSpectrum(Spectrum):
    """Static image (no eye movement). Provided for the spatial-only image
    factor; C(f, ω) raises since the static case is δ(ω) on a discrete grid.
    """
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "static_image")
        object.__setattr__(self, "reference", "Field 1987")

    def C(self, f, omega) -> np.ndarray:
        raise NotImplementedError(
            "Static image has no spatiotemporal spectrum on a finite ω grid; "
            "use a movement model (DriftSpectrum, SaccadeSpectrum, ...)."
        )


@dataclass(frozen=True)
class DriftSpectrum(Spectrum):
    """Brownian-drift fixational eye movement (Aytekin et al. 2014; Kuang
    et al. 2012). Diffusion coefficient D in (length unit)^2 / s.
    """
    D: float = 1.0
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "drift")
        object.__setattr__(self, "reference", "Kuang et al. 2012")

    def redistribution(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        return drift_lorentzian(f_arr[:, None], omega_arr[None, :], self.D)

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f_arr)[:, None] * self.redistribution(f_arr, omega)


@dataclass(frozen=True)
class LinearMotionSpectrum(Spectrum):
    """Linear motion with Gaussian-distributed isotropic velocity
    (Dong & Atick 1995). Speed s = std of velocity in any direction.
    """
    s: float = 1.0
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "linear_motion_gaussian")
        object.__setattr__(self, "reference", "Dong & Atick 1995")

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        return linear_motion_spectrum_gaussian(
            f_arr[:, None], omega_arr[None, :], s=self.s,
            beta=self.image.beta, A=self.image.A_image, k0=self.image.k0,
        )


@dataclass(frozen=True)
class SeparableMovieSpectrum(Spectrum):
    """Stationary separable natural-movie approximation.

    This is the explicit control for the approximation used by passive-movie
    efficient-coding models that factor the spectrum into independent spatial
    and temporal power laws:

        C_sep(f, omega) = C_I(f) Q_T(omega).

    This follows the paper-style power law S(k, omega) proportional to
    1/(|k|^2 |omega|^2).  The low-frequency floors are numerical cutoffs for
    the singularity, not fitted scales.
    """
    omega0: float = 0.05
    temporal_beta: float = 2.0
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "separable_movie")
        object.__setattr__(self, "reference", "stationary separable movie control")

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return separable_movie_spectrum(
            f_arr[:, None],
            np.atleast_1d(np.asarray(omega, dtype=float)).ravel()[None, :],
            omega0=self.omega0,
            beta=self.image.beta,
            A=self.image.A_image,
            k0=self.image.k0,
            temporal_beta=self.temporal_beta,
        )


@dataclass(frozen=True)
class SaccadeSpectrum(Spectrum):
    """Cumulative-Gaussian analytic saccade-transient approximation.

    Amplitude A is in degrees or the same spatial unit as f.
    """
    A: float = 2.5
    duration_divisor: float = 8.0
    omega_floor: float = 1e-15
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "saccade")
        object.__setattr__(self, "reference", "cumulative-Gaussian saccade approximation")

    @property
    def D_eff(self) -> float:
        """No stationary drift-equivalent diffusion exists for this transient."""
        return np.nan

    def redistribution(self, f, omega) -> np.ndarray:
        return saccade_redistribution(
            f,
            omega,
            A=self.A,
            duration_divisor=self.duration_divisor,
            omega_floor=self.omega_floor,
        )

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f_arr)[:, None] * self.redistribution(f_arr, omega)


__all__ = [
    # Free functions
    "TWOPI",
    "angular_spatial_frequency",
    "image_spectrum",
    "drift_lorentzian",
    "drift_spectrum",
    "linear_motion_spectrum_gaussian",
    "temporal_lorentzian",
    "temporal_power_law",
    "separable_movie_spectrum",
    "saccade_main_sequence_duration",
    "saccade_smoothing_sigma",
    "saccade_redistribution",
    "saccade_amplitude_average",
    "saccade_spectrum",
    # Class API
    "ImageParams",
    "DEFAULT_IMAGE",
    "Spectrum",
    "StaticImageSpectrum",
    "DriftSpectrum",
    "LinearMotionSpectrum",
    "SeparableMovieSpectrum",
    "SaccadeSpectrum",
]
