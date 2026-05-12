"""Signal power spectra for a moving retina.

Everything here uses public units:
    f      : cycles / degree
    tf_hz  : cycles / second = Hz

Angular factors appear only inside formulas:
    spatial phase = 2π f x
    temporal angular frequency = 2π tf_hz

The classes are intentionally direct.  To make a spectrum, instantiate the class
and call ``.C(f, tf_hz)``.
"""

from dataclasses import dataclass
import numpy as np
from scipy.special import j0

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class ImageParams:
    """Static natural-image spectrum C_I(f) = A/(f^2 + f0^2)^(beta/2)."""

    beta: float = 2.0
    A_image: float = 1.0
    f0: float = 1e-6

    def C(self, f):
        f = np.asarray(f, dtype=float)
        return self.A_image / (f * f + self.f0 * self.f0) ** (self.beta / 2.0)


@dataclass(frozen=True)
class DriftSpectrum:
    """Brownian fixational drift spectrum.

    D is the projected Brownian diffusion coefficient in deg^2/s.
    A commonly discussed human value is 40 arcmin^2/s = 40/3600 deg^2/s.

    For a spatial sinusoid with frequency f, Brownian drift decorrelates it with
    rate

        a(f) = D (2π f)^2     [rad/s].

    The temporal redistribution is the Lorentzian

        Q(f,ν) = 2a / (a^2 + (2πν)^2),

    which integrates to one under dν because dω/(2π) = dν.
    """

    D: float = 40.0 / 3600.0
    image: ImageParams = ImageParams()

    def redistribution(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        a = self.D * (TWOPI * f[:, None]) ** 2
        omega = TWOPI * tf_hz[None, :]
        return 2.0 * a / (a * a + omega * omega)

    def C(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f)[:, None] * self.redistribution(f, tf_hz)


# --- saccade duration helper -------------------------------------------------

SACCADE_SIGMA_DIVISOR = 5.5 * np.sqrt(TWOPI)


def saccade_main_sequence_duration(A_deg, base_s=0.021, slope_s_per_deg=0.0022):
    """Approximate saccade duration in seconds.

    This is the same simple Mostofi/Bahill-style linear main-sequence
    approximation used in earlier versions of the repo.
    """
    A_deg = np.asarray(A_deg, dtype=float)
    return base_s + slope_s_per_deg * A_deg


def saccade_smoothing_sigma(A_deg, base_s=0.021, slope_s_per_deg=0.0022):
    """Gaussian smoothing scale for a cumulative-Gaussian saccade trajectory."""
    return saccade_main_sequence_duration(A_deg, base_s, slope_s_per_deg) / SACCADE_SIGMA_DIVISOR


@dataclass(frozen=True)
class SaccadeSpectrum:
    """Analytic saccade-transient spectrum.

    A is saccade amplitude in degrees.  The model is a smoothed displacement
    step.  It is a finite-event spectrum used for an early post-saccadic epoch,
    not a globally stationary movie spectrum.

        Q(f,ν;A) = 2[1 - J0(2π f A)] exp[-(σ_A 2πν)^2] / (2πν)^2.
    """

    A: float = 3.5
    image: ImageParams = ImageParams()

    @property
    def sigma(self):
        return float(saccade_smoothing_sigma(self.A))

    def redistribution(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        spatial = 2.0 * (1.0 - j0(TWOPI * f[:, None] * self.A))
        omega = TWOPI * tf_hz[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            temporal = np.exp(-(self.sigma * omega) ** 2) / (omega * omega)
        temporal = np.where(np.isfinite(temporal), temporal, 0.0)
        return np.maximum(spatial * temporal, 0.0)

    def C(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f)[:, None] * self.redistribution(f, tf_hz)


@dataclass(frozen=True)
class LinearMotionSpectrum:
    """Stationary linear-motion control.

    s is the projected velocity scale in deg/s.  If a static image moves at a
    velocity v, spatial frequency f appears at temporal frequency ν = f v.
    With Gaussian velocities, this gives a Gaussian redistribution in ν with
    width s f.
    """

    s: float = 1.0
    image: ImageParams = ImageParams()
    f_floor: float = 1e-6

    def redistribution(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        sigma_nu = max(self.s, 1e-300) * np.maximum(f[:, None], self.f_floor)
        return (1.0 / (np.sqrt(2.0 * np.pi) * sigma_nu)) * np.exp(-0.5 * (tf_hz[None, :] / sigma_nu) ** 2)

    def C(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f)[:, None] * self.redistribution(f, tf_hz)


@dataclass(frozen=True)
class SeparableMovieSpectrum:
    """Stationary separable natural-movie control.

    This is the intentionally old-school approximation:

        C(f,ν) = C_I(f) / (|ν| + tf0_hz)^beta_t.
    """

    image: ImageParams = ImageParams()
    beta_t: float = 2.0
    tf0_hz: float = 0.05

    def C(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        temporal = 1.0 / (np.abs(tf_hz)[None, :] + self.tf0_hz) ** self.beta_t
        return self.image.C(f)[:, None] * temporal
