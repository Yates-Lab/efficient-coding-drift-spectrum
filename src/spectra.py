"""Input power spectra C_theta(k, omega) for moving-sensor efficient coding.

All spectra are written for spatial frequency magnitude f = ||k|| (radial form).
The 2D spectrum is recovered by rotational symmetry where applicable.

Two complementary APIs are provided:

1. Free functions (legacy). image_spectrum, drift_spectrum,
   linear_motion_spectrum_gaussian, combined_spectrum, saccade_template,
   saccade_redistribution, saccade_spectrum. These keep the original
   signatures so that existing tests and figures continue to work.

2. Spectrum classes. Each class stores its parameters and provenance and
   exposes a single C(f, omega) method that returns the on-retina
   spectrum on the given grid. Classes are the recommended API for new
   code. Available:
       StaticImageSpectrum            Field 1987
       DriftSpectrum                  Kuang et al. 2012, Aytekin et al. 2014
       LinearMotionSpectrum           Dong & Atick 1995
       SaccadeSpectrum                Mostofi et al. 2020
       DriftPlusSaccadeSpectrum       unified stationary; this work
       BoiCycleEarlySpectrum          Boi et al. 2017, early-fixation regime
       BoiCycleLateSpectrum           Boi et al. 2017, late-fixation regime

Conventions
-----------
- f: spatial frequency magnitude [cycles per unit length]
- omega: temporal angular frequency [rad/sec]
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

def drift_lorentzian(f, omega, D):
    """Drift Lorentzian factor 2 D f^2 / ((D f^2)^2 + omega^2).

    Integrates to 1 over dω / (2π). For D == 0 returns 0 everywhere.
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if D == 0:
        return np.zeros(np.broadcast_shapes(f.shape, omega.shape))
    Dk2 = D * f ** 2
    return 2.0 * Dk2 / (Dk2 ** 2 + omega ** 2)


def drift_spectrum(f, omega, D, beta=2.0, A=1.0, k0=0.05):
    """Brownian fixational drift spectrum.

    C_D(k, ω) = C_I(k) * 2 D ||k||^2 / ((D ||k||^2)^2 + ω^2)
    """
    return image_spectrum(f, beta, A, k0) * drift_lorentzian(f, omega, D)


# ---------------------------------------------------------------------------
# Linear motion with Gaussian velocity distribution (free function)
# ---------------------------------------------------------------------------

def linear_motion_spectrum_gaussian(f, omega, s, beta=2.0, A=1.0, k0=0.05,
                                    f_floor=1e-10):
    """Linear motion spectrum with isotropic Gaussian velocity distribution.

    C_s(f, ω) = (sqrt(2π) C_I(f) / (s f)) * exp(-ω^2 / (2 s^2 f^2))
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    f_safe = np.maximum(f, f_floor)
    C_I = image_spectrum(f, beta, A, k0)
    sf = s * f_safe
    return np.sqrt(2.0 * np.pi) * C_I / sf * np.exp(-omega ** 2 / (2.0 * sf ** 2))


# ---------------------------------------------------------------------------
# Saccades (free functions)
# ---------------------------------------------------------------------------

def saccade_template(t, peak_time=0.040, zeta=0.6):
    """Unit-amplitude saccade trajectory u(t) (dimensionless).

    Damped harmonic-oscillator step response, validated against the saccade
    traces in Mostofi, Zhao, Intoy, Boi, Victor & Rucci (2020) figure 5A.
    Settles at u → 1 for t much greater than peak_time, has a small
    overshoot at t = peak_time, and is identically 0 for t < 0.
    """
    t = np.asarray(t, dtype=float)
    if not (0 < zeta < 1):
        raise ValueError("zeta must be in (0, 1) for underdamped step response")
    omega_0 = np.pi / (peak_time * np.sqrt(1 - zeta ** 2))
    omega_d = omega_0 * np.sqrt(1 - zeta ** 2)
    out = np.zeros_like(t)
    active = t >= 0.0
    if active.any():
        decay = np.exp(-zeta * omega_0 * t[active])
        out[active] = 1.0 - decay * (
            np.cos(omega_d * t[active])
            + (zeta / np.sqrt(1 - zeta ** 2)) * np.sin(omega_d * t[active])
        )
    return out


def saccade_redistribution(f, omega, A, lam=3.0):
    """Saccade redistribution kernel Q_sac(f, ω; A, λ).

    Stationary Poisson-process model (closed-form):
        Q(k, ω) = 2 λ α(k) / ((λ α(k))² + ω²),
        α(k) = 1 - J_0(2π k A).
    """
    from scipy.special import j0

    f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()

    F = f_arr[:, None]
    W = omega_arr[None, :]
    alpha_k = 1.0 - j0(2.0 * np.pi * F * A)
    eff = lam * alpha_k
    Q = 2.0 * eff / (eff ** 2 + W ** 2)
    return Q


def saccade_spectrum(f, omega, A, lam=3.0, beta=2.0, A_image=1.0, k0=0.05):
    """Saccade-induced retinal input spectrum.

        C_sac(f, ω; A, λ) = C_I(f) · Q_sac(f, ω; A, λ)
    """
    f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    C_I = image_spectrum(f_arr, beta=beta, A=A_image, k0=k0)
    Q = saccade_redistribution(f_arr, omega, A, lam=lam)
    return C_I[:, None] * Q


def combined_spectrum(f, omega, D, s, beta=2.0, A=1.0, k0=0.05, n_a=129,
                      n_sigma=6.0):
    """Combined drift + Gaussian linear motion (eq. 66, projected to 1D).

    Numerically integrates the Lorentzian over the projected Gaussian velocity:

        C_{D,s}(f, ω) = C_I(f) * ∫ da P_||(a) * 2 D f^2 / ((D f^2)^2 + (ω - f a)^2)
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)

    if s == 0.0:
        return drift_spectrum(f, omega, D, beta, A, k0)
    if D == 0.0:
        return linear_motion_spectrum_gaussian(f, omega, s, beta, A, k0)

    a = np.linspace(-n_sigma * s, n_sigma * s, n_a)
    da = a[1] - a[0]
    P_a = np.exp(-a ** 2 / (2.0 * s ** 2)) / (np.sqrt(2.0 * np.pi) * s)
    if n_a % 2 == 0:
        a = a[:-1]
        da = a[1] - a[0]
        P_a = P_a[:-1]
        n_a -= 1
    w = np.ones(n_a)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= da / 3.0
    weights = w * P_a

    f_b = f[..., None]
    omega_b = omega[..., None]

    Dk2 = D * f_b ** 2
    L = 2.0 * Dk2 / (Dk2 ** 2 + (omega_b - f_b * a) ** 2)
    integrand = L * weights
    integral = integrand.sum(axis=-1)

    return image_spectrum(f, beta, A, k0) * integral


# ---------------------------------------------------------------------------
# Class-based API (recommended for new code)
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

    def image_factor(self, f) -> np.ndarray:
        return self.image.C(f)

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
class SaccadeSpectrum(Spectrum):
    """Stationary Poisson-process saccades (Mostofi et al. 2020; closed-form).

    Amplitude A in length units; rate lam in events / s.
    Drift limit (small kA): D_eff = π² λ A².
    """
    A: float = 2.5
    lam: float = 3.0
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "saccade")
        object.__setattr__(self, "reference", "Mostofi et al. 2020")

    @property
    def D_eff(self) -> float:
        """Drift-equivalent diffusion in the small-kA limit."""
        return np.pi ** 2 * self.lam * self.A ** 2

    def redistribution(self, f, omega) -> np.ndarray:
        return saccade_redistribution(f, omega, A=self.A, lam=self.lam)

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f_arr)[:, None] * self.redistribution(f_arr, omega)


@dataclass(frozen=True)
class DriftPlusSaccadeSpectrum(Spectrum):
    """Unified stationary spectrum for drift and Poisson saccades.

    Treating drift and saccade jumps as statistically independent
    displacement processes, the displacement autocorrelation factorizes
    and the cumulant rate adds inside the Lorentzian:

        Q_tot(k, ω) = 2 α_tot(k) / (α_tot(k)² + ω²),
        α_tot(k) = D k² + λ (1 - J_0(2π k A)).

    Reductions:
        - lam = 0 (or A = 0):   recovers DriftSpectrum exactly
        - D = 0:                recovers SaccadeSpectrum exactly
    """
    D: float = 1.0
    A: float = 2.5
    lam: float = 3.0
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "drift+saccade")
        object.__setattr__(self, "reference", "this work (unified stationary)")

    def alpha_tot(self, f) -> np.ndarray:
        from scipy.special import j0
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        drift_term = self.D * f_arr ** 2
        sacc_term = self.lam * (1.0 - j0(2.0 * np.pi * f_arr * self.A))
        return drift_term + sacc_term

    def redistribution(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        a = self.alpha_tot(f_arr)[:, None]
        w = omega_arr[None, :]
        return 2.0 * a / (a ** 2 + w ** 2)

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f_arr)[:, None] * self.redistribution(f_arr, omega)


# ---------------------------------------------------------------------------
# Boi et al. cycle spectra (non-stationary regime model)
# ---------------------------------------------------------------------------

def _windowed_saccade_redistribution(
    f, omega, A, T_win=0.512,
    peak_time=0.040, zeta=0.6, n_t=4096,
):
    """Windowed-FT redistribution for a single saccade event.

    Following Boi et al. 2017 supplementary procedure: each saccade is
    placed at the center of a T_win-second window in which the eye is
    otherwise immobile, and the spectrum of the resulting transient
    is computed.

    Angle-averaging the phase factor exp(-2πi k · A u(t) e_hat) over the
    saccade direction e_hat (uniform on the unit circle) gives
    J_0(2π k A u(t)). Subtracting the time-mean (which would only
    contribute to ω = 0, outside our band):

        Q_early(k, ω) = | FT_t [ J_0(2π k A u(t)) - <J_0> ] |^2 / T_win

    The result is non-negative by the |·|^2 construction, finite, and
    captures the post-saccade plateau (u → 1 holds inside the window
    after the saccade transient).

    Parameters
    ----------
    f : 1D array of spatial frequencies (cycles / length).
    omega : 1D centered uniform array of angular frequencies (rad / s).
    A : float, saccade amplitude (length units).
    T_win : float, window duration (s).
    peak_time, zeta : saccade template shape parameters.
    n_t : int, time-samples in the window. dt = T_win / n_t; pick large
          enough that 2π/(2 dt) covers omega.max() (Nyquist).

    Returns
    -------
    Q : ndarray, shape (len(f), len(omega))
    """
    from scipy.special import j0

    f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
    omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()

    # Time grid: dense, centered on saccade onset at t=0.
    dt = T_win / n_t
    # Place saccade onset at t=0, centered in window from -T_win/2 to T_win/2.
    t = (np.arange(n_t) - n_t // 2) * dt

    # Saccade trajectory u(t): 0 for t<0, transient toward 1 for t>0.
    u = saccade_template(t, peak_time=peak_time, zeta=zeta)  # (n_t,)

    # Angle-averaged phase factor: J_0(2π k A u(t)). Shape (Nf, n_t).
    j0_factor = j0(2.0 * np.pi * f_arr[:, None] * A * u[None, :])

    # Subtract time-mean (DC suppression: ω=0 carries the static-image
    # plateau, not a transient signal of interest).
    j0_factor = j0_factor - j0_factor.mean(axis=1, keepdims=True)

    # Centered-time FFT: shift to put t=0 at index 0 before FFT,
    # then fftshift in omega afterwards.
    g = np.fft.ifftshift(j0_factor, axes=1)
    G = np.fft.fft(g, axis=1) * dt  # FT in (rad/s) convention, per dt
    G = np.fft.fftshift(G, axes=1)

    # Native FFT omega grid (centered).
    domega_native = 2.0 * np.pi / (n_t * dt)
    omega_native = (np.arange(n_t) - n_t // 2) * domega_native

    # Power spectral density: |G|^2 / T_win has units of (length-units)^2
    # per (rad/s); multiplied later by C_I, gives the input spectrum.
    P_native = (np.abs(G) ** 2) / T_win  # (Nf, n_t)

    # Interpolate onto requested omega grid.
    Q = np.empty((f_arr.size, omega_arr.size), dtype=float)
    for i in range(f_arr.size):
        Q[i] = np.interp(omega_arr, omega_native, P_native[i],
                         left=0.0, right=0.0)
    np.maximum(Q, 0.0, out=Q)
    return Q


@dataclass(frozen=True)
class BoiCycleEarlySpectrum(Spectrum):
    """Boi et al. 2017 early-fixation regime: saccade-transient spectrum
    in a T_win-second window enclosing one saccade event.

    Default A = 4.4 deg matches the Boi et al. natural-viewing average.
    Default T_win = 0.512 s matches their isolation procedure; for a
    more focused early-fixation regime, use T_win ≈ 0.1-0.2 s.

    Note: this is a non-stationary regime model. It does NOT satisfy
    the same power-preserving identity as the stationary spectra.
    """
    A: float = 4.4
    T_win: float = 0.512
    peak_time: float = 0.040
    zeta: float = 0.6
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "boi_cycle_early")
        object.__setattr__(self, "reference", "Boi et al. 2017")

    def redistribution(self, f, omega) -> np.ndarray:
        return _windowed_saccade_redistribution(
            f, omega, A=self.A, T_win=self.T_win,
            peak_time=self.peak_time, zeta=self.zeta,
        )

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f_arr)[:, None] * self.redistribution(f_arr, omega)


@dataclass(frozen=True)
class BoiCycleLateSpectrum(Spectrum):
    """Boi et al. 2017 late-fixation regime: drift-only spectrum after the
    saccade transient has decayed. Identical to DriftSpectrum but tagged
    with the cycle reference for regime-comparison plots.
    """
    D: float = 1.0
    image: ImageParams = DEFAULT_IMAGE

    def __post_init__(self):
        object.__setattr__(self, "name", "boi_cycle_late")
        object.__setattr__(self, "reference", "Boi et al. 2017 (drift, Kuang 2012)")

    def redistribution(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        omega_arr = np.atleast_1d(np.asarray(omega, dtype=float)).ravel()
        return drift_lorentzian(f_arr[:, None], omega_arr[None, :], self.D)

    def C(self, f, omega) -> np.ndarray:
        f_arr = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        return self.image.C(f_arr)[:, None] * self.redistribution(f_arr, omega)


__all__ = [
    # Free functions (legacy)
    "image_spectrum",
    "drift_lorentzian",
    "drift_spectrum",
    "linear_motion_spectrum_gaussian",
    "saccade_template",
    "saccade_redistribution",
    "saccade_spectrum",
    "combined_spectrum",
    # Class API
    "ImageParams",
    "DEFAULT_IMAGE",
    "Spectrum",
    "StaticImageSpectrum",
    "DriftSpectrum",
    "LinearMotionSpectrum",
    "SaccadeSpectrum",
    "DriftPlusSaccadeSpectrum",
    "BoiCycleEarlySpectrum",
    "BoiCycleLateSpectrum",
]
