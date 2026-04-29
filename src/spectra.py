"""Input power spectra C_theta(k, omega) for moving-sensor efficient coding.

All spectra are written for spatial frequency magnitude f = ||k|| (radial form).
The 2D spectrum is recovered by rotational symmetry where applicable.

Conventions
-----------
- f: spatial frequency magnitude [cycles per unit length]
- omega: temporal angular frequency [rad/sec]
- The spectra are two-sided in omega (defined on (-inf, inf)).
- Image spectrum normalization: A is amplitude, k0 is low-freq regularizer.

The drift Lorentzian preserves total spatial power per spatial frequency:
    integral over dω/(2π) of C_D(f,ω) = C_I(f).
The Gaussian linear-motion spectrum preserves it as well.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Static image spectrum
# ---------------------------------------------------------------------------

def image_spectrum(f, beta=2.0, A=1.0, k0=0.05):
    """Regularized power-law image spectrum.

    C_I(f) = A / (f^2 + k0^2)^(beta/2)

    Parameters
    ----------
    f : array_like
        Spatial frequency magnitude.
    beta : float
        Power-law exponent. Natural images: beta ~ 2.
    A : float
        Amplitude.
    k0 : float
        Low-frequency regularizer (avoids divergence at f=0).

    Returns
    -------
    C_I : ndarray
    """
    f = np.asarray(f, dtype=float)
    return A / (f ** 2 + k0 ** 2) ** (beta / 2)


# ---------------------------------------------------------------------------
# Brownian drift
# ---------------------------------------------------------------------------

def drift_lorentzian(f, omega, D):
    """Drift Lorentzian factor 2 D f^2 / ((D f^2)^2 + omega^2).

    Integrates to 1 over dω / (2π). For D == 0 returns 0 everywhere
    (the true limit is 2 pi delta(omega) which can't be represented on a grid;
    the user should handle the static case separately or use D very small).
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if D == 0:
        return np.zeros(np.broadcast_shapes(f.shape, omega.shape))
    Dk2 = D * f ** 2
    return 2.0 * Dk2 / (Dk2 ** 2 + omega ** 2)


def drift_spectrum(f, omega, D, beta=2.0, A=1.0, k0=0.05):
    """Brownian fixational drift spectrum (eq. 33).

    C_D(k, ω) = C_I(k) * 2 D ||k||^2 / ((D ||k||^2)^2 + ω^2)
    """
    return image_spectrum(f, beta, A, k0) * drift_lorentzian(f, omega, D)


# ---------------------------------------------------------------------------
# Linear motion with Gaussian velocity distribution
# ---------------------------------------------------------------------------

def linear_motion_spectrum_gaussian(f, omega, s, beta=2.0, A=1.0, k0=0.05,
                                    f_floor=1e-10):
    """Linear motion spectrum with isotropic Gaussian velocity distribution.

    With P(u) ~ N(0, s^2 I_2), the projected-velocity distribution onto k-direction
    is N(0, s^2). Equation (53) gives:

        C_s(f, ω) = (sqrt(2π) C_I(f) / (s f)) * exp(-ω^2 / (2 s^2 f^2))

    The factor 1/f comes from the |J| of the delta(ω - k·u) integral.

    Parameters
    ----------
    f, omega : array_like
    s : float
        Speed parameter (std of velocity in any direction).
    f_floor : float
        Numerical floor on f to avoid divide-by-zero at f=0. Choose much
        smaller than the smallest f of interest.
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)
    f_safe = np.maximum(f, f_floor)
    C_I = image_spectrum(f, beta, A, k0)
    sf = s * f_safe
    return np.sqrt(2.0 * np.pi) * C_I / sf * np.exp(-omega ** 2 / (2.0 * sf ** 2))


# ---------------------------------------------------------------------------
# Combined drift + linear motion
# ---------------------------------------------------------------------------

def combined_spectrum(f, omega, D, s, beta=2.0, A=1.0, k0=0.05, n_a=129,
                      n_sigma=6.0):
    """Combined drift + Gaussian linear motion (eq. 66, projected to 1D).

    Numerically integrates the Lorentzian over the projected Gaussian velocity:

        C_{D,s}(f, ω) = C_I(f) * ∫ da P_||(a) * 2 D f^2 / ((D f^2)^2 + (ω - f a)^2)

    where P_||(a) = N(0, s^2). Reduces to drift if s -> 0 and to Gaussian linear
    motion if D -> 0 (with appropriate care at the boundaries).

    Parameters
    ----------
    n_a : int
        Number of points in the velocity-projection grid.
    n_sigma : float
        Half-width of the velocity-projection grid in units of s.
    """
    f = np.asarray(f, dtype=float)
    omega = np.asarray(omega, dtype=float)

    if s == 0.0:
        return drift_spectrum(f, omega, D, beta, A, k0)
    if D == 0.0:
        # Gaussian limit (no drift): we re-derive directly to avoid issues.
        return linear_motion_spectrum_gaussian(f, omega, s, beta, A, k0)

    # Build velocity grid (Simpson's rule for smoothness)
    a = np.linspace(-n_sigma * s, n_sigma * s, n_a)
    da = a[1] - a[0]
    P_a = np.exp(-a ** 2 / (2.0 * s ** 2)) / (np.sqrt(2.0 * np.pi) * s)
    # Simpson weights (1, 4, 2, 4, ..., 2, 4, 1) * da / 3
    if n_a % 2 == 0:
        # need odd; trim the highest |a| point
        a = a[:-1]
        da = a[1] - a[0]
        P_a = P_a[:-1]
        n_a -= 1
    w = np.ones(n_a)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    w *= da / 3.0
    weights = w * P_a  # integration weights times the density (still need da)

    # Actually we already absorbed da into w. Let's clean up:
    # integrand_total = sum over i of (Lorentzian_i * P_a_i) * w_i,  w_i has units of da.

    # Broadcast: f and omega will be shape (...,), velocity grid (n_a,).
    f_b = f[..., None]
    omega_b = omega[..., None]
    a_b = a[None] * np.ones_like(f_b)  # broadcast

    Dk2 = D * f_b ** 2
    L = 2.0 * Dk2 / (Dk2 ** 2 + (omega_b - f_b * a) ** 2)
    integrand = L * weights
    integral = integrand.sum(axis=-1)

    return image_spectrum(f, beta, A, k0) * integral
