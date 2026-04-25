"""
Input spectra: static image C_I(k) and drift-induced C_D(k, omega).

All quantities dimensionless. See notes Section 3.
"""

import numpy as np


def C_I(k, A=1.0, beta=2.0, k0=0.0):
    """
    Static image power spectrum, regularized power law.
        C_I(k) = A / (||k||^2 + k0^2)^(beta/2)

    Parameters
    ----------
    k : array_like
        Spatial frequency magnitude ||k||. Any shape.
    A : float
        Overall amplitude.
    beta : float
        Spatial power-law exponent. beta=2 is the standard natural-image value.
    k0 : float
        Low-frequency regularizer. k0=0 gives the pure power law
        (diverges at k=0); use k0 > 0 for numerics.

    Returns
    -------
    Array with the same shape as k.
    """
    k = np.asarray(k, dtype=float)
    return A / (k**2 + k0**2) ** (beta / 2.0)


def C_D(k, omega, A=1.0, beta=2.0, D=1.0, k0=0.0):
    """
    Drift-induced retinal input spectrum (Eq. 29 / 33).
        C_D(k, omega) = C_I(k) * 2*D*k^2 / ((D*k^2)^2 + omega^2)

    Parameters
    ----------
    k, omega : array_like
        Must broadcast. Typical usage: k is shape (Nk, 1), omega is (1, Nw).
    A, beta, k0 : see C_I.
    D : float
        Diffusion constant.

    Returns
    -------
    Array of shape broadcast(k, omega).
    """
    k = np.asarray(k, dtype=float)
    omega = np.asarray(omega, dtype=float)

    # Handle k=0 carefully: with k0>0, C_I is finite; the Lorentzian numerator
    # is 2*D*k^2 which vanishes, and the denominator is omega^2, so the whole
    # thing is 0 except exactly at omega=0 where it's 0/0. We set k=0 -> 0
    # (the delta at omega=0 has measure zero on a finite grid).
    k2 = k**2
    Dk2 = D * k2
    lorentz = np.where(
        k2 > 0,
        2.0 * Dk2 / (Dk2**2 + omega**2),
        0.0,
    )
    return C_I(k, A=A, beta=beta, k0=k0) * lorentz


def C_D_integrated_over_omega(k, A=1.0, beta=2.0, k0=0.0):
    """
    Analytic integral of C_D(k, omega) over temporal frequency.
    By Eq. (31), int d omega/(2*pi) C_D(k, omega) = C_I(k) for all D.
    Used as a ground truth for the normalization check.
    """
    return C_I(k, A=A, beta=beta, k0=k0)
