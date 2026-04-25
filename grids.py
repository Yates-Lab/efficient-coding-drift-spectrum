"""
Frequency grids and their integration weights.

Conventions: Fourier integrals carry (2 pi)^-1 per dimension (see Eq. 2).
We discretize int f(k) d^2k/(2pi)^2 as sum_ij f(k_ij) * (dk_x dk_y) / (2pi)^2.

Two grid types:
  - Radial 1D: assumes isotropy. Integral is 2*pi * int f(k) k dk / (2pi)^2.
  - Cartesian 2D: full (kx, ky) grid covering a square Brillouin zone.

Plus a shared temporal grid over omega in [-wmax, wmax].
"""

import numpy as np


# ---------- Temporal grid ----------

def omega_grid(wmax, Nw):
    """
    Symmetric temporal frequency grid in [-wmax, wmax], Nw points.

    Returns
    -------
    omega : (Nw,) array
    dw_over_2pi : float
        Integration weight per sample: d omega / (2 pi).
    """
    omega = np.linspace(-wmax, wmax, Nw)
    dw = omega[1] - omega[0]
    return omega, dw / (2.0 * np.pi)


# ---------- Radial 1D spatial grid ----------

def radial_grid(kmax, Nk, kmin=0.0):
    """
    Radial spatial-frequency grid k in [kmin, kmax], Nk points.
    Integration weight includes the polar Jacobian k and the (2 pi)^-2 factor:
        int f(k) d^2k/(2pi)^2 = int_0^inf f(k) * k dk / (2 pi).

    Returns
    -------
    k : (Nk,) array
    weight_k : (Nk,) array
        Weight such that sum(f * weight_k) approximates the 2D isotropic
        spatial integral.
    """
    k = np.linspace(kmin, kmax, Nk)
    dk = k[1] - k[0]
    weight_k = k * dk / (2.0 * np.pi)
    return k, weight_k


def build_radial_weights(weight_k, dw_over_2pi):
    """
    Outer-product weights for a (Nk, Nw) radial grid.
    weights[i, j] approximates dk * domega /(2pi)^2 * k_i (polar measure).
    """
    return weight_k[:, None] * dw_over_2pi


# ---------- 2D Cartesian spatial grid ----------

def cartesian_grid(kmax, Nk):
    """
    Square (kx, ky) grid on [-kmax, kmax]^2, Nk x Nk points.
    Treated as the Brillouin zone G_0 when we ignore aliasing, with
    lattice constant a = pi / kmax.

    Returns
    -------
    kx, ky : (Nk,) arrays
    Kmag : (Nk, Nk) array
        ||k|| at each grid point.
    weight_xy : float
        Per-cell weight dk_x dk_y / (2 pi)^2.
    """
    kx = np.linspace(-kmax, kmax, Nk)
    ky = np.linspace(-kmax, kmax, Nk)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    Kmag = np.sqrt(KX**2 + KY**2)
    dkx = kx[1] - kx[0]
    dky = ky[1] - ky[0]
    weight_xy = dkx * dky / (2.0 * np.pi) ** 2
    return kx, ky, Kmag, weight_xy


def build_cartesian_weights(weight_xy, dw_over_2pi, shape_kk):
    """
    Weights for a (Nkx, Nky, Nw) Cartesian grid.
    Scalar weight_xy broadcast with the temporal dw/(2pi).
    Returns a 3D array of shape (Nkx, Nky, Nw).
    """
    Nkx, Nky = shape_kk
    w = np.full((Nkx, Nky, 1), weight_xy) * dw_over_2pi
    return w
