"""
Information rate and mutual information (Section 3.5).

Per-cell information density (Eq. 54):
    I(k, omega) = log [ (|v|^2 (C_x + sigma_in^2) + sigma_out^2)
                        / (|v|^2 sigma_in^2 + sigma_out^2) ].

Total information (Eq. 55):
    I = integral over (k, omega) of I(k, omega), with the Fourier measure
    absorbed into the grid weights.

We drop convention-dependent constants as in the notes.
"""

import numpy as np

from optimizer import solve_lambda, v_star_sq


def info_density(v_mag_sq, Cx, sigma_in_sq, sigma_out_sq):
    """
    Per-cell information density from Eq. (54).

    Parameters
    ----------
    v_mag_sq, Cx : ndarray
        Same shape.
    sigma_in_sq, sigma_out_sq : float
    """
    num = v_mag_sq * (Cx + sigma_in_sq) + sigma_out_sq
    den = v_mag_sq * sigma_in_sq + sigma_out_sq
    return np.log(num / den)


def total_info(v_mag_sq, Cx, sigma_in_sq, sigma_out_sq, weights):
    """Total information, given weights that encode the Fourier measure."""
    I = info_density(v_mag_sq, Cx, sigma_in_sq, sigma_out_sq)
    return np.sum(I * weights)


def info_vs_drift(
    C_D_of_D,
    D_values,
    sigma_in_sq,
    sigma_out_sq,
    P_target,
    weights,
    verbose=False,
):
    """
    Sweep diffusion constant D and return (lambda(D), I(D)).

    Parameters
    ----------
    C_D_of_D : callable D -> ndarray
        Function returning the input power spectrum on the current grid.
    D_values : array_like
    sigma_in_sq, sigma_out_sq, P_target : float
    weights : ndarray
        Grid weights matching the output of C_D_of_D.

    Returns
    -------
    lambdas : (Nd,) array
    infos : (Nd,) array
    """
    lambdas = np.zeros(len(D_values))
    infos = np.zeros(len(D_values))
    for i, D in enumerate(D_values):
        Cx = C_D_of_D(D)
        lam = solve_lambda(Cx, sigma_in_sq, sigma_out_sq, P_target, weights)
        v2 = v_star_sq(Cx, sigma_in_sq, sigma_out_sq, lam)
        I = total_info(v2, Cx, sigma_in_sq, sigma_out_sq, weights)
        lambdas[i] = lam
        infos[i] = I
        if verbose:
            print(f"  D={D:>10.4g}  lam={lam:>12.4g}  I={I:>12.4g}")
    return lambdas, infos
