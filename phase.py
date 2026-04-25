"""
Minimum-phase reconstruction of the causal temporal filter (Sec. 2.6).

For each spatial frequency k, the optimization determines only |v(k, omega)|.
The causal filter with minimum delay has phase
    phi(k, omega) = -H_omega[log |v(k, omega)|],
where H is the Hilbert transform over temporal frequency.

Implementation notes:
  - log |v| is -inf on the inactive set. We add a small floor before the log
    to regularize; the floor value determines only the phase outside the
    active set (where |v| is already 0 so the filter is unaffected).
  - We use scipy.signal.hilbert, which returns the analytic signal
    x + i * H[x]. So Im(hilbert(log|v|)) = H[log|v|], and we negate to get phi.
  - The transform acts along the omega axis. We accept v2 of shape (..., Nw).
  - To reduce end-effects we zero-pad in omega before transforming.
"""

import numpy as np
from scipy.signal import hilbert


def minimum_phase(v_mag_sq, floor_rel=1e-10, pad_factor=2):
    """
    Given |v|^2 sampled on (..., Nw) omega axis, return the complex filter
    v(k, omega) = |v| exp(i phi) with minimum phase.

    Parameters
    ----------
    v_mag_sq : ndarray
        Nonnegative array, last axis is omega.
    floor_rel : float
        Relative floor added to |v|^2 before log, as a fraction of max|v|^2.
        Prevents log(0) on the inactive set. Set small.
    pad_factor : int
        The omega axis is zero-padded to pad_factor * Nw before the Hilbert
        transform to mitigate wrap-around. Set to 1 to disable.

    Returns
    -------
    v_complex : ndarray
        Same shape as v_mag_sq, complex dtype.
    """
    v_mag_sq = np.asarray(v_mag_sq, dtype=float)
    v_mag = np.sqrt(v_mag_sq)

    # Regularize the log.
    max_v2 = v_mag_sq.max()
    if max_v2 == 0:
        return np.zeros_like(v_mag_sq, dtype=complex)
    floor = floor_rel * max_v2
    log_mag = 0.5 * np.log(v_mag_sq + floor)

    # Zero-pad along omega (last axis).
    Nw = log_mag.shape[-1]
    if pad_factor > 1:
        pad_width = [(0, 0)] * (log_mag.ndim - 1) + [(0, Nw * (pad_factor - 1))]
        # Pad with the mean of log_mag so the Hilbert transform sees a smooth
        # extension rather than a sudden drop.
        edge_val = log_mag.mean(axis=-1, keepdims=True)
        padded = np.concatenate(
            [log_mag] + [np.broadcast_to(edge_val, log_mag.shape[:-1] + (Nw * (pad_factor - 1),))],
            axis=-1,
        )
    else:
        padded = log_mag

    analytic = hilbert(padded, axis=-1)
    H_log_mag = np.imag(analytic)[..., :Nw]

    # Sign convention: the notes use f(t) = int dw/(2pi) exp(+i omega t) f(w).
    # With this convention, a causal filter has v(w) analytic in Im(w) < 0.
    # scipy.signal.hilbert uses the exp(-i omega t) convention internally,
    # so its analytic signal has the "wrong" sign for us. Negating gives the
    # correct minimum phase.
    phi = -H_log_mag
    return v_mag * np.exp(1j * phi)


def ifft_to_time(v_complex, omega):
    """
    Inverse Fourier transform v(omega) -> v(t) along the last axis.

    Uses the convention of Eq. (2):
        f(t) = int d omega / (2 pi) exp(i omega t) f(omega).

    Parameters
    ----------
    v_complex : ndarray
        Filter sampled on the omega grid. Last axis is omega.
    omega : (Nw,) array
        Must be evenly spaced and symmetric around 0.

    Returns
    -------
    t : (Nw,) array
    v_time : ndarray
        Same leading shape as v_complex, last axis is t.
    """
    Nw = omega.shape[-1]
    dw = omega[1] - omega[0]
    assert np.isclose(omega[0], -omega[-1], atol=1e-10 * abs(omega[-1])), \
        "omega grid must be symmetric around 0 for this IFT."

    # Standard discrete inverse FFT with the Eq. 2 convention.
    # Re-order omega from [-wmax, ..., wmax] to FFT order [0, pos, neg].
    v_shifted = np.fft.ifftshift(v_complex, axes=-1)
    v_time_shifted = np.fft.ifft(v_shifted, axis=-1) * (Nw * dw) / (2.0 * np.pi)
    v_time = np.fft.fftshift(v_time_shifted, axes=-1)

    t_max = np.pi / dw
    t = np.linspace(-t_max, t_max, Nw, endpoint=False)
    return t, v_time
