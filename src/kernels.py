"""Recover causal temporal kernels and 2D spatial kernels from the optimized
filter magnitude |v*(k, ω)|.

The efficient-coding calculation determines |v*(k,ω)|^2; the temporal phase
is then recovered using the minimum-phase (causal) construction
(eqs. 24-25 of the appendix). The spatial part is taken as zero-phase
(rotationally symmetric, real receptive field profile).

Algorithms
----------
Temporal min-phase: cepstral method.
    1. Form log|V(ω)| on a uniform DFT grid.
    2. IFFT -> real cepstrum c[n].
    3. Apply causal "fold" window to get min-phase cepstrum.
    4. FFT -> log V_minphase(ω).
    5. exp -> V_minphase(ω); IFFT -> v(t), causal by construction.

Spatial: 2D inverse FFT of the (rotationally symmetric, zero-phase) magnitude
at fixed ω, then take a 1D radial cross-section for plotting.
"""

from __future__ import annotations

import numpy as np


__all__ = [
    "minimum_phase_log_filter",
    "minimum_phase_temporal_filter",
    "spatial_kernel_2d",
    "radial_cross_section",
    "soft_band_taper",
]


def soft_band_taper(omega, omega_min, omega_max, alpha=0.25):
    """A symmetric Tukey-style cosine taper on |ω| ∈ [omega_min, omega_max].

    The taper is exactly 1 in the central core, smoothly drops to 0 at the
    outer edges over a transition width of `alpha * omega_min` (low edge)
    and `alpha * omega_max` (high edge). Used to soften hard band cutoffs
    before cepstral minimum-phase reconstruction (which is unstable when
    log|H| has large negative spikes from a near-zero floor).

    Returns an array the same shape as omega. Combine with a finite floor
    (e.g. 1e-3 * peak) when feeding into log: replace
        v_t_smooth = soft_band_taper(omega, ωmin, ωmax) * v_t_raw
                     + floor * (1 - taper)
    or simply: v_t_smooth = max(taper * v_t_raw, floor).
    """
    abs_w = np.abs(omega)
    delta_lo = alpha * omega_min
    delta_hi = alpha * omega_max
    a = max(omega_min - delta_lo, 1e-9)
    b = omega_min + delta_lo
    c = omega_max - delta_hi
    d = omega_max + delta_hi
    out = np.zeros_like(omega, dtype=float)
    in_core = (abs_w >= b) & (abs_w <= c)
    out[in_core] = 1.0
    if delta_lo > 0:
        in_lo = (abs_w >= a) & (abs_w < b)
        out[in_lo] = 0.5 * (1.0 - np.cos(np.pi * (abs_w[in_lo] - a) / (b - a)))
    if delta_hi > 0:
        in_hi = (abs_w > c) & (abs_w <= d)
        out[in_hi] = 0.5 * (1.0 + np.cos(np.pi * (abs_w[in_hi] - c) / (d - c)))
    return out


# ---------------------------------------------------------------------------
# Minimum-phase temporal filter via cepstrum
# ---------------------------------------------------------------------------

def minimum_phase_log_filter(log_mag, axis=-1):
    """Convert a real log-magnitude array log|V(ω)| into the corresponding
    min-phase log-spectrum log V(ω) on the same DFT grid.

    The cepstral method (Oppenheim & Schafer): the cepstrum of a min-phase
    signal is causal, so we keep n=0, double 0<n<N/2, zero negative-time
    samples, then FFT back.

    `log_mag` should be sampled on a *DFT-ordered* grid (DC at index 0,
    positive frequencies first, then negative). If your data is centered
    (-ωmax ... +ωmax), call np.fft.ifftshift before passing in.
    """
    log_mag = np.asarray(log_mag, dtype=float)
    N = log_mag.shape[axis]
    cepstrum = np.fft.ifft(log_mag, axis=axis)

    # Build min-phase fold window in time domain.
    window = np.zeros(N)
    window[0] = 1.0
    if N % 2 == 0:
        window[1:N // 2] = 2.0
        window[N // 2] = 1.0  # Nyquist sample stays
    else:
        window[1:(N + 1) // 2] = 2.0

    shape = [1] * log_mag.ndim
    shape[axis] = N
    window = window.reshape(shape)

    minphase_cepstrum = cepstrum * window
    log_V = np.fft.fft(minphase_cepstrum, axis=axis)
    return log_V


def minimum_phase_temporal_filter(v_mag_centered, omega_centered, eps=1e-300):
    """Build the causal min-phase temporal impulse response for given |v(ω)|.

    Parameters
    ----------
    v_mag_centered : ndarray
        |v(ω)| sampled on a centered uniform grid (e.g. -ωmax..+ωmax,
        with ω=0 somewhere near the middle). Shape (..., N_omega).
    omega_centered : ndarray
        1D array of the centered ω samples (length N_omega, uniform).
    eps : float
        Floor used for log to keep |v| from going to -inf.

    Returns
    -------
    t : ndarray, shape (N_omega,)
        Time samples, t = 0 at index 0 (causal).
    v_t : ndarray, shape (..., N_omega)
        Causal real impulse response v(t).
    V_complex : ndarray, shape (..., N_omega)
        Complex frequency response V(ω) on the centered ω grid.
    """
    v_mag_centered = np.asarray(v_mag_centered, dtype=float)
    omega_centered = np.asarray(omega_centered, dtype=float)

    # Reorder centered grid to DFT order (DC at index 0).
    log_mag = np.log(np.maximum(v_mag_centered, eps))
    log_mag_dft = np.fft.ifftshift(log_mag, axes=-1)

    log_V_dft = minimum_phase_log_filter(log_mag_dft, axis=-1)
    V_dft = np.exp(log_V_dft)
    # IFFT -> time domain. Spacing in ω is dω, so dt = 2π/(N dω). The
    # continuous IFT integral is approximated by (dω/(2π)) * Σ V_n e^{i ω_n t_m}.
    # numpy IFFT gives (1/N) Σ V_n e^{2πi nm/N}. With ω_n = n dω, t_m = m dt,
    # ω_n t_m = 2π n m / N if dt = 2π/(N dω). So:
    #   Σ V_n e^{i ω_n t_m} = N * IFFT(V)_m
    # Thus continuous v(t_m) ≈ (dω/(2π)) * N * IFFT(V)_m = (1/dt) * IFFT(V)_m.
    dw = float(omega_centered[1] - omega_centered[0])
    N = omega_centered.size
    v_t = np.fft.ifft(V_dft, axis=-1) * (N * dw / (2.0 * np.pi))
    # Should be (numerically) real for min-phase filter.
    v_t = v_t.real

    dt = 2.0 * np.pi / (N * dw)
    t = np.arange(N) * dt

    # Reorder V back to centered ω grid for downstream plotting.
    V_complex = np.fft.fftshift(V_dft, axes=-1)
    return t, v_t, V_complex


# ---------------------------------------------------------------------------
# Spatial kernel
# ---------------------------------------------------------------------------

def spatial_kernel_2d(v_mag_radial_func, k_max, n_k=256):
    """Compute the 2D zero-phase spatial kernel from a rotationally symmetric
    magnitude |v(f)|.

    The continuous transform is
        v(r) = ∫ d²k/(2π)² V(||k||) e^{i k·r}.
    We approximate this by 2D IFFT on a centered (k_x, k_y) grid.

    Parameters
    ----------
    v_mag_radial_func : callable
        Takes f >= 0 and returns real-valued |v(f)|.
    k_max : float
        Half-extent of the (k_x, k_y) grid.
    n_k : int
        Grid size (per axis). Use even.

    Returns
    -------
    rx : ndarray, shape (n_k,)
    ry : ndarray, shape (n_k,)
    v_xy : ndarray, shape (n_k, n_k)
        The 2D real-space kernel, v_xy[i,j] at position (rx[j], ry[i]).
    """
    if n_k % 2:
        raise ValueError("n_k must be even.")
    dk = 2.0 * k_max / n_k
    # Centered grid (k=0 included if n_k is even at index n_k//2)
    kx = (np.arange(n_k) - n_k // 2) * dk
    ky = kx.copy()
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    F = np.sqrt(KX ** 2 + KY ** 2)
    V = v_mag_radial_func(F)  # zero-phase, real

    # Continuous-IFT scaling: v(r) ≈ (Δk^2/(2π)^2) * Σ V e^{i k·r}.
    # numpy ifft2 = (1/N²) Σ ..., so multiply by N² Δk² / (2π)².
    V_dft = np.fft.ifftshift(V)
    v_dft = np.fft.ifft2(V_dft)
    v_xy = np.fft.fftshift(v_dft).real
    scale = (n_k * dk / (2.0 * np.pi)) ** 2
    v_xy = v_xy * scale

    # Spatial coordinates: dr = 2π / (n_k dk)
    dr = 2.0 * np.pi / (n_k * dk)
    rx = (np.arange(n_k) - n_k // 2) * dr
    ry = rx.copy()
    return rx, ry, v_xy


def radial_cross_section(v_xy, rx, ry):
    """Take the y=0 horizontal slice through a 2D kernel."""
    iy0 = np.argmin(np.abs(ry))
    return rx, v_xy[iy0]
