"""Kernel reconstruction from filter magnitude.

The solver gives only |v(f,ν)|².  The spatial phase is chosen to be zero so the
spatial kernel is centered.  The temporal phase is chosen by the minimum-phase
Hilbert/cepstrum construction, giving a causal filter with minimal delay.
"""

import numpy as np


def soft_band_taper(tf_hz, tf_min_hz, tf_max_hz, alpha=0.15):
    """Smoothly taper a centered temporal-frequency grid to the band."""
    tf_abs = np.abs(np.asarray(tf_hz, dtype=float))
    out = np.zeros_like(tf_abs)
    lo0 = max(tf_min_hz * (1.0 - alpha), 1e-12)
    lo1 = tf_min_hz * (1.0 + alpha)
    hi0 = tf_max_hz * (1.0 - alpha)
    hi1 = tf_max_hz * (1.0 + alpha)
    core = (tf_abs >= lo1) & (tf_abs <= hi0)
    out[core] = 1.0
    low = (tf_abs >= lo0) & (tf_abs < lo1)
    if np.any(low):
        out[low] = 0.5 * (1 - np.cos(np.pi * (tf_abs[low] - lo0) / (lo1 - lo0)))
    high = (tf_abs > hi0) & (tf_abs <= hi1)
    if np.any(high):
        out[high] = 0.5 * (1 + np.cos(np.pi * (tf_abs[high] - hi0) / (hi1 - hi0)))
    return out


def _minimum_phase_log_spectrum(log_mag_dft_order):
    """Minimum-phase log spectrum from log magnitude in DFT order."""
    n = log_mag_dft_order.size
    cep = np.fft.ifft(log_mag_dft_order)
    lifter = np.zeros(n)
    lifter[0] = 1.0
    if n % 2 == 0:
        lifter[1:n // 2] = 2.0
        lifter[n // 2] = 1.0
    else:
        lifter[1:(n + 1) // 2] = 2.0
    return np.fft.fft(cep * lifter)


def minimum_phase_temporal_filter(v_mag_centered, tf_hz_centered, eps=1e-12):
    """Return t, h(t), complex spectrum from |V(ν)| on a uniform centered grid."""
    tf_hz_centered = np.asarray(tf_hz_centered, dtype=float)
    v_mag_centered = np.asarray(v_mag_centered, dtype=float)
    dnu = tf_hz_centered[1] - tf_hz_centered[0]
    if not np.allclose(np.diff(tf_hz_centered), dnu, rtol=1e-4, atol=1e-10):
        raise ValueError("tf_hz_centered must be uniformly spaced")
    n = tf_hz_centered.size
    log_mag = np.log(np.maximum(v_mag_centered, eps))
    log_mag_dft = np.fft.ifftshift(log_mag)
    log_V_dft = _minimum_phase_log_spectrum(log_mag_dft)
    V_dft = np.exp(log_V_dft)
    h = (np.fft.ifft(V_dft) * n * abs(dnu)).real
    t = np.arange(n) / (n * abs(dnu))
    V_centered = np.fft.fftshift(V_dft)
    return t, h, V_centered

def temporal_kernel_slice(
    f,
    tf_hz,
    v_sq,
    f0,
    *,
    tf_min_hz=0.01,
    tf_max_hz=120.0,
    n_uniform=2048,
    floor_rel=1e-4,
):
    """Minimum-phase temporal kernel at the spatial bin nearest f0."""

    # Convert the spatial-frequency grid to a NumPy array.
    f = np.asarray(f, dtype=float)

    # Convert the temporal-frequency grid to a NumPy array.
    tf_hz = np.asarray(tf_hz, dtype=float)

    # Convert the filter power array |v(f, tf)|^2 to a NumPy array.
    v_sq = np.asarray(v_sq, dtype=float)

    # Find the spatial-frequency bin closest to the requested f0.
    i = int(np.argmin(np.abs(f - float(f0))))

    # Keep only positive temporal frequencies.
    # The magnitude spectrum is assumed even, so negative frequencies can be
    # reconstructed by mirroring the positive side.
    pos = tf_hz > 0

    # Positive temporal-frequency samples.
    tf_pos = tf_hz[pos]

    # Convert power |v|^2 to magnitude |v| at the selected spatial bin.
    # max(..., 0) prevents tiny numerical negatives from creating NaNs.
    mag_pos = np.sqrt(np.maximum(v_sq[i, pos], 0.0))

    # Sort positive temporal frequencies, because np.interp expects an
    # increasing x-axis.
    order = np.argsort(tf_pos)

    # Reorder the positive frequency grid.
    tf_pos = tf_pos[order]

    # Reorder the magnitude values to match the sorted frequency grid.
    mag_pos = mag_pos[order]

    # Do not reconstruct beyond the data-supported frequency range or the
    # requested maximum. This avoids filling unsupported high frequencies with
    # a flat numerical floor, which can create an artificial impulse at t=0.
    tf_max = min(float(tf_pos.max()), float(tf_max_hz))

    # Build a centered, uniformly spaced temporal-frequency grid.
    # Minimum-phase reconstruction needs uniform spacing and both positive and
    # negative frequencies.
    tf_uniform = (
        np.arange(int(n_uniform)) - int(n_uniform) // 2
    ) * (2.0 * tf_max / int(n_uniform))

    # Interpolate the positive-frequency magnitude onto the centered grid.
    # np.abs(tf_uniform) mirrors the positive-frequency magnitude to negative
    # frequencies, making an even magnitude spectrum.
    mag_uniform = np.interp(
        np.abs(tf_uniform),
        tf_pos,
        mag_pos,
        left=mag_pos[0],
        right=0.0,
    )

    # Apply a smooth band taper.
    # This suppresses sharp edges near tf_min_hz and tf_max, which otherwise
    # create ringing in the reconstructed time-domain kernel.
    taper = soft_band_taper(tf_uniform, tf_min_hz, tf_max)

    # Apply the taper to the magnitude spectrum.
    mag_uniform = mag_uniform * taper

    # Choose a small numerical floor relative to the largest tapered magnitude.
    # The floor prevents log(0) inside the minimum-phase reconstruction.
    floor = floor_rel * max(float(np.nanmax(mag_uniform)), 1e-300)

    # Enforce the floor everywhere.
    # This is numerically helpful, but if floor_rel is too large it adds
    # broadband content and can increase h[0].
    mag_uniform = np.maximum(mag_uniform, floor)

    # Reconstruct the causal minimum-phase temporal filter from the magnitude.
    # Assumption for this version: minimum_phase_temporal_filter expects the
    # same frequency units as tf_uniform.
    t, h, V = minimum_phase_temporal_filter(mag_uniform, tf_uniform)

    # Return:
    # t          time samples
    # h          reconstructed causal temporal kernel
    # V          complex minimum-phase frequency response
    # tf_uniform frequency grid used for reconstruction
    return t, h, V, tf_uniform


# def temporal_kernel_slice(f, tf_hz, v_sq, f0, *, tf_min_hz=0.01, tf_max_hz=120.0, n_uniform=2048, floor_rel=1e-4):
#     """Minimum-phase temporal kernel at the spatial bin nearest f0."""
#     f = np.asarray(f, dtype=float)
#     tf_hz = np.asarray(tf_hz, dtype=float)
#     v_sq = np.asarray(v_sq, dtype=float)
#     i = int(np.argmin(np.abs(f - float(f0))))

#     pos = tf_hz > 0
#     tf_pos = tf_hz[pos]
#     mag_pos = np.sqrt(np.maximum(v_sq[i, pos], 0.0))

#     # Uniform centered grid required by the cepstrum/Hilbert construction.
#     tf_uniform = np.linspace(-tf_max_hz, tf_max_hz, int(n_uniform), endpoint=False)
#     mag_uniform = np.interp(np.abs(tf_uniform), tf_pos, mag_pos, left=mag_pos[0], right=0.0)
#     mag_uniform *= soft_band_taper(tf_uniform, tf_min_hz, tf_max_hz)
#     floor = floor_rel * max(float(np.max(mag_uniform)), 1e-300)
#     mag_uniform = np.maximum(mag_uniform, floor)
#     t, h, V = minimum_phase_temporal_filter(mag_uniform, tf_uniform)
#     return t, h, V, tf_uniform


def spatial_kernel_2d_from_radial_magnitude(f, mag_f, *, f_max=80.0, n=512):
    """Centered zero-phase 2D spatial kernel from radial magnitude |V(f)|."""
    if n % 2:
        raise ValueError("n must be even")
    f = np.asarray(f, dtype=float)
    mag_f = np.asarray(mag_f, dtype=float)
    df = 2.0 * float(f_max) / int(n)
    fx = (np.arange(int(n)) - int(n) // 2) * df
    fy = fx.copy()
    FX, FY = np.meshgrid(fx, fy, indexing="xy")
    FR = np.sqrt(FX * FX + FY * FY)
    V = np.interp(FR, f, mag_f, left=mag_f[0], right=0.0)
    v_xy = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(V))).real * (n * df) ** 2
    dx = 1.0 / (n * df)
    x = (np.arange(n) - n // 2) * dx
    y = x.copy()
    return x, y, v_xy


def spatial_kernel_slice(f, tf_hz, v_sq, tf0_hz, *, f_max=80.0, n=512):
    """Spatial cross-section at the temporal bin nearest tf0_hz."""
    f = np.asarray(f, dtype=float)
    tf_hz = np.asarray(tf_hz, dtype=float)
    v_sq = np.asarray(v_sq, dtype=float)
    j = int(np.argmin(np.abs(tf_hz - float(tf0_hz))))
    mag = np.sqrt(np.maximum(v_sq[:, j], 0.0))
    x, y, v_xy = spatial_kernel_2d_from_radial_magnitude(f, mag, f_max=f_max, n=n)
    iy0 = int(np.argmin(np.abs(y)))
    return x, v_xy[iy0], v_xy


def default_slice_frequencies(f, tf_hz, v_sq):
    """Pick representative f0 and tf0 from the filter magnitude."""
    v = np.asarray(v_sq, dtype=float)
    tf_pos = np.asarray(tf_hz) > 0
    spatial_mass = np.sum(v[:, tf_pos], axis=1)
    temporal_mass = np.sum(v[:, tf_pos], axis=0)
    f0 = float(np.asarray(f)[int(np.argmax(spatial_mass))])
    tf0 = float(np.asarray(tf_hz)[tf_pos][int(np.argmax(temporal_mass))])
    return f0, tf0
