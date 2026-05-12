"""Tiny compatibility layer for the interactive script."""

from .solver import Result, solve_on_grid
from .kernels import spatial_kernel_slice, temporal_kernel_slice, default_slice_frequencies


def extract_kernels(result, *, f0=None, tf0_hz=None, tf_min_hz=0.01, tf_max_hz=120.0):
    """Attach representative spatial and temporal slices to a Result.

    This keeps the Result object convenient in notebooks without hiding the math.
    """
    if f0 is None or tf0_hz is None:
        auto_f0, auto_tf0 = default_slice_frequencies(result.f, result.tf_hz, result.v_sq)
        if f0 is None:
            f0 = auto_f0
        if tf0_hz is None:
            tf0_hz = auto_tf0
    x, spatial, _ = spatial_kernel_slice(result.f, result.tf_hz, result.v_sq, tf0_hz)
    t, temporal, _, _ = temporal_kernel_slice(result.f, result.tf_hz, result.v_sq, f0, tf_min_hz=tf_min_hz, tf_max_hz=tf_max_hz)
    result.spatial_x = x
    result.spatial_v = spatial
    result.temporal_t = t
    result.temporal_v = temporal
    result.f_peak = f0
    result.tf_peak_hz = tf0_hz
    return result
