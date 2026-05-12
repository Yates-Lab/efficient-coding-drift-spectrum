"""Small grid/band helper.

Public units everywhere:
    f      : cycles / degree
    tf_hz  : cycles / second = Hz

This file intentionally contains only one small class.  It is a convenience for
interactive scripts, not an abstraction layer.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Band:
    """Representable spatial and temporal frequency band."""

    f_min: float = 0.01
    f_max: float = 100.0
    tf_min_hz: float = 0.01
    tf_max_hz: float = 120.0

    @property
    def edges(self):
        """Tuple consumed by the solver: (f_max, tf_min_hz, tf_max_hz)."""
        return (self.f_max, self.tf_min_hz, self.tf_max_hz)

    def log_symmetric_grid(self, n_f=96, n_tf_pos=320, tf_max_grid_hz=None):
        """Log-spaced f grid and symmetric log-spaced temporal-frequency grid.

        The temporal grid contains negative and positive frequencies but no zero.
        That is useful for minimum-phase temporal reconstruction and for symmetric
        Fourier-domain calculations.  Plots usually show only the positive branch.
        """
        if tf_max_grid_hz is None:
            tf_max_grid_hz = self.tf_max_hz
        f = np.geomspace(self.f_min, self.f_max, int(n_f))
        tf_pos = np.geomspace(self.tf_min_hz, float(tf_max_grid_hz), int(n_tf_pos))
        tf_hz = np.concatenate([-tf_pos[::-1], tf_pos])
        return f, tf_hz

    def log_positive_grid(self, n_f=180, n_tf=180):
        """Positive grid for clean spectrum displays."""
        f = np.geomspace(self.f_min, self.f_max, int(n_f))
        tf_hz = np.geomspace(self.tf_min_hz, self.tf_max_hz, int(n_tf))
        return f, tf_hz
