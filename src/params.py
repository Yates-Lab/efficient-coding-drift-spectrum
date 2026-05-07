"""Shared band/grid configuration used across figures and analyses.

The retinal analysis band B = [0, k_max] × {ω : ω_min ≤ |ω| ≤ ω_max}
is biologically motivated:
  k_max = 6 cpd  (upper retinal-ganglion spatial-frequency cutoff)
  ω_min = 0.5 rad/s ≈ 0.08 Hz  (slow temporal cutoff; adaptation)
  ω_max = 400 rad/s ≈ 64 Hz    (fast temporal cutoff)

The ``Band`` dataclass is the single source of truth for band edges and grid
construction.  Module-level constants (``K_MAX``, ``OMEGA_MIN``,
``OMEGA_MAX``) and the ``fast_grid()``/``hi_res_grid()`` functions are kept as
thin shims so existing scripts and tests keep working.

Hz conversion: temporal frequency = ω / (2π).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass(frozen=True, init=False)
class Band:
    """Analysis band B = [0, k_max] × {|ω| ∈ [ω_min, ω_max]}.

    k_max  : spatial-frequency cutoff [cpd]
    omega_min, omega_max : temporal angular-frequency band [rad/s]

    For grid construction we additionally need a low-edge spatial frequency
    k_min for log-spaced sampling of k.
    """
    k_min: float = 0.01
    k_max: float = 6.0
    omega_min: float = 0.5
    omega_max: float = 400.0

    # Default grid sizes used by fast_grid()/hi_res_grid()
    n_k_fast: int = 120
    n_omega_fast: int = 1024
    omega_max_grid_fast: float = 500.0

    n_k_hi: int = 220
    n_omega_hi: int = 2048
    omega_max_grid_hi: float = 800.0

    def __init__(
        self,
        k_min: float = 0.01,
        k_max: float = 6.0,
        omega_min: float = 0.5,
        omega_max: float = 400.0,
        n_k_fast: int = 120,
        n_omega_fast: int = 1024,
        omega_max_grid_fast: float = 500.0,
        n_k_hi: int = 220,
        n_omega_hi: int = 2048,
        omega_max_grid_hi: float = 800.0,
        *,
        f_min: float | None = None,
        f_max: float | None = None,
        n_f_fast: int | None = None,
        n_f_hi: int | None = None,
    ):
        # Legacy keyword aliases are accepted so older exploratory notebooks do
        # not break while the notation moves from f to k.
        if f_min is not None:
            k_min = f_min
        if f_max is not None:
            k_max = f_max
        if n_f_fast is not None:
            n_k_fast = n_f_fast
        if n_f_hi is not None:
            n_k_hi = n_f_hi
        object.__setattr__(self, "k_min", float(k_min))
        object.__setattr__(self, "k_max", float(k_max))
        object.__setattr__(self, "omega_min", float(omega_min))
        object.__setattr__(self, "omega_max", float(omega_max))
        object.__setattr__(self, "n_k_fast", int(n_k_fast))
        object.__setattr__(self, "n_omega_fast", int(n_omega_fast))
        object.__setattr__(self, "omega_max_grid_fast", float(omega_max_grid_fast))
        object.__setattr__(self, "n_k_hi", int(n_k_hi))
        object.__setattr__(self, "n_omega_hi", int(n_omega_hi))
        object.__setattr__(self, "omega_max_grid_hi", float(omega_max_grid_hi))

    @property
    def f_min(self) -> float:
        """Legacy alias for ``k_min``."""
        return self.k_min

    @property
    def f_max(self) -> float:
        """Legacy alias for ``k_max``."""
        return self.k_max

    @property
    def n_f_fast(self) -> int:
        """Legacy alias for ``n_k_fast``."""
        return self.n_k_fast

    @property
    def n_f_hi(self) -> int:
        """Legacy alias for ``n_k_hi``."""
        return self.n_k_hi

    @property
    def edges(self) -> Tuple[float, float, float]:
        """The (k_max, omega_min, omega_max) triple consumed by solver/pipeline."""
        return (self.k_max, self.omega_min, self.omega_max)

    def fast_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Coarser (k, ω) grid for I*(D) sweeps.

        n_omega = 1024, ω_max_grid = 500 rad/s ⇒ Δω ≈ 0.98 rad/s, T ≈ 6.4 s.
        Spatial grid is log-spaced from k_min to k_max so the solver actually
        sees the top of the band.
        """
        k = np.geomspace(self.k_min, self.k_max, self.n_k_fast)
        n = self.n_omega_fast
        domega = 2.0 * self.omega_max_grid_fast / n
        omega = (np.arange(n) - n // 2) * domega
        return k, omega

    def hi_res_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Wide (k, ω) grid for kernel reconstruction and 2D contours.

        n_omega = 2048, ω_max_grid = 800 rad/s ⇒ Δω ≈ 0.78 rad/s, T ≈ 8.0 s.
        """
        k = np.geomspace(self.k_min, self.k_max, self.n_k_hi)
        n = self.n_omega_hi
        domega = 2.0 * self.omega_max_grid_hi / n
        omega = (np.arange(n) - n // 2) * domega
        return k, omega

    def grid(self, kind: str = "fast") -> Tuple[np.ndarray, np.ndarray]:
        """Dispatcher: ``kind`` ∈ {'fast', 'hi_res'}."""
        if kind == "fast":
            return self.fast_grid()
        if kind == "hi_res":
            return self.hi_res_grid()
        raise ValueError(f"Unknown grid kind {kind!r}; use 'fast' or 'hi_res'.")


# Default band used everywhere unless overridden.
DEFAULT_BAND = Band()


# ---------------------------------------------------------------------------
# Back-compatibility constants and grid functions.
# Kept so existing imports continue to work without churn.
# ---------------------------------------------------------------------------

K_MAX: float = DEFAULT_BAND.k_max
OMEGA_MIN: float = DEFAULT_BAND.omega_min
OMEGA_MAX: float = DEFAULT_BAND.omega_max

# Legacy alias. New code should import K_MAX.
F_MAX: float = K_MAX


def fast_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Coarser grid for I*(D) sweeps. See ``Band.fast_grid``."""
    return DEFAULT_BAND.fast_grid()


def hi_res_grid() -> Tuple[np.ndarray, np.ndarray]:
    """Wide grid for kernel reconstruction. See ``Band.hi_res_grid``."""
    return DEFAULT_BAND.hi_res_grid()


__all__ = [
    "Band", "DEFAULT_BAND",
    "K_MAX", "OMEGA_MIN", "OMEGA_MAX",
    "F_MAX",
    "fast_grid", "hi_res_grid",
]
