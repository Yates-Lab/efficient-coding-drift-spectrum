"""Input/output noise power spectra.

Each noise object has one method:

    noise.power(f, tf_hz) -> array with shape (len(f), len(tf_hz))

The returned quantity is a power spectrum, not a standard deviation.
"""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class WhiteNoise:
    """Constant noise power."""

    power_level: float

    @classmethod
    def from_sigma(cls, sigma):
        return cls(float(sigma) ** 2)

    def power(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        return np.full((f.size, tf_hz.size), self.power_level, dtype=float)

    def describe(self):
        return f"white power={self.power_level:g}"


@dataclass(frozen=True)
class TemporalPowerLawNoise:
    """White floor plus low-frequency 1/f^alpha excess.

        S(ν) = σ² [1 + low_freq_gain * (corner/(|ν| + floor))^alpha]
    """

    sigma: float
    alpha: float = 1.0
    corner_hz: float = 1.0
    floor_hz: float = 0.03
    low_freq_gain: float = 1.0

    @classmethod
    def from_sigma(cls, sigma, alpha=1.0, corner_hz=1.0, floor_hz=0.03, low_freq_gain=1.0):
        return cls(float(sigma), float(alpha), float(corner_hz), float(floor_hz), float(low_freq_gain))

    def power(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        shape = 1.0 + self.low_freq_gain * (self.corner_hz / (np.abs(tf_hz) + self.floor_hz)) ** self.alpha
        return (self.sigma ** 2) * np.ones((f.size, 1)) * shape[None, :]

    def describe(self):
        return f"white + 1/f^{self.alpha:g}, sigma={self.sigma:g}, corner={self.corner_hz:g} Hz"


@dataclass(frozen=True)
class ConeLikeNoise:
    """Simple phenomenological cone-like temporal noise.

    This is deliberately only an approximation.  It gives a white floor, extra
    low-frequency noise, and an optional high-frequency pedestal:

        S(ν) = σ² [1 + a_low (corner/(|ν|+floor))^alpha
                    + a_high * |ν|^m/(|ν|^m + high_corner^m)].
    """

    sigma: float
    low_freq_gain: float = 1.0
    alpha: float = 1.0
    corner_hz: float = 1.0
    floor_hz: float = 0.03
    high_freq_gain: float = 0.15
    high_corner_hz: float = 30.0
    high_power: float = 2.0

    @classmethod
    def from_sigma(
        cls,
        sigma,
        low_freq_gain=1.0,
        alpha=1.0,
        corner_hz=1.0,
        floor_hz=0.03,
        high_freq_gain=0.15,
        high_corner_hz=30.0,
        high_power=2.0,
    ):
        return cls(
            float(sigma), float(low_freq_gain), float(alpha), float(corner_hz),
            float(floor_hz), float(high_freq_gain), float(high_corner_hz), float(high_power)
        )

    def power(self, f, tf_hz):
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        abs_tf = np.abs(tf_hz)
        low = self.low_freq_gain * (self.corner_hz / (abs_tf + self.floor_hz)) ** self.alpha
        high = self.high_freq_gain * (abs_tf ** self.high_power) / (abs_tf ** self.high_power + self.high_corner_hz ** self.high_power)
        shape = 1.0 + low + high
        return (self.sigma ** 2) * np.ones((f.size, 1)) * shape[None, :]

    def describe(self):
        return f"cone-like, sigma={self.sigma:g}, low 1/f^{self.alpha:g}, high pedestal={self.high_freq_gain:g}"


def evaluate_noise(noise, f, tf_hz, default_sigma=None):
    """Return a noise power array from a noise object, scalar, array, or None."""
    if noise is None:
        if default_sigma is None:
            raise ValueError("noise is None and no default_sigma was supplied")
        noise = WhiteNoise.from_sigma(default_sigma)
    if hasattr(noise, "power"):
        return noise.power(f, tf_hz)
    arr = np.asarray(noise, dtype=float)
    if arr.ndim == 0:
        f = np.atleast_1d(np.asarray(f, dtype=float)).ravel()
        tf_hz = np.atleast_1d(np.asarray(tf_hz, dtype=float)).ravel()
        return np.full((f.size, tf_hz.size), float(arr), dtype=float)
    return arr
