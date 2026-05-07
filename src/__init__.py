"""Efficient coding for moving sensors.

Core movement and movie spectra live in :mod:`src.spectra`.
"""

import numpy as np


if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
