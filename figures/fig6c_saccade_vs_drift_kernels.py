"""Figure 6c: shared-library saccade and drift comparison.

This script intentionally obtains spectra through the shared figure-facing
entrypoints so figure code does not drift away from the analysis library.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.params import DEFAULT_BAND
from src.power_spectrum_library import DEFAULT_DRIFT_D, drift_spectrum_specs, saccade_spectrum_specs


def main() -> None:
    k, omega = DEFAULT_BAND.fast_grid()
    specs = [
        saccade_spectrum_specs([4.4])[0],
        drift_spectrum_specs([DEFAULT_DRIFT_D])[0],
    ]

    fig, axes = plt.subplots(1, len(specs), figsize=(6.4, 2.8), sharex=True, sharey=True)
    for ax, spec in zip(axes, specs):
        C = np.maximum(spec.spectrum.C(k, omega), 1e-12)
        ax.imshow(
            np.log10(C.T),
            origin="lower",
            aspect="auto",
            extent=(k.min(), k.max(), omega.min(), omega.max()),
            cmap="magma",
        )
        ax.set_title(spec.title or spec.label)
        ax.set_xlabel("spatial frequency k (cpd)")
    axes[0].set_ylabel("temporal frequency (rad/s)")

    outdir = Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "fig6c_saccade_vs_drift_kernels.png", dpi=200)


if __name__ == "__main__":
    main()
