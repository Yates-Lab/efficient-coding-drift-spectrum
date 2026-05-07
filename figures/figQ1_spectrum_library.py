"""Figure Q1: spectra exposed by the shared power-spectrum library."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.params import DEFAULT_BAND
from src.power_spectrum_library import spectrum_comparison_specs


def main() -> None:
    k, omega = DEFAULT_BAND.fast_grid()
    specs = spectrum_comparison_specs(include_controls=True)

    fig, axes = plt.subplots(1, len(specs), figsize=(2.4 * len(specs), 2.8), sharex=True, sharey=True)
    if len(specs) == 1:
        axes = [axes]

    for ax, (label, spectrum, color) in zip(axes, specs):
        C = np.maximum(spectrum.C(k, omega), 1e-12)
        ax.imshow(
            np.log10(C.T),
            origin="lower",
            aspect="auto",
            extent=(k.min(), k.max(), omega.min(), omega.max()),
            cmap="magma",
        )
        ax.set_title(label, color=color)
        ax.set_xlabel("spatial frequency k (cpd)")
    axes[0].set_ylabel("temporal frequency (rad/s)")

    outdir = Path(__file__).resolve().parent / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outdir / "figQ1_spectrum_library.png", dpi=200)


if __name__ == "__main__":
    main()
