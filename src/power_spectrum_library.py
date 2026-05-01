"""Shared power-spectrum panel generation for figures.

This module is the figure-facing source of truth for movement spectra.  Figure
scripts should ask for named panels here instead of rebuilding Rucci/Boi cycle
arrays or choosing separate display grids locally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.rucci_cycle_spectra import (
    RucciCycleSpectra,
    make_figure7_rucci_cycle_spectra,
    spectra_as_wrappers,
)
from src.spectra import (
    DriftSpectrum,
    SaccadeSpectrum,
    DriftPlusSaccadeSpectrum,
    LinearMotionSpectrum,
)

TWOPI = 2.0 * np.pi


@dataclass(frozen=True)
class SpectrumPanel:
    """A prepared positive-temporal-frequency spectrum panel."""

    key: str
    title: str
    f: np.ndarray
    omega: np.ndarray
    temporal_hz: np.ndarray
    C: np.ndarray
    overlay_kind: Optional[str] = None
    overlay_param: Optional[float] = None


def canonical_positive_cycle_view() -> Tuple[RucciCycleSpectra, np.ndarray, np.ndarray, np.ndarray]:
    """Return the Figure 7 cycle and its positive temporal-frequency grid."""
    cycle = make_figure7_rucci_cycle_spectra()
    pos = cycle.omega > 0
    omega = cycle.omega[pos]
    temporal_hz = omega / TWOPI
    return cycle, cycle.f, omega, temporal_hz


def _normalize_panels(panels, normalize: str):
    normalize = normalize.lower()
    if normalize in {"none", "raw"}:
        return list(panels)
    if normalize == "shared":
        vmax = max(float(np.nanmax(p.C)) for p in panels)
        vmax = max(vmax, 1e-30)
        return [
            SpectrumPanel(
                p.key, p.title, p.f, p.omega, p.temporal_hz,
                p.C / vmax, p.overlay_kind, p.overlay_param,
            )
            for p in panels
        ]
    if normalize == "panel":
        out = []
        for p in panels:
            vmax = max(float(np.nanmax(p.C)), 1e-30)
            out.append(
                SpectrumPanel(
                    p.key, p.title, p.f, p.omega, p.temporal_hz,
                    p.C / vmax, p.overlay_kind, p.overlay_param,
                )
            )
        return out
    raise ValueError("normalize must be 'none', 'raw', 'shared', or 'panel'")


def cycle_decomposition_panels(*, normalize: str = "panel"):
    """Return Figure 7 early/late C=I(f)Q cycle panels.

    The returned arrays are direct positive-frequency slices of the canonical
    Figure 7 cycle, optionally normalized for plotting.
    """
    cycle, f, omega, temporal_hz = canonical_positive_cycle_view()
    pos = cycle.omega > 0
    panels = [
        SpectrumPanel(
            "cycle_early",
            "Early fixation\n(saccade traces, modulated)",
            f,
            omega,
            temporal_hz,
            cycle.C_early_mod[:, pos],
        ),
        SpectrumPanel(
            "cycle_late",
            f"Late fixation\n(OU drift, $D_{{eff}}={cycle.drift_D_eff_deg2_s:.3g}$)",
            f,
            omega,
            temporal_hz,
            cycle.C_late_total[:, pos],
        ),
    ]
    return _normalize_panels(panels, normalize)


def spectrum_library_panels(*, normalize: str = "panel"):
    """Return the shared Figure 1c spectrum library on the Figure 7 grid."""
    cycle, f, omega, temporal_hz = canonical_positive_cycle_view()
    pos = cycle.omega > 0
    drift_D = 2.0
    saccade_A = 2.5
    saccade_lam = 3.0
    linear_s = 1.0
    panels = [
        SpectrumPanel(
            "diffusion",
            "Diffusion",
            f,
            omega,
            temporal_hz,
            DriftSpectrum(D=drift_D).C(f, omega),
            overlay_kind="drift_legacy_hz",
            overlay_param=drift_D,
        ),
        SpectrumPanel(
            "saccades",
            "Saccades",
            f,
            omega,
            temporal_hz,
            SaccadeSpectrum(A=saccade_A, lam=saccade_lam).C(f, omega),
        ),
        SpectrumPanel(
            "cycle_early",
            "Early fixation cycle",
            f,
            omega,
            temporal_hz,
            cycle.C_early_mod[:, pos],
        ),
        SpectrumPanel(
            "cycle_late",
            "Late fixation cycle",
            f,
            omega,
            temporal_hz,
            cycle.C_late_total[:, pos],
            overlay_kind="drift_cycles_hz",
            overlay_param=cycle.drift_D_eff_deg2_s,
        ),
        SpectrumPanel(
            "linear",
            "Linear velocity distribution",
            f,
            omega,
            temporal_hz,
            LinearMotionSpectrum(s=linear_s).C(f, omega),
            overlay_kind="linear_hz",
            overlay_param=linear_s,
        ),
    ]
    return _normalize_panels(panels, normalize)


def cycle_solver_spectra(*, use_modulated_early: bool = True):
    """Return canonical Figure 7 spectra for filter reconstruction.

    All early/late Rucci-cycle filter and kernel figures should use this helper
    so the solver inputs and the Figure 1b/1c display panels are tied to the
    same cached Figure 7 cycle object.
    """
    cycle = make_figure7_rucci_cycle_spectra()
    return spectra_as_wrappers(cycle, use_modulated_early=use_modulated_early)


def spectrum_comparison_specs(*, include_controls: bool = True):
    """Return named spectra used by Q1's filter-reconstruction comparison."""
    early_cycle, late_cycle = cycle_solver_spectra(use_modulated_early=True)
    specs = []
    if include_controls:
        specs.extend([
            ("drift", DriftSpectrum(D=2.0), "tab:blue"),
            ("saccade", SaccadeSpectrum(A=2.5, lam=3.0), "tab:orange"),
            (
                "drift + saccade",
                DriftPlusSaccadeSpectrum(D=2.0, A=2.5, lam=3.0),
                "tab:green",
            ),
        ])
    specs.extend([
        ("late cycle (Rucci/Boi)", late_cycle, "tab:purple"),
        ("early cycle (Rucci/Boi)", early_cycle, "tab:red"),
    ])
    return specs


def overlay_curve_hz(panel: SpectrumPanel):
    """Return an overlay curve `(f, temporal_hz)` for panels that define one."""
    if panel.overlay_kind is None:
        return None
    if panel.overlay_param is None:
        return None
    f = panel.f
    p = float(panel.overlay_param)
    if panel.overlay_kind == "drift_legacy_hz":
        return f, p * f ** 2 / TWOPI
    if panel.overlay_kind == "drift_cycles_hz":
        return f, p * TWOPI * f ** 2
    if panel.overlay_kind == "linear_hz":
        return f, p * f / TWOPI
    raise ValueError(f"Unknown overlay kind {panel.overlay_kind!r}")


__all__ = [
    "SpectrumPanel",
    "canonical_positive_cycle_view",
    "cycle_decomposition_panels",
    "spectrum_library_panels",
    "cycle_solver_spectra",
    "spectrum_comparison_specs",
    "overlay_curve_hz",
]
