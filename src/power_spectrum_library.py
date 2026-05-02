"""Shared power-spectrum library for figures and analyses.

This is the human-readable place to add new movement spectra.  The pattern is:

1. Add or reuse a ``Spectrum`` class in ``src.spectra``.
2. Add a small factory below that returns ``SpectrumSpec`` objects.
3. Register that factory in ``SPECTRUM_SETS`` if it is a named collection that
   figures or scripts should be able to request.

Figure scripts should consume ``SpectrumSpec`` collections and then call the
shared pipeline, rather than rebuilding spectra, grids, weights, and solver
calls locally.  The older panel helpers remain below for display-specific
Figure 1/7 views.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from src.rucci_cycle_spectra import (
    RucciCycleSpectra,
    make_figure7_rucci_cycle_spectra,
    spectra_as_wrappers,
)
from src.spectra import (
    DriftSpectrum,
    SaccadeSpectrum,
    LinearMotionSpectrum,
    SeparableMovieSpectrum,
)

TWOPI = 2.0 * np.pi


DEFAULT_DRIFT_SWEEP = (0.05, 0.5, 2.0, 10.0, 50.0)
DEFAULT_SACCADE_SWEEP = (0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_EQUIVALENT_CASES = (0.3, 2.5, 7.0)
DEFAULT_SEPARABLE_OMEGA0 = 0.05


@dataclass(frozen=True)
class SpectrumSpec:
    """Human-readable description of one spectrum to run or plot.

    ``spectrum`` is the only field needed by the numerical pipeline.  The other
    fields are intentionally descriptive metadata so figure scripts can stay
    declarative and future spectra are easy to inspect.
    """

    key: str
    label: str
    spectrum: object
    family: str
    parameters: Dict[str, float] = field(default_factory=dict)
    color: Optional[str] = None
    title: Optional[str] = None
    reference: Optional[str] = None
    notes: str = ""

    def describe(self) -> str:
        return self.label


def _params(**values: float) -> Dict[str, float]:
    return {k: float(v) for k, v in values.items()}


def drift_spectrum_specs(
    D_values: Sequence[float] = DEFAULT_DRIFT_SWEEP,
    *,
    color: Optional[str] = None,
) -> list[SpectrumSpec]:
    """Brownian drift spectra for a sweep over diffusion coefficient ``D``."""
    return [
        SpectrumSpec(
            key=f"drift_D_{D:g}",
            label=rf"$D={D:g}$",
            title=rf"$D = {D:g}$",
            spectrum=DriftSpectrum(D=float(D)),
            family="drift",
            parameters=_params(D=D),
            color=color,
            reference="Kuang et al. 2012",
        )
        for D in D_values
    ]


def saccade_spectrum_specs(
    A_values: Sequence[float] = DEFAULT_SACCADE_SWEEP,
    *,
    color: Optional[str] = None,
) -> list[SpectrumSpec]:
    """Mostofi analytic saccade-transient spectra for a sweep over amplitude ``A``."""
    return [
        SpectrumSpec(
            key=f"saccade_A_{A:g}",
            label=rf"$A={A:g}^\circ$",
            title=rf"$A = {A:g}$",
            spectrum=SaccadeSpectrum(A=float(A)),
            family="saccade",
            parameters=_params(A=A),
            color=color,
            reference="Mostofi et al. 2020 analytic approximation",
        )
        for A in A_values
    ]


def linear_motion_spectrum_specs(
    s_values: Sequence[float] = (1.0,),
    *,
    color: Optional[str] = None,
) -> list[SpectrumSpec]:
    """Gaussian linear-motion spectra for a sweep over speed scale ``s``."""
    return [
        SpectrumSpec(
            key=f"linear_s_{s:g}",
            label=rf"$s={s:g}$",
            title=rf"$s = {s:g}$",
            spectrum=LinearMotionSpectrum(s=float(s)),
            family="linear_motion",
            parameters=_params(s=s),
            color=color,
            reference="Dong & Atick 1995",
        )
        for s in s_values
    ]


def separable_movie_spectrum_specs(
    omega0_values: Sequence[float] = (DEFAULT_SEPARABLE_OMEGA0,),
    *,
    color: Optional[str] = None,
) -> list[SpectrumSpec]:
    """Stationary separable natural-movie controls.

    These are the explicit old-school controls:
    S(k, omega) proportional to 1/(|k|^2 |omega|^2).  They remove the
    movement-induced f--omega coupling.
    """
    return [
        SpectrumSpec(
            key=f"separable_omega0_{omega0:g}",
            label=rf"separable $\omega_0={omega0:g}$",
            title=rf"Separable stationary ($\omega_0={omega0:g}$)",
            spectrum=SeparableMovieSpectrum(omega0=float(omega0)),
            family="separable_stationary",
            parameters=_params(omega0=omega0),
            color=color,
            reference="stationary separable movie control",
            notes="Paper-style separable power law: S(k,omega) proportional to 1/(|k|^2 |omega|^2).",
        )
        for omega0 in omega0_values
    ]


def stationary_vs_active_story_specs() -> list[SpectrumSpec]:
    """Core comparison set for the narrative figure.

    This intentionally contrasts: (i) a stationary separable approximation,
    (ii) the Dong--Atick linear-motion control, and (iii) the analytic
    early/late fixation selector.
    """
    cycle_specs = cycle_spectrum_specs(use_modulated_early=True)
    return [
        SpectrumSpec(
            key="separable_stationary",
            label="separable stationary",
            title="Separable stationary",
            spectrum=SeparableMovieSpectrum(omega0=DEFAULT_SEPARABLE_OMEGA0),
            family="stationary_control",
            parameters=_params(omega0=DEFAULT_SEPARABLE_OMEGA0),
            color="tab:gray",
            reference="stationary separable movie approximation",
        ),
        SpectrumSpec(
            key="dong_atick_linear",
            label="Dong--Atick linear motion",
            title="Stationary linear motion",
            spectrum=LinearMotionSpectrum(s=1.0),
            family="stationary_control",
            parameters=_params(s=1.0),
            color="tab:blue",
            reference="Dong & Atick 1995",
        ),
        cycle_specs[0],
        cycle_specs[1],
    ]


def cycle_spectrum_specs(*, use_modulated_early: bool = True) -> list[SpectrumSpec]:
    """Canonical early/late fixation selector: saccade transient vs drift."""
    early, late = cycle_solver_spectra(use_modulated_early=use_modulated_early)
    return [
        SpectrumSpec(
            key="cycle_early",
            label="early cycle (Mostofi saccade)",
            title="Early fixation",
            spectrum=early,
            family="fixation_cycle",
            parameters=_params(cycle_phase=0.0),
            color="tab:red",
            reference="Mostofi et al. 2020 analytic approximation",
            notes="Selector state: saccade transient.",
        ),
        SpectrumSpec(
            key="cycle_late",
            label="late cycle (drift)",
            title="Late fixation",
            spectrum=late,
            family="fixation_cycle",
            parameters=_params(cycle_phase=1.0),
            color="tab:purple",
            reference="Kuang et al. 2012 analytic drift",
            notes="Selector state: Brownian drift.",
        ),
    ]


def spectrum_comparison_spec_objects(*, include_controls: bool = True) -> list[SpectrumSpec]:
    """Named comparison set used by Q1 and general sanity checks."""
    specs: list[SpectrumSpec] = []
    if include_controls:
        specs.extend([
            SpectrumSpec(
                key="drift_control",
                label="drift",
                title="Drift",
                spectrum=DriftSpectrum(D=2.0),
                family="drift",
                parameters=_params(D=2.0),
                color="tab:blue",
            ),
            SpectrumSpec(
                key="saccade_control",
                label="saccade",
                title="Saccade",
                spectrum=SaccadeSpectrum(A=2.5),
                family="saccade",
                parameters=_params(A=2.5),
                color="tab:orange",
            ),
        ])
    cycle_specs = cycle_spectrum_specs(use_modulated_early=True)
    # Preserve the legacy Q1 display order: late-cycle control, then early.
    specs.extend([cycle_specs[1], cycle_specs[0]])
    return specs


SPECTRUM_SETS: Dict[str, Callable[..., list[SpectrumSpec]]] = {
    "drift_sweep": drift_spectrum_specs,
    "saccade_sweep": saccade_spectrum_specs,
    "linear_motion_sweep": linear_motion_spectrum_specs,
    "separable_movie": separable_movie_spectrum_specs,
    "stationary_vs_active_story": stationary_vs_active_story_specs,
    "cycle_early_late": cycle_spectrum_specs,
    "comparison_controls_and_cycle": spectrum_comparison_spec_objects,
}


SPECTRUM_SET_DESCRIPTIONS = {
    "drift_sweep": "Brownian drift spectra parameterized by D.",
    "saccade_sweep": "Mostofi analytic saccade-transient spectra parameterized by A.",
    "linear_motion_sweep": "Gaussian linear-motion spectra parameterized by s.",
    "separable_movie": "Stationary separable C_I(f) Q_T(omega) controls.",
    "stationary_vs_active_story": "Separable control, Dong--Atick linear motion, and analytic early/late selector spectra.",
    "cycle_early_late": "Canonical early/late selector: Mostofi saccade and Brownian drift.",
    "comparison_controls_and_cycle": "Drift, Mostofi saccade, and selector controls.",
}


def list_spectrum_sets() -> Dict[str, str]:
    """Return the named spectrum collections available to scripts."""
    return dict(SPECTRUM_SET_DESCRIPTIONS)


def get_spectrum_set(name: str, **kwargs) -> list[SpectrumSpec]:
    """Return a named spectrum collection.

    Examples
    --------
    get_spectrum_set("drift_sweep", D_values=np.geomspace(0.01, 200, 25))
    get_spectrum_set("cycle_early_late")
    """
    try:
        factory = SPECTRUM_SETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(SPECTRUM_SETS))
        raise ValueError(f"Unknown spectrum set {name!r}. Available: {available}") from exc
    return factory(**kwargs)


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
            "Early fixation\n(Mostofi saccade)",
            f,
            omega,
            temporal_hz,
            cycle.C_early_mod[:, pos],
        ),
        SpectrumPanel(
            "cycle_late",
            f"Late fixation\n(drift, $D={cycle.drift_D_eff_deg2_s:.3g}$)",
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
            "Mostofi saccade",
            f,
            omega,
            temporal_hz,
            SaccadeSpectrum(A=saccade_A).C(f, omega),
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
    return [
        (spec.label, spec.spectrum, spec.color)
        for spec in spectrum_comparison_spec_objects(include_controls=include_controls)
    ]


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
    "SpectrumSpec",
    "DEFAULT_DRIFT_SWEEP",
    "DEFAULT_SACCADE_SWEEP",
    "DEFAULT_EQUIVALENT_CASES",
    "drift_spectrum_specs",
    "saccade_spectrum_specs",
    "linear_motion_spectrum_specs",
    "separable_movie_spectrum_specs",
    "stationary_vs_active_story_specs",
    "cycle_spectrum_specs",
    "spectrum_comparison_spec_objects",
    "list_spectrum_sets",
    "get_spectrum_set",
    "SpectrumPanel",
    "canonical_positive_cycle_view",
    "cycle_decomposition_panels",
    "spectrum_library_panels",
    "cycle_solver_spectra",
    "spectrum_comparison_specs",
    "overlay_curve_hz",
]
