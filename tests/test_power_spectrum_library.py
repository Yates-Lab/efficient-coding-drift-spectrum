"""Tests for shared figure-facing power spectrum panel generation."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

from pathlib import Path

import numpy as np

from src.power_spectrum_library import (
    canonical_positive_cycle_view,
    cycle_decomposition_panels,
    cycle_solver_spectra,
    spectrum_comparison_specs,
    spectrum_library_panels,
)


def _panel_by_key(panels, key):
    for panel in panels:
        if panel.key == key:
            return panel
    raise AssertionError(f"missing panel {key!r}")


def test_cycle_decomposition_panels_are_figure7_arrays():
    cycle, _, _, _ = canonical_positive_cycle_view()
    pos = cycle.omega > 0
    panels = cycle_decomposition_panels(normalize="none")

    early = _panel_by_key(panels, "cycle_early")
    late = _panel_by_key(panels, "cycle_late")

    np.testing.assert_allclose(early.C, cycle.C_early_mod[:, pos], rtol=1e-12)
    np.testing.assert_allclose(late.C, cycle.C_late_total[:, pos], rtol=1e-12)
    np.testing.assert_allclose(early.temporal_hz, cycle.omega[pos] / (2.0 * np.pi))
    np.testing.assert_allclose(late.f, cycle.f)


def test_spectrum_library_reuses_cycle_decomposition_arrays():
    cycle_panels = cycle_decomposition_panels(normalize="none")
    library_panels = spectrum_library_panels(normalize="none")

    np.testing.assert_allclose(
        _panel_by_key(library_panels, "cycle_early").C,
        _panel_by_key(cycle_panels, "cycle_early").C,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        _panel_by_key(library_panels, "cycle_late").C,
        _panel_by_key(cycle_panels, "cycle_late").C,
        rtol=1e-12,
    )


def test_all_library_panels_share_grid_and_normalize_per_panel():
    panels = spectrum_library_panels(normalize="panel")
    f0 = panels[0].f
    t0 = panels[0].temporal_hz
    for panel in panels:
        np.testing.assert_allclose(panel.f, f0)
        np.testing.assert_allclose(panel.temporal_hz, t0)
        assert panel.C.shape == (f0.size, t0.size)
        assert np.all(np.isfinite(panel.C))
        assert np.all(panel.C >= 0.0)
        assert np.isclose(np.nanmax(panel.C), 1.0)


def test_cycle_solver_spectra_are_figure7_arrays():
    cycle, _, _, _ = canonical_positive_cycle_view()
    early, late = cycle_solver_spectra(use_modulated_early=True)
    nonzero = cycle.omega != 0.0

    np.testing.assert_allclose(
        early.C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_early_mod[:, nonzero],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        late.C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_late_total[:, nonzero],
        rtol=1e-12,
    )

    late_eval = late.C(cycle.f, cycle.omega)
    dc = cycle.omega == 0.0
    assert np.nanmax(late_eval[:, dc]) <= np.nanmax(cycle.C_late_total[:, nonzero])


def test_spectrum_comparison_specs_use_shared_cycle_solver_spectra():
    cycle, _, _, _ = canonical_positive_cycle_view()
    specs = spectrum_comparison_specs(include_controls=False)
    by_label = {label: spec for label, spec, _ in specs}
    nonzero = cycle.omega != 0.0

    np.testing.assert_allclose(
        by_label["early cycle (Rucci/Boi)"].C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_early_mod[:, nonzero],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        by_label["late cycle (Rucci/Boi)"].C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_late_total[:, nonzero],
        rtol=1e-12,
    )


def test_cycle_reconstruction_figures_use_shared_spectrum_entrypoints():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "figures/fig6_saccade_kernels.py": "cycle_solver_spectra",
        "figures/figQ3_magno_parvo.py": "cycle_solver_spectra",
        "figures/figQ1_spectrum_library.py": "spectrum_comparison_specs",
    }
    forbidden = {
        "figure7_cycle_wrappers",
        "make_figure7_rucci_cycle_spectra",
        "make_rucci_cycle_spectra",
    }

    for relpath, shared_entrypoint in expected.items():
        source = (root / relpath).read_text()
        assert shared_entrypoint in source
        for name in forbidden:
            assert name not in source
