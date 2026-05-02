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
    drift_spectrum_specs,
    get_spectrum_set,
    list_spectrum_sets,
    spectrum_comparison_specs,
    spectrum_comparison_spec_objects,
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


def test_spectrum_library_has_requested_four_panel_collection():
    library_panels = spectrum_library_panels(normalize="none")

    assert [p.key for p in library_panels] == [
        "brownian_drift_D_1",
        "saccade_A_3",
        "dong_atick_separable",
        "dong_atick_linear",
    ]
    assert len(library_panels) == 4


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
        by_label["early cycle (Mostofi saccade)"].C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_early_mod[:, nonzero],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        by_label["late cycle (drift)"].C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_late_total[:, nonzero],
        rtol=1e-12,
    )


def test_named_spectrum_sets_return_readable_specs():
    available = list_spectrum_sets()
    assert "drift_sweep" in available

    specs = get_spectrum_set("drift_sweep", D_values=[0.5, 2.0])
    assert [s.key for s in specs] == ["drift_D_0.5", "drift_D_2"]
    assert [s.family for s in specs] == ["drift", "drift"]
    assert [s.parameters["D"] for s in specs] == [0.5, 2.0]
    assert all(hasattr(s.spectrum, "C") for s in specs)


def test_spectrum_comparison_spec_objects_match_legacy_tuples():
    objects = spectrum_comparison_spec_objects(include_controls=True)
    tuples = spectrum_comparison_specs(include_controls=True)

    assert [s.label for s in objects] == [row[0] for row in tuples]
    assert [s.spectrum.describe() for s in objects] == [row[1].describe() for row in tuples]
    assert [s.color for s in objects] == [row[2] for row in tuples]


def test_direct_spec_factories_are_plain_to_extend():
    specs = drift_spectrum_specs([1.0])
    assert specs[0].label == r"$D=1$"
    assert specs[0].title == r"$D = 1$"


def test_cycle_reconstruction_figures_use_shared_spectrum_entrypoints():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "figures/fig6_saccade_kernels.py": "cycle_solver_spectra",
        "figures/fig6c_saccade_vs_drift_kernels.py": "saccade_spectrum_specs",
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
