"""Tests for shared figure-facing power spectrum definitions."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

from pathlib import Path

from src.power_spectrum_library import (
    drift_spectrum_specs,
    get_spectrum_set,
    list_spectrum_sets,
    saccade_drift_pair_specs,
    spectrum_comparison_specs,
    spectrum_comparison_spec_objects,
)


def test_saccade_drift_pair_uses_core_spectrum_classes():
    specs = saccade_drift_pair_specs(A=4.4, D=0.0375)
    assert [s.family for s in specs] == ["saccade", "drift"]
    assert [s.parameters for s in specs] == [{"A": 4.4}, {"D": 0.0375}]
    assert [s.spectrum.name for s in specs] == ["saccade", "drift"]


def test_named_spectrum_sets_return_readable_specs():
    available = list_spectrum_sets()
    assert "drift_sweep" in available
    assert "saccade_drift_pair" in available

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


def test_figure_scripts_use_shared_spectrum_entrypoints():
    root = Path(__file__).resolve().parents[1]
    expected = {
        "figures/fig6c_saccade_vs_drift_kernels.py": "saccade_spectrum_specs",
        "figures/figQ1_spectrum_library.py": "spectrum_comparison_specs",
    }
    forbidden = {
        "cycle_" + "solver_spectra",
        "canonical_" + "positive_cycle_view",
        "make_figure7_" + "ru" + "cci_cycle_spectra",
        "make_" + "ru" + "cci_cycle_spectra",
        "src." + "ru" + "cci_cycle_spectra",
    }

    for relpath, shared_entrypoint in expected.items():
        source = (root / relpath).read_text()
        assert shared_entrypoint in source
        for name in forbidden:
            assert name not in source
