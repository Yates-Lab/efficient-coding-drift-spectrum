"""Tests for production cell-learning movement-condition construction."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

from src.cell_class_learning import (
    build_cell_learning_conditions,
    build_named_cell_learning_conditions,
    conditions_from_spectrum_specs,
)
from src.power_spectrum_library import drift_spectrum_specs


def test_cell_learning_default_conditions_use_core_spectra():
    conditions, pi = build_cell_learning_conditions()

    assert [c.name for c in conditions] == ["saccade_A_4.4", "drift_D_0.0375"]
    assert [c.epoch for c in conditions] == ["saccade", "drift"]
    assert [c.spectrum.name for c in conditions] == ["saccade", "drift"]
    np.testing.assert_allclose(pi, [0.5, 0.5])


def test_conditions_can_be_built_from_spectrum_specs():
    specs = drift_spectrum_specs([0.5, 2.0])
    conditions, pi = conditions_from_spectrum_specs(specs)

    assert [c.name for c in conditions] == ["drift_D_0.5", "drift_D_2"]
    assert [c.epoch for c in conditions] == ["drift", "drift"]
    assert [c.parameter_name for c in conditions] == ["D", "D"]
    assert [c.parameter_value for c in conditions] == [0.5, 2.0]
    assert np.allclose(pi, [0.5, 0.5])


def test_named_movement_sweep_uses_core_saccades_and_cycles_aware_drift():
    saccade_A = (1.0, 2.0)
    drift_D = (0.0375, 0.075)
    conditions, pi = build_named_cell_learning_conditions(
        "movement_sweep",
        grid="fast",
        saccade_A_values=saccade_A,
        drift_D_values=drift_D,
    )

    assert [c.name for c in conditions] == [
        "saccade_A_1",
        "saccade_A_2",
        "drift_D_0.0375",
        "drift_D_0.075",
    ]
    assert [c.epoch for c in conditions] == ["saccade", "saccade", "drift", "drift"]
    np.testing.assert_allclose(pi, [0.25, 0.25, 0.25, 0.25])

    early = conditions[0].spectrum
    assert early.name == "saccade"
    assert early.A == 1.0

    late = conditions[-1].spectrum
    assert late.name == "drift"
    f = np.array([1.0])
    omega = np.array([0.0])
    q0 = late.redistribution(f, omega)[0, 0]
    expected_a = drift_D[-1] * (2.0 * np.pi * f[0]) ** 2
    np.testing.assert_allclose(q0, 2.0 / expected_a)
