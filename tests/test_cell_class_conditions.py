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
from src.power_spectrum_library import canonical_positive_cycle_view
from src.power_spectrum_library import drift_spectrum_specs


def test_cell_learning_default_conditions_use_figure7_cycle_spectra():
    conditions, pi = build_cell_learning_conditions()
    cycle, _, _, _ = canonical_positive_cycle_view()

    assert [c.name for c in conditions] == ["early_cycle", "late_cycle"]
    assert [c.epoch for c in conditions] == ["early", "late"]
    np.testing.assert_allclose(pi, [0.5, 0.5])
    nonzero = cycle.omega != 0.0
    np.testing.assert_allclose(
        conditions[0].spectrum.C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_early_mod[:, nonzero],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        conditions[1].spectrum.C(cycle.f, cycle.omega)[:, nonzero],
        cycle.C_late_total[:, nonzero],
        rtol=1e-12,
    )

    late_eval = conditions[1].spectrum.C(cycle.f, cycle.omega)
    dc = cycle.omega == 0.0
    assert np.nanmax(late_eval[:, dc]) <= np.nanmax(cycle.C_late_total[:, nonzero])


def test_conditions_can_be_built_from_spectrum_specs():
    specs = drift_spectrum_specs([0.5, 2.0])
    conditions, pi = conditions_from_spectrum_specs(specs)

    assert [c.name for c in conditions] == ["drift_D_0.5", "drift_D_2"]
    assert [c.epoch for c in conditions] == ["drift", "drift"]
    assert [c.parameter_name for c in conditions] == ["D", "D"]
    assert [c.parameter_value for c in conditions] == [0.5, 2.0]
    assert np.allclose(pi, [0.5, 0.5])


def test_named_movement_sweep_uses_fast_precomputed_saccades_and_cycles_aware_drift():
    early_A = (1.0, 2.0)
    late_D = (0.0375, 0.075)
    conditions, pi = build_named_cell_learning_conditions(
        "movement_sweep",
        grid="fast",
        early_A_values=early_A,
        late_D_values=late_D,
        saccade_n_saccades=2,
        saccade_n_orientations=2,
        saccade_T_win_s=0.150,
    )

    assert [c.name for c in conditions] == [
        "early_A_1",
        "early_A_2",
        "late_D_0.0375",
        "late_D_0.075",
    ]
    assert [c.epoch for c in conditions] == ["early", "early", "late", "late"]
    np.testing.assert_allclose(pi, [0.25, 0.25, 0.25, 0.25])

    early = conditions[0].spectrum
    assert hasattr(early, "_C")
    assert early._C.shape == (120, 1024)

    late = conditions[-1].spectrum
    assert late.f_is_cycles is True
    f = np.array([1.0])
    omega = np.array([0.0])
    q0 = late.redistribution(f, omega)[0, 0]
    expected_a = late_D[-1] * (2.0 * np.pi * f[0]) ** 2
    np.testing.assert_allclose(q0, 2.0 / expected_a)
