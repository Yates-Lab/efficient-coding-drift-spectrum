"""Tests for production cell-learning movement-condition construction."""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np

from src.cell_class_learning import build_cell_learning_conditions
from src.power_spectrum_library import canonical_positive_cycle_view


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
