# Refactor 1–5

## 1. Saccade calibration to Mostofi 2020

`src/spectra.py`. The cumulative-Gaussian saccade smoothing scale is now

    σ_A = T_A / (5.5 · √(2π))    [seconds]

where `T_A = 0.021 + 0.0022·A` (s, deg) is the main-sequence duration. The
`5.5·√(2π) ≈ 13.79` divisor is derived from Mostofi 2020 Fig 1G's `v_peak·T_A
≈ 5.5·A` regression: for `x(t) = A·Φ(t/σ_A)`, peak velocity is
`A/(σ_A·√(2π))`, so `v_peak·T_A = (A/(σ·√(2π)))·T_A = 5.5·A` gives the
expression above. The previous `duration_divisor=8` was arbitrary.

Verified: with the new σ_A, `v_peak·T_A = 5.5·A` to machine precision for
every amplitude.

## 2. omega_floor removed

`src/spectra.py`. The old `temporal = exp(-(σω)²) / max(ω², ω_floor²)` with
`ω_floor=1e-15` was a numerical hack: at ω=0 it produced `Q ~ 4×10³⁰`,
hidden by the band mask but otherwise meaningless. The new form divides by
`ω²` directly, with the spatial factor `2(1−J₀(2π k A))` taking care of the
k=0 limit and the band mask `|ω| ≥ ω_min` taking care of the ω=0 limit.
Verified: Q is finite everywhere on the analysis band, zero at k=0 and ω=0.

## 3. Physiological defaults

`src/power_spectrum_library.py`. `DEFAULT_DRIFT_SWEEP` was `(0.05, 0.5, 2,
10, 50)` deg²/s, which spans 30–200× past biological. New default is
`(0.005, 0.0125, 0.0375, 0.1, 0.25)` — covers the Aytekin/Mostofi range
0.005–0.05 plus a high-drift control. Added `DEFAULT_DRIFT_D = 0.0375`
(Mostofi 2020) as the canonical scalar default. `DriftSpectrum` class
default also moved from `D=1.0` to `D=0.0375`. The
`spectrum_comparison_spec_objects` controls now use `DEFAULT_DRIFT_D`.

## 4. Single Band / Grid object

`src/params.py`. New `Band` dataclass owns `(k_min, k_max, ω_min, ω_max)` and
the grid construction. `DEFAULT_BAND.fast_grid()` and `.hi_res_grid()` are
the new entry points; `K_MAX`, `OMEGA_MIN`, `OMEGA_MAX`, `fast_grid()`, and
`hi_res_grid()` are kept as module-level shims so existing callers in
`scripts/`, `figures/`, `tests/`, `src/cell_class_learning.py`, and
`src/pipeline.py` keep working unchanged.

Also fixed the off-by-one: the old `fast_grid()` and `hi_res_grid()` had
`k_max=5.0` while `K_MAX=6.0`, so the optimizer never saw the top of the
band. The new grids end exactly at `k_max=K_MAX=6.0`.

## 5. Free-function/class-API duplication trimmed

`src/spectra.py`. Each spectrum had both a free function and a class
wrapper that called the function. The class API is now the primary
implementation; the free functions (`drift_spectrum`,
`saccade_redistribution`, `saccade_amplitude_average`, etc.) remain as
back-compat shims that delegate to the classes. New `SaccadeAmplitudeMixture`
class replaces the loose `saccade_amplitude_average` function. Module
shrunk from 486 to 443 lines while gaining one new class and clearer
documentation.

## NumPy 2.x compatibility

`np.trapz` → `np.trapezoid` everywhere (6 call sites).

## Tests

32 passed, 2 skipped (pre-existing), 1 deselected (`test_figure_scripts_use_shared_spectrum_entrypoints` references a missing figure file `fig6c_saccade_vs_drift_kernels.py` that doesn't exist in the repo — broken before this refactor).
