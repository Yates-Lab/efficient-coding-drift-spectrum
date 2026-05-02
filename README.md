# Efficient coding for moving sensors

This repository contains a reproducible Python implementation of moving-sensor
efficient-coding analyses. It builds retinal input spectra from image statistics
and eye-movement models, solves the constrained optimal-filter problem, and
generates the figure and diagnostic artifacts in `outputs/`.

The main code path is:

1. Define an input spectrum `C_theta(f, omega)` with a `Spectrum.C(f, omega)`
   object from `src/spectra.py` or `src/rucci_cycle_spectra.py`.
2. Solve the Linsker/Jun efficient-coding problem with `src/solver.py`, usually
   through the shared pipeline in `src/pipeline.py`.
3. Optionally reconstruct spatial and temporal kernels with `src/kernels.py`.
4. Plot figures, run audits, or fit reusable cell-class filters with the
   scripts in `figures/` and `scripts/`.

## What It Computes

For a natural-image spectrum

```text
C_I(f) = A / (f^2 + k0^2)^(beta/2)
```

and a movement-generated temporal redistribution, the code solves

```text
maximize    integral_B 0.5 log(1 + |v|^2 C_theta / sigma_out^2) dmu
subject to  integral_B |v|^2 (C_theta + sigma_in^2) dmu <= P0
            |v|^2 >= 0
```

The solver uses the closed-form KKT solution with bisection on the dual
variable lambda. The integration band is controlled by `src/params.py`:

```text
F_MAX = 6.0 cyc/unit
OMEGA_MIN = 0.5 rad/s
OMEGA_MAX = 400.0 rad/s
```

The solver grids currently sample spatial frequencies from `0.05` to `5.0`
cyc/unit. The `hi_res` grid uses `220 x 2048` samples; the `fast` grid uses
`120 x 1024` samples.

## Repository Layout

```text
src/
  spectra.py                  Spectrum API plus static image, Brownian drift,
                              Mostofi saccade, Dong-Atick linear motion, and
                              separable stationary controls
  rucci_cycle_spectra.py      Canonical analytic early/late fixation-cycle
                              spectra used by the cycle figures and cell fits
  power_spectrum_library.py   Named SpectrumSpec collections shared by figures,
                              scripts, and cell-class condition builders
  solver.py                   Closed-form optimal-filter solver and information
                              objective
  pipeline.py                 Spectrum -> Result orchestration, grid selection,
                              kernel extraction, and slice helpers
  kernels.py                  Minimum-phase temporal reconstruction and radial
                              spatial-kernel reconstruction
  cell_class_learning.py      Reference information-aware reusable class-filter
                              model
  cell_class_learning_fast.py Faster Torch implementation used by production
                              cell-class scripts
  cell_class_figures.py       Plotting and diagnostics for learned cell classes
  spectrum_diagnostics.py     Separability and temporal-centroid diagnostics
  plotting.py                 Matplotlib style, radial weights, band masks, and
                              contour helpers
  params.py                   Shared band and grid parameters

figures/                      Figure-generating entry points
scripts/                      Audits, cell-class runs, and story figures
tests/                        Focused pytest suite
outputs/                      Generated PNG/PDF/NPZ/JSON artifacts
run_all.py                    Tests plus the standard figure pipeline
HANDOFF.md                    Historical notes; useful context, not source code
```

## Setup

Use Python 3.9+ from the repository root. There is no package metadata file, so
install the scientific stack directly:

```bash
python -m pip install numpy scipy matplotlib pytest
```

Cell-class learning and the related tests/scripts also need Torch:

```bash
python -m pip install torch
```

The scripts assume they are run from the repository root because they add `.`
to `sys.path`.

## Quick Start

Run the test suite:

```bash
python -m pytest tests/ -v
```

Generate the standard figure set:

```bash
python run_all.py --skip-tests
```

Run tests and generate the standard figures:

```bash
python run_all.py
```

Run the full production path, including the fast cell-class fit and the
cell-class story figures:

```bash
python run_all.py --with-cell-learning --with-cell-story
```

Useful `run_all.py` flags:

```text
--skip-tests
--skip-figures
--with-cell-learning
--with-cell-story
--cell-outdir outputs/cell_classes
--cell-story-outdir outputs/cell_classes_story
--cell-kmax 4
--cell-steps 1600
--cell-restarts 2
--cell-device auto
--cell-dtype float32
```

## Standard Figure Outputs

`python run_all.py --skip-tests` runs these figure scripts:

```text
figures/fig1_power_spectra.py
  outputs/fig1a_main.png
  outputs/fig1b_boi_cycle.png
  outputs/fig1c_library.png

figures/fig2_optimal_filter.py
  outputs/fig2_optimal_filter.png

figures/fig3_kernels.py
  outputs/fig3_kernels.png

figures/fig4_information_vs_D.py
  outputs/fig4_information_vs_D.png

figures/fig5_kernel_slices.py
  outputs/fig5_kernel_slices.png

figures/fig6_saccade_kernels.py
  outputs/fig6_saccade_kernels.png

figures/fig6b_saccade_diagnostics.py
  outputs/fig6b_saccade_diagnostics.png

figures/fig6c_saccade_vs_drift_kernels.py
  outputs/fig6c_saccade_vs_drift_kernels.png

figures/fig7_rucci_cycle_spectra.py
  outputs/fig7_rucci_cycle_spectra.png

figures/fig8_mostofi_saccade_approximation.py
  outputs/fig8_mostofi_saccade_approximation.png

figures/figQ1_spectrum_library.py
  outputs/figQ1_spectrum_library.png

figures/figQ2_information_sweeps.py
  outputs/figQ2_information_sweeps.png

figures/figQ3_magno_parvo.py
  outputs/figQ3_magno_parvo.png
```

The canonical early/late fixation-cycle spectra are generated once by
`make_figure7_rucci_cycle_spectra()` and reused through
`src.power_spectrum_library`. Use `cycle_solver_spectra()` for solver inputs
and `cycle_decomposition_panels()` / `spectrum_library_panels()` for display
panels. This keeps Figures 1, 6, 7, Q1, Q3, and the cell-class condition stack
tied to the same cycle object.

## Common Workflows

Run the aliasing sanity check:

```bash
python scripts/check_aliasing_negligible.py
```

Audit movement spectra and separability assumptions:

```bash
python scripts/run_spectrum_audit.py --outdir outputs/spectrum_audit
```

Create the stationary-vs-active story figures:

```bash
python scripts/make_stationary_vs_active_story.py \
  --grid fast \
  --no-kernels \
  --outdir outputs/stationary_vs_active_story
```

Run the default cell-class fit on the canonical early/late cycle pair:

```bash
python scripts/run_cell_class_learning.py \
  --grid fast \
  --kmax 4 \
  --steps 1600 \
  --restarts 2 \
  --device auto \
  --dtype float32 \
  --sigma-in 0.3 \
  --sigma-out 1.0 \
  --P0 50 \
  --outdir outputs/cell_classes
```

Turn a saved cell-class fit into publication-style story figures:

```bash
python scripts/make_cell_class_story_figures.py \
  --indir outputs/cell_classes \
  --outdir outputs/cell_classes_story \
  --K 2
```

Run the input-noise sweep for reusable cell classes:

```bash
python scripts/run_cell_class_noise_sweep.py \
  --grid fast \
  --sigma-in-values 0.1,0.3,0.6,1.0 \
  --kmax 3 \
  --steps 1200 \
  --restarts 2 \
  --device auto \
  --dtype float32 \
  --outdir outputs/cell_classes_noise_sweep
```

Run the larger movement-sweep cell-class experiment:

```bash
python scripts/run_cell_class_learning.py \
  --condition-set movement_sweep \
  --grid fast \
  --kmax 4 \
  --steps 1600 \
  --restarts 2 \
  --torch-threads 1 \
  --outdir outputs/cell_classes_movement_sweep
```

Compute information-vs-power curves for oracle and reusable class models:

```bash
python scripts/run_information_power_curve.py \
  --condition-set movement_sweep \
  --grid fast \
  --k-values 1,2,3 \
  --fit-mode fixed-H \
  --alpha-mode bounded_log_gain \
  --gain-delta-max 0.5 \
  --P-ref 50 \
  --steps-H 1500 \
  --steps-alpha 400 \
  --restarts 2 \
  --device auto \
  --dtype float32 \
  --torch-threads 1 \
  --outdir outputs/information_power_curve
```

For slower optimizer comparisons in the cell-class scripts, pass
`--optimizer reference`.

## Using the Pipeline in Code

```python
from src.pipeline import SolveConfig, run_many, extract_kernels
from src.power_spectrum_library import drift_spectrum_specs

specs = drift_spectrum_specs([0.05, 0.5, 5.0, 50.0])
config = SolveConfig(sigma_in=0.3, sigma_out=1.0, P0=50.0, grid="hi_res")

results = run_many(specs, config, kernels=True)
for spec, result in zip(specs, results):
    print(spec.key, result.I, result.f_peak)
```

For a single custom spectrum:

```python
from src.pipeline import run, extract_kernels
from src.spectra import SaccadeSpectrum

result = run(SaccadeSpectrum(A=2.5), sigma_in=0.3, sigma_out=1.0, P0=50.0)
extract_kernels(result)
```

## Adding a Movement Spectrum

New analyses should define spectra once and reuse the shared pipeline.

1. Add or reuse a `Spectrum` class in `src/spectra.py` with a `C(f, omega)`
   method.
2. Add a small factory in `src/power_spectrum_library.py` that returns
   `SpectrumSpec` objects with readable keys, labels, parameters, and
   references.
3. Register the factory in `SPECTRUM_SETS` if scripts should request it by
   name.
4. Run the specs with `src.pipeline.run_many(...)`, or convert them to
   cell-class conditions with `conditions_from_spectrum_specs(...)`.

Avoid rebuilding grids, masks, solver calls, or cycle spectra inside figure
scripts when a shared entry point already exists.

## Cell-Class Model

The cell-class workflow learns a small number of reusable nonnegative filter
power spectra `H_c(f, omega)`. For each movement condition `q`, the model forms
a condition-dependent mixture

```text
G_raw[q] = sum_c alpha[q, c] H_c
```

and rescales it so every condition spends the same response-power budget `P0`.
The objective is the same Gaussian mutual information used by the oracle
efficient-coding solver, not a squared-error fit to the oracle filters.

The default condition stack is the canonical Figure 7 pair:

```text
early_cycle = I(f) Q_saccade
late_cycle  = I(f) Q_drift
```

`--condition-set movement_sweep` instead fits five early fixed-amplitude
Mostofi saccade spectra and five late Brownian-drift spectra.

## Tests

The current test suite collects 52 tests:

```text
tests/test_cell_class_conditions.py
tests/test_constrained_gain_learning.py
tests/test_pipeline.py
tests/test_power_spectrum_library.py
tests/test_rucci_cycle_spectra.py
tests/test_separable_stationary_control.py
tests/test_spectrum_classes.py
```

`tests/test_constrained_gain_learning.py` uses `pytest.importorskip("torch")`,
so the Torch-specific tests are skipped when Torch is not installed.

## Numerical Conventions

- Spatial frequency `f` is radial frequency in cycles per unit length.
- Temporal frequency `omega` is angular frequency in rad/s; Hz is
  `omega / (2*pi)`.
- Radial integration uses the 2D spatial measure collapsed into `f df d omega`.
- The temporal grid is centered and uniform; kernel reconstruction uses
  `ifftshift` before FFT operations where needed.
- The temporal kernel phase is recovered by cepstral minimum-phase
  reconstruction after softening hard band edges with `soft_band_taper`.

## Aliasing Note

The main solver uses the direct, unaliased spectrum, following the
dominant-copy assumption in the appendix. `scripts/check_aliasing_negligible.py`
quantifies the size of folded copies for the shared band and parameter ranges.

The short version: the assumption is more reliable at higher drift and with
oversampling. At very low drift under critical sampling, folded high-spatial
frequency power can be comparable to the direct in-band contribution. Modeling
that case properly requires choosing an explicit mosaic structure, which this
repository does not currently do.
