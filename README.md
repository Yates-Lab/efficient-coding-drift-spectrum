# Efficient coding for moving sensors

Reproducible Python implementation of the analytic results in the
moving-sensor efficient-coding appendix (Jun 2026 draft) with
publication-quality figures and a validating test suite.

## What this code computes

Given a power-law image spectrum

    C_I(f) = A / (f^2 + k0^2)^(β/2),

a movement model (Brownian drift, Gaussian linear motion, stationary
saccade controls, or trace-based Rucci/Boi fixation-cycle spectra), and a
fixed band B = { (f, ω) : 0 ≤ f ≤ f_max,
ω_min ≤ |ω| ≤ ω_max }, this codebase

1. constructs the spatiotemporal input spectrum C_θ(f, ω) (`src/spectra.py`);
2. solves the constrained efficient-coding problem (Linsker / Jun)

       maximise  ∫_B (1/2) log( 1 + |v|^2 C_θ / σ_out^2 )  dω df
       s.t.      ∫_B |v|^2 (C_θ + σ_in^2)  ≤  P0,
                 |v|^2 ≥ 0

   in closed form via dual bisection on λ (`src/solver.py`);
3. reconstructs causal minimum-phase temporal kernels and 2D spatial
   kernels from the magnitude solution (`src/kernels.py`).

The main fixation-cycle path is trace based (`src/rucci_cycle_spectra.py`).
It generates isolated saccade traces and smooth drift traces, estimates
both movement redistribution spectra with the same orientation-averaged
Fourier-power estimator, and returns the two inputs needed by the solver:
early fixation `C_early = I(f) Q_saccade` and late fixation
`C_late = I(f) Q_drift`. By default, late drift uses the smooth Brownian
displacement-probability limit implied by the OU trace parameters, avoiding
finite-window periodogram bands. The older stationary Poisson saccade formulas
remain in `src/spectra.py` as analytic controls.

`make_figure7_rucci_cycle_spectra()` is the canonical Rucci-style cycle
source for the paper figures. Figure-facing spectrum panels are centralized in
`src/power_spectrum_library.py`, so Figure 1b and 1c use the same positive-Hz
Figure 7 arrays. Figure 6, Q1, and Q3 get their solver inputs from
`cycle_solver_spectra()` / `spectrum_comparison_specs()` in that same module,
so filter reconstructions and spectrum panels stay tied to the same cached
Figure 7 cycle object instead of regenerating per-figure trace estimates.

The optimization is done on the direct (unaliased) spectrum, following
the appendix's working assumption that the m=0 copy dominates. See the
*aliasing* note below for caveats.

## Layout

```
src/
    spectra.py     drift, Gaussian linear motion, saccade redistribution,
                   combined drift+motion spectra, analytic controls
    rucci_cycle_spectra.py
                   trace-generated Rucci/Boi early and late cycle spectra
    power_spectrum_library.py
                   shared Figure 1b/1c panels and solver spectra from the
                   canonical Figure 7 cycle source
    solver.py      KKT solver, mutual information, water-filling reduction
    kernels.py     cepstral min-phase temporal filter, 2D spatial kernel,
                   soft-band Tukey taper
    plotting.py    publication style, log-contour helper, palettes
    params.py      shared band and grid parameters
tests/
    test_spectra.py    21 tests (12 drift/linear/combined + 9 saccade)
    test_solver.py     10 tests
    test_kernels.py     8 tests
figures/
    fig1_power_spectra.py        movement-induced C_θ(f, ω)
    fig2_optimal_filter.py       |v*(f, ω)|^2 across D and σ_in sweeps
    fig3_kernels.py              spatial v_s(r) and temporal v_t(t)
                                 kernels under drift
    fig4_information_vs_D.py     I*(D) inverted-U for varying σ_in
    fig5_kernel_slices.py        kernels at fixed ω₀ and f₀ slices
    fig6_saccade_kernels.py      spatial and temporal kernels under the
                                 Rucci/Boi early and late cycle spectra
    fig6b_saccade_diagnostics.py  diagnostic: 2D Q kernel, spatial /
                                 temporal profiles, and saccade-vs-drift
                                 spectrum comparison
    fig7_rucci_cycle_spectra.py   trace-estimated Q and C for early saccade
                                 and late drift cycle regimes
    figQ1_spectrum_library.py     supplemental spectrum/solver library
    figQ2_information_sweeps.py   supplemental information sweeps
    figQ3_magno_parvo.py          Rucci/Boi cycle kernels and summary
scripts/
    check_aliasing_negligible.py validation that the m=0 copy assumption
                                 is acceptable for the band/parameters used
outputs/   PNG artifacts at 320 dpi
```

## Running

```bash
pip install numpy scipy matplotlib pytest

python run_all.py                                 # tests + all figure scripts
python run_all.py --with-cell-learning --with-cell-story
                                                  # full production artifacts

python scripts/check_aliasing_negligible.py       # sanity check
```

## Adding a movement spectrum

New analyses should define spectra once and then reuse the shared pipeline.
The intended path is:

1. Add or reuse a `Spectrum` class in `src/spectra.py` with a `C(f, omega)` method.
2. Add a readable factory in `src/power_spectrum_library.py` that returns
   `SpectrumSpec` objects with keys, labels, parameters, and references.
3. Register the factory in `SPECTRUM_SETS` if scripts should request it by name.
4. Run it with `src.pipeline.run_many(...)` or convert the specs to cell-class
   conditions with `conditions_from_spectrum_specs(...)`.

Figure scripts should specify which spectrum set they need; they should not
rebuild grids, masks, Lagrange multiplier solves, or kernel reconstruction.

## What each figure shows

**Figure 1.** Brownian drift Lorentzian, stationary saccade controls,
Gaussian-linear-motion spectra, and the trace-based Rucci/Boi early/late
cycle decomposition. The white line marks characteristic crossovers
where they are meaningful.

**Figure 2.** Optimal filter |v*(f, ω)|^2 in two parameter sweeps. Top
row varies D from 0.05 to 50 at σ_in = 0.3; the band of high-gain
frequencies expands as drift pushes spatial power into temporal
frequencies, and the inverted-U is visible from the I* values in the
titles. Bottom row varies σ_in from 0.05 to 2 at D = 5; gain collapses
as input noise grows.

**Figure 3.** Spatial kernel v_s(r) (radial cross-section, symmetric
around r = 0) and temporal kernel v_t(t) (causal min-phase IFT after a
soft Tukey taper at the band edges). The temporal kernel peak time
scales monotonically with 1/D — faster drift selects faster filters.
The spatial kernel narrows with D and shows classic Mexican-hat
center-surround structure.

A note on the temporal kernels: cepstral minimum-phase reconstruction
is unstable for spectra with hard zeros. The hard band edges produce
log|v_t| spikes outside the band that contaminate the cepstrum. The
soft Tukey taper in `src.kernels.soft_band_taper` keeps the
reconstruction stable; it leaves the in-band filter untouched and
smooths the edge transition.

**Figure 4.** I*(D) curves for σ_in ∈ {0.03, 0.06, 0.11, 0.22, 0.42,
0.81, 1.6, 3.0}, on linear and log y-axes. The inverted-U is preserved
across all noise levels; D* shifts to higher values as σ_in grows.

**Figure 5.** Slices of the optimal filter at fixed ω₀ ∈ {1, 2.3, 5.1,
12, 26, 59, 130, 300} rad/s and at fixed f₀ ∈ {0.08, 0.13, 0.23, 0.38,
0.63, 1.1, 1.8, 3.0} cyc/u, for σ_in ∈ {0.1, 0.5} at D = 5. High-ω₀
slices broaden in space; high-f₀ slices peak earlier in time.

**Figure 6.** Spatial and temporal kernels of the optimal filter under
the canonical Figure 7 trace-based Rucci/Boi cycle spectra. The early condition uses
`C_early_mod = I(f) Q_saccade_mod`; the late condition uses
`C_late_total = I(f) Q_drift_total`. The figure sweeps σ_in and compares
the resulting early and late spatial/temporal kernels.

**Figure 6b (diagnostic).** Sanity check on the saccade redistribution
kernel Q_sac.
- Top row: 2D log-contour of Q at A ∈ {0.5°, 2.5°, 7°} with the
  predicted crossover f_c = 1/(2A) marked.
- Middle row left: spatial profile Q(f, ω₀=8 Hz) for several A,
  showing the f² rise and saturation.
- Middle row right: temporal profile Q(f₀=0.5 cyc/u, ω), showing
  the ~1/ω² falloff at high ω.
- Bottom row: full spectra C_sac vs C_drift for comparison, plus a
  spatial-power conservation check confirming that drift exactly
  preserves C_I(f) and the saccade spectrum integrates proportionally
  to C_I.
The figure also prints numerical sanity checks: low-f log-log slope
(predicted 2.0; measured 1.98), and 1/A scaling of the crossover
(predicted ratio 4.0 between A=1° and A=4°; measured 4.01).

**Figure 7.** The trace-based Rucci/Boi cycle spectra used for the
early-vs-late fixation question. It plots `Q_saccade_mod`,
`I(f) Q_saccade_mod`, `Q_drift_total`, and `I(f) Q_drift_total`; it also
saves `outputs/rucci_cycle_spectra_demo.npz`. This same source spectrum is
used by the other cycle figures through `src.power_spectrum_library`.

**Note on finite-window estimation.** The trace-based saccade estimator
uses finite 512 ms saccade-centered segments, matching the Rucci/Mostofi
procedure. Small synthetic sample counts can reveal periodogram sidelobes,
so the implementation applies mild temporal smoothing while preserving
row-wise temporal power. Late drift defaults to the smooth Brownian
displacement-probability limit implied by the OU trace parameters.

## Aliasing

The appendix assumes the m=0 copy of the periodic-replication sum
dominates the represented spectrum. The figures here are generated under
that assumption. `scripts/check_aliasing_negligible.py` quantifies how
well the assumption holds for the band/parameter ranges we use. The
findings:

- At high drift (D ≥ 5), aliased contributions are < 25% of the in-band
  signal power for a critically-sampled mosaic, and < 6% at 1.5×
  over-sampling. The dominant-copy approximation works.
- At low drift (D ≤ 0.5), aliased contributions can be comparable to or
  larger than the direct contribution under critical sampling. The
  dominant-copy approximation breaks down here, because the spectrum
  hasn't been spread across temporal frequencies yet — high-spatial-
  frequency power still has substantial weight at ω inside the band.

The honest version of this story would compute the aliased spectrum
under a global square mosaic with a chosen k_s. Doing this properly
requires choosing what the cell mosaic actually is (a single global k_s,
or per-cell-type k_s in a multi-class model) — and that's a modeling
choice the appendix doesn't make. We don't model it here.

## Numerical conventions

- Frequencies in cycles/unit; angular frequency in rad/s. Hz = ω / 2π.
- 2D Fourier convention: V(k) = ∫ d²r v(r) e^{-i k·r}, with inverse
  v(r) = ∫ d²k/(2π)² V(k) e^{i k·r}, and analogously in time.
- Spatial integration weight: 2π f df dω/(2π) = f df dω (radial in 2D).
- Centered ω grid: ω ∈ {(n − N/2) Δω}, with ifftshift before the FFT.
- Cepstral minimum-phase fold: keep n=0 and n=N/2, double 0 < n < N/2,
  zero out n > N/2.

## Parameters used in the figures

β = 2, A = 1, k0 = 0.05, σ_out = 1, P0 = 50.
Band: f_max = 4 cyc/u, ω_min = 0.5 rad/s ≈ 0.08 Hz, ω_max = 400 rad/s
≈ 64 Hz.
Frequency grids: f ∈ geomspace(0.05, 5, 220) cyc/u; ω ∈ centered uniform
2048 points spanning ±800 rad/s, giving Δω ≈ 0.78 rad/s and Δt ≈ 3.93 ms.
Figure 4 uses a coarser grid (1024 × 120) since it requires hundreds of
optimisation solves.

## Validation

All 45 tests pass in the local environment. The suite covers:

- spectrum normalisation (drift Lorentzian integrates to C_I(f)) and
  power-preservation in ω for both drift and linear motion;
- saccade template causality (zero before t=0), settling to 1, and
  overshoot magnitude consistent with zeta=0.6;
- saccade Q kernel non-negativity, low-f f² rise (whitening regime),
  and crossover scaling as 1/A (matching Mostofi et al. fig 3D);
- saccade spectrum factorization C_sac = C_I · Q;
- saccade trajectory windowing reduces spurious sinc-lobe ringing;
- KKT optimality of the closed-form solver (relative residual ≤ 1e-10
  on the active set);
- water-filling reduction in the σ_in → 0 limit;
- monotonicity of mutual information in P0;
- causality and magnitude preservation of the cepstral min-phase
  reconstruction;
- soft Tukey band taper produces an early-peaked min-phase impulse
  response (peak < 0.3 s for the test band);
- analytic agreement of 2D spatial-kernel widths and peak values for a
  Gaussian input.
- trace-based Rucci/Boi cycle generation, image-factorization, total vs
  modulated saccade spectra, and `ArraySpectrum` compatibility with the
  existing pipeline.
- shared figure-facing spectrum panels, including exact reuse of the Figure 7
  early/late cycle arrays in Figure 1b and Figure 1c.


## Cell class learning

The production cell-class workflow uses the fast optimizer by default. It keeps
the same information objective and budget normalization as the reference
optimizer, but runs only active in-band frequencies, initializes from the oracle
filter stack, supports `device=auto`, and uses early stopping. Its default
condition stack is the canonical Figure 7 Rucci/Boi cycle split:
`early_cycle = I(f)Q_saccade` and `late_cycle = I(f)Q_drift`.

```
python scripts/run_cell_class_learning.py \
  --grid fast \
  --kmax 4 \
  --steps 1600 \
  --restarts 2 \
  --device auto \
  --dtype float32 \
  --sigma-in 0.3 \
  --P0 50 \
  --outdir outputs/cell_classes
```

```
python scripts/make_cell_class_story_figures.py \
  --indir outputs/cell_classes \
  --outdir outputs/cell_classes_story \
  --K 2
```

```
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

For a slower optimizer comparison, add `--optimizer reference`.
