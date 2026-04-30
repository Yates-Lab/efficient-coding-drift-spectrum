# Efficient coding for moving sensors

Reproducible Python implementation of the analytic results in the
moving-sensor efficient-coding appendix (Jun 2026 draft) with
publication-quality figures and a validating test suite.

## What this code computes

Given a power-law image spectrum

    C_I(f) = A / (f^2 + k0^2)^(β/2),

a movement model (Brownian drift, Gaussian linear motion, or both), and a
fixed band B = { (f, ω) : 0 ≤ f ≤ f_max, ω_min ≤ |ω| ≤ ω_max }, this
codebase

1. constructs the spatiotemporal input spectrum C_θ(f, ω) (`src/spectra.py`);
2. solves the constrained efficient-coding problem (Linsker / Jun)

       maximise  ∫_B (1/2) log( 1 + |v|^2 C_θ / σ_out^2 )  dω df
       s.t.      ∫_B |v|^2 (C_θ + σ_in^2)  ≤  P0,
                 |v|^2 ≥ 0

   in closed form via dual bisection on λ (`src/solver.py`);
3. reconstructs causal minimum-phase temporal kernels and 2D spatial
   kernels from the magnitude solution (`src/kernels.py`).

The optimization is done on the direct (unaliased) spectrum, following
the appendix's working assumption that the m=0 copy dominates. See the
*aliasing* note below for caveats.

## Layout

```
src/
    spectra.py     drift, Gaussian linear motion, combined spectra
    solver.py      KKT solver, mutual information, water-filling reduction
    kernels.py     cepstral min-phase temporal filter, 2D spatial kernel,
                   soft-band Tukey taper
    plotting.py    publication style, log-contour helper, palettes
    params.py      shared band and grid parameters
tests/
    test_spectra.py    12 tests
    test_solver.py     10 tests
    test_kernels.py     8 tests
figures/
    fig1_power_spectra.py        movement-induced C_θ(f, ω)
    fig2_optimal_filter.py       |v*(f, ω)|^2 across D and σ_in sweeps
    fig3_kernels.py              spatial v_s(r) and temporal v_t(t)
                                 kernels (ω- and f-integrated)
    fig4_information_vs_D.py     I*(D) inverted-U for varying σ_in
    fig5_kernel_slices.py        kernels at fixed ω₀ and f₀ slices
scripts/
    check_aliasing_negligible.py validation that the m=0 copy assumption
                                 is acceptable for the band/parameters used
outputs/   PNG artifacts at 320 dpi
```

## Running

```bash
pip install numpy scipy matplotlib pytest

python -m pytest tests/                           # 30 tests, ~3 s
python figures/fig1_power_spectra.py
python figures/fig2_optimal_filter.py
python figures/fig3_kernels.py
python figures/fig4_information_vs_D.py
python figures/fig5_kernel_slices.py

python scripts/check_aliasing_negligible.py       # sanity check
```

## What each figure shows

**Figure 1.** Brownian drift Lorentzian and Gaussian-linear-motion
spectra, swept over D, β, and s. The white line marks the characteristic
crossover ω ∝ Df² (drift) or ω = sf (linear motion).

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

All 30 tests pass on numpy 2.4 / scipy 1.17. The suite covers:

- spectrum normalisation (drift Lorentzian integrates to C_I(f)) and
  power-preservation in ω for both drift and linear motion;
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
