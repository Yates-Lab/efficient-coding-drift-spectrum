# Efficient coding for moving sensors

A reproducible Python implementation of the analytic results of Jun et al.
(2026 draft, "Efficient coding under moving sensors") with publication-quality
figures and a validating test suite.

## What this code computes

Given a power-law image spectrum

    C_I(f) = A / (f^2 + k0^2)^(β/2),

a movement model (Brownian drift, Gaussian linear motion, or both), and a
band of cell tunings B = { (f, ω) : 0 ≤ f ≤ f_max, ω_min ≤ |ω| ≤ ω_max },
this codebase

1. constructs the spatiotemporal input spectrum C_θ(f, ω) (`src/spectra.py`);
2. solves the constrained efficient-coding problem (Linsker / Jun)

       maximise  ∫_B (1/2) log( 1 + |v|^2 C_θ / σ_out^2 )  dω df
       s.t.      ∫_B |v|^2 (C_θ + σ_in^2)  ≤  P0,
                 |v|^2 ≥ 0

   in closed form via dual bisection on λ (`src/solver.py`);
3. reconstructs causal minimum-phase temporal kernels and 2D spatial kernels
   from the magnitude solution (`src/kernels.py`);
4. handles per-cell-Nyquist aliasing — the cell tuned to f0 sees an aliased
   spectrum C^sample(f0, ω) = C(f0, ω) + 2 Σ_{q≥1} C((2q+1) f0, ω)
   (`src/aliasing.py`).

## Layout

```
src/
    spectra.py     drift + linear-motion + combined spectra
    solver.py      KKT solver, mutual information, water-filling reduction
    kernels.py     cepstral min-phase temporal filter, 2D spatial kernel,
                   soft-band Tukey taper
    aliasing.py    per-cell-Nyquist mosaic aliasing
    plotting.py    publication style, log contour helper, palettes
    params.py      shared band parameters and grids
tests/
    test_spectra.py    12 tests
    test_solver.py     10 tests
    test_kernels.py     7 tests
    test_aliasing.py    5 tests
figures/
    fig1_power_spectra.py        movement-induced C_θ(f, ω)
    fig2_optimal_filter.py       |v*(f, ω)|^2 vs (D, σ_in), direct vs aliased
    fig3_aliasing.py             per-cell-Nyquist aliasing effects
    fig4_kernels.py              spatial v_s(r) and temporal v_t(t) kernels,
                                 ω-integrated, direct vs aliased
    fig5_information_vs_D.py     I*(D) inverted-U for varying σ_in,
                                 direct vs aliased
    fig6_kernel_slices.py        spatial kernels at fixed ω₀ slices and
                                 temporal kernels at fixed f₀ slices,
                                 direct vs aliased at two σ_in levels
outputs/   PNG artifacts at 320 dpi
```

## Running

```bash
# Install deps if needed
pip install numpy scipy matplotlib pytest

# Run the validation suite (35 tests, ~3 s)
python -m pytest tests/

# Generate the six figures
python figures/fig1_power_spectra.py
python figures/fig2_optimal_filter.py
python figures/fig3_aliasing.py
python figures/fig4_kernels.py
python figures/fig5_information_vs_D.py
python figures/fig6_kernel_slices.py
```

Outputs land in `outputs/`.

## What each figure shows

**Figure 1.** The Brownian drift Lorentzian and the Gaussian-linear-motion
Doppler-like spectrum, swept over D, β, and s. The white line marks the
characteristic crossover ω ∝ Df² (drift) or ω = sf (linear motion) that
separates the spatial and temporal scaling regimes.

**Figure 2.** Two parameter sweeps of the optimal filter |v*(f, ω)|^2 in a
4×4 grid: top two rows vary D at fixed σ_in (direct then aliased); bottom
two rows vary σ_in at fixed D (direct then aliased). The aliased rows use
C^sample(f, ω) — the per-cell-Nyquist folded spectrum each cell actually
sees in a Nyquist-matched mosaic. The aliased input has higher total power
in the band, so I^* is uniformly larger; the inverted-U structure (peak at
D = 5) is preserved in both versions.

**Figure 3.** The aliasing comparison itself. Top row: direct C(f₀, ω),
aliased C^sample(f₀, ω), and their ratio (which can exceed 100 at high
f₀). Bottom row: optimal filters under each input model and the
ω-integrated gain per spatial frequency.

**Figure 4.** Spatial kernel v_s(r) (radial cross-section of the 2D inverse
FT, plotted symmetric around r = 0) and temporal kernel v_t(t) (minimum-
phase causal IFT after a soft Tukey taper at the band edges). Each row
contrasts direct vs aliased input, sweeping over D in rows 1, 3 and over
σ_in in rows 2, 4. The spatial kernels show classic Mexican-hat
center-surround structure that broadens with D; the temporal kernels show
an impulse with peak at 50–150 ms, becoming sharper / more biphasic at
low σ_in.

A note on the temporal kernels: cepstral minimum-phase reconstruction is
unstable for spectra with hard zeros, because log|v_t| has large negative
spikes outside the band that contaminate the cepstrum and shift the
recovered impulse-response peak by ~0.5 s. Applying a smooth Tukey taper
at the band edges (`src.kernels.soft_band_taper`) keeps the cepstrum well
behaved and recovers physically reasonable impulse responses. The taper
is purely a numerical regularizer — it does not change the band-pass
structure of the optimal filter inside the band.

**Figure 5.** I*(D) curves for σ_in ∈ {0.05, 0.15, 0.4, 1.0, 2.5}, both
linear and log y-axes, showing direct (solid, circle markers at peaks) and
aliased (dashed, diamond markers at peaks). Aliasing uniformly raises
I^* and shifts the peak D^* to lower values. The aliasing folds in
free signal power at the cell's preferred frequency, so less drift is
needed to optimally fill the band.

**Figure 6.** Spatial kernel slices at fixed ω₀ and temporal kernel slices
at fixed f₀, for two σ_in levels (0.1 and 0.5) under direct and aliased
inputs (4 conditions, 8 panels total). Spatial slices show how the
receptive field profile depends on temporal frequency: low-ω₀ slices
show a tight Mexican hat; high-ω₀ slices broaden, and under aliasing
become Gaussian-like at the top end of the band (because aliasing folds
low-spatial-frequency power into high-f₀ cells). Temporal slices show
how the impulse response depends on spatial frequency: high-f₀ slices
peak earlier and have less negative undershoot, especially under
aliasing where high spatial frequencies carry high-temporal-frequency
content folded in from the low-f power.

## Validation

All 35 tests pass on numpy 2.4 / scipy 1.17. The suite covers:

- spectrum normalisation (drift Lorentzian integrates to C_I(f)) and
  power-preservation in ω for both drift and linear motion;
- KKT optimality of the closed-form solver (relative residual ≤ 1e-10
  on the active set);
- water-filling reduction in the σ_in → 0 limit;
- monotonicity of mutual information in P0;
- causality and magnitude preservation of the cepstral min-phase
  reconstruction;
- soft Tukey band taper produces an early-peaked min-phase impulse
  response (peak < 0.3 s for the test band), validating the
  cepstral-stability fix;
- analytic agreement of 2D spatial-kernel widths and peak values for a
  Gaussian input;
- per-cell aliasing reproduces the closed-form series π²/4 for the
  unregularised 1/f² spectrum.

## Numerical conventions

- Frequencies in cycles/unit; angular frequency in rad/s. The Fourier
  conventions are V(k) = ∫ d²r v(r) e^{-i k·r}, v(r) = ∫ d²k/(2π)² V(k) e^{i k·r}
  (and analogously in time), so a Gaussian C(k) = exp(-||k||² σ²/2) has
  inverse FT v(r) = (1/2π σ²) exp(-||r||²/(2σ²)).
- The spatial integration weight is 2π f df dω/(2π) = f df dω (radial in 2D
  Fourier space).
- All temporal grids are centered: ω ∈ {(n − N/2) Δω : n = 0..N-1}, and
  ifftshift is applied before the FFT in `src/kernels.py`.
- The cepstral fold for minimum-phase reconstruction follows the standard
  recipe: keep n=0 and n=N/2, double 0 < n < N/2, zero out n > N/2.

## Parameters used in the figures

Default unless stated:  β = 2, A = 1, k0 = 0.05, σ_out = 1, P0 = 50.
Band B = {(f, ω) : 0 ≤ f ≤ 4 cyc/u, 0.5 ≤ |ω| ≤ 400 rad/s}, i.e.
temporal frequency in [0.08, 64] Hz (since f_Hz = ω/2π) — the full
biologically relevant range for visual neurons.
Frequency grids: f ∈ geomspace(0.05, 5, 220) cyc/u.
Temporal grid: 2048 centered uniform points in ω spanning ±800 rad/s,
giving Δω ≈ 0.78 rad/s and Δt ≈ 3.93 ms (T ≈ 8.05 s).
Figure 5 uses a coarser grid (1024×120) since it requires hundreds of
optimisation solves.
