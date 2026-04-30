# Project handoff: efficient coding for moving sensors with saccades

## Where we are

Working codebase at `/mnt/user-data/outputs/efficient_coding.tar.gz` (also available unbundled at `/mnt/user-data/outputs/efficient_coding/`). 39 tests passing. Eight figures produced. The saccade extension is implemented correctly and matches Mostofi et al. (2020) predictions to within ~5%.

Primary deliverable: a closed-form saccade redistribution kernel $Q_{\text{sac}}(f, \omega; A, \lambda)$ that plugs into the existing efficient-coding optimization framework, alongside the drift and Gaussian-linear-motion spectra you already had.

## The core formula (this is the punch line)

$$Q_{\text{sac}}(k, \omega) = \frac{2 \lambda \alpha(k)}{(\lambda \alpha(k))^2 + \omega^2}, \quad \alpha(k) = 1 - J_0(2\pi k A)$$

derived from the temporal autocorrelation of the phase factor $e^{-2\pi i k \cdot \Delta x(\tau)}$ under a stationary Poisson process of saccade events with rate $\lambda$ and isotropic step magnitude $A$. This is the standard formula used in Aytekin et al. 2014 and is what underlies Mostofi et al.'s factorization analysis.

Limits:
- **Small $kA$** (microsaccades, low spatial frequencies): $\alpha(k) \to \pi^2 (kA)^2$, giving $Q \to 2 D_{\text{eff}} k^2 / ((D_{\text{eff}} k^2)^2 + \omega^2)$ — a pure Brownian-drift Lorentzian with effective diffusion $D_{\text{eff}} = \pi^2 \lambda A^2$
- **Large $kA$**: $\alpha(k)$ bounded in $[0, 2]$, oscillating — saturation regime
- **Crossover**: $f_c = 1/(2A)$, matching Mostofi fig 3D

## What we learned about saccades vs drift

The user's central question across the conversation: do saccades behave like drift?

**Answer: yes in the small-$kA$ regime, no in the large-$kA$ regime.** The key results:

1. **Mapping**: large saccades = high $D_{\text{eff}}$ (because $D_{\text{eff}} \propto A^2$). Microsaccades = low $D_{\text{eff}}$. This goes opposite to a naive "small saccade = high drift" intuition we initially confused ourselves with.

2. **Spatial kernels diverge between saccades and equivalent-$D_{\text{eff}}$ drift at moderate-to-large $A$**: drift kernels keep getting wider as $D$ grows; saccade kernels stay narrow because $\alpha(k)$ saturates and the optimizer has no high-$f$ signal to gain from. This is *the* qualitative difference Mostofi calls out in the paper.

3. **Temporal kernels match perfectly across the entire $A$ range when paired with equivalent-$D_{\text{eff}}$ drift.** This matters for the unified-model story.

`figures/fig6c_saccade_vs_drift_kernels.py` makes this visual: three columns ($A = 0.3°, 2.5°, 7°$) with saccade kernel and matched-$D_{\text{eff}}$ drift kernel overlaid. Spatial kernels match at $A = 0.3°$, diverge at $A = 7°$. Temporal kernels match at all three.

## Critical implementation arc (what NOT to redo)

We went through three iterations of the saccade implementation. Don't repeat the dead ends:

### Iteration 1 (wrong, deleted)
Single-saccade transient FT: $|FT_t[e^{-2\pi i k A u(t)}]|^2$ with the damped-harmonic-oscillator template. Validated against Mostofi fig 5A.

**Problem**: this computes the energy spectrum of *one* saccade transient, not the power spectrum the cell sees from a continuous saccade stream. The trajectory $u(t)$ asymptotes to 1 (not to 0), giving a near-step structure whose FT has sinc lobes at $\omega = 2\pi/(T/2)$ inside our band.

### Iteration 2 (also wrong, deleted)
Same FT formula but with a Tukey window (200 ms half-width, 100 ms taper) on the trajectory to suppress sinc lobes.

**Problem**: the window forces $u(t) \to 0$ at the edges, killing the *physical* low-omega content of the step (the post-saccade plateau is real — the eye holds the new fixation position). This shifts power away from low $\omega$, distorts the kernels, makes the crossover position 2× wrong.

### Iteration 3 (correct, current)
Closed-form Poisson-process Lorentzian as written above. No FFT, no windowing, no template. Just $J_0$ and the algebraic Lorentzian.

**Why this works**: it's the autocorrelation of a stationary process (the sequence of saccade events), which is the right object for what the cell sees. It naturally has drift-like behavior at low $kA$, saturation at high $kA$, no window artifacts, and is fast (instant evaluation, no FFT).

### Why the saccade template is still in the codebase
`saccade_template(t)` is kept as a utility for reproducing Mostofi's fig 5 (a sanity check we did early on showing the damped-harmonic-oscillator template with $\zeta = 0.6$, peak at 40 ms reproduces real saccade trajectories). It's no longer used by `saccade_redistribution` but is exported.

## File layout

```
src/
  spectra.py        image, drift, linear-motion, combined, saccade_template,
                    saccade_redistribution, saccade_spectrum
  solver.py         KKT closed-form solver + bisection on lambda
  kernels.py        cepstral min-phase temporal filter, 2D spatial kernel,
                    soft-band Tukey taper for kernel reconstruction
  plotting.py       palettes, log-contour, radial weights, band masks
  params.py         F_MAX = 4 cyc/deg, OMEGA_MIN = 0.5, OMEGA_MAX = 400 rad/s
                    hi_res_grid (220 f x 2048 omega), fast_grid (120 x 1024)

tests/              39 tests, ~3 s
  test_spectra.py   21 (12 drift/linear/combined + 9 saccade incl. drift-limit
                    and Lorentzian-normalization tests)
  test_solver.py    10
  test_kernels.py   8

figures/
  fig1_power_spectra.py        drift and linear-motion C_theta(f, omega)
  fig2_optimal_filter.py       |v*|^2 sweeps over D and sigma_in
  fig3_kernels.py              spatial v_s(r) and temporal v_t(t) under drift
  fig4_information_vs_D.py     I*(D) inverted-U for varying sigma_in
  fig5_kernel_slices.py        kernels at fixed omega_0 and f_0
  fig6_saccade_kernels.py      saccade kernels, sweep A and sigma_in
  fig6b_saccade_diagnostics.py Q diagnostic: 2D contour, spatial+temporal
                               profiles, saccade-vs-drift band-integrated
  fig6c_saccade_vs_drift_kernels.py  side-by-side kernel comparison at
                               matched D_eff (the answer to "why don't
                               large saccades give wide spatial kernels")

scripts/
  check_aliasing_negligible.py  verifies the m=0-dominant-copy assumption
                                across the parameter ranges used

outputs/                         eight PNGs at 320 dpi
```

## Key parameters

- Image: $\beta = 2$, $A_{\text{image}} = 1$, $k_0 = 0.05$
- Constraint: $\sigma_{\text{out}} = 1$, $P_0 = 50$
- Band: $f_{\max} = 4$ cyc/deg (foveal acuity), $\omega_{\min} = 0.5$ rad/s ($\approx 0.08$ Hz), $\omega_{\max} = 400$ rad/s ($\approx 64$ Hz, RGC bandwidth)
- Saccade rate default: $\lambda = 3$/s (Mostofi)
- Saccade amplitudes swept: $A \in \{0.3°, 0.47°, 0.74°, 1.2°, 1.8°, 2.8°, 4.5°, 7°\}$
- $\sigma_{\text{in}}$ swept: $\{0.03, 0.055, 0.1, 0.18, 0.33, 0.6, 1.1, 2\}$

## Unit convention (this caused real confusion mid-conversation)

The codebase's spatial unit is **degrees of visual angle**. With this:
- $f_{\max} = 4$ cyc/deg = upper end of foveal acuity
- $D$ values 0.05–50 deg²/s span weak-fixational to strong-saccadic-equivalent drift
- $A$ values plug in directly as Mostofi's amplitudes

Mid-conversation we got confused because the codebase's other quantities don't strictly require this unit — the framework is unit-agnostic and only ties to physics through the $D$ values. But for the saccade extension, treating "unit = degree" is the correct interpretation and what makes the figures match Mostofi quantitatively.

## I* values from current saccade sweeps

For $\sigma_{\text{in}} = 0.3$, $\sigma_{\text{out}} = 1$, $P_0 = 50$, $\lambda = 3$/s:

```
A = 0.3°:  I* = 1.69 nats   (microsaccade)
A = 0.47°: I* = 1.83
A = 0.74°: I* = 1.92
A = 1.2°:  I* = 1.98
A = 1.8°:  I* = 2.01
A = 2.8°:  I* = 2.03
A = 4.5°:  I* = 2.04
A = 7°:    I* = 2.04        (large saccade)
```

Roughly flat in $A$ — saccade amplitude redistributes power without changing the band's total signal-to-noise budget much. The structural difference shows up in kernel shapes, not $I^*$.

For $A = 2.5°$, varying $\sigma_{\text{in}}$:

```
sigma_in = 0.03:  I* = 16.4 nats
sigma_in = 0.10:  I* = 7.06
sigma_in = 0.33:  I* = 1.78
sigma_in = 1.1:   I* = 0.28
sigma_in = 2.0:   I* = 0.10
```

Sigmoidal collapse, as expected.

## Open threads / what comes next

These are the directions I'd suggest considering for continuation:

1. **The unified-model framing.** Mostofi argues the right framework treats drift and saccades as a continuum parameterized by a single "scale of movement" knob. With our $D_{\text{eff}} = \pi^2 \lambda A^2$ result, this is now mathematically explicit. Could be worth writing up as a short methods section in the appendix: "saccades are Brownian-like below the crossover, with $D_{\text{eff}}$ given by..."

2. **Combined drift + saccade dynamics during natural viewing.** During fixation the eye drifts; saccades punctuate fixations. The cell sees both. Mathematically: total displacement is sum of drift Brownian motion plus saccade jumps; the autocorrelation factorizes if we assume independence. The combined spectrum would be $Q_{\text{total}}(k, \omega)$ from a Levy-process-like model. Could plug directly into the same Lorentzian framework with $\alpha_{\text{total}}(k) = D k^2 + \lambda (1 - J_0(2\pi k A))$ inside the autocorrelation exponent. This would give *one* spectrum for natural viewing.

3. **Cycle-averaged dynamics.** Mostofi notes the natural saccade-fixation cycle modulates the bandwidth of whitening over time (broad post-saccade with drift dominating, narrow during saccade transitions). Could compute time-resolved $Q$ and corresponding optimal filter dynamics — predicts coarse-to-fine dynamics in cell receptive fields over the fixation cycle.

4. **Cell-class-specific predictions.** Mostofi notes magnocellular vs parvocellular tuning differences could match the saccade-vs-drift differences. With our framework, optimize for two cell classes with different $f_{\max}$ values and see if the saccade-induced spatial kernel distinction predicts magno-vs-parvo.

5. **Aliasing under critical sampling.** From a previous turn, `scripts/check_aliasing_negligible.py` showed the m=0-dominant assumption doesn't hold at low $D$ under critical mosaic sampling. We left this honestly noted in the README rather than fixed. A proper treatment would require committing to a specific mosaic structure (single $k_s$ or per-cell-type), which is a modeling choice not yet made.

## Quick orientation for the new conversation

To pick up where we left off, the most useful files to read first are:

1. **`src/spectra.py`** — `saccade_redistribution` and `saccade_spectrum` are the new additions. Look at the docstrings; the math is fully documented there.

2. **`outputs/fig6c_saccade_vs_drift_kernels.png`** — answers the user's most-asked question (why don't saccade kernels look like drift kernels at high $D_{\text{eff}}$).

3. **`outputs/fig6b_saccade_diagnostics.png`** — confirms $Q$ has the right shape: $f^2$ rise, saturation, $1/A$ crossover scaling, drift-like comparison.

4. **`tests/test_spectra.py`** — the saccade tests in particular validate: drift limit at small $kA$, Lorentzian normalization, $f^2$ rise slope, $1/A$ crossover scaling.

A reasonable opening prompt for the next conversation: *"Continue work on the moving-sensor efficient-coding codebase at `/mnt/user-data/outputs/efficient_coding`. I have the saccade extension implemented per Mostofi et al. 2020 with the closed-form Poisson autocorrelation formula $Q = 2\lambda\alpha/((\lambda\alpha)^2 + \omega^2)$ where $\alpha = 1 - J_0(2\pi k A)$. Read the README and the saccade-related tests to orient. I want to [next thing]."*

## Things the new instance should NOT redo

- **Don't reimplement saccade Q with FFT and templates.** That was iterations 1 and 2; both had artifacts. The current closed-form Lorentzian is correct.
- **Don't add windowing parameters** to `saccade_redistribution`. The current API is `(f, omega, A, lam)`. No template, no window, no `n_theta`. Removed deliberately.
- **Don't try to "fix" the spatial-kernel difference between saccades and equivalent-$D_{\text{eff}}$ drift.** It's the correct physical prediction (saturation regime).

## Style preferences

The user has a strict writing blacklist (no "It's not just X; it's Y", no Corrective Escalation, no Forced Exaggeration, no Lexical Gentrification, etc.). Calm, direct, somewhat information-dense prose. Adjectives earn their place. No decorative sentences. Keep it rigorous but with some conversational tone where it helps clarity. The user pushes back hard when something looks wrong — that's productive, not adversarial. Treat unexpected results as signals to investigate, not to defend the implementation.
