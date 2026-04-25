# drift_filter

Numerical exploration of the efficient-coding filter under Brownian retinal
drift, following the notes in `Jun_2022_analytic_results_notes.pdf`.

## Scope

Two questions:

1. How does drift change the optimal encoding?
2. Given a fixed encoding, how does drift change mutual information?

Joint optimization of drift and representation is a downstream goal and is
not addressed here.

## Layout

Core modules:
- `spectrum.py` — regularized static image spectrum C_I(k) and drift spectrum
  C_D(k, omega).
- `grids.py` — 1D radial frequency grid and integration weights.
- `optimizer.py` — Eq. (10) filter magnitude and the lambda(D) constraint
  solver.
- `phase.py` — minimum-phase reconstruction (Eq. 20) and inverse Fourier
  transform with the e^{+i omega t} convention used in the notes.
- `information.py` — MI density (Eq. 54) and I(D) sweep.
- `style.py` — matplotlib publication defaults.

Validation:
- `validate.py` — Lorentzian normalization, rectification, lambda solver,
  whitening limit, scale invariance. All pass.
- `validate_phase.py` — minimum-phase reconstruction against a closed-form
  Lorentzian case. Passes.

Figures:
- `fig1_filter_magnitude.py` — log |v*|^2 on (k, omega) for D in {0.08, 0.8,
  8, 80}. Shows how drift shifts power from high-omega / all-k to wider-k /
  lower-omega.
- `fig2_temporal_kernels.py` — causal temporal kernels v*(k, t) at three
  representative spatial frequencies, for the same D values. Kernels
  sharpen by ~25x as D grows four decades.
- `fig3_info_vs_drift.py` — I(D) with the optimal filter re-trained at each
  D (envelope), alongside I(D) curves for filters trained at fixed D_fit
  and rescaled to meet the power budget at each test D. Recovers the
  information-maximizing D* predicted in Section 3.5.
- `fig4_scaling.py` — empirical active-set scalings k_max ~ D^-1/4 and
  omega_max ~ D^1/2 (Eq. 49 / Section 3.3). Match predicted exponents to
  three significant figures over the fit region.

## Running

```
python run_all.py
```

or individually. All scripts are plain numpy/scipy + matplotlib.

## Two notes for the paper

1. The whitening limit of Section 2.5 actually requires three conditions,
   not two:
     (a) C_x >> sigma_in^2                   (stated)
     (b) 4 sigma_in^2 / (lam sigma_out^2 C_x) << 1    (stated)
     (c) lam sigma_out^2 << 1                (not stated)
   Condition (c) is needed because the next-order expansion gives
   |v*|^2 ~ 1/(lam C_x) - sigma_out^2/C_x, so the whitening term dominates
   only when 1/lam >> sigma_out^2.

2. The minimum-phase construction in Section 2.6 requires regularization
   of log|v(k, omega)| on the inactive set before the Hilbert transform.
   Our implementation adds a floor of 1e-10 * max(|v|^2).
```
