# Bare-bones moving-retina efficient coding

This repo is intentionally small.  It keeps only the pieces needed to understand
and manipulate the current model:

1. signal spectra: drift, saccade, linear motion, separable movie;
2. noise spectra: white, 1/f, cone-like approximation;
3. the closed-form optimal-filter solver under a response-power budget;
4. plotting and minimum-phase/zero-phase kernel reconstruction helpers.

The human-facing source of truth is:

```bash
scripts/interactive_moving_retina.py
```

Run it cell-by-cell in VS Code, Spyder, Jupyter, or any editor that understands
`#%%` cell markers.

## Units

Public units are fixed everywhere:

```text
spatial frequency f : cycles / degree
frequency ν         : Hz = cycles / second
```

The only place angular factors appear is inside formulas that need phase:

```text
spatial phase = 2π f x
temporal angular frequency = 2πν
```

## Main files

```text
src/params.py      small Band helper and grid construction
src/spectra.py     DriftSpectrum, SaccadeSpectrum, LinearMotionSpectrum, SeparableMovieSpectrum
src/noise.py       WhiteNoise, TemporalPowerLawNoise, ConeLikeNoise
src/solver.py      analytic water-filling / Linsker-Jun solver
src/kernels.py     zero-phase spatial and minimum-phase temporal reconstruction
src/plotting.py    plotting and integration weights
```

There are no hyperparameter sweeps, no cell-class optimizer, no spectrum library
indirection, and no mosaic model in this stripped-down pass.

## Run

```bash
python -m pytest -q
MPLBACKEND=Agg python scripts/interactive_moving_retina.py
```

The script is written for interactive use, so running it as a script mostly checks
that the code path is clean.
