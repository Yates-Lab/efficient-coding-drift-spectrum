#%%
"""Bare-bones moving-retina efficient-coding walkthrough.

Run this file cell-by-cell.  The whole point is that a human can see every
piece of the calculation:

1. make a frequency grid;
2. instantiate a signal spectrum directly;
3. instantiate a noise spectrum directly;
4. solve the analytic optimal-filter problem;
5. plot the signal, noise, filter, and reconstructed kernel slices.

Public units throughout:
    spatial frequency f : cycles / degree
    temporal frequency ν : Hz
"""

#%% Imports
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.params import Band
from src.spectra import (
    ImageParams,
    DriftSpectrum,
    SaccadeSpectrum,
    LinearMotionSpectrum,
    SeparableMovieSpectrum,
    saccade_main_sequence_duration,
    saccade_smoothing_sigma,
)
from src.noise import WhiteNoise, TemporalPowerLawNoise, ConeLikeNoise
from src.solver import solve_on_grid, response_power_spend, mutual_information_density
from src.plotting import setup_style, panel_loglog, plot_spectrum_filter_pair, radial_weights, band_mask_radial
from src.kernels import spatial_kernel_slice, temporal_kernel_slice, default_slice_frequencies
from src.mp_kernels import mp_filter_power

setup_style()

#%% Main knobs
band = Band(
    f_min=0.01,
    f_max=100.0,
    tf_min_hz=0.01,
    tf_max_hz=120.0,
)

n_f = 120
n_tf_pos = 360
f, tf_hz = band.log_symmetric_grid(n_f=n_f, n_tf_pos=n_tf_pos, tf_max_grid_hz=band.tf_max_hz)

image = ImageParams(beta=2.0, A_image=1.0, f0=1e-6)

# Physiological movement defaults.  Keep these editable.
D_arcmin2_per_s = 40.0
D_deg2_per_s = D_arcmin2_per_s / 3600.0
saccade_A_deg = 3.5

# Efficient-coding knobs.  These are not yet biologically pinned.
P0 = 50.0

output_sigma = 0.05

print("units")
print("  f      : cycles/degree")
print("  ν      : Hz")
print(f"  drift : {D_arcmin2_per_s:g} arcmin²/s = {D_deg2_per_s:.8f} deg²/s")
print(f"  saccade A={saccade_A_deg:g} deg, T={1000*saccade_main_sequence_duration(saccade_A_deg):.2f} ms, sigma={1000*saccade_smoothing_sigma(saccade_A_deg):.2f} ms")

#%% Make the four signal spectra directly
# No library indirection.  These are the four class calls.
drift = DriftSpectrum(D=D_deg2_per_s, image=image)
saccade = SaccadeSpectrum(A=saccade_A_deg, image=image)
linear_motion = LinearMotionSpectrum(s=1.0, image=image)
separable = SeparableMovieSpectrum(image=image, beta_t=2.0, tf0_hz=0.05)

spectra = [
    ("drift", drift),
    ("saccade", saccade),
    ("linear motion", linear_motion),
    ("separable movie", separable),
]

#%% Make the noise spectra directly
input_sigma = 0.01
white_noise = WhiteNoise.from_sigma(input_sigma)
powerlaw_noise = TemporalPowerLawNoise.from_sigma(
    input_sigma,
    alpha=1.0,
    corner_hz=1.0,
    floor_hz=0.03,
)
cone_noise = ConeLikeNoise.from_sigma(
    input_sigma,
    low_freq_gain=1.0,
    alpha=1.0,
    corner_hz=1.0,
    floor_hz=0.03,
    high_freq_gain=0.15,
    high_corner_hz=30.0,
)
output_noise = WhiteNoise.from_sigma(output_sigma)

# Pick the one you want for the oracle.
input_noise = cone_noise

#%% Plot signal-spectrum library
fig, axes = plt.subplots(1, len(spectra), figsize=(3.0 * len(spectra), 3.2), constrained_layout=True)
for ax, (name, spec) in zip(axes, spectra):
    panel_loglog(
        ax, f, tf_hz, spec.C(f, tf_hz),
        f_min=band.f_min, f_max=band.f_max,
        tf_min_hz=band.tf_min_hz, tf_max_hz=band.tf_max_hz,
        cmap="magma", normalize=True, label=name,
        floor_rel=1e-10,
    )
fig.suptitle("signal spectra C(f,ν), normalized per panel")

#%% Plot the noise models
fig, axs = plt.subplots(1, 3, figsize=(7.5, 3.3), constrained_layout=True)
tf_pos = np.geomspace(band.tf_min_hz, band.tf_max_hz, 600)
for name, ax, noise in [("white", axs[0], white_noise), ("1/f", axs[1], powerlaw_noise), ("cone-like", axs[2], cone_noise)]:
    ax.plot(tf_pos, drift.C(10*np.ones_like(tf_pos), tf_pos)[0], label="drift @ 10 cpd")
    ax.plot(tf_pos, noise.power(np.array([1.0]), tf_pos)[0], label='noise')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()
    ax.set_xlabel("temporal frequency ν (Hz)")
    ax.set_ylabel("input-noise power")
    ax.set_title(name)

#%% Solve one spectrum/noise pair
# Change these two lines while exploring.
spec = drift
input_noise = cone_noise

r = solve_on_grid(
    spec,
    f,
    tf_hz,
    P0=P0,
    input_noise=input_noise,
    output_noise=output_noise,
    band=band.edges,
)

weights = radial_weights(f, tf_hz) * band_mask_radial(f, tf_hz, *band.edges)
spend = response_power_spend(r.C, r.v_sq, weights, r.input_noise_power)
print(f"information I = {r.I:.6g} nats")
print(f"lambda        = {r.lam:.6g}")
print(f"budget spend  = {spend:.6g} / P0={P0:g}")

fig, axes = plot_spectrum_filter_pair(f, tf_hz, r.C, r.v_sq, band, title="one oracle solve", normalize=True)

#%% Compare several oracle filters side by side
cases = [
    ("saccade + white", saccade, white_noise),
    ("saccade + cone", saccade, cone_noise),
    ("drift + white", drift, white_noise),
    ("drift + cone", drift, cone_noise),
]

fig, axes = plt.subplots(2, len(cases), figsize=(3.1 * len(cases), 6.0), constrained_layout=True)
results = []
for j, (name, spec_i, noise_i) in enumerate(cases):
    ri = solve_on_grid(spec_i, f, tf_hz, P0=P0, input_noise=noise_i, output_noise=output_noise, band=band.edges)
    results.append((name, ri))
    panel_loglog(axes[0, j], f, tf_hz, ri.C, f_min=band.f_min, f_max=band.f_max, tf_min_hz=band.tf_min_hz, tf_max_hz=band.tf_max_hz, cmap="magma", normalize=True, label=name + "\nC")
    panel_loglog(axes[1, j], f, tf_hz, ri.v_sq, f_min=band.f_min, f_max=band.f_max, tf_min_hz=band.tf_min_hz, tf_max_hz=band.tf_max_hz, cmap="coolwarm", normalize=True, label="|v|²")

#%% Visualize published M/P kernels on the same grid
mp_gamma = 1.0
mp_rho = 1.0
fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.2), constrained_layout=True)
for ax, cell in zip(axes, ["M", "P"]):
    panel_loglog(
        ax,
        f,
        tf_hz,
        mp_filter_power(f, tf_hz, cell, gamma=mp_gamma, rho=mp_rho),
        f_min=band.f_min,
        f_max=band.f_max,
        tf_min_hz=band.tf_min_hz,
        tf_max_hz=band.tf_max_hz,
        cmap="coolwarm",
        normalize=True,
        label=f"{cell} kernel |K(f)H(ν)|²",
    )
fig.suptitle(f"published M/P kernels, normalized per panel (gamma={mp_gamma:g}, rho={mp_rho:g})")

#%% Reconstruct spatial and temporal slices from |v|²
# Spatial: zero phase, centered at x=0.
# Temporal: minimum phase, using the Hilbert/cepstrum construction.
name, ri = results[2]  # edit this index or use r from the single-solve cell
f0 = 10
tf0 = 7
print(f"representative slice frequencies for {name}: f0={f0:.3g} cpd, ν0={tf0:.3g} Hz")

x, spatial_slice, spatial_image = spatial_kernel_slice(ri.f, ri.tf_hz, ri.v_sq, tf0, f_max=band.f_max, n=512)
t, temporal_slice, temporal_spectrum, tf_uniform = temporal_kernel_slice(
    ri.f, ri.tf_hz, ri.v_sq, f0,
    tf_min_hz=band.tf_min_hz,
    tf_max_hz=band.tf_max_hz,
    n_uniform=4096,
)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
axes[0].plot(x, spatial_slice / max(np.max(np.abs(spatial_slice)), 1e-300))
axes[0].axhline(0, color="0.75", lw=0.7)
axes[0].axvline(0, color="0.85", lw=0.7)
axes[0].set_xlim(-.5, .5)
axes[0].set_ylim(-.25)
axes[0].set_xlabel("space (deg)")
axes[0].set_ylabel("normalized amplitude")
axes[0].set_title(f"spatial slice at ν≈{tf0:.2g} Hz")

axes[1].plot(t, temporal_slice / max(np.max(np.abs(temporal_slice)), 1e-300))
axes[1].axhline(0, color="0.75", lw=0.7)
axes[1].set_xlim(0, 0.35)
axes[1].set_xlabel("time (s)")
axes[1].set_ylabel("normalized amplitude")
axes[1].set_title(f"minimum-phase temporal slice at f≈{f0:.2g} cpd")

#%% Show information density for the selected solve

fig, ax = plt.subplots(1, len(results),figsize=(10.4, 3.5), constrained_layout=True)
for j, (name, r) in enumerate(results):
    info_density = mutual_information_density(r.C, r.v_sq, r.input_noise_power, r.output_noise_power)
    panel_loglog(
        ax[j], f, tf_hz, info_density,
        f_min=band.f_min, f_max=band.f_max,
        tf_min_hz=band.tf_min_hz, tf_max_hz=band.tf_max_hz,
        cmap="viridis", normalize=True, label="information density",
    )

# %%
