#%%
"""Figure 1: on-retina power spectra C_theta(f, omega).

This file is intentionally written as an interactive percent-cell script.
Open it in VS Code, Spyder, or another editor that understands ``#%%`` cells,
then run cells one at a time. The parameter cell below is the main place to
tinker; rerun any downstream figure cell to redraw after edits.

Running the whole file as a normal script still writes the standard outputs:
  outputs/fig1a_main.png
  outputs/fig1b_boi_cycle.png
  outputs/fig1c_library.png
  outputs/fig1d_optimal_filters.png
"""

#%%
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.spectra import (
    DriftSpectrum,
    LinearMotionSpectrum,
    SaccadeSpectrum,
    SeparableMovieSpectrum,
)

from src.plotting import (
    add_log_colorbar,
    band_mask_radial,
    panel_loglog,
    parameter_palette,
    radial_log_grid,
    radial_weights,
    setup_style,
    shared_lims,
)
from src.pipeline import solve_on_grid, spatial_kernel_slice
from src.kernels import minimum_phase_temporal_filter, soft_band_taper
from src.cell_class_localized import fit_cell_classes_localized

setup_style()
%matplotlib inline

#%%
# Editable plotting parameters. Change these, then rerun the downstream cells.
F_MIN, F_MAX = 0.1, 10.0
OMEGA_MIN, OMEGA_MAX = 0.1, 400.0
N_F, N_OMEGA = 200, 200
CMAP = "magma"
N_LEVELS = 24
FLOOR = 1e-6

DRIFT_DS = [0.005, 0.05, 0.5, 1, 2.0]
SACCADE_AS = [0.5, 1.0, 2.0, 4.0, 8.0]

SAVE_FIGS = True
SHOW_FIGS = True
CLOSE_AFTER_SAVE = True
OUTDIR = Path("outputs")

CBAR_LABEL = (
    r"$C_\theta(f,\omega) / \max_{\mathrm{fig}}\,C_\theta$"
    r"  (per-figure max normalization)"
)


#%%
def save_and_display(fig, filename):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    if SAVE_FIGS:
        out = OUTDIR / filename
        fig.savefig(out)
        print(f"wrote {out}")
    if SHOW_FIGS:
        plt.show(block=False)
    if CLOSE_AFTER_SAVE and not SHOW_FIGS:
        plt.close(fig)
    return fig


def make_grid():
    return radial_log_grid(
        N_F,
        N_OMEGA,
        f_min=F_MIN,
        f_max=F_MAX,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
    )


def plot_panels(
    f,
    omega,
    arrays,
    labels,
    *,
    filename=None,
    title=None,
    cmap=CMAP,
    colorbar_label=None,
):
    arrays = list(arrays)
    labels = list(labels)
    _, vmax = shared_lims(arrays, floor=FLOOR)

    fig, axes = plt.subplots(
        1,
        len(arrays),
        figsize=(2.55 * len(arrays), 3.1),
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={"wspace": 0.20},
    )
    axes = axes.ravel()

    for ax, array, label in zip(axes, arrays, labels):
        panel_loglog(
            ax,
            f,
            omega,
            array / vmax,
            FLOOR,
            1.0,
            n_levels=N_LEVELS,
            cmap=cmap,
            f_min=F_MIN,
            f_max=F_MAX,
            omega_min=OMEGA_MIN,
            omega_max=OMEGA_MAX,
        )
        ax.set_title(label, pad=4)

    for ax in axes:
        ax.set_xlabel(r"$f$ (cycles/unit)")
    axes[0].set_ylabel(r"$\omega$ (rad/s)")

    fig.subplots_adjust(left=0.06, right=0.93, top=0.84, bottom=0.18)
    if title:
        fig.suptitle(title, y=0.99, fontsize=10.5)
    add_log_colorbar(
        fig,
        [0.945, 0.20, 0.010, 0.60],
        cmap=cmap,
        vmin=FLOOR,
        vmax=1.0,
        label=colorbar_label,
    )

    if filename is not None:
        save_and_display(fig, filename)
    return fig, axes


#%% Figure 1a: compute spectra
f, omega = make_grid()

row_drift = [DriftSpectrum(D=D).C(f, omega) for D in DRIFT_DS]
row_sacc = [SaccadeSpectrum(A=A).C(f, omega) for A in SACCADE_AS]
_, fig1a_vmax = shared_lims(row_drift + row_sacc, floor=FLOOR)

#%% Figure 1a: draw and inspect
n_cols = max(len(DRIFT_DS), len(SACCADE_AS))
fig, axes = plt.subplots(
    2,
    n_cols,
    figsize=(2.0 * n_cols, 4.6),
    sharex=True,
    sharey=True,
    squeeze=False,
    gridspec_kw={"hspace": 0.45, "wspace": 0.18},
)

for ax, C, D in zip(axes[0], row_drift, DRIFT_DS):
    ax.set_visible(True)
    panel_loglog(
        ax,
        f,
        omega,
        C / fig1a_vmax,
        FLOOR,
        1.0,
        n_levels=N_LEVELS,
        cmap=CMAP,
        f_min=F_MIN,
        f_max=F_MAX,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
    )
    ax.set_title(rf"$D = {D:g}$", pad=2)

for ax, C, A in zip(axes[1], row_sacc, SACCADE_AS):
    ax.set_visible(True)
    panel_loglog(
        ax,
        f,
        omega,
        C / fig1a_vmax,
        FLOOR,
        1.0,
        n_levels=N_LEVELS,
        cmap=CMAP,
        f_min=F_MIN,
        f_max=F_MAX,
        omega_min=OMEGA_MIN,
        omega_max=OMEGA_MAX,
    )
    ax.set_title(rf"$A = {A:g}$", pad=2)

# add labels
for ax in axes[-1]:
    if ax.get_visible():
        ax.set_xlabel(r"$f$ (cycles/unit)")

for row in axes:
    visible = [ax for ax in row if ax.get_visible()]
    if visible:
        visible[0].set_ylabel(r"$\omega$ (rad/s)")

fig.text(0.015, 0.715, "Diffusion\n(vary $D$)", rotation=90,
         fontsize=9, va="center", ha="center")
fig.text(0.015, 0.305, "Saccades\n(vary $A$)",
         rotation=90, fontsize=9, va="center", ha="center")
fig.subplots_adjust(left=0.07, right=0.91, top=0.91, bottom=0.10)
fig.suptitle(
    r"Figure 1a  on-retina power spectra $C_\theta(f, \omega)$",
    y=0.985,
    fontsize=10.5,
)
add_log_colorbar(
    fig,
    [0.93, 0.13, 0.012, 0.74],
    cmap=CMAP,
    vmin=FLOOR,
    vmax=1.0,
    label=CBAR_LABEL,
)


_ = save_and_display(fig, "fig1a_main.png")

#%% Figure 1c: choose spectra as objects
spectra_library = [
    ("Separable", SeparableMovieSpectrum(omega0=0.05)),
    ("Drift", DriftSpectrum(D=2)),
    ("Saccade", SaccadeSpectrum(A=3.0)),
    ("Linear", LinearMotionSpectrum(s=5.0)),
]
labels = [label for label, _ in spectra_library]

#%% Figure 1c: evaluate spectra on this grid
C_list = [spec.C(f, omega) for _, spec in spectra_library]

#%% Figure 1c: plot spectra
fig, axes = plot_panels(
    f,
    omega,
    C_list,
    labels,
    filename="fig1c_library.png",
    title=r"Figure 1c  spectrum library $C_\theta(f, \omega)$",
    colorbar_label=r"$C_\theta(f,\omega) / \max_\mathrm{set}\,C_\theta$",
)

#%% Choose efficient-coding parameters
sigma_in = 0.2
sigma_out = 2.0
P0 = 5.0


row_drift = [(f'Drift (D={D})' ,DriftSpectrum(D=D)) for D in DRIFT_DS]
row_sacc = [(f'Saccade (A={A})', SaccadeSpectrum(A=A)) for A in SACCADE_AS]
spectra = row_drift + row_sacc

C_list = [spec.C(f, omega) for _, spec in spectra]

#%% Solve optimal filters on this same grid
results = [
    solve_on_grid(
        spec,
        f,
        omega,
        sigma_in,
        sigma_out,
        P0,
        band=(F_MAX, OMEGA_MIN, OMEGA_MAX),
    )
    for _, spec in spectra
]

#%% Plot optimal filters
labels = [label for label, _ in spectra]

fig, axes = plot_panels(
    f,
    omega,
    C_list,
    labels,
    filename="fig1c_library.png",
    title=r"Figure 1c  spectrum library $C_\theta(f, \omega)$",
    colorbar_label=r"$C_\theta(f,\omega) / \max_\mathrm{set}\,C_\theta$",
)

fig, axes = plot_panels(
    f,
    omega,
    [r.v_sq for r in results],
    labels,
    filename="fig1d_optimal_filters.png",
    title=r"Optimal filters $|v^*(f,\omega)|^2$",
    cmap="viridis",
    colorbar_label=r"$|v^*|^2 / \max_\mathrm{set}\,|v^*|^2$",
)

filter_results = results
filter_labels = labels
filter_spectra = spectra
filter_C_list = C_list

# %%
DRIFT_DS = np.geomspace(.005, 500, 25)
P0 = 10.0
sigma_out = 2.0


sweep_spectra = [(f'Drift (D={D})' ,DriftSpectrum(D=D)) for D in DRIFT_DS]

for sigma_in in [0.1, 0.5, 1.0]:

    sweep_results = [
        solve_on_grid(
            spec,
            f,
            omega,
            sigma_in,
            sigma_out,
            P0,
            band=(F_MAX, OMEGA_MIN, OMEGA_MAX),
        )
        for _, spec in sweep_spectra
    ]

    I = np.array([r.I for r in sweep_results])
    h = plt.plot(DRIFT_DS, I/I.max(), label=f'$\sigma_\mathrm{{in}}={sigma_in}$')
    plt.plot(DRIFT_DS[np.where(I==I.max())[0][0]], 1.0, 'o', color=h[0].get_color()) 

plt.legend()
# xscale log
plt.xscale('log')

#%% Plot kernels in the space and time domain
KERNEL_F_SLICES = [0.15, 0.5, 1.5, 4.0]      # cycles/degree
KERNEL_TF_HZ_SLICES = [1.0, 4.0, 16.0, 32.0] # Hz
KERNEL_TIME_MAX = 0.35
KERNEL_SPACE_MAX = 8.0
KERNEL_N_OMEGA = 2048
KERNEL_TAPER_ALPHA = 0.25
KERNEL_FLOOR_REL = 1e-3


def temporal_kernel_at_f(result, f0):
    """Minimum-phase temporal kernel at the spatial bin nearest ``f0``.

    The display grid above is positive and log-spaced, while minimum-phase
    reconstruction wants a centered uniform omega grid.  For inspection we
    interpolate the positive-frequency magnitude onto a symmetric uniform grid.
    """
    i_f = int(np.argmin(np.abs(result.f - float(f0))))
    v_mag_pos = np.sqrt(np.maximum(result.v_sq[i_f], 0.0))

    omega_max = min(float(result.omega.max()), float(OMEGA_MAX))
    domega = 2.0 * omega_max / int(KERNEL_N_OMEGA)
    omega_centered = (np.arange(int(KERNEL_N_OMEGA)) - int(KERNEL_N_OMEGA) // 2) * domega
    v_mag = np.interp(
        np.abs(omega_centered),
        result.omega,
        v_mag_pos,
        left=v_mag_pos[0],
        right=0.0,
    )
    taper = soft_band_taper(
        omega_centered,
        OMEGA_MIN,
        omega_max,
        alpha=KERNEL_TAPER_ALPHA,
    )
    floor = KERNEL_FLOOR_REL * max(float(np.nanmax(v_mag * taper)), 1e-30)
    v_mag = np.maximum(v_mag * taper, floor)
    t, h, _ = minimum_phase_temporal_filter(v_mag, omega_centered)
    return t, h


def normalize_curve(y):
    denom = max(float(np.nanmax(np.abs(y))), 1e-30)
    return y / denom


colors = parameter_palette(len(filter_results), cmap="tab20", lo=0.02, hi=0.98)
fig, axes = plt.subplots(
    2,
    max(len(KERNEL_F_SLICES), len(KERNEL_TF_HZ_SLICES)),
    figsize=(3.0 * max(len(KERNEL_F_SLICES), len(KERNEL_TF_HZ_SLICES)), 5.6),
    squeeze=False,
    gridspec_kw={"hspace": 0.42, "wspace": 0.30},
)

for ax, f0 in zip(axes[0], KERNEL_F_SLICES):
    for result, label, color in zip(filter_results, filter_labels, colors):
        t, h = temporal_kernel_at_f(result, f0)
        ax.plot(t, normalize_curve(h), color=color, label=label)
    ax.set_xlim(0.0, KERNEL_TIME_MAX)
    ax.axhline(0.0, color="0.75", lw=0.5)
    ax.set_title(rf"temporal kernel at $f={f0:g}$")
    ax.set_xlabel(r"$t$ (s)")
    ax.set_ylabel(r"$v_t(t)$ / max")

for ax, tf_hz in zip(axes[1], KERNEL_TF_HZ_SLICES):
    omega0 = 2.0 * np.pi * float(tf_hz)
    for result, label, color in zip(filter_results, filter_labels, colors):
        r, v = spatial_kernel_slice(result, omega0=omega0)
        ax.plot(r, normalize_curve(v), color=color, label=label)
    ax.set_xlim(-KERNEL_SPACE_MAX, KERNEL_SPACE_MAX)
    ax.axhline(0.0, color="0.75", lw=0.5)
    ax.axvline(0.0, color="0.85", lw=0.5)
    ax.set_title(rf"spatial kernel at {tf_hz:g} Hz")
    ax.set_xlabel(r"$r$ (deg)")
    ax.set_ylabel(r"$v_s(r)$ / max")

for ax in axes[0, len(KERNEL_F_SLICES):]:
    ax.set_visible(False)
for ax in axes[1, len(KERNEL_TF_HZ_SLICES):]:
    ax.set_visible(False)

axes[0, 0].legend(loc="best", fontsize=6.5, ncol=1)
fig.suptitle("Kernel slices from the optimal filters", y=0.995, fontsize=11)
_ = save_and_display(fig, "fig1e_kernel_slices.png")


#%% Approximate Non-stationary Oracle with cell classes with localized bandwidth
K_CLASSES = 3
LOC_WEIGHT = 0.5
DELTA_MAX = 0.5
LEARN_BASELINE_SHARE = True
RETUNE_CLASSES = False
CELL_N_STEPS = 600
CELL_N_RESTARTS = 1
CELL_LR = 5e-2
CELL_DEVICE = "auto"
CELL_DTYPE = "float32"

oracle_C_stack = np.stack([r.C for r in filter_results], axis=0)
oracle_G_stack = np.stack([r.v_sq for r in filter_results], axis=0)
oracle_weights = radial_weights(f, omega) * band_mask_radial(
    f,
    omega,
    F_MAX,
    OMEGA_MIN,
    OMEGA_MAX,
)
condition_weights = np.ones(len(filter_results), dtype=float) / len(filter_results)

cell_fit = fit_cell_classes_localized(
    oracle_C_stack,
    oracle_weights,
    f,
    sigma_in=filter_results[0].sigma_in,
    sigma_out=filter_results[0].sigma_out,
    P0=filter_results[0].P0,
    K=K_CLASSES,
    condition_weights=condition_weights,
    G_star=oracle_G_stack,
    loc_weight=LOC_WEIGHT,
    delta_max=DELTA_MAX,
    learn_baseline_share=LEARN_BASELINE_SHARE,
    retune=RETUNE_CLASSES,
    n_steps=CELL_N_STEPS,
    n_restarts=CELL_N_RESTARTS,
    lr=CELL_LR,
    device=CELL_DEVICE,
    dtype=CELL_DTYPE,
    seed=1,
    check_every=25,
)

oracle_I = np.array([r.I for r in filter_results], dtype=float)
oracle_J = float(np.sum(condition_weights * oracle_I))
regret = (oracle_J - cell_fit.J) / max(abs(oracle_J), 1e-300)
print(f"Oracle J*: {oracle_J:.4g}")
print(f"K={K_CLASSES} localized-cell J: {cell_fit.J:.4g}")
print(f"Relative regret: {100 * regret:.2f}%")
print("Baseline class shares:", np.round(cell_fit.rho0, 3))
print("Class f centroids:", np.round(cell_fit.f_centroid_cpd, 3))

fig, axes = plot_panels(
    f,
    omega,
    [cell_fit.H[k] for k in range(K_CLASSES)],
    [f"class {k + 1}" for k in range(K_CLASSES)],
    filename="fig1f_cell_classes.png",
    title=rf"Localized cell classes, $K={K_CLASSES}$",
    cmap="viridis",
    colorbar_label=r"$H_c(f,\omega) / \max_c H_c$",
)

fig, axes = plot_panels(
    f,
    omega,
    list(cell_fit.G),
    filter_labels,
    filename="fig1g_cell_class_filters.png",
    title=rf"Cell-class approximation to oracle filters, $K={K_CLASSES}$",
    cmap="viridis",
    colorbar_label=r"$G_q(f,\omega) / \max_q G_q$",
)

# %%
