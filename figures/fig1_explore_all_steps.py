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
    panel_loglog,
    radial_log_grid,
    setup_style,
    shared_lims,
)
from src.pipeline import solve_on_grid

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

# %%
DRIFT_DS = np.geomspace(.005, 500, 25)
P0 = 10.0
sigma_out = 2.0


spectra = [(f'Drift (D={D})' ,DriftSpectrum(D=D)) for D in DRIFT_DS]

for sigma_in in [0.1, 0.5, 1.0]:

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

    I = np.array([r.I for r in results])
    h = plt.plot(DRIFT_DS, I/I.max(), label=f'$\sigma_\mathrm{{in}}={sigma_in}$')
    plt.plot(DRIFT_DS[np.where(I==I.max())[0][0]], 1.0, 'o', color=h[0].get_color()) 

plt.legend()
# xscale log
plt.xscale('log')

#%% Plot kernels in the space and time domain



    

    
#%% Approximate Non-stationary Oracle with cell classes with localized bandwidth
