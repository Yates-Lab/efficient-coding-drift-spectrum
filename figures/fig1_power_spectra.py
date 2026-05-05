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

from src.spectra import DriftSpectrum, SaccadeSpectrum
from src.power_spectrum_library import (
    cycle_decomposition_panels,
    overlay_curve_hz,
    spectrum_library_panels,
)
from src.plotting import (
    add_log_colorbar,
    overlay_drift,
    panel_loglog,
    panel_loglog_hz,
    radial_log_grid,
    setup_style,
    shared_lims,
)

setup_style()
%matplotlib inline

#%%
# Editable plotting parameters. Change these, then rerun the downstream cells.
F_MIN, F_MAX = 0.1, 10.0
OMEGA_MIN, OMEGA_MAX = 0.25, 400.0
N_F, N_OMEGA = 200, 200
CMAP = "magma"
N_LEVELS = 24
FLOOR = 1e-6

DRIFT_DS = [0.05, 0.5, 2.0, 10.0, 50.0]
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

#%% Figure 1c: compute panels
from src.spectra import (
    DriftSpectrum,
    LinearMotionSpectrum,
    SaccadeSpectrum,
    SeparableMovieSpectrum,
)

spectrum_examples = [SeparableMovieSpectrum(omega0=0.05).C(f, omega),
    DriftSpectrum(D=2).C(f, omega),
    SaccadeSpectrum(A=3.0).C(f, omega),
    LinearMotionSpectrum(s=1.0).C(f, omega),
 ]

titles = ['Separable', 'Drift', 'Saccade', 'Linear']

_, fig1c_vmax = shared_lims(spectrum_examples, floor=FLOOR)

#%% Figure 1c: draw and inspect
fig, axes = plt.subplots(
    1,
    4,
    figsize=(10.2, 3.1),
    sharex=True,
    sharey=True,
    gridspec_kw={"wspace": 0.20},
)

for ax, C, title in zip(axes, spectrum_examples, titles):
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
    ax.set_title(title, pad=4)

for ax in axes:
    ax.set_xlabel(r"spatial frequency $f$ (cpd)")
axes[0].set_ylabel("temporal frequency (Hz)")

fig.subplots_adjust(left=0.06, right=0.93, top=0.84, bottom=0.18)
fig.suptitle(
    r"Figure 1c  spectrum library $C_\theta(f, \omega)$",
    y=0.99,
    fontsize=10.5,
)
add_log_colorbar(
    fig,
    [0.945, 0.20, 0.010, 0.60],
    cmap=CMAP,
    vmin=FLOOR,
    vmax=1.0,
    label=r"$C_\theta(f,\omega) / \max_\mathrm{panel}\,C_\theta$",
)
save_and_display(fig, "fig1c_library.png")

# %% Get the optimal filter
from src.pipeline import SolveConfig, run_many

spectrum_examples = [SeparableMovieSpectrum(omega0=0.05),
    DriftSpectrum(D=2),
    SaccadeSpectrum(A=3.0),
    LinearMotionSpectrum(s=1.0),
 ]

sigma_in = 0.2
sigma_out = 2.0
P0 = 50.0

s = SolveConfig(
            sigma_in=sigma_in, sigma_out=sigma_out, P0=P0,
            grid="hi_res",
        )

Res = run_many(spectrum_examples, s)



# %%
ax = plt.gca()
panel_loglog(ax, f, omega, Res[0].v_sq)
# %%
