"""Figure 7: trace-based Rucci/Boi cycle spectra.

This is the operational saccade/fixation-cycle input for the efficient-coding
solver:

    early fixation: C_early(f, omega) = I(f) Q_saccade(f, omega)
    late fixation:  C_late(f, omega)  = I(f) Q_drift(f, omega)

Both Q_saccade and Q_drift are estimated from explicit eye-position traces
with the same orientation-averaged Fourier-power estimator.  The stationary
Poisson saccade spectrum remains in src.spectra as an analytic control, but it
is not used here.
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from src.rucci_cycle_spectra import (
    make_figure7_rucci_cycle_spectra,
    temporal_power_integral,
    spatial_slope_loglog,
)
from src.plotting import setup_style

setup_style()


def _positive_omega(omega):
    mask = omega > 0
    return mask, omega[mask] / (2.0 * np.pi)


def _log_panel(ax, f, omega, Z, title, ylabel=False):
    mask, nu = _positive_omega(omega)
    Zp = np.asarray(Z)[:, mask]
    good = np.isfinite(Zp) & (Zp > 0)
    if not np.any(good):
        vmin, vmax = 1e-12, 1.0
    else:
        vmax = float(np.nanpercentile(Zp[good], 99.5))
        vmin = max(vmax * 1e-6, float(np.nanpercentile(Zp[good], 1.0)))
    levels = np.geomspace(vmin, vmax, 24)
    cf = ax.contourf(
        f,
        nu,
        np.maximum(Zp.T, vmin),
        levels=levels,
        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="magma",
        extend="both",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("spatial frequency f (cpd)")
    if ylabel:
        ax.set_ylabel("temporal frequency (Hz)")
    ax.set_title(title)
    return cf


def fig7():
    cycle = make_figure7_rucci_cycle_spectra()
    f = cycle.f
    omega = cycle.omega
    cycle.save_npz("outputs/rucci_cycle_spectra_demo.npz")

    p_sac = temporal_power_integral(cycle.Q_saccade_total, omega)
    nu_target = 8.0
    i_w = int(np.argmin(np.abs(omega / (2.0 * np.pi) - nu_target)))
    print("Rucci/Boi trace-cycle diagnostics")
    print(f"  saccade amplitude mean={cycle.saccade_amplitudes_deg.mean():.2f} deg, "
          f"sd={cycle.saccade_amplitudes_deg.std():.2f} deg")
    print(f"  drift D_eff={cycle.drift_D_eff_deg2_s:.4f} deg^2/s")
    print(f"  median integral Q_saccade_total={np.median(p_sac):.3f}")
    print("  Q_drift_total uses the smooth Brownian displacement-probability limit")
    print(f"  C_early_mod slope at {nu_target:g} Hz, 0.05-0.2 cpd: "
          f"{spatial_slope_loglog(f, cycle.C_early_mod[:, i_w], 0.05, 0.2):.2f}")
    print(f"  C_late_total slope at {nu_target:g} Hz, 0.2-5 cpd: "
          f"{spatial_slope_loglog(f, cycle.C_late_total[:, i_w], 0.2, 5.0):.2f}")

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2),
                             constrained_layout=True)
    _log_panel(axes[0, 0], f, omega, cycle.Q_saccade_mod,
               "early: Q_saccade, movement modulation", ylabel=True)
    cf = _log_panel(axes[0, 1], f, omega, cycle.C_early_mod,
                    "early: C = I(f) Q_saccade")
    _log_panel(axes[1, 0], f, omega, cycle.Q_drift_total,
               "late: Q_drift", ylabel=True)
    _log_panel(axes[1, 1], f, omega, cycle.C_late_total,
               "late: C = I(f) Q_drift")
    for ax in axes.ravel():
        ax.axhline(8.0, color="white", lw=0.6, ls=":", alpha=0.8)
    fig.colorbar(cf, ax=axes[:, 1], shrink=0.72,
                 label="spectral density, arbitrary units")

    out = "outputs/fig7_rucci_cycle_spectra.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    fig7()
