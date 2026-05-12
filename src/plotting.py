"""Small plotting and integration helpers."""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_style():
    mpl.rcParams.update({
        "font.size": 9,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 220,
    })


def trapezoid_weights_1d(x):
    x = np.asarray(x, dtype=float)
    if x.size == 1:
        return np.ones_like(x)
    w = np.empty_like(x)
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    return np.abs(w)


def radial_weights(f, tf_hz):
    """Simple radial integration weights for 2D spatial frequency + time.

    The constant 2π factors are omitted because they only rescale information and
    budget units.  The important part is the radial spatial measure f df dν.
    """
    f = np.asarray(f, dtype=float)
    tf_hz = np.asarray(tf_hz, dtype=float)
    wf = trapezoid_weights_1d(f)
    wt = trapezoid_weights_1d(tf_hz)
    return (f * wf)[:, None] * wt[None, :]


def band_mask_radial(f, tf_hz, f_max, tf_min_hz, tf_max_hz):
    f = np.asarray(f, dtype=float)
    tf_hz = np.asarray(tf_hz, dtype=float)
    return (f[:, None] <= f_max) & (np.abs(tf_hz)[None, :] >= tf_min_hz) & (np.abs(tf_hz)[None, :] <= tf_max_hz)


def radial_log_grid(n_f=180, n_tf=180, f_min=0.01, f_max=100.0, tf_min_hz=0.01, tf_max_hz=120.0):
    return np.geomspace(f_min, f_max, int(n_f)), np.geomspace(tf_min_hz, tf_max_hz, int(n_tf))


def _positive_branch(tf_hz, Z):
    tf_hz = np.asarray(tf_hz, dtype=float)
    Z = np.asarray(Z, dtype=float)
    if np.any(tf_hz < 0):
        keep = tf_hz > 0
        return tf_hz[keep], Z[:, keep]
    return tf_hz, Z


def panel_loglog(
    ax,
    f,
    tf_hz,
    Z,
    *,
    f_min=None,
    f_max=None,
    tf_min_hz=None,
    tf_max_hz=None,
    cmap="magma",
    vmin=None,
    vmax=None,
    n_levels=24,
    normalize=False,
    floor_rel=1e-10,
    colorbar=False,
    label=None,
):
    """Log-log contour plot for arrays with shape (len(f), len(tf_hz)).

    If ``tf_hz`` is symmetric, only positive temporal frequencies are shown.
    """
    f = np.asarray(f, dtype=float)
    tf_plot, Zp = _positive_branch(tf_hz, Z)
    Zp = np.asarray(Zp, dtype=float)

    good = np.isfinite(Zp) & (Zp > 0)
    zmax = np.nanmax(np.where(good, Zp, np.nan))
    if not np.isfinite(zmax) or zmax <= 0:
        zmax = 1.0

    if normalize:
        Zp = Zp / zmax
        zmax = 1.0

    if vmax is None:
        vmax = zmax
    if vmin is None:
        vmin = max(floor_rel * float(vmax), 1e-300)

    Zp = np.where(np.isfinite(Zp) & (Zp > 0), Zp, vmin)
    Zp = np.clip(Zp, vmin, vmax)
    levels = np.geomspace(max(vmin, 1e-300), max(vmax, vmin * 1.001), int(n_levels))

    cf = ax.contourf(f, tf_plot, Zp.T, levels=levels, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, extend="both")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(f_min if f_min is not None else f.min(), f_max if f_max is not None else f.max())
    ax.set_ylim(tf_min_hz if tf_min_hz is not None else tf_plot.min(), tf_max_hz if tf_max_hz is not None else tf_plot.max())
    ax.set_xlabel("spatial frequency f (cycles/deg)")
    ax.set_ylabel("temporal frequency ν (Hz)")
    if label is not None:
        ax.set_title(label)
    if colorbar:
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    return cf


def plot_spectrum_filter_pair(f, tf_hz, C, v_sq, band, *, title="", normalize=True):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    panel_loglog(axes[0], f, tf_hz, C, f_min=band.f_min, f_max=band.f_max, tf_min_hz=band.tf_min_hz, tf_max_hz=band.tf_max_hz, cmap="magma", normalize=normalize, label="Signal spectrum")
    panel_loglog(axes[1], f, tf_hz, v_sq, f_min=band.f_min, f_max=band.f_max, tf_min_hz=band.tf_min_hz, tf_max_hz=band.tf_max_hz, cmap="coolwarm", normalize=normalize, label="Optimal filter |v|²")
    if title:
        fig.suptitle(title)
    return fig, axes
