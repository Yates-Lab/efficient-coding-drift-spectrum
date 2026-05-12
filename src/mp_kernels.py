"""M- and P-cell retinal ganglion-cell kernels.

The parameters and equations here follow the difference-of-Gaussians spatial
kernel and Victor/Benardete-Kaplan temporal cascade used by Casile, Victor &
Rucci (2019), eLife 8:e40924.

Public units:
    spatial frequency f : cycles / degree
    temporal frequency  : Hz

Default scalings use the unscaled Kaplan-style parameterization:
    gamma = 1.0 for the Croner/Kaplan spatial DoG
    rho   = 1.0 for the Benardete/Kaplan temporal cascade

Casile et al. used gamma=0.5 and rho=1/1.6 for their foveal/large-stimulus
adjustments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class SpatialParams:
    """Difference-of-Gaussians spatial parameters."""

    rc: float
    Kc: float
    rs: float
    Ks: float


@dataclass(frozen=True)
class TemporalParams:
    """Temporal cascade parameters."""

    N: int
    A: float
    D_ms: float
    Hs: float
    tau_L_ms: float
    tau_S_ms: float


SPATIAL: Dict[str, SpatialParams] = {
    "M": SpatialParams(rc=0.10, Kc=148.0, rs=0.72, Ks=1.1),
    "P": SpatialParams(rc=0.03, Kc=353.2, rs=0.18, Ks=4.4),
}

TEMPORAL: Dict[str, TemporalParams] = {
    "M": TemporalParams(N=30, A=499.77, D_ms=2.0, Hs=1.0, tau_L_ms=1.1, tau_S_ms=2.23),
    "P": TemporalParams(N=38, A=67.59, D_ms=3.5, Hs=0.69, tau_L_ms=1.27, tau_S_ms=29.36),
}


def _cell_key(cell: str) -> str:
    key = str(cell).upper()
    if key not in SPATIAL:
        raise ValueError("cell must be 'M' or 'P'.")
    return key


def spatial_kernel_K(
    f_cpd: np.ndarray | float,
    cell: str,
    gamma: float = 1.0,
    C: float = 1.0,
    dog_form: str = "fourier",
) -> np.ndarray:
    """Spatial frequency kernel K(f), Eq. 3 in Casile et al.

    ``dog_form="fourier"`` uses the conventional Fourier-domain circular
    Gaussian exponent and matches the figure-like profiles. ``"printed"``
    keeps the exponent as it appears in the paper text.
    """

    p = SPATIAL[_cell_key(cell)]
    f_eff = gamma * np.asarray(f_cpd, dtype=float)
    f2 = np.abs(f_eff) ** 2

    if dog_form == "fourier":
        center_exp = -((np.pi * p.rc) ** 2) * f2
        surround_exp = -((np.pi * p.rs) ** 2) * f2
    elif dog_form == "printed":
        center_exp = -np.pi * p.rc * f2
        surround_exp = -np.pi * p.rs * f2
    else:
        raise ValueError("dog_form must be 'fourier' or 'printed'.")

    center = p.Kc * np.pi * p.rc**2 * np.exp(center_exp)
    surround = p.Ks * np.pi * p.rs**2 * np.exp(surround_exp)
    return C * (center - surround)


def temporal_kernel_H(
    tf_hz: np.ndarray | float,
    cell: str,
    rho: float = 1.0,
) -> np.ndarray:
    """Complex temporal frequency kernel H(nu), Eq. 4 in Casile et al."""

    p = TEMPORAL[_cell_key(cell)]
    f = np.asarray(tf_hz, dtype=float)
    s = 1j * rho * 2.0 * np.pi * f

    D = p.D_ms * 1e-3
    tau_L = p.tau_L_ms * 1e-3
    tau_S = p.tau_S_ms * 1e-3

    delay = np.exp(s * D)
    high_pass = 1.0 - p.Hs / (1.0 + s * tau_S)
    low_pass_cascade = (1.0 / (1.0 + s * tau_L)) ** p.N
    return p.A * delay * high_pass * low_pass_cascade


def spatiotemporal_RF(
    sf_cpd: np.ndarray,
    tf_hz: np.ndarray,
    cell: str,
    gamma: float = 1.0,
    rho: float = 1.0,
    dog_form: str = "fourier",
) -> np.ndarray:
    """Separable transfer function RF(f,nu)=K(f)H(nu).

    This preserves the shape convention of the original standalone script:
    ``(n_temporal_frequency, n_spatial_frequency)``.
    """

    K = spatial_kernel_K(sf_cpd, cell, gamma=gamma, dog_form=dog_form)[None, :]
    H = temporal_kernel_H(tf_hz, cell, rho=rho)[:, None]
    return K * H


def mp_filter_power(
    sf_cpd: np.ndarray,
    tf_hz: np.ndarray,
    cell: str,
    gamma: float = 1.0,
    rho: float = 1.0,
    dog_form: str = "fourier",
) -> np.ndarray:
    """Return |K(f)H(nu)|^2 shaped like repo grids: ``(len(f), len(tf_hz))``."""

    return np.abs(
        spatiotemporal_RF(sf_cpd, tf_hz, cell, gamma=gamma, rho=rho, dog_form=dog_form)
    ).T ** 2


def power_to_db(power: np.ndarray, floor_db: float = -50.0) -> np.ndarray:
    """Normalize a nonnegative power array to its maximum and convert to dB."""

    power = np.asarray(power, dtype=float)
    ref = np.nanmax(power)
    if not np.isfinite(ref) or ref <= 0:
        raise ValueError("Power array must contain at least one positive finite value.")
    floor_linear = 10.0 ** (floor_db / 10.0)
    return 10.0 * np.log10(np.maximum(power / ref, floor_linear))


def raised_cosine_envelope(t: np.ndarray, ramp_s: float = 0.6) -> np.ndarray:
    """Smooth contrast onset/offset envelope."""

    t = np.asarray(t, dtype=float)
    duration_s = float(t[-1] - t[0] + (t[1] - t[0]))
    env = np.ones_like(t)

    if ramp_s <= 0:
        return env
    if 2.0 * ramp_s > duration_s:
        raise ValueError("ramp_s must be no more than half the stimulus duration.")

    up = t < ramp_s
    down = t > (duration_s - ramp_s)
    env[up] = 0.5 - 0.5 * np.cos(np.pi * t[up] / ramp_s)
    env[down] = 0.5 - 0.5 * np.cos(np.pi * (duration_s - t[down]) / ramp_s)
    return env


def simulate_brownian_drift(
    t: np.ndarray,
    D_arcmin2_s: float = 250.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate 2D fixational drift as Brownian retinal image motion."""

    t = np.asarray(t, dtype=float)
    if rng is None:
        rng = np.random.default_rng()
    dt = float(np.median(np.diff(t)))
    D_deg2_s = D_arcmin2_s / 60.0**2

    step_sd = np.sqrt(2.0 * D_deg2_s * dt)
    x = np.cumsum(rng.normal(0.0, step_sd, size=t.size))
    y = np.cumsum(rng.normal(0.0, step_sd, size=t.size))
    x -= x.mean()
    y -= y.mean()
    return x, y


def retinal_input_power_map(
    sf_cpd: np.ndarray,
    duration_s: float = 3.2,
    sample_rate_hz: float = 1000.0,
    temporal_mod_hz: float | None = 6.0,
    D_arcmin2_s: float = 250.0,
    n_trials: int = 16,
    n_orientations: int = 8,
    n_phases: int = 4,
    ramp_s: float = 0.6,
    seed: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate retinal input power for gratings jittered by Brownian drift.

    Returns ``tf_hz`` and an input-power array shaped
    ``(n_temporal_frequency, n_spatial_frequency)``.
    """

    sf_cpd = np.asarray(sf_cpd, dtype=float)
    rng = np.random.default_rng(seed)

    t = np.arange(0.0, duration_s, 1.0 / sample_rate_hz)
    tf_hz = np.fft.rfftfreq(t.size, d=1.0 / sample_rate_hz)
    env = raised_cosine_envelope(t, ramp_s=ramp_s)

    if temporal_mod_hz is None or temporal_mod_hz == 0:
        temporal_mod = np.ones_like(t)
    else:
        temporal_mod = np.sin(2.0 * np.pi * temporal_mod_hz * t)

    orientations = np.linspace(0.0, 2.0 * np.pi, n_orientations, endpoint=False)
    phases = np.linspace(0.0, 2.0 * np.pi, n_phases, endpoint=False)
    phase_grid = phases[:, None]

    P_input = np.zeros((tf_hz.size, sf_cpd.size), dtype=float)
    denom = n_trials * n_orientations * n_phases

    for _ in range(n_trials):
        eye_x, eye_y = simulate_brownian_drift(t, D_arcmin2_s=D_arcmin2_s, rng=rng)

        for alpha in orientations:
            projection_deg = eye_x * np.cos(alpha) + eye_y * np.sin(alpha)

            for j, f in enumerate(sf_cpd):
                phase_t = 2.0 * np.pi * f * projection_deg
                stim = np.sin(phase_t[None, :] + phase_grid)
                stim *= temporal_mod[None, :] * env[None, :]

                F = np.fft.rfft(stim, axis=1)
                P_input[:, j] += np.mean(np.abs(F) ** 2, axis=0) / denom

    return tf_hz, P_input


def response_power_from_input(
    sf_cpd: np.ndarray,
    tf_hz: np.ndarray,
    P_input: np.ndarray,
    cell: str,
    low_tf_cutoff_hz: float = 0.63,
    gamma: float = 1.0,
    rho: float = 1.0,
    dog_form: str = "fourier",
) -> np.ndarray:
    """Compute response power ``P_input * |RF_cell|^2``.

    Frequencies below ``low_tf_cutoff_hz`` are set to zero, matching the paper's
    decision to discard the first two temporal samples so the integral began
    near 0.63 Hz.
    """

    O = np.asarray(P_input, dtype=float) * np.abs(
        spatiotemporal_RF(sf_cpd, tf_hz, cell, gamma=gamma, rho=rho, dog_form=dog_form)
    ) ** 2
    O[np.asarray(tf_hz, dtype=float) < low_tf_cutoff_hz, :] = 0.0
    return O


def format_log_axes(ax, xlabel: str | None = None, ylabel: str | None = None) -> None:
    """Apply plain-number tick labels to log axes used by M/P plots."""

    from matplotlib.ticker import ScalarFormatter

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())


def plot_raw_kernels(
    out_png: str = "mp_raw_kernels.png",
    gamma: float = 1.0,
    rho: float = 1.0,
) -> None:
    """Visualize K(f), H(nu), and raw |K(f)H(nu)|^2."""

    import matplotlib.pyplot as plt

    sf = np.logspace(np.log10(0.3), np.log10(60.0), 240)
    tf = np.logspace(np.log10(0.3), np.log10(100.0), 260)

    fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.2), constrained_layout=True)

    for row, cell in enumerate(["M", "P"]):
        K = np.abs(spatial_kernel_K(sf, cell, gamma=gamma, dog_form="fourier"))
        H = np.abs(temporal_kernel_H(tf, cell, rho=rho))
        RF_power = np.abs(
            spatiotemporal_RF(sf, tf, cell, gamma=gamma, rho=rho, dog_form="fourier")
        ) ** 2
        RF_db = power_to_db(RF_power, floor_db=-50)

        ax = axes[row, 0]
        ax.plot(sf, K / K.max(), linewidth=2)
        ax.set_xscale("log")
        ax.set_title(f"{cell} spatial kernel K(f)")
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0.3, 1, 3, 10, 30, 60])
        format_log_axes(ax, "spatial frequency (cpd)", "normalized |K|")

        ax = axes[row, 1]
        ax.plot(tf, H / H.max(), linewidth=2)
        ax.set_xscale("log")
        ax.set_title(f"{cell} temporal kernel H(nu)")
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0.3, 1, 3, 10, 30, 100])
        format_log_axes(ax, "temporal frequency (Hz)", "normalized |H|")

        ax = axes[row, 2]
        SF, TF = np.meshgrid(sf, tf)
        pcm = ax.pcolormesh(SF, TF, RF_db, shading="auto", cmap="hot", vmin=-50, vmax=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{cell} raw RF power |K*H|^2")
        ax.set_xticks([0.3, 1, 3, 10, 30, 60])
        ax.set_yticks([0.3, 1, 3, 10, 30, 100])
        format_log_axes(ax, "spatial frequency (cpd)", "temporal frequency (Hz)")
        fig.colorbar(pcm, ax=ax, label="power (dB)")

    fig.savefig(out_png, dpi=180)
    print(f"Saved {out_png}")


def plot_figure4c_like_response(
    out_png: str = "mp_figure4c_like_response.png",
    gamma: float = 1.0,
    rho: float = 1.0,
) -> None:
    """Make a Figure-4C-like M/P response-power visualization."""

    import matplotlib.pyplot as plt

    sf = np.logspace(np.log10(0.3), np.log10(30.0), 90)

    print("Simulating retinal input power; this can take ~10-30 seconds...")
    tf_fft, P_input = retinal_input_power_map(
        sf,
        duration_s=3.2,
        sample_rate_hz=1000.0,
        temporal_mod_hz=6.0,
        D_arcmin2_s=250.0,
        n_trials=16,
        n_orientations=8,
        n_phases=4,
        ramp_s=0.6,
        seed=4,
    )

    keep = (tf_fft >= 0.5) & (tf_fft <= 50.0)
    tf = tf_fft[keep]

    fig, axes = plt.subplots(2, 1, figsize=(5.6, 7.0), constrained_layout=True)

    for ax, cell in zip(axes, ["M", "P"]):
        O = response_power_from_input(
            sf,
            tf_fft,
            P_input.copy(),
            cell,
            gamma=gamma,
            rho=rho,
        )[keep, :]
        O_db = power_to_db(O, floor_db=-50)
        SF, TF = np.meshgrid(sf, tf)
        pcm = ax.pcolormesh(SF, TF, O_db, shading="auto", cmap="hot", vmin=-50, vmax=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{cell} cells")
        ax.set_xlim(0.3, 30)
        ax.set_ylim(0.5, 50)
        ax.set_xticks([1, 3, 10, 30])
        ax.set_yticks([1, 3, 10, 50])
        format_log_axes(ax, "spatial frequency (cpd)", "temporal frequency (Hz)")

    fig.colorbar(pcm, ax=axes, label="power (dB)")
    fig.savefig(out_png, dpi=180)
    print(f"Saved {out_png}")
