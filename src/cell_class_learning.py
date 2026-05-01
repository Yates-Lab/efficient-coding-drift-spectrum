"""Information-aware learning of reusable cell-class filters.

This module is designed to drop into the current moving-sensor efficient-coding
repo. It learns a small set of fixed filter-power spectra H_c(f, omega) whose
condition-dependent mixtures approximate the oracle efficient-coding filters
for a stack of movement conditions.

The learned object is not a space-time separable filter. Each class spectrum
H_c(f, omega) is fully nonseparable unless you add that constraint later.

Model
-----
For condition q and class c,

    G_raw[q] = sum_c alpha[q, c] * H[c]

where H[c] >= 0 and alpha[q] lies on the simplex. The effective filter power is
then rescaled so each condition uses the same response-power budget P0:

    G[q] = scale[q] * G_raw[q]
    sum G[q] * (C[q] + sigma_in^2) * weights = P0

The objective maximizes the actual Gaussian mutual information under C[q], not
squared-error reconstruction of the oracle filters.

This file imports torch only inside the optimizer functions, so the rest of the
repo can still import this module without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.params import F_MAX, OMEGA_MIN, OMEGA_MAX
from src.pipeline import Result, run
from src.plotting import radial_weights, band_mask_radial
from src.power_spectrum_library import cycle_solver_spectra


Array = np.ndarray


@dataclass(frozen=True)
class Condition:
    """One movement condition in the class-learning stack."""

    name: str
    epoch: str
    parameter_name: str
    parameter_value: float
    spectrum: object


@dataclass
class OracleStack:
    """Oracle one-filter solutions for all conditions."""

    conditions: List[Condition]
    results: List[Result]
    f: Array
    omega: Array
    weights: Array
    C_stack: Array
    G_star: Array
    I_star_q: Array
    J_star: float
    condition_weights: Array


@dataclass
class ClassFit:
    """Fit for one number of cell classes K."""

    K: int
    J: float
    I_q: Array
    H: Array
    alpha: Array
    scale: Array
    G: Array
    history: Optional[Dict[str, List[float]]] = None


@dataclass
class SweepResult:
    """Fits and regrets for a sweep over K."""

    J_star: float
    I_star_q: Array
    fits: Dict[int, ClassFit]
    regret: Dict[int, float]


# ---------------------------------------------------------------------------
# Basic information and budget functions
# ---------------------------------------------------------------------------


def _flatten_stack(C_stack: Array, weights: Array) -> Tuple[Array, Array, Tuple[int, ...]]:
    C_stack = np.asarray(C_stack, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if C_stack.ndim < 2:
        raise ValueError("C_stack must have shape (Q, *freq_shape)")
    if weights.shape != C_stack.shape[1:]:
        raise ValueError(
            f"weights shape {weights.shape} must match spectral shape {C_stack.shape[1:]}"
        )
    if np.any(~np.isfinite(C_stack)) or np.any(C_stack < 0):
        raise ValueError("C_stack must be finite and nonnegative")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")
    return C_stack.reshape(C_stack.shape[0], -1), weights.reshape(-1), C_stack.shape[1:]


def normalize_condition_weights(condition_weights: Optional[Array], Q: int) -> Array:
    if condition_weights is None:
        pi = np.ones(Q, dtype=float) / Q
    else:
        pi = np.asarray(condition_weights, dtype=float).ravel()
        if pi.shape != (Q,):
            raise ValueError(f"condition_weights must have shape {(Q,)}, got {pi.shape}")
        if np.any(pi < 0) or pi.sum() <= 0:
            raise ValueError("condition_weights must be nonnegative and sum to a positive value")
        pi = pi / pi.sum()
    return pi


def information_from_filter_power(
    C_stack: Array,
    G_stack: Array,
    weights: Array,
    sigma_in: float,
    sigma_out: float,
) -> Array:
    """Information I_q(G_q) for each condition q, in nats."""
    C, W, freq_shape = _flatten_stack(C_stack, weights)
    Q, F = C.shape
    G = np.asarray(G_stack, dtype=float)
    if G.shape == freq_shape:
        G = np.broadcast_to(G.reshape(1, F), (Q, F))
    elif G.shape == C_stack.shape:
        G = G.reshape(Q, F)
    elif G.shape != (Q, F):
        raise ValueError(f"G_stack has incompatible shape {G.shape}")
    if np.any(G < 0):
        raise ValueError("G_stack must be nonnegative filter power")
    s_in2 = float(sigma_in) ** 2
    s_out2 = float(sigma_out) ** 2
    num = G * (C + s_in2) + s_out2
    den = G * s_in2 + s_out2
    return np.sum(np.log(num / den) * W[None, :], axis=1)


def response_power_budget(C_stack: Array, G_stack: Array, weights: Array, sigma_in: float) -> Array:
    """Response-power spend for each condition."""
    C, W, freq_shape = _flatten_stack(C_stack, weights)
    Q, F = C.shape
    G = np.asarray(G_stack, dtype=float)
    if G.shape == freq_shape:
        G = np.broadcast_to(G.reshape(1, F), (Q, F))
    elif G.shape == C_stack.shape:
        G = G.reshape(Q, F)
    elif G.shape != (Q, F):
        raise ValueError(f"G_stack has incompatible shape {G.shape}")
    return np.sum(G * (C + float(sigma_in) ** 2) * W[None, :], axis=1)


# ---------------------------------------------------------------------------
# Condition construction
# ---------------------------------------------------------------------------


def build_rucci_cycle_conditions(
    *,
    early_weight: float = 0.5,
    late_weight: float = 0.5,
    use_modulated_early: bool = True,
) -> Tuple[List[Condition], Array]:
    """Create the production early/late stack from the canonical Figure 7 spectra.

    This is the cell-learning source of truth for the Rucci/Boi fixation-cycle
    question. It uses the same `C_early = I(f) Q_saccade` and
    `C_late = I(f) Q_drift` ArraySpectrum wrappers that feed the Figure 6/Q1/Q3
    filter reconstructions.
    """
    early, late = cycle_solver_spectra(use_modulated_early=use_modulated_early)
    conditions = [
        Condition(
            name="early_cycle",
            epoch="early",
            parameter_name="cycle_phase",
            parameter_value=0.0,
            spectrum=early,
        ),
        Condition(
            name="late_cycle",
            epoch="late",
            parameter_name="cycle_phase",
            parameter_value=1.0,
            spectrum=late,
        ),
    ]
    pi = np.asarray([early_weight, late_weight], dtype=float)
    if np.any(pi < 0) or pi.sum() <= 0:
        raise ValueError("early_weight and late_weight must be nonnegative and sum positive")
    pi = pi / pi.sum()
    return conditions, pi


def build_cell_learning_conditions(
    *,
    early_weight: float = 0.5,
    late_weight: float = 0.5,
) -> Tuple[List[Condition], Array]:
    """Build the canonical Figure 7 Rucci/Boi condition stack."""
    return build_rucci_cycle_conditions(
        early_weight=early_weight,
        late_weight=late_weight,
        use_modulated_early=True,
    )


def conditions_from_spectrum_specs(
    specs: Sequence,
    *,
    condition_weights: Optional[Array] = None,
    default_epoch: str = "condition",
) -> Tuple[List[Condition], Array]:
    """Convert SpectrumSpec-like objects into cell-class learning conditions.

    A spec only needs ``key``, ``spectrum``, and optional metadata fields such
    as ``family`` and ``parameters``.  This keeps cell-class experiments tied to
    the same human-readable spectrum library used by the figure scripts.
    """
    conditions: List[Condition] = []
    for i, spec in enumerate(specs):
        params = getattr(spec, "parameters", {}) or {}
        if params:
            parameter_name, parameter_value = next(iter(params.items()))
        else:
            parameter_name, parameter_value = "index", float(i)
        conditions.append(
            Condition(
                name=getattr(spec, "key", f"condition_{i}"),
                epoch=getattr(spec, "family", default_epoch),
                parameter_name=str(parameter_name),
                parameter_value=float(parameter_value),
                spectrum=getattr(spec, "spectrum", spec),
            )
        )
    pi = normalize_condition_weights(condition_weights, len(conditions))
    return conditions, pi


def solve_oracle_stack(
    conditions: Sequence[Condition],
    *,
    sigma_in: float = 0.3,
    sigma_out: float = 1.0,
    P0: float = 50.0,
    grid: str = "fast",
    band: Tuple[float, float, float] = (F_MAX, OMEGA_MIN, OMEGA_MAX),
    condition_weights: Optional[Array] = None,
) -> OracleStack:
    """Run the repo's one-filter solver on every condition."""
    results: List[Result] = []
    for cond in conditions:
        r = run(
            cond.spectrum,
            sigma_in=sigma_in,
            sigma_out=sigma_out,
            P0=P0,
            grid=grid,
            band=band,
        )
        results.append(r)

    f = results[0].f
    omega = results[0].omega
    f_max, omega_min, omega_max = band
    weights = radial_weights(f, omega) * band_mask_radial(f, omega, f_max, omega_min, omega_max)
    C_stack = np.stack([r.C for r in results], axis=0)
    G_star = np.stack([r.v_sq for r in results], axis=0)

    pi = normalize_condition_weights(condition_weights, len(results))
    I_star_q = information_from_filter_power(C_stack, G_star, weights, sigma_in, sigma_out)
    J_star = float(np.sum(pi * I_star_q))
    return OracleStack(
        conditions=list(conditions),
        results=results,
        f=f,
        omega=omega,
        weights=weights,
        C_stack=C_stack,
        G_star=G_star,
        I_star_q=I_star_q,
        J_star=J_star,
        condition_weights=pi,
    )


# ---------------------------------------------------------------------------
# Class learning optimizer
# ---------------------------------------------------------------------------


def fit_cell_classes(
    C_stack: Array,
    weights: Array,
    *,
    sigma_in: float,
    sigma_out: float,
    P0: float,
    K: int,
    condition_weights: Optional[Array] = None,
    n_steps: int = 5000,
    n_restarts: int = 8,
    lr: float = 3e-2,
    smooth_weight: float = 1e-4,
    entropy_weight: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> ClassFit:
    """Learn K class spectra H_c and condition mixtures alpha_qc.

    Requires PyTorch. Install with: pip install torch
    """
    import torch
    import torch.nn.functional as Fnn

    if K < 1:
        raise ValueError("K must be >= 1")
    if P0 <= 0:
        raise ValueError("P0 must be positive")

    torch.set_default_dtype(torch.float64)

    C_np, W_np, freq_shape = _flatten_stack(C_stack, weights)
    Q, F = C_np.shape
    pi_np = normalize_condition_weights(condition_weights, Q)
    support_np = W_np > 0

    C = torch.tensor(C_np, device=device)
    W = torch.tensor(W_np, device=device)
    pi = torch.tensor(pi_np, device=device)
    support = torch.tensor(support_np.astype(float), device=device)

    s_in2 = float(sigma_in) ** 2
    s_out2 = float(sigma_out) ** 2
    eps = 1e-12
    best: Optional[ClassFit] = None

    def smooth_penalty(H_flat: "torch.Tensor") -> "torch.Tensor":
        if smooth_weight <= 0:
            return torch.tensor(0.0, device=device)
        H_grid = H_flat.reshape((K,) + freq_shape)
        logH = torch.log(H_grid + eps)
        penalty = torch.tensor(0.0, device=device)
        # Dim 0 is class; remaining dims are spectral axes.
        for dim in range(1, logH.ndim):
            if logH.shape[dim] > 1:
                penalty = penalty + torch.mean(torch.diff(logH, dim=dim) ** 2)
        return penalty

    def entropy_penalty(alpha: "torch.Tensor") -> "torch.Tensor":
        if entropy_weight <= 0:
            return torch.tensor(0.0, device=device)
        # Positive entropy penalty encourages conditions to use fewer classes.
        entropy = -torch.sum(alpha * torch.log(alpha + eps), dim=1)
        return torch.sum(pi * entropy)

    for restart in range(n_restarts):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed + restart)

        U = torch.nn.Parameter(0.01 * torch.randn(K, F, generator=gen, device=device))
        Z = torch.nn.Parameter(0.01 * torch.randn(Q, K, generator=gen, device=device))
        opt = torch.optim.Adam([U, Z], lr=lr)
        history: Dict[str, List[float]] = {"J": [], "loss": []}

        for step in range(n_steps):
            opt.zero_grad()

            # Positive class powers, constrained to the represented band.
            H = (Fnn.softplus(U) + eps) * support[None, :]
            # Remove scale degeneracy: H_c integrates to one under the same weights.
            H = H / (torch.sum(H * W[None, :], dim=1, keepdim=True) + eps)

            alpha = torch.softmax(Z, dim=1)
            G_raw = alpha @ H

            spend = torch.sum(G_raw * (C + s_in2) * W[None, :], dim=1) + eps
            scale = P0 / spend
            G = G_raw * scale[:, None]

            num = G * (C + s_in2) + s_out2
            den = G * s_in2 + s_out2
            I_q = torch.sum(torch.log(num / den) * W[None, :], dim=1)
            J = torch.sum(pi * I_q)

            loss = -J + smooth_weight * smooth_penalty(H) + entropy_weight * entropy_penalty(alpha)
            loss.backward()
            opt.step()

            if verbose and (step % max(1, n_steps // 10) == 0 or step == n_steps - 1):
                print(f"restart={restart} step={step} J={float(J):.6g} loss={float(loss):.6g}")
            if step % max(1, n_steps // 200) == 0 or step == n_steps - 1:
                history["J"].append(float(J.detach().cpu()))
                history["loss"].append(float(loss.detach().cpu()))

        with torch.no_grad():
            H = (Fnn.softplus(U) + eps) * support[None, :]
            H = H / (torch.sum(H * W[None, :], dim=1, keepdim=True) + eps)
            alpha = torch.softmax(Z, dim=1)
            G_raw = alpha @ H
            spend = torch.sum(G_raw * (C + s_in2) * W[None, :], dim=1) + eps
            scale = P0 / spend
            G = G_raw * scale[:, None]
            num = G * (C + s_in2) + s_out2
            den = G * s_in2 + s_out2
            I_q = torch.sum(torch.log(num / den) * W[None, :], dim=1)
            J = torch.sum(pi * I_q)

            fit = ClassFit(
                K=K,
                J=float(J.cpu().numpy()),
                I_q=I_q.cpu().numpy(),
                H=H.cpu().numpy().reshape((K,) + freq_shape),
                alpha=alpha.cpu().numpy(),
                scale=scale.cpu().numpy(),
                G=G.cpu().numpy().reshape((Q,) + freq_shape),
                history=history,
            )

        if best is None or fit.J > best.J:
            best = fit

    assert best is not None
    return best


def sweep_cell_classes(
    oracle: OracleStack,
    *,
    K_values: Sequence[int] = (1, 2, 3, 4),
    n_steps: int = 5000,
    n_restarts: int = 8,
    lr: float = 3e-2,
    smooth_weight: float = 1e-4,
    entropy_weight: float = 0.0,
    device: str = "cpu",
    seed: int = 0,
    verbose: bool = False,
) -> SweepResult:
    """Fit K=1,2,3,... and compute information regret versus the oracle."""
    fits: Dict[int, ClassFit] = {}
    regret: Dict[int, float] = {}
    for K in K_values:
        fit = fit_cell_classes(
            oracle.C_stack,
            oracle.weights,
            sigma_in=oracle.results[0].sigma_in,
            sigma_out=oracle.results[0].sigma_out,
            P0=oracle.results[0].P0,
            K=K,
            condition_weights=oracle.condition_weights,
            n_steps=n_steps,
            n_restarts=n_restarts,
            lr=lr,
            smooth_weight=smooth_weight,
            entropy_weight=entropy_weight,
            device=device,
            seed=seed + 100 * K,
            verbose=verbose,
        )
        fits[K] = fit
        regret[K] = (oracle.J_star - fit.J) / abs(oracle.J_star)
    return SweepResult(
        J_star=oracle.J_star,
        I_star_q=oracle.I_star_q,
        fits=fits,
        regret=regret,
    )


# ---------------------------------------------------------------------------
# Summaries for interpreting learned classes
# ---------------------------------------------------------------------------


def class_centroids(H: Array, f: Array, omega: Array, weights: Array) -> List[Dict[str, float]]:
    """Return spatial and temporal centroids for each learned class."""
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    F_grid = f[:, None]
    W_grid = np.abs(omega)[None, :]
    out: List[Dict[str, float]] = []
    for c in range(H.shape[0]):
        mass = float(np.sum(H[c] * weights)) + 1e-12
        f_bar = float(np.sum(H[c] * F_grid * weights) / mass)
        omega_bar = float(np.sum(H[c] * W_grid * weights) / mass)
        out.append({"class": c, "f_centroid": f_bar, "omega_centroid": omega_bar})
    return out


def condition_table(oracle: OracleStack) -> List[Dict[str, float | str]]:
    """Small serializable table describing the condition stack."""
    rows = []
    for i, (cond, I) in enumerate(zip(oracle.conditions, oracle.I_star_q)):
        rows.append(
            {
                "index": i,
                "name": cond.name,
                "epoch": cond.epoch,
                "parameter_name": cond.parameter_name,
                "parameter_value": cond.parameter_value,
                "weight": float(oracle.condition_weights[i]),
                "I_star": float(I),
            }
        )
    return rows



def effective_class_gains(fit: ClassFit) -> Array:
    """Condition-dependent effective gain factors s_q alpha_qc.

    ``alpha`` is a simplex weight. The scalar ``scale`` then enforces the
    response-power budget for the whole mixture in each condition. Their
    product is therefore the amplitude multiplying class c in condition q.
    """
    return np.asarray(fit.scale, dtype=float)[:, None] * np.asarray(fit.alpha, dtype=float)


def class_budget_shares(
    C_stack: Array,
    H: Array,
    alpha: Array,
    scale: Array,
    weights: Array,
    sigma_in: float,
) -> Tuple[Array, Array]:
    """Response-power share carried by each class in each condition.

    Parameters
    ----------
    C_stack : ndarray, shape (Q, *freq_shape)
        Input spectra.
    H : ndarray, shape (K, *freq_shape)
        Learned class filter-power spectra.
    alpha : ndarray, shape (Q, K)
        Condition-dependent simplex mixture weights.
    scale : ndarray, shape (Q,)
        Per-condition budget-normalization factors.
    weights : ndarray, shape (*freq_shape)
        Integration weights.
    sigma_in : float
        Input-noise standard deviation.

    Returns
    -------
    rho : ndarray, shape (Q, K)
        Fraction of the response-power budget carried by each class.
    spend : ndarray, shape (Q, K)
        Unnormalized response-power spend per class. Rows sum to P0 up to
        numerical precision when the fit was budget-normalized.
    """
    C, W, freq_shape = _flatten_stack(C_stack, weights)
    Q, F = C.shape
    H_arr = np.asarray(H, dtype=float)
    K = H_arr.shape[0]
    if H_arr.shape[1:] != freq_shape:
        raise ValueError(f"H has shape {H_arr.shape}; expected (K, {freq_shape})")
    H_flat = H_arr.reshape(K, F)
    alpha = np.asarray(alpha, dtype=float)
    scale = np.asarray(scale, dtype=float).ravel()
    if alpha.shape != (Q, K):
        raise ValueError(f"alpha must have shape {(Q, K)}, got {alpha.shape}")
    if scale.shape != (Q,):
        raise ValueError(f"scale must have shape {(Q,)}, got {scale.shape}")

    s_in2 = float(sigma_in) ** 2
    spend = np.zeros((Q, K), dtype=float)
    for q in range(Q):
        drive = (C[q] + s_in2) * W
        for c in range(K):
            spend[q, c] = scale[q] * alpha[q, c] * np.sum(H_flat[c] * drive)
    denom = np.maximum(spend.sum(axis=1, keepdims=True), 1e-300)
    rho = spend / denom
    return rho, spend


def per_condition_regret(I_star_q: Array, I_q: Array) -> Array:
    """Relative information regret per condition."""
    I_star_q = np.asarray(I_star_q, dtype=float)
    I_q = np.asarray(I_q, dtype=float)
    return (I_star_q - I_q) / np.maximum(np.abs(I_star_q), 1e-300)


def class_summary_table(
    H: Array,
    f: Array,
    omega: Array,
    weights: Array,
    *,
    f_cut: float = 1.0,
    omega_cut: float = 50.0,
) -> List[Dict[str, float]]:
    """Quantitative summaries for learned class spectra.

    Centroids and high-frequency mass fractions are computed from H itself;
    because H is normalized inside the optimizer, these are shape summaries,
    not absolute gain summaries.
    """
    H = np.asarray(H, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float)
    F_grid = f[:, None]
    W_grid = np.abs(omega)[None, :]
    rows: List[Dict[str, float]] = []
    for c in range(H.shape[0]):
        mass = float(np.sum(H[c] * weights)) + 1e-300
        f_bar = float(np.sum(H[c] * F_grid * weights) / mass)
        omega_bar = float(np.sum(H[c] * W_grid * weights) / mass)
        high_f = float(np.sum(H[c] * weights * (F_grid >= f_cut)) / mass)
        high_w = float(np.sum(H[c] * weights * (W_grid >= omega_cut)) / mass)
        rows.append(
            {
                "class": int(c),
                "f_centroid": f_bar,
                "omega_centroid": omega_bar,
                "high_f_fraction": high_f,
                "high_omega_fraction": high_w,
                "f_cut": float(f_cut),
                "omega_cut": float(omega_cut),
            }
        )
    return rows


def log_separability_residual(
    Z: Array,
    *,
    mask: Optional[Array] = None,
    floor_rel: float = 1e-8,
) -> float:
    """How far a positive 2D spectrum is from multiplicative separability.

    A multiplicatively separable spectrum has the form Z(f, omega)=A(f)B(omega),
    so log Z is additively separable: a(f)+b(omega).  We fit the best additive
    row-plus-column model to log Z by double-centering and return the fraction
    of log-variance left in the residual.

    The score is zero for an exactly separable positive matrix and approaches
    one when row/column factors explain little of the log-spectrum structure.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.ndim != 2:
        raise ValueError("Z must be a 2D spectrum")
    if mask is None:
        mask = np.isfinite(Z) & (Z > 0)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(Z)
    if not np.any(mask):
        return float("nan")

    zpos = Z[mask]
    zmax = np.nanmax(zpos)
    floor = max(float(floor_rel) * zmax, 1e-300)
    L = np.log(np.maximum(Z, floor))

    # Restrict to rows and columns that have at least one represented sample.
    # This avoids all-NaN row/column means when a band mask excludes entire rows
    # or columns.
    row_keep = np.any(mask, axis=1)
    col_keep = np.any(mask, axis=0)
    Ls = L[np.ix_(row_keep, col_keep)]
    Ms = mask[np.ix_(row_keep, col_keep)]

    # Fill masked-out samples with NaN and use nan means. This supports the
    # rectangular positive-frequency band used in the figures while remaining
    # robust to band masks.
    Lm = np.where(Ms, Ls, np.nan)
    grand = float(np.nanmean(Lm))
    row = np.nanmean(Lm, axis=1, keepdims=True)
    col = np.nanmean(Lm, axis=0, keepdims=True)
    pred = row + col - grand
    resid = np.where(Ms, Lm - pred, np.nan)
    total = np.where(Ms, Lm - grand, np.nan)
    rss = float(np.nansum(resid ** 2))
    tss = float(np.nansum(total ** 2))
    return rss / max(tss, 1e-300)


def temporal_kernel_from_filter_power(
    v_sq: Array,
    f: Array,
    omega: Array,
    *,
    taper_alpha: float = 0.25,
    floor_rel: float = 1e-3,
) -> Tuple[Array, Array, float]:
    """Minimum-phase temporal kernel for a filter-power spectrum.

    The temporal magnitude is taken at the spatial frequency with largest
    integrated filter power, matching ``pipeline.extract_temporal_kernel``.
    """
    from src.kernels import minimum_phase_temporal_filter, soft_band_taper
    from src.params import OMEGA_MIN, OMEGA_MAX

    v_sq = np.asarray(v_sq, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    domega = np.gradient(omega)
    energy_per_f = np.sum(v_sq * np.abs(domega)[None, :], axis=1)
    i_peak_f = int(np.argmax(energy_per_f))
    f_peak = float(f[i_peak_f])
    v_t_mag = np.sqrt(np.maximum(v_sq[i_peak_f, :], 0.0))
    taper = soft_band_taper(omega, OMEGA_MIN, OMEGA_MAX, alpha=taper_alpha)
    v_t_smooth = v_t_mag * taper
    floor = floor_rel * max(float(v_t_smooth.max()), 1e-30)
    v_t_smooth = np.maximum(v_t_smooth, floor)
    t, h_t, _ = minimum_phase_temporal_filter(v_t_smooth, omega)
    return t, h_t, f_peak


def spatial_kernel_from_filter_power(
    v_sq: Array,
    f: Array,
    omega: Array,
    *,
    k_max: float = 8.0,
    n_k: int = 512,
    n_f_fine: int = 1024,
) -> Tuple[Array, Array]:
    """Zero-phase radial spatial kernel for a filter-power spectrum."""
    from src.kernels import spatial_kernel_2d, radial_cross_section

    v_sq = np.asarray(v_sq, dtype=float)
    f = np.asarray(f, dtype=float).ravel()
    omega = np.asarray(omega, dtype=float).ravel()
    domega = np.gradient(omega)
    v_s_sq = np.sum(v_sq * np.abs(domega)[None, :], axis=1) / (2.0 * np.pi)
    v_s = np.sqrt(np.maximum(v_s_sq, 0.0))
    f_fine = np.linspace(0.0, f.max() * 1.2, n_f_fine)
    v_s_interp = np.interp(f_fine, f, v_s, left=v_s[0], right=0.0)

    def vmag(k):
        return np.interp(k, f_fine, v_s_interp, left=v_s_interp[0], right=0.0)

    rx, ry, v_xy = spatial_kernel_2d(vmag, k_max=k_max, n_k=n_k)
    r_radial, v_radial = radial_cross_section(v_xy, rx, ry)
    return r_radial, v_radial
