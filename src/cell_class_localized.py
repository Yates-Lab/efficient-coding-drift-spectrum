"""Localized cell-class learning with bounded population-share modulation.

This module is a companion to :mod:`src.cell_class_learning_fast`.  It keeps
the fast optimizer's masked-frequency implementation, but changes the class
allocation model:

* class spectra are spatially localized by a log-frequency variance penalty;
* condition-dependent class shares are bounded log-gain modulations around a
  baseline population share;
* each class spends its own share of the response-power budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.cell_class_learning import (
    Condition,
    OracleStack,
    SweepResult,
    normalize_condition_weights,
    solve_oracle_stack,
)
from src.cell_class_learning_fast import (
    _as_torch_device,
    _as_torch_dtype,
    _flatten_and_mask,
    _normalize_rows_under_weights,
    _softplus_inverse_np,
    oracle_initialization,
)
from src.kernels import radial_cross_section, spatial_kernel_2d
from src.power_spectrum_library import canonical_positive_cycle_view
from src.rucci_cycle_spectra import (
    ArraySpectrum,
    ImageParams as RucciImageParams,
    image_spectrum as rucci_image_spectrum,
)
from src.spectra import BoiLateDriftApprox, mostofi_saccade_redistribution

Array = np.ndarray


@dataclass
class LocalizedClassFit:
    K: int
    J: float
    I_q: Array
    H: Array
    H_qc: Optional[Array]
    Delta: Optional[Array]
    rho0: Array
    rho_qc: Array
    delta_qc: Array
    G: Array
    G_class: Array
    f_centroid: Array
    f_centroid_cpd: Array
    f_log_std: Array
    spatial_rf_width_deg: Array
    R_loc: float
    R_adapt: float
    history: Optional[Dict[str, List[float]]] = None


def _validate_f_grid(f: Array, freq_shape: Tuple[int, ...]) -> Array:
    f = np.asarray(f, dtype=np.float64).ravel()
    if len(freq_shape) != 2:
        raise ValueError("localized cell classes currently require a (F, T) spectral grid")
    if f.shape != (freq_shape[0],):
        raise ValueError(f"f must have shape {(freq_shape[0],)}, got {f.shape}")
    if not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("f must be finite and strictly positive for log-frequency localization")
    return f


def _active_f_indices(freq_shape: Tuple[int, ...], support: Array) -> Array:
    F, T = freq_shape
    return np.repeat(np.arange(F, dtype=np.int64), T)[np.asarray(support, dtype=bool)]


def _full_from_active(active: Array, support: Array, shape: Tuple[int, ...]) -> Array:
    out = np.zeros(active.shape[:-1] + (int(np.prod(shape)),), dtype=np.float64)
    out[..., np.asarray(support, dtype=bool)] = np.asarray(active, dtype=np.float64)
    return out.reshape(active.shape[:-1] + shape)


def _share_from_logits_torch(Z, B, *, delta_max: float, learn_baseline_share: bool, eps: float):
    import torch

    K = Z.shape[1]
    if learn_baseline_share:
        rho0 = torch.softmax(B, dim=0)
    else:
        rho0 = torch.full((K,), 1.0 / K, device=Z.device, dtype=Z.dtype)
    delta = float(delta_max) * torch.tanh(Z)
    rho = rho0[None, :] * torch.exp(delta)
    rho = rho / (torch.sum(rho, dim=1, keepdim=True) + eps)
    return rho0, delta, rho


def _spatial_marginal_torch(H_active, W_active, f_index, F_total: int):
    import torch

    K = H_active.shape[0]
    out = torch.zeros((K, F_total), device=H_active.device, dtype=H_active.dtype)
    out.scatter_add_(1, f_index[None, :].expand(K, -1), H_active * W_active[None, :])
    out = out / (torch.sum(out, dim=1, keepdim=True) + 1e-30)
    return out


def spatial_stats_from_H(H: Array, f: Array, weights: Array) -> Tuple[Array, Array, Array, Array]:
    """Return log-f centroids, cpd centroids, log std, and weighted marginals."""
    H = np.asarray(H, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64)
    if H.ndim != 3 or weights.shape != H.shape[1:] or f.shape != (H.shape[1],):
        raise ValueError("expected H=(K,F,T), f=(F,), weights=(F,T)")
    marginal = np.sum(H * weights[None, :, :], axis=2)
    marginal = marginal / np.maximum(np.sum(marginal, axis=1, keepdims=True), 1e-300)
    log_f = np.log(f)
    mu = np.sum(marginal * log_f[None, :], axis=1)
    var = np.sum(marginal * (log_f[None, :] - mu[:, None]) ** 2, axis=1)
    return mu, np.exp(mu), np.sqrt(np.maximum(var, 0.0)), marginal


def _rf_width_from_spatial_marginal(marginal: Array, f: Array, *, k_max: float = 8.0, n_k: int = 512) -> Array:
    """Stable RF-width estimate from class spatial marginals."""
    widths = np.zeros(marginal.shape[0], dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    for c, m in enumerate(np.asarray(marginal, dtype=np.float64)):
        v_s = np.sqrt(np.maximum(m, 0.0))
        if not np.any(v_s > 0):
            widths[c] = np.nan
            continue

        def vmag(k):
            return np.interp(k, f, v_s, left=v_s[0], right=0.0)

        rx, ry, v_xy = spatial_kernel_2d(vmag, k_max=k_max, n_k=n_k)
        r, v = radial_cross_section(v_xy, rx, ry)
        power = np.maximum(np.abs(v) ** 2, 0.0)
        denom = np.trapz(power, r)
        if not np.isfinite(denom) or abs(denom) <= 1e-300:
            widths[c] = np.nan
        else:
            widths[c] = float(np.sqrt(abs(np.trapz((r ** 2) * power, r) / denom)))
    return widths


def spatial_rf_width_from_H(
    H: Array,
    f: Array,
    omega: Array,
    weights: Array,
    *,
    method: str = "second_moment",
    k_max: float = 8.0,
    n_k: int = 512,
) -> Array:
    """Reconstruct class spatial kernels and return RF widths in degrees."""
    if method != "second_moment":
        raise ValueError("only method='second_moment' is currently supported")
    del omega
    _, _, _, marginal = spatial_stats_from_H(H, f, weights)
    return _rf_width_from_spatial_marginal(marginal, f, k_max=k_max, n_k=n_k)


def _compute_fit_arrays(
    *,
    H_active: Array,
    H_qc_active: Optional[Array],
    Delta_active: Optional[Array],
    rho0: Array,
    rho_qc: Array,
    delta_qc: Array,
    C_np: Array,
    W_np: Array,
    freq_shape: Tuple[int, ...],
    support_np: Array,
    f: Array,
    sigma_in: float,
    sigma_out: float,
    P0: float,
    condition_weights: Array,
    R_loc: float,
    R_adapt: float,
    history: Optional[Dict[str, List[float]]],
) -> LocalizedClassFit:
    Q = C_np.shape[0]
    K = H_active.shape[0]
    F_total = int(np.prod(freq_shape))
    H_qc_use = H_qc_active if H_qc_active is not None else np.broadcast_to(H_active[None, :, :], (Q, K, H_active.shape[1]))
    s_in2 = float(sigma_in) ** 2
    s_out2 = float(sigma_out) ** 2
    E = np.sum(H_qc_use * (C_np[:, None, :] + s_in2) * W_np[None, None, :], axis=2)
    G_class_active = (rho_qc[:, :, None] * float(P0) / np.maximum(E, 1e-300)[:, :, None]) * H_qc_use
    G_active = np.sum(G_class_active, axis=1)
    I_q = np.sum(np.log1p((G_active * C_np) / (G_active * s_in2 + s_out2)) * W_np[None, :], axis=1)
    J = float(np.sum(condition_weights * I_q))

    H = _full_from_active(H_active, support_np, freq_shape)
    H_qc = None if H_qc_active is None else _full_from_active(H_qc_active, support_np, freq_shape)
    Delta = None if Delta_active is None else _full_from_active(Delta_active, support_np, freq_shape)
    G = _full_from_active(G_active, support_np, freq_shape)
    G_class = _full_from_active(G_class_active, support_np, freq_shape)
    weights_full = _full_from_active(W_np[None, :], support_np, freq_shape)[0]
    mu, mu_cpd, log_std, marginal = spatial_stats_from_H(H, f, weights_full)
    width = _rf_width_from_spatial_marginal(marginal, f)
    return LocalizedClassFit(
        K=K,
        J=J,
        I_q=I_q,
        H=H,
        H_qc=H_qc,
        Delta=Delta,
        rho0=np.asarray(rho0, dtype=np.float64),
        rho_qc=np.asarray(rho_qc, dtype=np.float64),
        delta_qc=np.asarray(delta_qc, dtype=np.float64),
        G=G,
        G_class=G_class,
        f_centroid=mu,
        f_centroid_cpd=mu_cpd,
        f_log_std=log_std,
        spatial_rf_width_deg=width,
        R_loc=float(R_loc),
        R_adapt=float(R_adapt),
        history=history,
    )


def fit_cell_classes_localized(
    C_stack: Array,
    weights: Array,
    f: Array,
    *,
    sigma_in: float,
    sigma_out: float,
    P0: float,
    K: int,
    condition_weights: Optional[Array] = None,
    G_star: Optional[Array] = None,
    loc_weight: float = 0.0,
    delta_max: float = 0.5,
    learn_baseline_share: bool = True,
    retune: bool = False,
    adapt_weight: float = 0.0,
    adapt_smooth_weight: float = 0.0,
    smooth_weight: float = 0.0,
    n_steps: int = 1500,
    n_restarts: int = 2,
    lr: float = 5e-2,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 0,
    patience: int = 25,
    check_every: int = 25,
    min_delta: float = 1e-7,
    jitter: float = 0.05,
    verbose: bool = False,
) -> LocalizedClassFit:
    """Fit localized class spectra with bounded condition modulation."""
    import torch
    import torch.nn.functional as Fnn

    if K < 1:
        raise ValueError("K must be >= 1")
    if P0 <= 0:
        raise ValueError("P0 must be positive")
    if delta_max < 0:
        raise ValueError("delta_max must be nonnegative")
    if loc_weight < 0 or smooth_weight < 0 or adapt_weight < 0 or adapt_smooth_weight < 0:
        raise ValueError("regularization weights must be nonnegative")

    C_np, W_np, freq_shape, support_np = _flatten_and_mask(C_stack, weights)
    f = _validate_f_grid(f, freq_shape)
    Q, F_active = C_np.shape
    F_total = int(np.prod(freq_shape))
    pi_np = normalize_condition_weights(condition_weights, Q)

    torch_device = _as_torch_device(device)
    torch_dtype = _as_torch_dtype(dtype)
    C = torch.as_tensor(C_np, device=torch_device, dtype=torch_dtype)
    W = torch.as_tensor(W_np, device=torch_device, dtype=torch_dtype)
    pi = torch.as_tensor(pi_np, device=torch_device, dtype=torch_dtype)
    log_f = torch.as_tensor(np.log(f), device=torch_device, dtype=torch_dtype)
    f_index = torch.as_tensor(_active_f_indices(freq_shape, support_np), device=torch_device, dtype=torch.long)
    support_idx = torch.as_tensor(np.flatnonzero(support_np), device=torch_device, dtype=torch.long)

    s_in2 = float(sigma_in) ** 2
    s_out2 = float(sigma_out) ** 2
    eps = 1e-8 if torch_dtype == torch.float32 else 1e-12

    H0_np, Z0_np = oracle_initialization(
        G_star,
        weights,
        support_np,
        K,
        condition_weights=condition_weights,
        eps=max(eps, 1e-12),
    )

    def smooth_penalty(H_active):
        if smooth_weight <= 0:
            return torch.zeros((), device=torch_device, dtype=torch_dtype)
        full = torch.zeros((K, F_total), device=torch_device, dtype=torch_dtype)
        full[:, support_idx] = H_active
        logH = torch.log(full.reshape((K,) + freq_shape) + eps)
        penalty = torch.zeros((), device=torch_device, dtype=torch_dtype)
        for dim in range(1, logH.ndim):
            if logH.shape[dim] > 1:
                penalty = penalty + torch.mean(torch.diff(logH, dim=dim) ** 2)
        return penalty

    def localization_penalty(H_active, rho0):
        marginal = _spatial_marginal_torch(H_active, W, f_index, freq_shape[0])
        mu = torch.sum(marginal * log_f[None, :], dim=1)
        var = torch.sum(marginal * (log_f[None, :] - mu[:, None]) ** 2, dim=1)
        return torch.sum(rho0 * var), mu, var

    def normalize_retuned(H_active, D):
        H_qk = H_active[None, :, :] * torch.exp(D)
        return H_qk / (torch.sum(H_qk * W[None, None, :], dim=2, keepdim=True) + eps)

    def adapt_penalty(D):
        if not retune or adapt_weight <= 0:
            return torch.zeros((), device=torch_device, dtype=torch_dtype)
        penalty = torch.sum(pi[:, None, None] * (D ** 2) * W[None, None, :])
        if adapt_smooth_weight > 0:
            full = torch.zeros((Q, K, F_total), device=torch_device, dtype=torch_dtype)
            full[:, :, support_idx] = D
            grid = full.reshape((Q, K) + freq_shape)
            smooth = torch.zeros((), device=torch_device, dtype=torch_dtype)
            for dim in range(2, grid.ndim):
                if grid.shape[dim] > 1:
                    smooth = smooth + torch.mean(torch.diff(grid, dim=dim) ** 2)
            penalty = penalty + float(adapt_smooth_weight) * smooth
        return penalty

    best_fit: Optional[LocalizedClassFit] = None
    best_J_global = -np.inf

    for restart in range(n_restarts):
        torch.manual_seed(seed + 1009 * restart + 17 * K)
        if H0_np is not None:
            H_init = H0_np.copy()
            if restart > 0 and jitter > 0:
                rng = np.random.default_rng(seed + 1009 * restart + 17 * K)
                H_init = H_init * np.exp(jitter * rng.standard_normal(H_init.shape))
                H_init = _normalize_rows_under_weights(H_init, W_np)
            U = torch.nn.Parameter(torch.as_tensor(_softplus_inverse_np(np.maximum(H_init, eps)), device=torch_device, dtype=torch_dtype))
        else:
            U = torch.nn.Parameter(0.01 * torch.randn(K, F_active, device=torch_device, dtype=torch_dtype))

        if Z0_np is not None:
            Z_init = Z0_np.copy()
            if restart > 0 and jitter > 0:
                rng = np.random.default_rng(seed + 7919 * restart + K)
                Z_init = Z_init + jitter * rng.standard_normal(Z_init.shape)
            Z = torch.nn.Parameter(torch.as_tensor(Z_init, device=torch_device, dtype=torch_dtype))
        else:
            Z = torch.nn.Parameter(0.01 * torch.randn(Q, K, device=torch_device, dtype=torch_dtype))

        params = [U, Z]
        if learn_baseline_share:
            B = torch.nn.Parameter(torch.zeros(K, device=torch_device, dtype=torch_dtype))
            params.append(B)
        else:
            B = None
        if retune:
            D = torch.nn.Parameter(torch.zeros(Q, K, F_active, device=torch_device, dtype=torch_dtype))
            params.append(D)
        else:
            D = None

        opt = torch.optim.Adam(params, lr=lr)
        history: Dict[str, List[float]] = {"J": [], "loss": [], "R_loc": [], "R_adapt": []}
        best_state = None
        best_J = -np.inf
        no_improve = 0

        for step in range(n_steps):
            opt.zero_grad(set_to_none=True)
            H = Fnn.softplus(U) + eps
            H = H / (torch.sum(H * W[None, :], dim=1, keepdim=True) + eps)
            rho0, delta, rho = _share_from_logits_torch(
                Z,
                B,
                delta_max=delta_max,
                learn_baseline_share=learn_baseline_share,
                eps=eps,
            )
            H_qk = normalize_retuned(H, D) if retune else H[None, :, :].expand(Q, -1, -1)
            E = torch.sum(H_qk * (C[:, None, :] + s_in2) * W[None, None, :], dim=2) + eps
            G_class = (rho[:, :, None] * float(P0) / E[:, :, None]) * H_qk
            G = torch.sum(G_class, dim=1)
            I_q = torch.sum(torch.log1p((G * C) / (G * s_in2 + s_out2)) * W[None, :], dim=1)
            J = torch.sum(pi * I_q)
            R_loc, _, _ = localization_penalty(H, rho0)
            R_adapt = adapt_penalty(D) if retune else torch.zeros((), device=torch_device, dtype=torch_dtype)
            loss = -J + float(loc_weight) * R_loc + float(smooth_weight) * smooth_penalty(H) + float(adapt_weight) * R_adapt
            loss.backward()
            opt.step()

            if step % check_every == 0 or step == n_steps - 1:
                J_float = float(J.detach().cpu())
                history["J"].append(J_float)
                history["loss"].append(float(loss.detach().cpu()))
                history["R_loc"].append(float(R_loc.detach().cpu()))
                history["R_adapt"].append(float(R_adapt.detach().cpu()))
                if verbose:
                    print(f"K={K} restart={restart} step={step} J={J_float:.6g} loss={float(loss.detach().cpu()):.6g}")
                if J_float > best_J + min_delta:
                    best_J = J_float
                    no_improve = 0
                    state = [U.detach().clone(), Z.detach().clone()]
                    if B is not None:
                        state.append(B.detach().clone())
                    if D is not None:
                        state.append(D.detach().clone())
                    best_state = state
                else:
                    no_improve += 1
                if patience > 0 and no_improve >= patience:
                    break

        if best_state is not None:
            with torch.no_grad():
                U.copy_(best_state[0])
                Z.copy_(best_state[1])
                idx = 2
                if B is not None:
                    B.copy_(best_state[idx])
                    idx += 1
                if D is not None:
                    D.copy_(best_state[idx])

        with torch.no_grad():
            H = Fnn.softplus(U) + eps
            H = H / (torch.sum(H * W[None, :], dim=1, keepdim=True) + eps)
            rho0, delta, rho = _share_from_logits_torch(
                Z,
                B,
                delta_max=delta_max,
                learn_baseline_share=learn_baseline_share,
                eps=eps,
            )
            H_qk = normalize_retuned(H, D) if retune else None
            H_for_budget = H_qk if H_qk is not None else H[None, :, :].expand(Q, -1, -1)
            R_loc, _, _ = localization_penalty(H, rho0)
            R_adapt = adapt_penalty(D) if retune else torch.zeros((), device=torch_device, dtype=torch_dtype)
            fit = _compute_fit_arrays(
                H_active=H.detach().cpu().numpy().astype(np.float64),
                H_qc_active=None if H_qk is None else H_qk.detach().cpu().numpy().astype(np.float64),
                Delta_active=None if D is None else D.detach().cpu().numpy().astype(np.float64),
                rho0=rho0.detach().cpu().numpy().astype(np.float64),
                rho_qc=rho.detach().cpu().numpy().astype(np.float64),
                delta_qc=delta.detach().cpu().numpy().astype(np.float64),
                C_np=C_np,
                W_np=W_np,
                freq_shape=freq_shape,
                support_np=support_np,
                f=f,
                sigma_in=sigma_in,
                sigma_out=sigma_out,
                P0=P0,
                condition_weights=pi_np,
                R_loc=float(R_loc.detach().cpu()),
                R_adapt=float(R_adapt.detach().cpu()),
                history=history,
            )
            del H_for_budget
        if fit.J > best_J_global:
            best_J_global = fit.J
            best_fit = fit

    assert best_fit is not None
    return best_fit


def refit_modulation_for_fixed_classes(
    C_stack: Array,
    weights: Array,
    f: Array,
    H_fixed: Array,
    rho0: Array,
    *,
    sigma_in: float,
    sigma_out: float,
    P0: float,
    condition_weights: Optional[Array] = None,
    delta_max: float = 0.5,
    retune: bool = False,
    adapt_weight: float = 0.0,
    adapt_smooth_weight: float = 0.0,
    n_steps: int = 400,
    lr: float = 5e-2,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 0,
    patience: int = 20,
    check_every: int = 25,
    min_delta: float = 1e-7,
    verbose: bool = False,
) -> LocalizedClassFit:
    """Refit only bounded condition modulation for frozen class spectra."""
    import torch

    C_np, W_np, freq_shape, support_np = _flatten_and_mask(C_stack, weights)
    f = _validate_f_grid(f, freq_shape)
    H_fixed = np.asarray(H_fixed, dtype=np.float64)
    if H_fixed.ndim != 3 or H_fixed.shape[1:] != freq_shape:
        raise ValueError(f"H_fixed must have shape (K, {freq_shape}), got {H_fixed.shape}")
    K = H_fixed.shape[0]
    Q, F_active = C_np.shape
    pi_np = normalize_condition_weights(condition_weights, Q)
    rho0_np = np.asarray(rho0, dtype=np.float64).ravel()
    if rho0_np.shape != (K,) or np.any(rho0_np < 0) or rho0_np.sum() <= 0:
        raise ValueError(f"rho0 must be a nonnegative vector with shape {(K,)}")
    rho0_np = rho0_np / rho0_np.sum()

    torch_device = _as_torch_device(device)
    torch_dtype = _as_torch_dtype(dtype)
    C = torch.as_tensor(C_np, device=torch_device, dtype=torch_dtype)
    W = torch.as_tensor(W_np, device=torch_device, dtype=torch_dtype)
    pi = torch.as_tensor(pi_np, device=torch_device, dtype=torch_dtype)
    H_active = torch.as_tensor(H_fixed.reshape(K, -1)[:, support_np], device=torch_device, dtype=torch_dtype)
    H_active = H_active / (torch.sum(H_active * W[None, :], dim=1, keepdim=True) + 1e-12)
    rho0_t = torch.as_tensor(rho0_np, device=torch_device, dtype=torch_dtype)
    s_in2 = float(sigma_in) ** 2
    s_out2 = float(sigma_out) ** 2
    eps = 1e-8 if torch_dtype == torch.float32 else 1e-12
    F_total = int(np.prod(freq_shape))
    support_idx = torch.as_tensor(np.flatnonzero(support_np), device=torch_device, dtype=torch.long)

    torch.manual_seed(seed)
    Z = torch.nn.Parameter(0.01 * torch.randn(Q, K, device=torch_device, dtype=torch_dtype))
    params = [Z]
    if retune:
        D = torch.nn.Parameter(torch.zeros(Q, K, F_active, device=torch_device, dtype=torch_dtype))
        params.append(D)
    else:
        D = None
    opt = torch.optim.Adam(params, lr=lr)
    history: Dict[str, List[float]] = {"J": [], "loss": [], "R_adapt": []}
    best_state = None
    best_J = -np.inf
    no_improve = 0

    def normalized_H_qk():
        if D is None:
            return H_active[None, :, :].expand(Q, -1, -1)
        H_qk = H_active[None, :, :] * torch.exp(D)
        return H_qk / (torch.sum(H_qk * W[None, None, :], dim=2, keepdim=True) + eps)

    def adapt_penalty():
        if D is None or adapt_weight <= 0:
            return torch.zeros((), device=torch_device, dtype=torch_dtype)
        penalty = torch.sum(pi[:, None, None] * (D ** 2) * W[None, None, :])
        if adapt_smooth_weight > 0:
            full = torch.zeros((Q, K, F_total), device=torch_device, dtype=torch_dtype)
            full[:, :, support_idx] = D
            grid = full.reshape((Q, K) + freq_shape)
            smooth = torch.zeros((), device=torch_device, dtype=torch_dtype)
            for dim in range(2, grid.ndim):
                if grid.shape[dim] > 1:
                    smooth = smooth + torch.mean(torch.diff(grid, dim=dim) ** 2)
            penalty = penalty + float(adapt_smooth_weight) * smooth
        return penalty

    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        delta = float(delta_max) * torch.tanh(Z)
        rho = rho0_t[None, :] * torch.exp(delta)
        rho = rho / (torch.sum(rho, dim=1, keepdim=True) + eps)
        H_qk = normalized_H_qk()
        E = torch.sum(H_qk * (C[:, None, :] + s_in2) * W[None, None, :], dim=2) + eps
        G_class = (rho[:, :, None] * float(P0) / E[:, :, None]) * H_qk
        G = torch.sum(G_class, dim=1)
        I_q = torch.sum(torch.log1p((G * C) / (G * s_in2 + s_out2)) * W[None, :], dim=1)
        J = torch.sum(pi * I_q)
        R_adapt = adapt_penalty()
        loss = -J + float(adapt_weight) * R_adapt
        loss.backward()
        opt.step()
        if step % check_every == 0 or step == n_steps - 1:
            J_float = float(J.detach().cpu())
            history["J"].append(J_float)
            history["loss"].append(float(loss.detach().cpu()))
            history["R_adapt"].append(float(R_adapt.detach().cpu()))
            if verbose:
                print(f"fixed localized step={step} J={J_float:.6g}")
            if J_float > best_J + min_delta:
                best_J = J_float
                no_improve = 0
                best_state = [Z.detach().clone()] if D is None else [Z.detach().clone(), D.detach().clone()]
            else:
                no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    if best_state is not None:
        with torch.no_grad():
            Z.copy_(best_state[0])
            if D is not None:
                D.copy_(best_state[1])

    with torch.no_grad():
        delta = float(delta_max) * torch.tanh(Z)
        rho = rho0_t[None, :] * torch.exp(delta)
        rho = rho / (torch.sum(rho, dim=1, keepdim=True) + eps)
        H_qk = normalized_H_qk()
        return _compute_fit_arrays(
            H_active=H_active.detach().cpu().numpy().astype(np.float64),
            H_qc_active=None if D is None else H_qk.detach().cpu().numpy().astype(np.float64),
            Delta_active=None if D is None else D.detach().cpu().numpy().astype(np.float64),
            rho0=rho0_np,
            rho_qc=rho.detach().cpu().numpy().astype(np.float64),
            delta_qc=delta.detach().cpu().numpy().astype(np.float64),
            C_np=C_np,
            W_np=W_np,
            freq_shape=freq_shape,
            support_np=support_np,
            f=f,
            sigma_in=sigma_in,
            sigma_out=sigma_out,
            P0=P0,
            condition_weights=pi_np,
            R_loc=0.0,
            R_adapt=float(adapt_penalty().detach().cpu()),
            history=history,
        )


def sweep_cell_classes_localized(
    oracle: OracleStack,
    *,
    K_values: Sequence[int] = (1, 2, 3),
    loc_weight: float = 0.5,
    delta_max: float = 0.5,
    learn_baseline_share: bool = True,
    n_steps: int = 1500,
    n_restarts: int = 2,
    lr: float = 5e-2,
    smooth_weight: float = 0.0,
    device: str = "auto",
    dtype: str = "float32",
    patience: int = 25,
    check_every: int = 25,
    seed: int = 0,
    verbose: bool = False,
) -> SweepResult:
    fits: Dict[int, LocalizedClassFit] = {}
    regret: Dict[int, float] = {}
    for K in K_values:
        fit = fit_cell_classes_localized(
            oracle.C_stack,
            oracle.weights,
            oracle.f,
            sigma_in=oracle.results[0].sigma_in,
            sigma_out=oracle.results[0].sigma_out,
            P0=oracle.results[0].P0,
            K=K,
            condition_weights=oracle.condition_weights,
            G_star=oracle.G_star,
            loc_weight=loc_weight,
            delta_max=delta_max,
            learn_baseline_share=learn_baseline_share,
            n_steps=n_steps,
            n_restarts=n_restarts,
            lr=lr,
            smooth_weight=smooth_weight,
            device=device,
            dtype=dtype,
            patience=patience,
            check_every=check_every,
            seed=seed + 100 * K,
            verbose=verbose,
        )
        fits[K] = fit
        regret[K] = (oracle.J_star - fit.J) / max(abs(oracle.J_star), 1e-300)
    return SweepResult(oracle.J_star, oracle.I_star_q, fits, regret)


def build_strategy_conditions(
    *,
    D: float,
    A: float,
    grid: str = "fast",
    early_weight: float = 0.5,
    late_weight: float = 0.5,
) -> Tuple[List[Condition], Array]:
    """Build an explicit early-saccade/late-drift strategy pair."""
    cycle, _, _, _ = canonical_positive_cycle_view()
    f_grid, omega_grid = cycle.f, cycle.omega
    if grid != "fast":
        from src.params import hi_res_grid

        f_grid, omega_grid = hi_res_grid()
    image_params = RucciImageParams(beta=2.0, f0=0.03, high_cut_cpd=60.0)
    C_early = rucci_image_spectrum(f_grid, image_params)[:, None] * mostofi_saccade_redistribution(f_grid, omega_grid, A=float(A))
    conditions = [
        Condition(
            name=f"early_A_{float(A):g}",
            epoch="early",
            parameter_name="A",
            parameter_value=float(A),
            spectrum=ArraySpectrum(f_grid, omega_grid, C_early, f"strategy early A={float(A):g}", ignore_dc_for_interp=True),
        ),
        Condition(
            name=f"late_D_{float(D):g}",
            epoch="late",
            parameter_name="D",
            parameter_value=float(D),
            spectrum=BoiLateDriftApprox(D=float(D), f_is_cycles=True),
        ),
    ]
    pi = normalize_condition_weights(np.asarray([early_weight, late_weight], dtype=float), 2)
    return conditions, pi


def solve_strategy_oracle(
    *,
    D: float,
    A: float,
    grid: str = "fast",
    sigma_in: float = 0.3,
    sigma_out: float = 1.0,
    P0: float = 50.0,
) -> OracleStack:
    conditions, pi = build_strategy_conditions(D=D, A=A, grid=grid)
    return solve_oracle_stack(
        conditions,
        sigma_in=sigma_in,
        sigma_out=sigma_out,
        P0=P0,
        grid=grid,
        condition_weights=pi,
    )
