"""Faster optimizers for information-aware cell-class learning.

This module is intended as a drop-in companion to ``src.cell_class_learning``.
It keeps the same model and objective, but makes the computation friendlier on
laptops by:

1. optimizing only in-band frequencies, not zero-weight grid points;
2. using float32 by default, which is much faster on CPU and required for MPS;
3. supporting Apple Silicon MPS with ``device='auto'``;
4. initializing class spectra from the oracle filter stack instead of random
   restarts;
5. adding early stopping;
6. optionally disabling the smoothness penalty for fast exploratory runs.

The mathematical objective is unchanged. For condition q,

    G_q = s_q sum_c alpha_qc H_c

and s_q is chosen so G_q spends exactly the response-power budget P0 under
C_q. The optimizer maximizes the actual Gaussian mutual information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.cell_class_learning import ClassFit, SweepResult, normalize_condition_weights

Array = np.ndarray


@dataclass
class FastFitDiagnostics:
    """Small diagnostic record for the fast optimizer."""

    device: str
    dtype: str
    n_active_freqs: int
    n_total_freqs: int
    steps_run: int
    restart: int


def _flatten_and_mask(C_stack: Array, weights: Array) -> Tuple[Array, Array, Tuple[int, ...], Array]:
    C_stack = np.asarray(C_stack, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if C_stack.ndim < 2:
        raise ValueError("C_stack must have shape (Q, *freq_shape)")
    if weights.shape != C_stack.shape[1:]:
        raise ValueError(f"weights shape {weights.shape} must match {C_stack.shape[1:]}")
    if not np.all(np.isfinite(C_stack)) or np.any(C_stack < 0):
        raise ValueError("C_stack must be finite and nonnegative")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("weights must be finite and nonnegative")

    freq_shape = C_stack.shape[1:]
    C_full = C_stack.reshape(C_stack.shape[0], -1)
    W_full = weights.reshape(-1)
    support = W_full > 0
    if not np.any(support):
        raise ValueError("No positive-weight frequency bins found")
    return C_full[:, support], W_full[support], freq_shape, support


def _as_torch_device(device: str):
    import torch

    if device == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def _as_torch_dtype(dtype: str):
    import torch

    if dtype in ("float32", "fp32", "single"):
        return torch.float32
    if dtype in ("float64", "fp64", "double"):
        return torch.float64
    raise ValueError("dtype must be float32 or float64")


def _validate_alpha_args(mode: str, K: int, alpha_floor: float, gain_delta_max: Optional[float]) -> str:
    mode = str(mode).lower()
    if mode not in {"softmax", "floor", "bounded_log_gain"}:
        raise ValueError("alpha_mode must be 'softmax', 'floor', or 'bounded_log_gain'")
    if not (0.0 <= float(alpha_floor) < 1.0 / K):
        raise ValueError("alpha_floor must satisfy 0 <= alpha_floor < 1/K")
    if mode == "bounded_log_gain":
        if gain_delta_max is None or float(gain_delta_max) < 0:
            raise ValueError("gain_delta_max must be nonnegative for bounded_log_gain")
    return mode


def _apply_alpha_floor(alpha, alpha_floor: float):
    if alpha_floor <= 0:
        return alpha
    K = alpha.shape[1]
    return alpha_floor + (1.0 - K * alpha_floor) * alpha


def _alpha_and_baseline(
    Z,
    *,
    mode: str = "softmax",
    alpha_floor: float = 0.0,
    gain_delta_max: Optional[float] = 0.5,
    baseline_logits=None,
    eps: float = 1e-8,
):
    """Return condition mixtures and the baseline mixture used to form them."""
    import torch

    K = Z.shape[1]
    mode = _validate_alpha_args(mode, K, alpha_floor, gain_delta_max)
    if baseline_logits is None:
        b = torch.full((K,), 1.0 / K, device=Z.device, dtype=Z.dtype)
    else:
        b = torch.softmax(baseline_logits, dim=0)

    if mode == "softmax":
        alpha = torch.softmax(Z, dim=1)
    elif mode == "floor":
        alpha = torch.softmax(Z, dim=1)
    else:
        m = float(gain_delta_max) * torch.tanh(Z)
        alpha = b[None, :] * torch.exp(m)
        alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + eps)

    alpha = _apply_alpha_floor(alpha, float(alpha_floor))
    alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + eps)
    return alpha, b


def make_alpha(
    Z,
    *,
    mode: str = "softmax",
    alpha_floor: float = 0.0,
    gain_delta_max: Optional[float] = 0.5,
    baseline_logits=None,
    learn_baseline: bool = False,
    eps: float = 1e-8,
):
    """Build condition-specific class mixtures.

    ``learn_baseline`` is accepted for API symmetry; callers that learn the
    baseline should pass the learned ``baseline_logits`` tensor.
    """
    del learn_baseline
    alpha, _ = _alpha_and_baseline(
        Z,
        mode=mode,
        alpha_floor=alpha_floor,
        gain_delta_max=gain_delta_max,
        baseline_logits=baseline_logits,
        eps=eps,
    )
    return alpha


def _budget_share_torch(alpha, H_active, C, W, s_in2: float, eps: float):
    import torch

    class_raw_spend = (
        alpha[:, :, None]
        * H_active[None, :, :]
        * (C[:, None, :] + s_in2)
        * W[None, None, :]
    )
    class_raw_spend = torch.sum(class_raw_spend, dim=2)
    return class_raw_spend / (torch.sum(class_raw_spend, dim=1, keepdim=True) + eps)


def _softplus_inverse_np(x: Array) -> Array:
    """Numerically stable inverse softplus for positive x."""
    x = np.asarray(x, dtype=np.float64)
    # For large x, softplus^{-1}(x) ≈ x. For small x, use log(expm1(x)).
    return np.where(x > 20.0, x, np.log(np.expm1(np.maximum(x, 1e-30))))


def _normalize_rows_under_weights(X: Array, W: Array, eps: float = 1e-30) -> Array:
    X = np.maximum(np.asarray(X, dtype=np.float64), eps)
    mass = np.sum(X * W[None, :], axis=1, keepdims=True)
    return X / np.maximum(mass, eps)


def _weighted_sqdist(A: Array, B: Array, W: Array) -> Array:
    """Return pairwise weighted squared distances between rows of A and B."""
    # Shapes: A=(Qa,F), B=(Qb,F). Returns (Qa,Qb).
    diff = A[:, None, :] - B[None, :, :]
    return np.sum(diff * diff * W[None, None, :], axis=2)


def oracle_initialization(
    G_star: Optional[Array],
    weights: Array,
    support: Array,
    K: int,
    condition_weights: Optional[Array] = None,
    *,
    eps: float = 1e-12,
) -> Tuple[Optional[Array], Optional[Array]]:
    """Initialize H and alpha from the oracle filter stack.

    Returns
    -------
    H0_active : ndarray or None, shape (K, F_active)
        Initial class spectra, normalized under the active weights.
    Z0 : ndarray or None, shape (Q, K)
        Initial logits for alpha=softmax(Z). If K=1 this is all zeros.
    """
    if G_star is None:
        return None, None

    G = np.asarray(G_star, dtype=np.float64)
    Q = G.shape[0]
    G_full = G.reshape(Q, -1)
    W_full = np.asarray(weights, dtype=np.float64).reshape(-1)
    W = W_full[support]
    G_active = np.maximum(G_full[:, support], eps)
    Gn = _normalize_rows_under_weights(G_active, W, eps=eps)
    pi = normalize_condition_weights(condition_weights, Q)

    if K == 1:
        H0 = np.sum(pi[:, None] * Gn, axis=0, keepdims=True)
        H0 = _normalize_rows_under_weights(H0, W, eps=eps)
        Z0 = np.zeros((Q, 1), dtype=np.float64)
        return H0, Z0

    # Greedy farthest-point initialization in the oracle-filter space.
    # First pick the condition farthest from the weighted mean, then keep adding
    # the condition whose distance to its nearest selected prototype is largest.
    mean = np.sum(pi[:, None] * Gn, axis=0, keepdims=True)
    dist_to_mean = _weighted_sqdist(Gn, mean, W).ravel()
    selected = [int(np.argmax(dist_to_mean))]

    while len(selected) < K:
        prototypes = Gn[selected]
        d = _weighted_sqdist(Gn, prototypes, W)
        nearest = np.min(d, axis=1)
        nearest[selected] = -np.inf
        selected.append(int(np.argmax(nearest)))

    H0 = Gn[selected]
    H0 = _normalize_rows_under_weights(H0, W, eps=eps)

    d = _weighted_sqdist(Gn, H0, W)
    tau = np.median(d[np.isfinite(d) & (d > 0)])
    if not np.isfinite(tau) or tau <= 0:
        tau = 1.0
    Z0 = -d / (tau + eps)
    # Center logits to avoid huge offsets.
    Z0 = Z0 - Z0.mean(axis=1, keepdims=True)
    return H0, Z0


def fit_cell_classes_fast(
    C_stack: Array,
    weights: Array,
    *,
    sigma_in: float,
    sigma_out: float,
    P0: float,
    K: int,
    condition_weights: Optional[Array] = None,
    G_star: Optional[Array] = None,
    n_steps: int = 1200,
    n_restarts: int = 2,
    lr: float = 5e-2,
    smooth_weight: float = 0.0,
    entropy_weight: float = 0.0,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 0,
    patience: int = 20,
    check_every: int = 25,
    min_delta: float = 1e-7,
    jitter: float = 0.05,
    torch_threads: Optional[int] = None,
    alpha_mode: str = "softmax",
    alpha_floor: float = 0.0,
    gain_delta_max: Optional[float] = 0.5,
    learn_baseline_mix: bool = False,
    baseline_mix_weight: float = 0.0,
    kl_to_baseline_weight: float = 0.0,
    share_floor: float = 0.0,
    share_floor_weight: float = 0.0,
    verbose: bool = False,
) -> ClassFit:
    """Fast version of ``fit_cell_classes``.

    The result has the same ``ClassFit`` fields as the original optimizer, so it
    can be used by the existing plotting code.
    """
    import torch
    import torch.nn.functional as Fnn

    if K < 1:
        raise ValueError("K must be >= 1")
    alpha_mode = _validate_alpha_args(alpha_mode, K, alpha_floor, gain_delta_max)
    if P0 <= 0:
        raise ValueError("P0 must be positive")
    if share_floor < 0 or share_floor >= 1.0 / K:
        raise ValueError("share_floor must satisfy 0 <= share_floor < 1/K")
    if torch_threads is not None and torch_threads > 0:
        torch.set_num_threads(int(torch_threads))

    C_np, W_np, freq_shape, support_np = _flatten_and_mask(C_stack, weights)
    Q, F_active = C_np.shape
    F_total = int(np.prod(freq_shape))
    pi_np = normalize_condition_weights(condition_weights, Q)

    torch_device = _as_torch_device(device)
    torch_dtype = _as_torch_dtype(dtype)

    C = torch.as_tensor(C_np, device=torch_device, dtype=torch_dtype)
    W = torch.as_tensor(W_np, device=torch_device, dtype=torch_dtype)
    pi = torch.as_tensor(pi_np, device=torch_device, dtype=torch_dtype)

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

    def smooth_penalty(H_active: "torch.Tensor") -> "torch.Tensor":
        if smooth_weight <= 0:
            return torch.zeros((), device=torch_device, dtype=torch_dtype)
        # Reconstruct full grid for a simple finite-difference penalty. This is
        # intentionally optional: set --smooth 0 for fastest exploratory runs.
        full = torch.zeros((K, F_total), device=torch_device, dtype=torch_dtype)
        support_idx = torch.as_tensor(np.flatnonzero(support_np), device=torch_device, dtype=torch.long)
        full[:, support_idx] = H_active
        H_grid = full.reshape((K,) + freq_shape)
        logH = torch.log(H_grid + eps)
        penalty = torch.zeros((), device=torch_device, dtype=torch_dtype)
        for dim in range(1, logH.ndim):
            if logH.shape[dim] > 1:
                penalty = penalty + torch.mean(torch.diff(logH, dim=dim) ** 2)
        return penalty

    def entropy_penalty(alpha: "torch.Tensor") -> "torch.Tensor":
        if entropy_weight <= 0:
            return torch.zeros((), device=torch_device, dtype=torch_dtype)
        entropy = -torch.sum(alpha * torch.log(alpha + eps), dim=1)
        return torch.sum(pi * entropy)

    best_fit: Optional[ClassFit] = None
    best_J_global = -np.inf

    for restart in range(n_restarts):
        torch.manual_seed(seed + 1009 * restart + 17 * K)

        if H0_np is not None:
            H_init = H0_np.copy()
            # Jitter in multiplicative/log space keeps positivity and breaks ties.
            if restart > 0 and jitter > 0:
                rng = np.random.default_rng(seed + 1009 * restart + 17 * K)
                H_init = H_init * np.exp(jitter * rng.standard_normal(H_init.shape))
                H_init = _normalize_rows_under_weights(H_init, W_np)
            U_init = _softplus_inverse_np(np.maximum(H_init, eps))
            U = torch.nn.Parameter(torch.as_tensor(U_init, device=torch_device, dtype=torch_dtype))
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

        opt_params = [U, Z]
        if learn_baseline_mix:
            B = torch.nn.Parameter(torch.zeros(K, device=torch_device, dtype=torch_dtype))
            opt_params.append(B)
        else:
            B = None

        opt = torch.optim.Adam(opt_params, lr=lr)
        history: Dict[str, List[float]] = {"J": [], "loss": []}
        best_state = None
        best_J = -np.inf
        no_improve = 0
        steps_run = 0

        for step in range(n_steps):
            steps_run = step + 1
            opt.zero_grad(set_to_none=True)

            H = Fnn.softplus(U) + eps
            H = H / (torch.sum(H * W[None, :], dim=1, keepdim=True) + eps)
            alpha, baseline_mix = _alpha_and_baseline(
                Z,
                mode=alpha_mode,
                alpha_floor=alpha_floor,
                gain_delta_max=gain_delta_max,
                baseline_logits=B,
                eps=eps,
            )
            G_raw = alpha @ H
            spend = torch.sum(G_raw * (C + s_in2) * W[None, :], dim=1) + eps
            scale = P0 / spend
            G = G_raw * scale[:, None]

            # log(num/den) written as log1p where possible. This is both stable
            # and usually a little faster for small ratios.
            den = G * s_in2 + s_out2
            signal = G * C
            I_q = torch.sum(torch.log1p(signal / den) * W[None, :], dim=1)
            J = torch.sum(pi * I_q)
            loss = -J
            if smooth_weight > 0:
                loss = loss + smooth_weight * smooth_penalty(H)
            if entropy_weight > 0:
                loss = loss + entropy_weight * entropy_penalty(alpha)
            if baseline_mix_weight > 0 and B is not None:
                loss = loss + baseline_mix_weight * (-torch.sum(torch.log(baseline_mix + eps)))
            if kl_to_baseline_weight > 0:
                kl = torch.sum(alpha * (torch.log(alpha + eps) - torch.log(baseline_mix[None, :] + eps)), dim=1)
                loss = loss + kl_to_baseline_weight * torch.sum(pi * kl)
            if share_floor_weight > 0 and share_floor > 0:
                budget_share = _budget_share_torch(alpha, H, C, W, s_in2, eps)
                share_penalty = torch.sum(pi[:, None] * torch.relu(float(share_floor) - budget_share) ** 2)
                loss = loss + share_floor_weight * share_penalty

            loss.backward()
            opt.step()

            if step % check_every == 0 or step == n_steps - 1:
                J_float = float(J.detach().cpu())
                loss_float = float(loss.detach().cpu())
                history["J"].append(J_float)
                history["loss"].append(loss_float)
                if verbose:
                    print(
                        f"K={K} restart={restart} step={step} "
                        f"J={J_float:.6g} loss={loss_float:.6g}"
                    )
                if J_float > best_J + min_delta:
                    best_J = J_float
                    no_improve = 0
                    if B is None:
                        best_state = (
                            U.detach().clone(),
                            Z.detach().clone(),
                        )
                    else:
                        best_state = (
                            U.detach().clone(),
                            Z.detach().clone(),
                            B.detach().clone(),
                        )
                else:
                    no_improve += 1
                if patience > 0 and no_improve >= patience:
                    if verbose:
                        print(f"early stop K={K} restart={restart} at step={step}")
                    break

        if best_state is not None:
            with torch.no_grad():
                U.copy_(best_state[0])
                Z.copy_(best_state[1])
                if B is not None and len(best_state) > 2:
                    B.copy_(best_state[2])

        with torch.no_grad():
            H_active = Fnn.softplus(U) + eps
            H_active = H_active / (torch.sum(H_active * W[None, :], dim=1, keepdim=True) + eps)
            alpha, baseline_mix = _alpha_and_baseline(
                Z,
                mode=alpha_mode,
                alpha_floor=alpha_floor,
                gain_delta_max=gain_delta_max,
                baseline_logits=B,
                eps=eps,
            )
            G_raw = alpha @ H_active
            spend = torch.sum(G_raw * (C + s_in2) * W[None, :], dim=1) + eps
            scale = P0 / spend
            G_active = G_raw * scale[:, None]
            den = G_active * s_in2 + s_out2
            signal = G_active * C
            I_q = torch.sum(torch.log1p(signal / den) * W[None, :], dim=1)
            J = torch.sum(pi * I_q)
            budget_share = _budget_share_torch(alpha, H_active, C, W, s_in2, eps)

            H_full = np.zeros((K, F_total), dtype=np.float64)
            G_full = np.zeros((Q, F_total), dtype=np.float64)
            H_full[:, support_np] = H_active.detach().cpu().numpy().astype(np.float64)
            G_full[:, support_np] = G_active.detach().cpu().numpy().astype(np.float64)

            fit = ClassFit(
                K=K,
                J=float(J.detach().cpu()),
                I_q=I_q.detach().cpu().numpy().astype(np.float64),
                H=H_full.reshape((K,) + freq_shape),
                alpha=alpha.detach().cpu().numpy().astype(np.float64),
                scale=scale.detach().cpu().numpy().astype(np.float64),
                G=G_full.reshape((Q,) + freq_shape),
                history=history,
            )
            # Attach diagnostics dynamically; ClassFit is a dataclass but not frozen.
            fit.fast_diagnostics = FastFitDiagnostics(
                device=str(torch_device),
                dtype=str(torch_dtype).replace("torch.", ""),
                n_active_freqs=F_active,
                n_total_freqs=F_total,
                steps_run=steps_run,
                restart=restart,
            )
            fit.budget_share = budget_share.detach().cpu().numpy().astype(np.float64)
            fit.baseline_mix = baseline_mix.detach().cpu().numpy().astype(np.float64)
            fit.alpha_mode = alpha_mode
            fit.gain_delta_max = None if gain_delta_max is None else float(gain_delta_max)
            fit.alpha_floor = float(alpha_floor)

        if fit.J > best_J_global:
            best_J_global = fit.J
            best_fit = fit

    assert best_fit is not None
    return best_fit


def refit_alpha_for_fixed_H_fast(
    C_stack: Array,
    weights: Array,
    H_fixed: Array,
    *,
    sigma_in: float,
    sigma_out: float,
    P0: float,
    condition_weights: Optional[Array] = None,
    alpha_mode: str = "bounded_log_gain",
    alpha_floor: float = 0.0,
    gain_delta_max: Optional[float] = 0.5,
    learn_baseline_mix: bool = False,
    baseline_mix_weight: float = 0.0,
    kl_to_baseline_weight: float = 0.0,
    share_floor: float = 0.0,
    share_floor_weight: float = 0.0,
    n_steps: int = 400,
    lr: float = 5e-2,
    device: str = "auto",
    dtype: str = "float32",
    seed: int = 0,
    patience: int = 20,
    check_every: int = 25,
    min_delta: float = 1e-7,
    torch_threads: Optional[int] = None,
    verbose: bool = False,
) -> ClassFit:
    """Optimize only condition-dependent gains for fixed class spectra."""
    import torch

    if P0 <= 0:
        raise ValueError("P0 must be positive")
    if torch_threads is not None and torch_threads > 0:
        torch.set_num_threads(int(torch_threads))

    C_np, W_np, freq_shape, support_np = _flatten_and_mask(C_stack, weights)
    Q, F_active = C_np.shape
    F_total = int(np.prod(freq_shape))
    H_fixed = np.asarray(H_fixed, dtype=np.float64)
    if H_fixed.ndim < 2 or H_fixed.shape[1:] != freq_shape:
        raise ValueError(f"H_fixed must have shape (K, {freq_shape}), got {H_fixed.shape}")
    K = H_fixed.shape[0]
    alpha_mode = _validate_alpha_args(alpha_mode, K, alpha_floor, gain_delta_max)
    if share_floor < 0 or share_floor >= 1.0 / K:
        raise ValueError("share_floor must satisfy 0 <= share_floor < 1/K")

    pi_np = normalize_condition_weights(condition_weights, Q)
    torch_device = _as_torch_device(device)
    torch_dtype = _as_torch_dtype(dtype)
    C = torch.as_tensor(C_np, device=torch_device, dtype=torch_dtype)
    W = torch.as_tensor(W_np, device=torch_device, dtype=torch_dtype)
    pi = torch.as_tensor(pi_np, device=torch_device, dtype=torch_dtype)
    H_active = torch.as_tensor(
        H_fixed.reshape(K, F_total)[:, support_np],
        device=torch_device,
        dtype=torch_dtype,
    )

    s_in2 = float(sigma_in) ** 2
    s_out2 = float(sigma_out) ** 2
    eps = 1e-8 if torch_dtype == torch.float32 else 1e-12

    torch.manual_seed(seed)
    Z = torch.nn.Parameter(0.01 * torch.randn(Q, K, device=torch_device, dtype=torch_dtype))
    opt_params = [Z]
    if learn_baseline_mix:
        B = torch.nn.Parameter(torch.zeros(K, device=torch_device, dtype=torch_dtype))
        opt_params.append(B)
    else:
        B = None
    opt = torch.optim.Adam(opt_params, lr=lr)

    history: Dict[str, List[float]] = {"J": [], "loss": []}
    best_state = None
    best_J = -np.inf
    no_improve = 0
    steps_run = 0

    for step in range(n_steps):
        steps_run = step + 1
        opt.zero_grad(set_to_none=True)
        alpha, baseline_mix = _alpha_and_baseline(
            Z,
            mode=alpha_mode,
            alpha_floor=alpha_floor,
            gain_delta_max=gain_delta_max,
            baseline_logits=B,
            eps=eps,
        )
        G_raw = alpha @ H_active
        spend = torch.sum(G_raw * (C + s_in2) * W[None, :], dim=1) + eps
        scale = P0 / spend
        G = G_raw * scale[:, None]
        den = G * s_in2 + s_out2
        signal = G * C
        I_q = torch.sum(torch.log1p(signal / den) * W[None, :], dim=1)
        J = torch.sum(pi * I_q)
        loss = -J
        if baseline_mix_weight > 0 and B is not None:
            loss = loss + baseline_mix_weight * (-torch.sum(torch.log(baseline_mix + eps)))
        if kl_to_baseline_weight > 0:
            kl = torch.sum(alpha * (torch.log(alpha + eps) - torch.log(baseline_mix[None, :] + eps)), dim=1)
            loss = loss + kl_to_baseline_weight * torch.sum(pi * kl)
        if share_floor_weight > 0 and share_floor > 0:
            budget_share = _budget_share_torch(alpha, H_active, C, W, s_in2, eps)
            share_penalty = torch.sum(pi[:, None] * torch.relu(float(share_floor) - budget_share) ** 2)
            loss = loss + share_floor_weight * share_penalty

        loss.backward()
        opt.step()

        if step % check_every == 0 or step == n_steps - 1:
            J_float = float(J.detach().cpu())
            loss_float = float(loss.detach().cpu())
            history["J"].append(J_float)
            history["loss"].append(loss_float)
            if verbose:
                print(f"fixed-H step={step} J={J_float:.6g} loss={loss_float:.6g}")
            if J_float > best_J + min_delta:
                best_J = J_float
                no_improve = 0
                if B is None:
                    best_state = (Z.detach().clone(),)
                else:
                    best_state = (Z.detach().clone(), B.detach().clone())
            else:
                no_improve += 1
            if patience > 0 and no_improve >= patience:
                break

    if best_state is not None:
        with torch.no_grad():
            Z.copy_(best_state[0])
            if B is not None and len(best_state) > 1:
                B.copy_(best_state[1])

    with torch.no_grad():
        alpha, baseline_mix = _alpha_and_baseline(
            Z,
            mode=alpha_mode,
            alpha_floor=alpha_floor,
            gain_delta_max=gain_delta_max,
            baseline_logits=B,
            eps=eps,
        )
        G_raw = alpha @ H_active
        spend = torch.sum(G_raw * (C + s_in2) * W[None, :], dim=1) + eps
        scale = P0 / spend
        G_active = G_raw * scale[:, None]
        den = G_active * s_in2 + s_out2
        signal = G_active * C
        I_q = torch.sum(torch.log1p(signal / den) * W[None, :], dim=1)
        J = torch.sum(pi * I_q)
        budget_share = _budget_share_torch(alpha, H_active, C, W, s_in2, eps)

        G_full = np.zeros((Q, F_total), dtype=np.float64)
        G_full[:, support_np] = G_active.detach().cpu().numpy().astype(np.float64)

        fit = ClassFit(
            K=K,
            J=float(J.detach().cpu()),
            I_q=I_q.detach().cpu().numpy().astype(np.float64),
            H=H_fixed.copy(),
            alpha=alpha.detach().cpu().numpy().astype(np.float64),
            scale=scale.detach().cpu().numpy().astype(np.float64),
            G=G_full.reshape((Q,) + freq_shape),
            history=history,
        )
        fit.fast_diagnostics = FastFitDiagnostics(
            device=str(torch_device),
            dtype=str(torch_dtype).replace("torch.", ""),
            n_active_freqs=F_active,
            n_total_freqs=F_total,
            steps_run=steps_run,
            restart=0,
        )
        fit.budget_share = budget_share.detach().cpu().numpy().astype(np.float64)
        fit.baseline_mix = baseline_mix.detach().cpu().numpy().astype(np.float64)
        fit.alpha_mode = alpha_mode
        fit.gain_delta_max = None if gain_delta_max is None else float(gain_delta_max)
        fit.alpha_floor = float(alpha_floor)
        return fit


def sweep_cell_classes_fast(
    oracle,
    *,
    K_values: Sequence[int] = (1, 2, 3),
    n_steps: int = 1200,
    n_restarts: int = 2,
    lr: float = 5e-2,
    smooth_weight: float = 0.0,
    entropy_weight: float = 0.0,
    device: str = "auto",
    dtype: str = "float32",
    patience: int = 20,
    check_every: int = 25,
    torch_threads: Optional[int] = None,
    alpha_mode: str = "softmax",
    alpha_floor: float = 0.0,
    gain_delta_max: Optional[float] = 0.5,
    learn_baseline_mix: bool = False,
    baseline_mix_weight: float = 0.0,
    kl_to_baseline_weight: float = 0.0,
    share_floor: float = 0.0,
    share_floor_weight: float = 0.0,
    seed: int = 0,
    verbose: bool = False,
) -> SweepResult:
    """Fast sweep over K using oracle initialization."""
    fits: Dict[int, ClassFit] = {}
    regret: Dict[int, float] = {}
    for K in K_values:
        fit = fit_cell_classes_fast(
            oracle.C_stack,
            oracle.weights,
            sigma_in=oracle.results[0].sigma_in,
            sigma_out=oracle.results[0].sigma_out,
            P0=oracle.results[0].P0,
            K=K,
            condition_weights=oracle.condition_weights,
            G_star=oracle.G_star,
            n_steps=n_steps,
            n_restarts=n_restarts,
            lr=lr,
            smooth_weight=smooth_weight,
            entropy_weight=entropy_weight,
            device=device,
            dtype=dtype,
            patience=patience,
            check_every=check_every,
            torch_threads=torch_threads,
            alpha_mode=alpha_mode,
            alpha_floor=alpha_floor,
            gain_delta_max=gain_delta_max,
            learn_baseline_mix=learn_baseline_mix,
            baseline_mix_weight=baseline_mix_weight,
            kl_to_baseline_weight=kl_to_baseline_weight,
            share_floor=share_floor,
            share_floor_weight=share_floor_weight,
            seed=seed + 100 * K,
            verbose=verbose,
        )
        fits[K] = fit
        regret[K] = (oracle.J_star - fit.J) / max(abs(oracle.J_star), 1e-300)
    return SweepResult(
        J_star=oracle.J_star,
        I_star_q=oracle.I_star_q,
        fits=fits,
        regret=regret,
    )
