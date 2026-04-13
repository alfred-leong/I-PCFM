"""
ipcfm_sampling.py — Inequality-constrained PCFM (I-PCFM) strategies A, B, C.

Inequality constraint: Oleinik entropy condition for Burgers equation.
  g_{i,n}(u) = u[i+1,n] - u[i,n] - dx/(n*dt) <= 0   for n >= 1, i = 0..nx-2
  Active set: A = {(i,n) : g_{i,n}(u) > -eps}
  J_g: sparse, each row has +1 at u[i+1,n] and -1 at u[i,n]

Three strategies:
  A — Slack Variable Reformulation
  B — Log-Barrier Augmentation
  C — Active-Set Projection
"""

import gc
import math
import json
import os
from typing import Callable, List, Optional

import torch
from torch import Tensor

from .pcfm_sampling import (
    pcfm_sample,
    compute_jacobian,
    relaxed_penalty_constraint_interp_linear_detached,
)


# ---------------------------------------------------------------------------
# Entropy inequality constraint (Oleinik E-condition for Burgers)
# ---------------------------------------------------------------------------

class EntropyIneq:
    """
    Oleinik entropy condition for inviscid Burgers equation.

    For t > 0: (u(x+h) - u(x)) / h <= 1/t.
    Discrete form: u[i+1, n] - u[i, n] <= dx / (n * dt)  for n >= 1.

    Constraint: g_{i,n}(u) = u[i+1,n] - u[i,n] - dx/(n*dt) <= 0
    """

    def __init__(self, nx: int, nt: int, dx: float, dt_pde: float):
        self.nx = nx
        self.nt = nt
        self.dx = float(dx)
        self.dt_pde = float(dt_pde)
        self.n_constraints = (nx - 1) * (nt - 1)

    def values(self, u_flat: Tensor) -> Tensor:
        """Compute g(u) for all constraints. g <= 0 means feasible.
        Returns: ((nx-1)*(nt-1),) tensor."""
        u = u_flat.view(self.nx, self.nt)
        du = u[1:, 1:] - u[:-1, 1:]  # (nx-1, nt-1)
        n_idx = torch.arange(1, self.nt, device=u.device, dtype=u.dtype)
        bounds = self.dx / (n_idx * self.dt_pde)  # (nt-1,)
        return (du - bounds.unsqueeze(0)).flatten()

    def jacobian_rows(self, active_indices: Tensor) -> Tensor:
        """Build dense Jacobian rows for active entropy constraints.
        Each g_{i,n} has dg/du[i+1,n] = +1, dg/du[i,n] = -1.
        Returns: (n_active, nx*nt) tensor."""
        n_active = active_indices.shape[0]
        device = active_indices.device
        n_total = self.nx * self.nt

        # Convert flat constraint index → (spatial_idx, time_idx)
        # Constraints are stored row-major over (nx-1) spatial interfaces x (nt-1) time steps
        i_idx = active_indices // (self.nt - 1)       # spatial index (0..nx-2)
        n_idx = active_indices % (self.nt - 1) + 1    # PDE time index (1..nt-1)

        # Flat indices into u_flat.view(nx, nt): u[i, n] = u_flat[i*nt + n]
        idx_left = i_idx * self.nt + n_idx
        idx_right = (i_idx + 1) * self.nt + n_idx

        J = torch.zeros(n_active, n_total, device=device)
        row = torch.arange(n_active, device=device)
        J[row, idx_right] = 1.0
        J[row, idx_left] = -1.0
        return J


# ---------------------------------------------------------------------------
# Strategy A — Slack Variable Reformulation
# ---------------------------------------------------------------------------

def ipcfm_a_sample(
    u_flat: Tensor,
    v_flat: Tensor,
    t,
    u0_flat: Tensor,
    dt: float,
    hfunc: Callable,
    ineq: EntropyIneq,
    newtonsteps: int = 1,
    eps: float = 1e-6,
    slack_threshold: float = 0.05,
    guided_interpolation: bool = False,
    interpolation_params: dict = {},
) -> Tensor:
    """
    Single-sample Strategy A (Slack Variable Reformulation).

    Augments equality constraints h(u)=0 with slack constraints g_k(u)+s_k=0
    for near-active entropy constraints (g_k > -slack_threshold), where s >= 0.

    Uses a single closed-form Newton step (matching original PCFM style):
      z = z - J_aug^T (J_aug J_aug^T)^{-1} [h(u); g_A(u)+s_A]
    followed by clamping s >= 0.
    """
    n = u_flat.shape[0]          # nx * nt
    n_ineq = ineq.n_constraints  # (nx-1) * (nt-1)
    device = u_flat.device

    # Predicted state at next time (uncorrected)
    ut1 = u_flat + (1.0 - t) * v_flat

    # Compute entropy constraint values for initial guess
    g_vals = ineq.values(ut1)

    # Initialize slack: s_k = max(-g_k, 0) so g_k + s_k ≈ 0 for active/feasible
    s = torch.clamp(-g_vals, min=0.0)

    # Active set: constraints near or violating the boundary
    active_mask = g_vals > -slack_threshold
    active_indices = active_mask.nonzero(as_tuple=True)[0]
    n_active = active_indices.shape[0]

    z = torch.cat([ut1, s], dim=0)  # (n + n_ineq,)

    u_p = z[:n]
    s_p = z[n:]

    # Equality residual and Jacobian
    h_eq = hfunc(u_p)            # (m,)
    m = h_eq.shape[0]
    J_eq = compute_jacobian(hfunc, u_p)  # (m, n)

    if n_active > 0:
        # Inequality residual (active): g_k(u) + s_k  (want = 0)
        h_ineq = g_vals[active_indices] + s_p[active_indices]  # (n_active,)

        # Augmented Jacobian: (m + n_active) x (n + n_ineq)
        J_aug = torch.zeros(m + n_active, n + n_ineq, device=device)
        J_aug[:m, :n] = J_eq  # equality rows depend only on u

        # Inequality rows: d(g_k + s_k)/du = dg_k/du, d/ds_k = +1
        J_g_active = ineq.jacobian_rows(active_indices)  # (n_active, n)
        J_aug[m:, :n] = J_g_active
        row_idx = torch.arange(n_active, device=device)
        J_aug[m + row_idx, n + active_indices] = 1.0

        h_aug = torch.cat([h_eq, h_ineq], dim=0)
    else:
        # No active constraints — equality-only Newton step
        J_aug = torch.cat([J_eq, torch.zeros(m, n_ineq, device=device)], dim=1)
        h_aug = h_eq

    JJt = J_aug @ J_aug.T
    # Guard: skip correction if NaN/Inf in system
    if torch.isfinite(JJt).all() and torch.isfinite(h_aug).all():
        lam = _safe_solve(JJt, h_aug, eps, device)
        correction = J_aug.T @ lam
        z = z - correction

    # Project slack back to R+
    u_corr = z[:n]
    z_s = torch.clamp(z[n:], min=0.0)  # noqa: F841 (slack kept for potential future use)

    t_next = t + dt

    if guided_interpolation and interpolation_params:
        lam_param = interpolation_params.get('custom_lam', 1e0)
        step_size = interpolation_params.get('step_size', 1e-2)
        num_steps = interpolation_params.get('num_steps', 20)
        ut_interp = relaxed_penalty_constraint_interp_linear_detached(
            u0=u0_flat, u1_proj=u_corr, v_flat=v_flat,
            t=t.item() if hasattr(t, 'item') else float(t),
            dt=dt, hfunc=hfunc,
            lam=lam_param, step_size=step_size, num_steps=num_steps,
        )
    else:
        ut_interp = (1.0 - t_next) * u0_flat + t_next * u_corr

    return ((ut_interp - u_flat) / dt).detach()


def ipcfm_a_batched(
    ut: Tensor,
    vf: Tensor,
    t,
    u0: Tensor,
    dt: float,
    hfunc: Callable,
    ineq: EntropyIneq = None,
    newtonsteps: int = 1,
    eps: float = 1e-6,
    slack_threshold: float = 0.05,
    guided_interpolation: bool = False,
    interpolation_params: dict = {},
    **kwargs,
) -> Tensor:
    """Batched Strategy A. Drop-in replacement for pcfm_batched()."""
    B, nx, nt = ut.shape
    n = nx * nt

    # Normalise hfunc to a list
    if not isinstance(hfunc, (list, tuple)):
        hfunc = [hfunc] * B

    u_flat_b = ut.view(B, n).detach().clone()
    v_flat_b = vf.view(B, n)
    u0_flat_b = u0.view(B, n)

    out = []
    for i in range(B):
        gc.collect()
        torch.cuda.empty_cache()
        v_proj = ipcfm_a_sample(
            u_flat_b[i], v_flat_b[i], t, u0_flat_b[i], dt, hfunc[i], ineq,
            newtonsteps=newtonsteps, eps=eps, slack_threshold=slack_threshold,
            guided_interpolation=guided_interpolation,
            interpolation_params=interpolation_params,
        )
        out.append(v_proj)

    return torch.stack(out, dim=0).view(B, nx, nt)


# ---------------------------------------------------------------------------
# Strategy B — Log-Barrier Augmentation
# ---------------------------------------------------------------------------

def _log_barrier_interp(
    u0_flat: Tensor,
    u1_proj: Tensor,
    v_flat: Tensor,
    t: float,
    dt: float,
    hfunc: Callable,
    ineq: EntropyIneq,
    mu: float,
    lam: float = 1e-2,
    step_size: float = 1e-2,
    num_steps: int = 10,
    safe_clamp: float = 1e-3,
    barrier_eps: float = 1e-6,
) -> Tensor:
    """
    Penalty interpolation with log-barrier for entropy condition.

    Minimizes:
        L(u) = ||u - hat_u||^2 + lam * ||h(u + gamma*v)||^2
               - mu * sum_k log(max(-g_k(u), barrier_eps))

    The barrier term pushes u away from entropy violation boundaries.
    """
    t_prime = t + dt
    gamma = max(1.0 - t_prime, safe_clamp)
    hat_u = (1.0 - t_prime) * u0_flat + t_prime * u1_proj
    u = hat_u.detach().clone().requires_grad_(True)

    for _ in range(num_steps):
        u_ext = u + gamma * v_flat
        eq_loss = hfunc(u_ext).pow(2).sum()
        smooth = (u - hat_u).pow(2).sum()
        # Log-barrier on entropy: -mu * sum log(-g_k(u))
        # -g > 0 when feasible; clamp for infeasible/boundary cases
        g_vals = ineq.values(u)
        slack = (-g_vals).clamp(min=barrier_eps)
        barrier = -mu * torch.log(slack).sum()
        loss = smooth + lam * eq_loss + barrier
        grad = torch.autograd.grad(loss, u)[0]
        u = (u - step_size * grad).detach().clone().requires_grad_(True)

    return u.detach()


def ipcfm_b_sample(
    u_flat: Tensor,
    v_flat: Tensor,
    t,
    u0_flat: Tensor,
    dt: float,
    hfunc: Callable,
    ineq: EntropyIneq,
    mu_0: float = 0.1,
    decay_rate: float = 3.0,
    eps: float = 1e-6,
    newtonsteps: int = 1,
    lam: float = 1e-2,
    step_size: float = 1e-2,
    num_steps: int = 10,
    guided_interpolation: bool = True,
    barrier_eps: float = 1e-6,
) -> Tensor:
    """
    Single-sample Strategy B (Log-Barrier Augmentation).

    Barrier coefficient decays with flow time: mu(tau) = mu_0 * exp(-decay_rate * tau).
    """
    tau = t.item() if hasattr(t, 'item') else float(t)
    mu = mu_0 * math.exp(-decay_rate * tau)

    # Guard: if u_flat is already NaN/Inf, return v_flat unchanged
    if not torch.isfinite(u_flat).all():
        return v_flat.detach()

    # Standard equality-constraint Newton projection
    ut1 = u_flat + (1.0 - t) * v_flat
    u_corr = ut1.clone()

    for _ in range(newtonsteps):
        res = hfunc(u_corr)
        J = compute_jacobian(hfunc, u_corr)
        JJt = J @ J.T
        # Guard: skip correction if NaN/Inf
        if not (torch.isfinite(JJt).all() and torch.isfinite(res).all()):
            break
        lam_newton = _safe_solve(JJt, res, max(eps, 1e-4), u_flat.device)
        correction = J.T @ lam_newton
        # Trust-region: cap correction magnitude
        corr_norm = correction.norm()
        u_ref_norm = u_corr.norm().clamp(min=1.0)
        if corr_norm > 10.0 * u_ref_norm:
            correction = correction * (10.0 * u_ref_norm / corr_norm)
        u_corr = u_corr - correction

    # Guard: Newton correction produced NaN -> fall back
    if not torch.isfinite(u_corr).all():
        u_corr = ut1.clone()

    t_next = t + dt

    if guided_interpolation:
        ut_interp = _log_barrier_interp(
            u0_flat=u0_flat, u1_proj=u_corr, v_flat=v_flat,
            t=tau, dt=dt, hfunc=hfunc, ineq=ineq, mu=mu,
            lam=lam, step_size=step_size, num_steps=num_steps,
            barrier_eps=barrier_eps,
        )
        # Guard: barrier interp blew up
        if not torch.isfinite(ut_interp).all():
            ut_interp = (1.0 - t_next) * u0_flat + t_next * u_corr
    else:
        ut_interp = (1.0 - t_next) * u0_flat + t_next * u_corr

    return ((ut_interp - u_flat) / dt).detach()


def ipcfm_b_batched(
    ut: Tensor,
    vf: Tensor,
    t,
    u0: Tensor,
    dt: float,
    hfunc: Callable,
    ineq: EntropyIneq = None,
    mu_0: float = 0.1,
    decay_rate: float = 3.0,
    eps: float = 1e-6,
    newtonsteps: int = 1,
    guided_interpolation: bool = True,
    lam: float = 1e-2,
    step_size: float = 1e-2,
    num_steps: int = 10,
    barrier_eps: float = 1e-6,
    **kwargs,
) -> Tensor:
    """Batched Strategy B. Drop-in replacement for pcfm_batched()."""
    B, nx, nt = ut.shape
    n = nx * nt

    u_flat_b = ut.view(B, n).detach().clone()
    v_flat_b = vf.view(B, n)
    u0_flat_b = u0.view(B, n)

    # Normalise hfunc to a list
    if not isinstance(hfunc, (list, tuple)):
        hfunc = [hfunc] * B

    out = []
    for i in range(B):
        gc.collect()
        torch.cuda.empty_cache()
        v_proj = ipcfm_b_sample(
            u_flat_b[i], v_flat_b[i], t, u0_flat_b[i], dt, hfunc[i], ineq,
            mu_0=mu_0, decay_rate=decay_rate, eps=eps,
            newtonsteps=newtonsteps, lam=lam, step_size=step_size,
            num_steps=num_steps, guided_interpolation=guided_interpolation,
            barrier_eps=barrier_eps,
        )
        out.append(v_proj)

    return torch.stack(out, dim=0).view(B, nx, nt)


# ---------------------------------------------------------------------------
# Strategy C — Active-Set Projection
# ---------------------------------------------------------------------------

def _safe_solve(JJt: Tensor, rhs: Tensor, reg: float, device) -> Tensor:
    """Solve (JJt + reg*I) x = rhs using solve. Returns zeros if non-finite."""
    # Guard: NaN/Inf in inputs means the Jacobian blew up — skip correction
    if torch.isnan(JJt).any() or torch.isinf(JJt).any() or \
       torch.isnan(rhs).any() or torch.isinf(rhs).any():
        return torch.zeros_like(rhs)
    k = JJt.shape[0]
    A = JJt + reg * torch.eye(k, device=device)
    sol = torch.linalg.solve(A, rhs)
    if not torch.isfinite(sol).all():
        return torch.zeros_like(rhs)
    return sol


def _combined_newton_project(
    u_flat: Tensor,
    hfunc: Callable,
    ineq: EntropyIneq,
    eps: float = 1e-3,
    reg: float = 1e-6,
    active_set_log: Optional[List] = None,
) -> Tensor:
    """
    Project u onto the intersection of equality constraints h(u)=0 and
    active entropy inequality constraints g_A(u)=0, using a single
    closed-form Newton step (matching original PCFM style):

      u = u - J_combined^T (J_combined J_combined^T)^{-1} [h(u); g_A(u)]

    Active set: A = {k : g_k(u) > -eps}
    """
    u = u_flat.clone()
    n = u.shape[0]
    device = u.device

    # Entropy constraint values
    g_vals = ineq.values(u)
    active = g_vals > -eps
    active_indices = active.nonzero(as_tuple=True)[0]
    n_active = int(active.sum().item())

    if active_set_log is not None:
        active_set_log.append(n_active)

    # Equality residual and Jacobian
    res_eq = hfunc(u)
    m = res_eq.shape[0]
    J_eq = compute_jacobian(hfunc, u)

    if n_active == 0:
        # Pure equality Newton step
        JJt = J_eq @ J_eq.T
        if not (torch.isfinite(JJt).all() and torch.isfinite(res_eq).all()):
            return u_flat.clone()
        lam = _safe_solve(JJt, res_eq, reg, device)
        u = u - J_eq.T @ lam
        return u

    # Combined residual: [h(u); g_A(u)]
    res_ineq = g_vals[active_indices]                    # (n_active,)
    res_combined = torch.cat([res_eq, res_ineq])         # (m + n_active,)

    # Combined Jacobian: [J_h; J_g_A]
    J_ineq = ineq.jacobian_rows(active_indices)          # (n_active, n)
    J_combined = torch.cat([J_eq, J_ineq], dim=0)       # (m + n_active, n)

    JJt = J_combined @ J_combined.T
    if not (torch.isfinite(JJt).all() and torch.isfinite(res_combined).all()):
        return u_flat.clone()
    lam = _safe_solve(JJt, res_combined, reg, device)
    u = u - J_combined.T @ lam

    return u


def ipcfm_c_sample(
    u_flat: Tensor,
    v_flat: Tensor,
    t,
    u0_flat: Tensor,
    dt: float,
    hfunc: Callable,
    ineq: EntropyIneq,
    eps: float = 1e-3,
    solve_eps: float = 1e-6,
    newtonsteps: int = 1,
    active_set_log: Optional[List] = None,
    guided_interpolation: bool = False,
    interpolation_params: dict = {},
    **kwargs,
) -> Tensor:
    """
    Single-sample Strategy C (Active-Set Projection) — single closed-form Newton step.

    At each flow substep:
      1. Compute the Newton target: ut1 = u_flat + (1-t)*v_flat.
      2. Single Newton step on [h(u)=0; g_A(u)=0].
      3. Interpolate and return corrected vector field.
    """
    # Newton target (unconstrained endpoint)
    ut1 = u_flat + (1.0 - t) * v_flat

    # Single combined Newton step: enforce h(u)=0 and g_A(u)=0
    u_corr = _combined_newton_project(
        ut1, hfunc, ineq, eps=eps,
        reg=solve_eps, active_set_log=active_set_log,
    )

    # Linear interpolation to next flow time
    t_next = t + dt
    ut_interp = (1.0 - t_next) * u0_flat + t_next * u_corr

    return ((ut_interp - u_flat) / dt).detach()


def ipcfm_c_batched(
    ut: Tensor,
    vf: Tensor,
    t,
    u0: Tensor,
    dt: float,
    hfunc: Callable,
    ineq: EntropyIneq = None,
    eps: float = 1e-3,
    active_set_log: Optional[List] = None,
    newtonsteps: int = 1,
    guided_interpolation: bool = False,
    interpolation_params: dict = {},
    solve_eps: float = 1e-6,
    **kwargs,
) -> Tensor:
    """Batched Strategy C. Drop-in replacement for pcfm_batched()."""
    B, nx, nt = ut.shape
    n = nx * nt

    u_flat_b = ut.view(B, n).detach().clone()
    v_flat_b = vf.view(B, n)
    u0_flat_b = u0.view(B, n)

    # Normalise hfunc to a list
    if not isinstance(hfunc, (list, tuple)):
        hfunc = [hfunc] * B

    out = []
    for i in range(B):
        gc.collect()
        torch.cuda.empty_cache()
        sample_log = [] if active_set_log is not None else None
        v_proj = ipcfm_c_sample(
            u_flat_b[i], v_flat_b[i], t, u0_flat_b[i], dt, hfunc[i], ineq,
            eps=eps, solve_eps=solve_eps,
            newtonsteps=newtonsteps, active_set_log=sample_log,
            guided_interpolation=guided_interpolation,
            interpolation_params=interpolation_params,
        )
        if active_set_log is not None and sample_log:
            active_set_log.extend(sample_log)
        out.append(v_proj)

    return torch.stack(out, dim=0).view(B, nx, nt)


# ---------------------------------------------------------------------------
# Active-set logging utility (for Exp 4)
# ---------------------------------------------------------------------------

def log_active_set_size(
    tau: float,
    active_set_sizes: list,
    log_file: str = 'results/active_set_log.json',
):
    """
    Append active-set sizes for a given flow time tau to a JSONL file.
    Each line: {"tau": float, "active_set_sizes": [int, ...], "mean": float, "std": float}
    """
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    if active_set_sizes:
        import statistics
        mean_size = statistics.mean(active_set_sizes)
        std_size = statistics.stdev(active_set_sizes) if len(active_set_sizes) > 1 else 0.0
    else:
        mean_size, std_size = 0.0, 0.0
    entry = {
        'tau': float(tau),
        'active_set_sizes': active_set_sizes,
        'mean': mean_size,
        'std': std_size,
    }
    with open(log_file, 'a') as f:
        f.write(json.dumps(entry) + '\n')
