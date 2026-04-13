"""
baselines.py — Baseline sampling methods for I-PCFM comparison.

Three baselines:
  1. vanilla_ffm_batched     — Pure Euler integration, no constraints
  2. pcfm_equality_batched   — Equality-only PCFM (the original method)
  3. soft_penalty_batched    — Equality + soft L2 penalty for g(u) = -u <= 0
"""

import torch
from torch import Tensor
from typing import Callable

from .pcfm_sampling import pcfm_batched


def vanilla_ffm_batched(
    ut: Tensor,
    vf: Tensor,
    t,
    u0: Tensor,
    dt: float,
    hfunc: Callable = None,
    **kwargs,
) -> Tensor:
    """
    Baseline 1: Vanilla FFM.
    Returns the raw vector field with no projection or constraint enforcement.
    The caller applies: u = u + dt * vanilla_ffm_batched(...)
    hfunc is accepted but ignored for interface compatibility.
    Returns: (B, nx, nt)
    """
    return vf


def pcfm_equality_batched(
    ut: Tensor,
    vf: Tensor,
    t,
    u0: Tensor,
    dt: float,
    hfunc: Callable,
    **kwargs,
) -> Tensor:
    """
    Baseline 2: Equality-only PCFM (the original pcfm_batched).
    Enforces h(u)=0 (IC + dynamics + mass conservation) but ignores g(u)<=0.
    Drop-in replacement for pcfm_batched with the same interface.
    Returns: (B, nx, nt)
    """
    return pcfm_batched(
        ut, vf, t, u0, dt, hfunc,
        mode=kwargs.get('mode', 'root'),
        newtonsteps=kwargs.get('newtonsteps', 1),
        guided_interpolation=kwargs.get('guided_interpolation', False),
        interpolation_params=kwargs.get('interpolation_params', {}),
        eps=kwargs.get('eps', 1e-6),
        use_vmap=False,
    )


def soft_penalty_batched(
    ut: Tensor,
    vf: Tensor,
    t,
    u0: Tensor,
    dt: float,
    hfunc: Callable,
    lam_soft: float = 10.0,
    **kwargs,
) -> Tensor:
    """
    Baseline 3: Soft penalty for the inequality constraint g(u) = -u <= 0.

    Augments the equality residual with sqrt(lam_soft) * max(0, -u_i) for each
    grid point, so that the penalty in the Newton solve is lam_soft * ||max(-u,0)||^2.
    Uses guided_interpolation=True to propagate the penalty through the interpolation.

    The augmented residual passed to pcfm_batched:
        h_aug(u) = [h_eq(u), sqrt(lam_soft) * clamp(-u, min=0)]

    Args:
        lam_soft: weight for the inequality penalty (default 10.0)
    Returns: (B, nx, nt)
    """
    # Normalise hfunc to a list
    if not isinstance(hfunc, (list, tuple)):
        B = ut.shape[0]
        hfunc_list = [hfunc] * B
    else:
        hfunc_list = list(hfunc)

    def _make_aug(hf):
        def augmented_hfunc(u_flat: Tensor) -> Tensor:
            h_eq = hf(u_flat)
            violations = torch.clamp(-u_flat, min=0.0)
            return torch.cat([h_eq, lam_soft ** 0.5 * violations], dim=0)
        return augmented_hfunc

    aug_hfuncs = [_make_aug(hf) for hf in hfunc_list]

    return pcfm_batched(
        ut, vf, t, u0, dt, aug_hfuncs,
        mode=kwargs.get('mode', 'root'),
        newtonsteps=kwargs.get('newtonsteps', 1),
        guided_interpolation=kwargs.get('guided_interpolation', True),
        interpolation_params=kwargs.get('interpolation_params', {}),
        eps=kwargs.get('eps', 1e-6),
        use_vmap=False,
    )
