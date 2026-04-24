"""
baselines.py — Baseline sampling methods for I-PCFM comparison.

Two baselines:
  1. vanilla_ffm_batched     — Pure Euler integration, no constraints
  2. pcfm_equality_batched   — Equality-only PCFM (the original method)
"""

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


