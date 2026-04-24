"""
evaluate_ipcfm_rd.py — Evaluation script for I-PCFM on the 1D Fisher-KPP
reaction-diffusion equation.

Two inequality-constraint branches, selected via --ineq:
    linear:    Max principle u(x,t) in [0, 1] (global, invariant interval)
    nonlinear: L^2 Gronwall envelope  int u^2 dx <= E_0 * exp(2 rho t)

Usage:
    python evaluate_ipcfm_rd.py --method all --exp1_main --no_wandb \
        --ckpt logs/rd_ic/20000.pt --n_samples 30 --ineq linear
    python evaluate_ipcfm_rd.py --method all --exp1_main --no_wandb \
        --ckpt logs/rd_ic/20000.pt --n_samples 30 --ineq nonlinear
"""
import argparse
import json
import os
import random
import sys
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from models import get_flow_model
from scripts.training.utils import load_config
from pcfm.constraints import Residuals
from pcfm.pcfm_sampling import make_grid
from pcfm.ipcfm_sampling import (
    ipcfm_a_batched,
    ipcfm_b_batched,
    ipcfm_c_batched,
    HeatMaxPrincipleIneq,
    RDEnergyGronwallIneq,
)
from pcfm.baselines import (
    vanilla_ffm_batched,
    pcfm_equality_batched,
)
from datasets.rd1d import RD1DDataset

ALL_METHODS = ['vanilla', 'pcfm_equality', 'ipcfm_a', 'ipcfm_b', 'ipcfm_c']
METHOD_FN_MAP = {
    'vanilla':       vanilla_ffm_batched,
    'pcfm_equality': pcfm_equality_batched,
    'ipcfm_a':       ipcfm_a_batched,
    'ipcfm_b':       ipcfm_b_batched,
    'ipcfm_c':       ipcfm_c_batched,
}

# Fisher-KPP coefficients (must match datasets/generate_RD1d_data.py)
RHO_DEFAULT = 0.01
NU_DEFAULT = 0.005
T_MAX_DEFAULT = 0.99  # fin_time in the generator


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='I-PCFM RD Evaluation Script')

    parser.add_argument('--method', type=str, default='pcfm_equality',
        choices=ALL_METHODS + ['all'])
    parser.add_argument('--skip_methods', type=str, default='')
    parser.add_argument('--n_steps', type=int, default=100)
    parser.add_argument('--n_samples', type=int, default=30)
    parser.add_argument('--ineq', type=str, default='linear',
        choices=['linear', 'nonlinear'],
        help="Inequality constraint: 'linear' = max principle [0,1], "
             "'nonlinear' = L^2 Gronwall envelope.")

    parser.add_argument('--ckpt', type=str, default='logs/rd_ic/20000.pt')
    parser.add_argument('--config', type=str, default='configs/rd1d.yml')
    parser.add_argument('--data', type=str,
        default='datasets/data/RD_neumann_test_nIC30_nBC30.h5')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--results_dir', type=str, default='results')

    parser.add_argument('--wandb_project', type=str, default='ipcfm')
    parser.add_argument('--no_wandb', action='store_true')

    parser.add_argument('--exp1_main', action='store_true')

    parser.add_argument('--mu_0', type=float, default=0.001)
    parser.add_argument('--decay_rate', type=float, default=3.0)
    parser.add_argument('--eps', type=float, default=1e-3)
    parser.add_argument('--slack_threshold', type=float, default=0.05)
    parser.add_argument('--solve_eps', type=float, default=1e-4)
    parser.add_argument('--newtonsteps', type=int, default=1)
    parser.add_argument('--result_suffix', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model and data loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, cfg, device: torch.device):
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_path)
        alt = os.path.join(ckpt_dir, 'latest.pt')
        if os.path.exists(alt):
            print(f'[WARNING] {ckpt_path} not found, using {alt}')
            ckpt_path = alt
        else:
            raise FileNotFoundError(
                f'No checkpoint found at {ckpt_path} or {alt}.\n'
                f'Run training first: python scripts/training/main.py configs/rd1d.yml ...'
            )
    model = get_flow_model(cfg.model, cfg.encoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = {k: v for k, v in ckpt['model'].items() if k != '_metadata'}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f'Loaded model from {ckpt_path} (step {ckpt.get("step", "?")})')
    return model


def load_rd_data(data_path: str, n_samples: int, seed: int,
                 device: torch.device):
    """Load RD test trajectories — random subset of shape (N, Nx, nt)."""
    root, data_file = os.path.split(data_path)
    ds = RD1DDataset(root=root, split='test', data_file=data_file)
    total = len(ds)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(total), min(n_samples, total)))
    samples = [ds[i] for i in idx]
    u_true = torch.stack(samples).to(device)
    print(f'Loaded {u_true.shape[0]} RD test samples, shape {tuple(u_true.shape)}')
    return u_true


# ---------------------------------------------------------------------------
# Grid + residual helpers (shared across all branches)
# ---------------------------------------------------------------------------

def _rd_grid(nx: int, nt: int, device: torch.device):
    """Cell-centered x on [0,1] and t on [0, t_max], matching the RD generator."""
    x = (torch.linspace(0., 1., nx + 1, device=device)[:-1]
         + (1.0 / nx) * 0.5)
    t_grid = torch.linspace(0., T_MAX_DEFAULT, nt, device=device)
    return x, t_grid


def build_hfunc_rd(u_true_sample: torch.Tensor, device: torch.device,
                   nx: int, nt: int,
                   rho: float = RHO_DEFAULT, nu: float = NU_DEFAULT):
    """Per-sample residual callable: IC + mass-with-reaction equality."""
    x, t_grid = _rd_grid(nx, nt, device)
    dx = x[1] - x[0]
    dt = t_grid[1] - t_grid[0]
    data = u_true_sample.unsqueeze(0).to(device)
    res_obj = Residuals(data=data, x=x, t_grid=t_grid, dx=dx, dt=dt,
                        nx=nx, nt=nt, rho=rho, nu=nu, bc=None, left_bc=None)
    return lambda u_flat: res_obj.full_residual_rd(u_flat)


# ---------------------------------------------------------------------------
# Inequality builders: linear (max principle) vs nonlinear (L^2 Gronwall)
# ---------------------------------------------------------------------------

def build_rd_ineqs_linear(u_true: torch.Tensor, nx: int, nt: int):
    """Per-sample linear max-principle box with global [0,1] (Fisher-KPP invariant)."""
    B = u_true.shape[0]
    return [HeatMaxPrincipleIneq(nx, nt, u0_min=0.0, u0_max=1.0)
            for _ in range(B)]


def build_global_rd_ineq_linear(u_true: torch.Tensor, nx: int, nt: int):
    """Shared linear ineq for the batched loop (global [0,1])."""
    return HeatMaxPrincipleIneq(nx, nt, u0_min=0.0, u0_max=1.0)


def build_rd_ineqs_nonlinear(u_true: torch.Tensor, nx: int, nt: int,
                             device: torch.device, rho: float = RHO_DEFAULT):
    """Per-sample L^2 Gronwall envelope using each sample's own E_0."""
    B = u_true.shape[0]
    x, t_grid = _rd_grid(nx, nt, device)
    dx = float(x[1] - x[0])
    ineqs = []
    for i in range(B):
        u0 = u_true[i, :, 0]
        E0 = float((u0 * u0).sum() * dx)
        ineqs.append(RDEnergyGronwallIneq(nx=nx, nt=nt, dx=dx,
                                          t_grid=t_grid.cpu(),
                                          rho=rho, E0=E0))
    return ineqs


def build_global_rd_ineq_nonlinear(u_true: torch.Tensor, nx: int, nt: int,
                                   device: torch.device,
                                   rho: float = RHO_DEFAULT):
    """Shared nonlinear ineq for the batched loop.
    Use the max E_0 across samples so the envelope is a valid upper bound for all."""
    x, t_grid = _rd_grid(nx, nt, device)
    dx = float(x[1] - x[0])
    u0_all = u_true[:, :, 0]
    E0_max = float((u0_all * u0_all).sum(dim=1).max() * dx)
    return RDEnergyGronwallIneq(nx=nx, nt=nt, dx=dx,
                                t_grid=t_grid.cpu(),
                                rho=rho, E0=E0_max)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics_rd(u_pred: torch.Tensor, u_true: torch.Tensor,
                       ineqs, nx: int, nt: int,
                       rho: float = RHO_DEFAULT,
                       nu: float = NU_DEFAULT) -> dict:
    B = u_pred.shape[0]
    device = u_pred.device

    u_flat_all = u_pred.view(B, -1)
    valid_mask = (torch.isfinite(u_flat_all).all(dim=1)
                  & (u_flat_all.abs().max(dim=1).values < 1e6))
    n_valid = int(valid_mask.sum().item())

    # Per-sample residuals: equalities + inequality max violation
    inf = float('inf')
    ic_per = torch.full((B,), inf, device=device)
    cl_per = torch.full((B,), inf, device=device)
    max_ineq_per = torch.full((B,), inf, device=device)
    ineq_viol_norm_per = torch.full((B,), inf, device=device)

    x, t_grid = _rd_grid(nx, nt, device)
    dx = x[1] - x[0]
    dt = t_grid[1] - t_grid[0]

    for i in range(B):
        if not valid_mask[i]:
            continue
        # Inequality
        g = ineqs[i].values(u_pred[i].flatten())
        viol = g.clamp(min=0.0)
        max_ineq_per[i] = viol.max()
        ineq_viol_norm_per[i] = viol.norm()
        # Equality (IC and mass-with-reaction)
        res_obj = Residuals(
            data=u_true[i:i+1], x=x, t_grid=t_grid, dx=dx, dt=dt,
            nx=nx, nt=nt, rho=rho, nu=nu, bc=None, left_bc=None,
        )
        u_flat = u_pred[i].flatten()
        ic_per[i] = res_obj.ic_residual(u_flat).norm()
        cl_per[i] = res_obj.mass_residual_rd(u_flat).norm()

    # Aggregates over valid samples
    finite_mask = valid_mask
    ce_ineq = ineq_viol_norm_per[finite_mask].mean().item() if finite_mask.any() else float('nan')
    ce_ic = ic_per[finite_mask].mean().item() if finite_mask.any() else float('nan')
    ce_cl = cl_per[finite_mask].mean().item() if finite_mask.any() else float('nan')

    # Feasibility metrics
    thr = 1e-3
    feasibility_rate = (max_ineq_per <= thr).float().mean().item()
    joint_feasibility_rate = (
        (max_ineq_per <= thr) & (ic_per <= thr) & (cl_per <= thr)
    ).float().mean().item()

    # Distributional moment-matching vs ground truth
    if n_valid > 0:
        u_valid = u_pred[valid_mask]
        u_true_valid = u_true[valid_mask]
        mu_gen = u_valid.mean(0)
        mu_gt = u_true_valid.mean(0)
        mmse = (mu_gen - mu_gt).pow(2).mean().item()
        sig_gen = u_valid.std(0)
        sig_gt = u_true_valid.std(0)
        smse = (sig_gen - sig_gt).pow(2).mean().item()
    else:
        mmse = float('nan')
        smse = float('nan')

    return {
        'ce_ic': ce_ic,
        'ce_cl': ce_cl,
        'ce_ineq': ce_ineq,
        'feasibility_rate': feasibility_rate,
        'joint_feasibility_rate': joint_feasibility_rate,
        'mmse': mmse,
        'smse': smse,
        'n_valid': n_valid,
    }


# ---------------------------------------------------------------------------
# Core sampling loop
# ---------------------------------------------------------------------------

def run_sampling_rd(
    model,
    u_true: torch.Tensor,
    method_name: str,
    n_steps: int,
    hfuncs,
    global_ineq,
    method_kwargs: dict,
    device: torch.device,
    silent: bool = False,
) -> tuple:
    B, nx, nt = u_true.shape
    method_fn = METHOD_FN_MAP[method_name]

    grid = make_grid((nx, nt), device=device)
    torch.manual_seed(42)
    with torch.no_grad():
        u0 = model.gp.sample(grid, (nx, nt), n_samples=B)
    u0 = u0.to(device)
    u = u0.clone()

    dt = 1.0 / n_steps
    ts = torch.linspace(0, 1, n_steps + 1, device=device)[:-1]

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_start = time.perf_counter()

    for tau in (tqdm(ts, desc=f'{method_name}') if not silent else ts):
        with torch.no_grad():
            vf = model(tau, u)

        if method_name == 'vanilla':
            with torch.no_grad():
                v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=hfuncs)
        elif method_name in ('ipcfm_a', 'ipcfm_b', 'ipcfm_c'):
            v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=hfuncs,
                               ineq=global_ineq, **method_kwargs)
        else:
            v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=hfuncs,
                               **method_kwargs)

        u = u + dt * v_proj

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.perf_counter() - t_start
    time_per_sample = elapsed / B

    return u.detach(), time_per_sample


# ---------------------------------------------------------------------------
# Utilities: wandb, JSON, summary
# ---------------------------------------------------------------------------

def init_wandb(args, run_name_suffix=''):
    try:
        import wandb
        run_name = f'rd_{args.ineq}_{args.method}_steps{args.n_steps}_n{args.n_samples}{run_name_suffix}'
        wandb.init(project=args.wandb_project, name=run_name,
                   config=vars(args), reinit=True)
        return wandb
    except Exception as e:
        print(f'[WARNING] wandb init failed: {e}')
        return None


def wandb_log(wb, metrics, step=None, prefix=''):
    if wb is None:
        return
    try:
        wb.log({f'{prefix}{k}': v for k, v in metrics.items()}, step=step)
    except Exception:
        pass


def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.',
                exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'Saved: {path}')


def append_summary(results_dir, text):
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'summary.md'), 'a') as f:
        f.write(text + '\n')


def save_error_log(results_dir, exp_name, error):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f'{exp_name}_error.log')
    with open(path, 'w') as f:
        f.write(f'Error in {exp_name}:\n{error}\n\n{traceback.format_exc()}')
    print(f'Error log saved: {path}')


def _build_method_kwargs(args, method_name: str) -> dict:
    base = {'newtonsteps': args.newtonsteps}
    if method_name == 'ipcfm_a':
        return {**base, 'slack_threshold': args.slack_threshold, 'eps': args.solve_eps}
    elif method_name == 'ipcfm_b':
        return {**base, 'mu_0': args.mu_0, 'decay_rate': args.decay_rate,
                'guided_interpolation': True}
    elif method_name == 'ipcfm_c':
        return {**base, 'eps': args.eps, 'solve_eps': args.solve_eps}
    return {}


# ---------------------------------------------------------------------------
# Experiment 1: Main comparison table
# ---------------------------------------------------------------------------

def run_exp1_main_rd(args, model, u_true, device):
    print('\n' + '=' * 60)
    print(f'Experiment 1 (RD, ineq={args.ineq}): Main Comparison Table')
    print('=' * 60)

    results = {}
    methods = ALL_METHODS if args.method == 'all' else [args.method]
    skip = [m.strip() for m in args.skip_methods.split(',') if m.strip()]
    methods = [m for m in methods if m not in skip]

    B, nx, nt = u_true.shape
    hfuncs = [build_hfunc_rd(u_true[j], device, nx=nx, nt=nt) for j in range(B)]
    if args.ineq == 'linear':
        ineqs = build_rd_ineqs_linear(u_true, nx=nx, nt=nt)
        global_ineq = build_global_rd_ineq_linear(u_true, nx=nx, nt=nt)
    else:
        ineqs = build_rd_ineqs_nonlinear(u_true, nx=nx, nt=nt, device=device)
        global_ineq = build_global_rd_ineq_nonlinear(u_true, nx=nx, nt=nt,
                                                     device=device)

    for method_name in methods:
        print(f'\nRunning {method_name}...')
        try:
            method_kwargs = _build_method_kwargs(args, method_name)
            u_pred, tps = run_sampling_rd(
                model, u_true, method_name, args.n_steps,
                hfuncs, global_ineq, method_kwargs, device,
            )
            metrics = compute_metrics_rd(u_pred, u_true, ineqs=ineqs,
                                         nx=nx, nt=nt)
            metrics['time_per_sample_s'] = tps
            results[method_name] = metrics
            print(f'  {method_name}: {metrics}')
        except Exception as e:
            print(f'  ERROR in {method_name}: {e}')
            save_error_log(args.results_dir, f'exp1_rd_{args.ineq}_{method_name}', str(e))
            results[method_name] = {'error': str(e)}

    out_path = os.path.join(
        args.results_dir,
        f'exp1_rd_{args.ineq}_main_table_n{args.n_samples}{args.result_suffix}.json',
    )
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    save_json(results, out_path)

    _print_latex_table(results)

    summary = f'\n## Exp 1 (RD, ineq={args.ineq}): Main Comparison Table\n'
    for m, v in results.items():
        summary += f'- {m}: {v}\n'
    append_summary(args.results_dir, summary)

    return results


def _print_latex_table(results):
    print('\n\\begin{tabular}{lrrrrrrrr}')
    print(r'\hline')
    print(r'Method & CE(IC) & CE(CL) & CE-Ineq & Feas(ineq) & Feas(joint) & MMSE & SMSE & Time(s/sample) \\')
    print(r'\hline')
    for method, m in results.items():
        if 'error' in m:
            print(f'{method} & \\multicolumn{{8}}{{c}}{{ERROR}} \\\\')
        else:
            print(
                f'{method} & '
                f'{m.get("ce_ic", float("nan")):.4f} & '
                f'{m.get("ce_cl", float("nan")):.4f} & '
                f'{m.get("ce_ineq", float("nan")):.4f} & '
                f'{m.get("feasibility_rate", float("nan")):.4f} & '
                f'{m.get("joint_feasibility_rate", float("nan")):.4f} & '
                f'{m.get("mmse", float("nan")):.6f} & '
                f'{m.get("smse", float("nan")):.6f} & '
                f'{m.get("time_per_sample_s", float("nan")):.3f} \\\\'
            )
    print(r'\hline')
    print(r'\end{tabular}')


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    wb = None
    if not args.no_wandb:
        wb = init_wandb(args)

    cfg = load_config(args.config)
    model = load_model(args.ckpt, cfg, device)

    nx = int(cfg.sample_dims[0]) if hasattr(cfg, 'sample_dims') else 128
    nt = int(cfg.sample_dims[1]) if hasattr(cfg, 'sample_dims') else 100
    u_true = load_rd_data(args.data, args.n_samples, args.seed, device)

    if args.exp1_main:
        try:
            results = run_exp1_main_rd(args, model, u_true, device)
            if wb:
                for method, metrics in results.items():
                    if 'error' not in metrics:
                        wandb_log(wb, metrics, prefix=f'{method}/')
        except Exception as e:
            save_error_log(args.results_dir, f'exp1_rd_{args.ineq}', str(e))
            print(f'[ERROR] Exp 1 (RD, {args.ineq}) failed: {e}')

    if wb:
        try:
            wb.finish()
        except Exception:
            pass

    print(f'\nAll results saved to: {args.results_dir}/')


if __name__ == '__main__':
    main()
