"""
evaluate_ipcfm.py — Unified evaluation script for I-PCFM experiments.

Usage examples:
    python evaluate_ipcfm.py --method pcfm_equality --n_steps 10 --n_samples 4
    python evaluate_ipcfm.py --method ipcfm_b --n_samples 64 --exp2_sweep mu0
    python evaluate_ipcfm.py --method ipcfm_c --n_samples 64 --exp2_sweep eps
    python evaluate_ipcfm.py --method ipcfm_a --n_samples 64
    python evaluate_ipcfm.py --method all --n_steps 100 --n_samples 32 --exp3_timing
    python evaluate_ipcfm.py --method ipcfm_c --n_samples 64 --n_steps 100 --exp4_active_set
    python evaluate_ipcfm.py --method all --n_steps 200 --n_samples 512 --exp1_main
"""

import argparse
import json
import os
import sys
import time
import traceback

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add repo root to path
sys.path.insert(0, os.path.dirname(__file__))

from models import get_flow_model
from scripts.training.utils import load_config
from pcfm.constraints import Residuals
from pcfm.pcfm_sampling import make_grid
from pcfm.ipcfm_sampling import (
    ipcfm_a_batched,
    ipcfm_b_batched,
    ipcfm_c_batched,
    log_active_set_size,
    _combined_newton_project,
    EntropyIneq,
)
from pcfm.baselines import (
    vanilla_ffm_batched,
    pcfm_equality_batched,
    soft_penalty_batched,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_METHODS = ['vanilla', 'pcfm_equality', 'soft_penalty', 'ipcfm_a', 'ipcfm_b', 'ipcfm_c']

METHOD_FN_MAP = {
    'vanilla':       vanilla_ffm_batched,
    'pcfm_equality': pcfm_equality_batched,
    'soft_penalty':  soft_penalty_batched,
    'ipcfm_a':       ipcfm_a_batched,
    'ipcfm_b':       ipcfm_b_batched,
    'ipcfm_c':       ipcfm_c_batched,
}

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='I-PCFM Evaluation Script')

    # Core args
    parser.add_argument('--method', type=str, default='pcfm_equality',
        choices=ALL_METHODS + ['all'],
        help='Sampling method to evaluate')
    parser.add_argument('--skip_methods', type=str, default='',
        help='Comma-separated list of methods to skip (e.g. soft_penalty)')
    parser.add_argument('--n_steps', type=int, default=100,
        help='Number of flow integration steps')
    parser.add_argument('--n_samples', type=int, default=64,
        help='Number of samples to generate')
    parser.add_argument('--good_indices_file', type=str, default=None,
        help='JSON file with pre-screened good sample indices (overrides n_samples)')

    # File paths
    parser.add_argument('--ckpt', type=str,
        default='/external1/alfred/pcfm_logs/burgers_ic/20000.pt',
        help='Path to model checkpoint (falls back to latest.pt)')
    parser.add_argument('--data', type=str,
        default='datasets/data/burgers_test_nIC30_nBC30.h5',
        help='Path to test HDF5 data file')
    parser.add_argument('--config', type=str,
        default='configs/burgers1d.yml',
        help='Path to model config YAML')
    parser.add_argument('--device', type=str, default='cuda:0',
        help='PyTorch device (default: cuda:0)')
    parser.add_argument('--results_dir', type=str, default='results',
        help='Directory to save results')

    # Wandb
    parser.add_argument('--wandb_project', type=str, default='ipcfm',
        help='W&B project name')
    parser.add_argument('--no_wandb', action='store_true',
        help='Disable wandb logging')

    # Experiment flags
    parser.add_argument('--exp1_main', action='store_true',
        help='Run Exp 1: main comparison table')
    parser.add_argument('--exp2_sweep', type=str, default=None,
        choices=['mu0', 'eps', 'slack_threshold', 'solve_eps'],
        help='Run Exp 2: sweep mu0 (Strategy B), eps (Strategy C), '
             'slack_threshold (Strategy A), or solve_eps (A and C)')
    parser.add_argument('--exp3_timing', action='store_true',
        help='Run Exp 3: runtime breakdown')
    parser.add_argument('--exp4_active_set', action='store_true',
        help='Run Exp 4: active-set analysis')

    # Method-specific hyperparameters
    parser.add_argument('--mu_0', type=float, default=0.001,
        help='Strategy B: initial barrier coefficient')
    parser.add_argument('--decay_rate', type=float, default=3.0,
        help='Strategy B: barrier decay rate')
    parser.add_argument('--eps', type=float, default=1e-3,
        help='Strategy C active set tolerance (constraints with g_k > -eps are active)')
    parser.add_argument('--slack_threshold', type=float, default=0.05,
        help='Strategy A near-active band: constraints with g_k > -slack_threshold get slacks')
    parser.add_argument('--solve_eps', type=float, default=1e-4,
        help='Tikhonov regularizer for the Newton solve in Strategies A and C')
    parser.add_argument('--lam_soft', type=float, default=10.0,
        help='Soft penalty baseline: penalty weight')
    parser.add_argument('--newtonsteps', type=int, default=1,
        help='Number of Newton steps in projection')
    parser.add_argument('--result_suffix', type=str, default='',
        help='Suffix appended to output filenames to avoid overwriting (e.g. _v2)')

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model and data loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, cfg, device: torch.device):
    """Load pretrained FFM model from checkpoint."""
    # Try the specified path, fall back to latest.pt
    if not os.path.exists(ckpt_path):
        ckpt_dir = os.path.dirname(ckpt_path)
        alt = os.path.join(ckpt_dir, 'latest.pt')
        if os.path.exists(alt):
            print(f'[WARNING] {ckpt_path} not found, using {alt}')
            ckpt_path = alt
        else:
            raise FileNotFoundError(
                f'No checkpoint found at {ckpt_path} or {alt}.\n'
                f'Run training first: python scripts/training/main.py {cfg} ...'
            )

    model = get_flow_model(cfg.model, cfg.encoder).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt['model']
    # GPyTorch may include a '_metadata' key that load_state_dict doesn't expect
    state_dict = {k: v for k, v in state_dict.items() if k != '_metadata'}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f'Loaded model from {ckpt_path} (step {ckpt.get("step", "?")})')
    return model


def load_test_data(data_path: str, n_samples: int, device: torch.device):
    """
    Load Burgers test data from HDF5 file.
    Returns: (n_samples, nx, nt) float32 tensor on device
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Test data not found: {data_path}')

    with h5py.File(data_path, 'r') as f:
        u = f['u'][:]   # (N_ic, N_bc, nx, nt)

    N_ic, N_bc, nx, nt = u.shape
    total = N_ic * N_bc
    u_flat = u.reshape(total, nx, nt)

    if n_samples > total:
        print(f'[WARNING] Requested {n_samples} samples but only {total} available. Using {total}.')
        n_samples = total

    u_tensor = torch.from_numpy(u_flat[:n_samples].astype(np.float32)).to(device)
    print(f'Loaded {n_samples} test samples from {data_path}, shape: {u_tensor.shape}')
    return u_tensor


def build_hfunc(u_true_sample: torch.Tensor, device: torch.device, nx: int = 101, nt: int = 101, k: int = 20):
    """
    Build the Burgers IC constraint residual function for a single sample.

    Args:
        u_true_sample: (nx, nt) ground truth solution for this IC
        k: number of unrolled local flux collocation points
    Returns:
        callable: u_flat (n,) -> residual (m,)
    """
    x = torch.linspace(0, 1, nx, device=device)
    t_grid = torch.linspace(0, 1, nt, device=device)
    dx = x[1] - x[0]
    dt = t_grid[1] - t_grid[0]

    # data must be (1, nx, nt) so that data[0][:, 0] gives the IC
    data = u_true_sample.unsqueeze(0).to(device)

    res_obj = Residuals(
        data=data, x=x, t_grid=t_grid,
        dx=dx, dt=dt, nx=nx, nt=nt,
        rho=None, nu=None, bc=None, left_bc=None,
    )
    return lambda u_flat: res_obj.full_residual_burgers(u_flat, k=k)


def build_entropy_ineq(device: torch.device, nx: int = 101, nt: int = 101):
    """Build the EntropyIneq object for Burgers equation."""
    dx = 1.0 / (nx - 1)
    dt_pde = 1.0 / (nt - 1)
    return EntropyIneq(nx, nt, dx, dt_pde)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(u_pred: torch.Tensor, u_true: torch.Tensor,
                    ineq: 'EntropyIneq' = None,
                    hfuncs=None) -> dict:
    """
    Compute all I-PCFM metrics.

    Args:
        u_pred: (B, nx, nt) generated samples
        u_true: (B, nx, nt) ground truth samples
        ineq:   EntropyIneq object for computing entropy violations
        hfuncs: list of per-sample hfunc callables for IC/CL metrics
    Returns:
        dict with keys: ce_ineq, feasibility_rate, ce_ic, ce_cl, mmse, smse
    """
    B, nx, nt = u_pred.shape
    dx = 1.0 / (nx - 1)
    dt_pde = 1.0 / (nt - 1)

    # Filter out samples with NaN/Inf or huge values (diverged but finite)
    u_flat_all = u_pred.view(B, -1)
    valid_mask = torch.isfinite(u_flat_all).all(dim=1) & (u_flat_all.abs().max(dim=1).values < 1e6)
    n_valid = valid_mask.sum().item()

    # Entropy condition violation: g_k(u) > 0 means violation
    if ineq is None:
        ineq = EntropyIneq(nx, nt, dx=dx, dt_pde=dt_pde)
    viol_norms = []
    max_viols = []
    for i in range(B):
        if not valid_mask[i]:
            max_viols.append(torch.tensor(float('inf'), device=u_pred.device))
            continue
        g = ineq.values(u_pred[i].flatten())
        viol = g.clamp(min=0.0)  # only positive g = violation
        viol_norms.append(viol.norm())
        max_viols.append(viol.max())
    max_viols = torch.stack(max_viols)
    ce_ineq = torch.stack(viol_norms).mean().item() if viol_norms else float('nan')
    feasibility_rate = (max_viols <= 1e-3).float().mean().item()

    # CE(IC): IC residual norm, CE(CL): mass conservation residual norm
    ce_ic = float('nan')
    ce_cl = float('nan')
    if hfuncs is not None:
        ic_norms = []
        cl_norms = []
        for i in range(B):
            if not valid_mask[i]:
                continue
            # Build a Residuals object to get IC and mass residuals separately
            res_obj = Residuals(
                data=u_true[i:i+1], x=torch.linspace(0, 1, nx, device=u_pred.device),
                t_grid=torch.linspace(0, 1, nt, device=u_pred.device),
                dx=torch.tensor(dx), dt=torch.tensor(dt_pde),
                nx=nx, nt=nt, rho=None, nu=None, bc=None, left_bc=None,
            )
            u_flat = u_pred[i].flatten()
            ic_res = res_obj.ic_residual(u_flat)
            mass_res = res_obj.mass_residual_burgers(u_flat)[1:]
            ic_norms.append(ic_res.norm())
            cl_norms.append(mass_res.norm())
        ce_ic = torch.stack(ic_norms).mean().item() if ic_norms else float('nan')
        ce_cl = torch.stack(cl_norms).mean().item() if cl_norms else float('nan')

    # MMSE / SMSE: computed only on valid samples
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
        'mmse': mmse,
        'smse': smse,
    }


# ---------------------------------------------------------------------------
# Core sampling loop
# ---------------------------------------------------------------------------

def run_sampling(
    model,
    u_true: torch.Tensor,
    method_name: str,
    n_steps: int,
    hfuncs,
    method_kwargs: dict,
    device: torch.device,
    active_set_log_ref: list = None,
    silent: bool = False,
    ineq: 'EntropyIneq' = None,
) -> tuple:
    """
    Run the Euler flow integration using the specified method.

    Returns:
        (u_final, time_per_sample_s): final samples (B, nx, nt) and wall time per sample
    """
    B, nx, nt = u_true.shape
    method_fn = METHOD_FN_MAP[method_name]

    # Sample initial noise from GP prior — fixed seed so metrics are reproducible
    # and consistent across experiments (exp1/exp2/exp4 all draw the same 51 noise
    # realizations for a given method).
    grid = make_grid((nx, nt), device=device)
    torch.manual_seed(42)
    with torch.no_grad():
        u0 = model.gp.sample(grid, (nx, nt), n_samples=B)
    u0 = u0.to(device)
    u = u0.clone()

    dt = 1.0 / n_steps
    ts = torch.linspace(0, 1, n_steps + 1, device=device)[:-1]

    # Active-set log for Strategy C Exp 4
    step_active_log = {} if (active_set_log_ref is not None and method_name == 'ipcfm_c') else None

    torch.cuda.synchronize() if device.type == 'cuda' else None
    t_start = time.perf_counter()

    for tau in (tqdm(ts, desc=f'{method_name}') if not silent else ts):
        with torch.no_grad():
            vf = model(tau, u)

        # Build per-step active-set log list for Strategy C
        step_log = [] if step_active_log is not None else None
        if method_name == 'ipcfm_c' and step_log is not None:
            method_kwargs_step = dict(method_kwargs, active_set_log=step_log)
        else:
            method_kwargs_step = method_kwargs

        # Dispatch to the appropriate batched method
        if method_name == 'vanilla':
            with torch.no_grad():
                v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=hfuncs)
        elif method_name in ('ipcfm_a', 'ipcfm_b', 'ipcfm_c'):
            v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=hfuncs,
                               ineq=ineq, **method_kwargs_step)
        else:
            v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=hfuncs,
                               **method_kwargs_step)

        u = u + dt * v_proj

        # Free intermediate CUDA memory to prevent fragmentation/deadlock
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Accumulate active-set log
        if step_active_log is not None and step_log:
            tau_val = float(tau.item())
            step_active_log[tau_val] = step_log
            if active_set_log_ref is not None:
                active_set_log_ref.append({'tau': tau_val, 'sizes': step_log})

    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = time.perf_counter() - t_start
    time_per_sample = elapsed / B

    # Post-loop cleanup for Strategy B: active-set equality projection at tau=1.
    # Identifies active entropy constraints and enforces g_A(u)=0 jointly
    # with the equality constraints h(u)=0 via a single combined Newton solve.
    if method_name == 'ipcfm_b' and ineq is not None:
        B2, nx2, nt2 = u.shape
        n2 = nx2 * nt2
        # Normalise hfuncs to a list
        if not isinstance(hfuncs, (list, tuple)):
            hfuncs_list = [hfuncs] * B2
        else:
            hfuncs_list = hfuncs
        u_flat_b = u.view(B2, n2)
        u_list = []
        for i in range(B2):
            try:
                u_corr = _combined_newton_project(
                    u_flat_b[i].detach().clone(), hfuncs_list[i], ineq,
                    eps=1e-3, reg=1e-4,
                )
            except Exception:
                u_corr = u_flat_b[i].detach().clone()
            u_list.append(u_corr)
        u = torch.stack(u_list).view(B2, nx2, nt2)

    return u.detach(), time_per_sample


# ---------------------------------------------------------------------------
# Wandb utilities
# ---------------------------------------------------------------------------

def init_wandb(args, run_name_suffix=''):
    try:
        import wandb
        run_name = f'{args.method}_steps{args.n_steps}_n{args.n_samples}{run_name_suffix}'
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            reinit=True,
        )
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


# ---------------------------------------------------------------------------
# Result saving
# ---------------------------------------------------------------------------

def save_json(data, path):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
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


# ---------------------------------------------------------------------------
# Experiment 1: Main comparison table
# ---------------------------------------------------------------------------

def run_exp1_main(args, model, u_true, device):
    """Run all 6 methods and produce the main comparison table."""
    print('\n' + '='*60)
    print('Experiment 1: Main Comparison Table')
    print('='*60)

    results = {}
    methods = ALL_METHODS if args.method == 'all' else [args.method]
    skip = [m.strip() for m in args.skip_methods.split(',') if m.strip()]
    methods = [m for m in methods if m not in skip]

    # Build per-sample hfuncs: k=5 for vanilla/pcfm_equality, k=20 for ipcfm methods
    B = u_true.shape[0]
    EQUALITY_METHODS = {'vanilla', 'pcfm_equality', 'soft_penalty'}
    hfuncs_k5 = [build_hfunc(u_true[j], device, nx=101, nt=101, k=5) for j in range(B)]
    hfuncs_k20 = [build_hfunc(u_true[j], device, nx=101, nt=101, k=20) for j in range(B)]
    ineq = build_entropy_ineq(device, nx=101, nt=101)

    for method_name in methods:
        print(f'\nRunning {method_name}...')
        hfuncs = hfuncs_k5 if method_name in EQUALITY_METHODS else hfuncs_k20
        try:
            method_kwargs = _build_method_kwargs(args, method_name)
            u_pred, tps = run_sampling(
                model, u_true, method_name, args.n_steps,
                hfuncs, method_kwargs, device, ineq=ineq,
            )
            metrics = compute_metrics(u_pred, u_true, ineq=ineq, hfuncs=hfuncs)
            metrics['time_per_sample_s'] = tps
            results[method_name] = metrics
            print(f'  {method_name}: {metrics}')
        except Exception as e:
            print(f'  ERROR in {method_name}: {e}')
            save_error_log(args.results_dir, f'exp1_{method_name}', str(e))
            results[method_name] = {'error': str(e)}

    # Save results
    out_path = os.path.join(args.results_dir, f'exp1_main_table{args.result_suffix}.json')
    save_json(results, out_path)

    # Print LaTeX table
    _print_latex_table(results)

    # Summary
    summary = f'\n## Exp 1: Main Comparison Table\n'
    for m, v in results.items():
        summary += f'- {m}: {v}\n'
    append_summary(args.results_dir, summary)

    return results


def _print_latex_table(results):
    header = r'\begin{tabular}{lrrrrrrr}'
    cols = r'Method & CE(IC) & CE(CL) & CE-Ineq & Feasibility & MMSE & SMSE & Time(s/sample) \\'
    print('\n' + header)
    print(r'\hline')
    print(cols)
    print(r'\hline')
    for method, m in results.items():
        if 'error' in m:
            print(f'{method} & \\multicolumn{{7}}{{c}}{{ERROR}} \\\\')
        else:
            print(
                f'{method} & '
                f'{m.get("ce_ic", float("nan")):.4f} & '
                f'{m.get("ce_cl", float("nan")):.4f} & '
                f'{m.get("ce_ineq", float("nan")):.4f} & '
                f'{m.get("feasibility_rate", float("nan")):.4f} & '
                f'{m.get("mmse", float("nan")):.6f} & '
                f'{m.get("smse", float("nan")):.6f} & '
                f'{m.get("time_per_sample_s", float("nan")):.3f} \\\\'
            )
    print(r'\hline')
    print(r'\end{tabular}')


# ---------------------------------------------------------------------------
# Experiment 2: Constraint-quality tradeoff
# ---------------------------------------------------------------------------

def run_exp2_sweep(args, model, u_true, device):
    """Hyperparameter sweep for Exp 2."""
    print('\n' + '='*60)
    print('Experiment 2: Constraint-Quality Tradeoff')
    print('='*60)

    B = u_true.shape[0]
    hfuncs = [build_hfunc(u_true[j], device, nx=101, nt=101) for j in range(B)]
    ineq = build_entropy_ineq(device, nx=101, nt=101)
    results = {}

    if args.method == 'ipcfm_b' and args.exp2_sweep == 'mu0':
        mu0_values = [1e-4, 1e-3, 1e-2, 0.1]
        print(f'Sweeping mu_0 for ipcfm_b: {mu0_values}')
        for mu_0 in mu0_values:
            key = f'mu0={mu_0}'
            print(f'\n  mu_0={mu_0}')
            try:
                kwargs = {'mu_0': mu_0, 'decay_rate': args.decay_rate,
                          'newtonsteps': args.newtonsteps, 'guided_interpolation': True}
                u_pred, tps = run_sampling(
                    model, u_true, 'ipcfm_b', args.n_steps, hfuncs, kwargs, device,
                    ineq=ineq,
                )
                metrics = compute_metrics(u_pred, u_true, ineq=ineq, hfuncs=hfuncs)
                metrics['time_per_sample_s'] = tps
                results[key] = metrics
                print(f'    {metrics}')
            except Exception as e:
                print(f'    ERROR: {e}')
                save_error_log(args.results_dir, f'exp2_ipcfm_b_{mu_0}', str(e))
                results[key] = {'error': str(e)}

    elif args.method == 'ipcfm_c' and args.exp2_sweep == 'eps':
        eps_values = [1e-4, 1e-3, 1e-2, 0.1]
        print(f'Sweeping eps for ipcfm_c: {eps_values}')
        for eps_val in eps_values:
            key = f'eps={eps_val}'
            print(f'\n  eps={eps_val}')
            try:
                kwargs = {'eps': eps_val, 'solve_eps': args.solve_eps,
                          'newtonsteps': args.newtonsteps}
                u_pred, tps = run_sampling(
                    model, u_true, 'ipcfm_c', args.n_steps, hfuncs, kwargs, device,
                    ineq=ineq,
                )
                metrics = compute_metrics(u_pred, u_true, ineq=ineq, hfuncs=hfuncs)
                metrics['time_per_sample_s'] = tps
                results[key] = metrics
                print(f'    {metrics}')
            except Exception as e:
                print(f'    ERROR: {e}')
                save_error_log(args.results_dir, f'exp2_ipcfm_c_{eps_val}', str(e))
                results[key] = {'error': str(e)}

    elif args.method == 'ipcfm_a':
        print('Running ipcfm_a with default settings (no hyperparameter sweep)')
        try:
            kwargs = {'newtonsteps': args.newtonsteps,
                      'slack_threshold': args.slack_threshold,
                      'eps': args.solve_eps}
            u_pred, tps = run_sampling(
                model, u_true, 'ipcfm_a', args.n_steps, hfuncs, kwargs, device,
                ineq=ineq,
            )
            metrics = compute_metrics(u_pred, u_true, ineq=ineq, hfuncs=hfuncs)
            metrics['time_per_sample_s'] = tps
            results['ipcfm_a_default'] = metrics
            print(f'  {metrics}')
        except Exception as e:
            print(f'  ERROR: {e}')
            save_error_log(args.results_dir, 'exp2_ipcfm_a', str(e))
            results['ipcfm_a_default'] = {'error': str(e)}

    # Save results
    out_path = os.path.join(args.results_dir, f'exp2_tradeoff{args.result_suffix}.json')
    # Load existing results and merge
    if os.path.exists(out_path):
        with open(out_path, 'r') as f:
            existing = json.load(f)
        existing.update(results)
        results = existing
    save_json(results, out_path)

    # Generate plot
    _plot_exp2_tradeoff(results, os.path.join(args.results_dir, f'exp2_tradeoff{args.result_suffix}.png'))

    summary = f'\n## Exp 2: Constraint-Quality Tradeoff\n'
    for k, v in results.items():
        summary += f'- {k}: {v}\n'
    append_summary(args.results_dir, summary)

    return results


def _plot_exp2_tradeoff(results, out_path):
    """Plot feasibility_rate vs MMSE for all strategies."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {'ipcfm_a': 'blue', 'ipcfm_b': 'red', 'ipcfm_c': 'green',
              'pcfm_equality': 'gray', 'vanilla': 'black', 'soft_penalty': 'orange'}

    for key, metrics in results.items():
        if 'error' in metrics or 'feasibility_rate' not in metrics:
            continue
        method_color = 'purple'
        for m, c in colors.items():
            if m in key:
                method_color = c
                break
        ax.scatter(metrics['mmse'], metrics['feasibility_rate'],
                   color=method_color, s=80, label=key, zorder=5)
        ax.annotate(key, (metrics['mmse'], metrics['feasibility_rate']),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)

    ax.set_xlabel('MMSE')
    ax.set_ylabel('Feasibility Rate')
    ax.set_title('Constraint-Quality Tradeoff (Exp 2)')
    ax.legend(loc='best', fontsize=7)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved plot: {out_path}')


# ---------------------------------------------------------------------------
# Experiment 3: Runtime breakdown
# ---------------------------------------------------------------------------

def run_exp3_timing(args, model, u_true, device):
    """Benchmark wall-clock time for all methods."""
    print('\n' + '='*60)
    print('Experiment 3: Runtime Breakdown')
    print('='*60)

    methods = ALL_METHODS if args.method == 'all' else [args.method]
    skip = [m.strip() for m in args.skip_methods.split(',') if m.strip()]
    methods = [m for m in methods if m not in skip]
    B = u_true.shape[0]
    hfuncs = [build_hfunc(u_true[j], device, nx=101, nt=101, k=5) for j in range(B)]
    ineq = build_entropy_ineq(device, nx=101, nt=101)
    n_trials = 3
    results = {}

    for method_name in methods:
        print(f'\nTiming {method_name} ({n_trials} trials)...')
        method_kwargs = _build_method_kwargs(args, method_name)
        times = []
        for trial in range(n_trials):
            try:
                _, tps = run_sampling(
                    model, u_true, method_name, args.n_steps,
                    hfuncs, method_kwargs, device, silent=True,
                    ineq=ineq,
                )
                times.append(tps * 1000)  # ms/sample
                print(f'  Trial {trial+1}: {tps*1000:.1f} ms/sample')
            except Exception as e:
                print(f'  Trial {trial+1} ERROR: {e}')
                save_error_log(args.results_dir, f'exp3_{method_name}', str(e))

        if times:
            results[method_name] = {
                'mean_ms_per_sample': float(np.mean(times)),
                'std_ms_per_sample': float(np.std(times)),
                'raw_ms': times,
            }
        else:
            results[method_name] = {'error': 'all trials failed'}

    # Print summary table
    print('\nRuntime Summary:')
    print(f'{"Method":<20} {"Mean (ms/sample)":>18} {"Std":>10}')
    print('-' * 52)
    for m, v in results.items():
        if 'error' not in v:
            print(f'{m:<20} {v["mean_ms_per_sample"]:>18.1f} {v["std_ms_per_sample"]:>10.1f}')

    out_path = os.path.join(args.results_dir, f'exp3_runtime{args.result_suffix}.json')
    save_json(results, out_path)

    summary = f'\n## Exp 3: Runtime Breakdown\n'
    for m, v in results.items():
        summary += f'- {m}: {v}\n'
    append_summary(args.results_dir, summary)

    return results


# ---------------------------------------------------------------------------
# Experiment 4: Active-set analysis
# ---------------------------------------------------------------------------

def run_exp4_active_set(args, model, u_true, device):
    """Run Strategy C and log active-set sizes over flow time."""
    print('\n' + '='*60)
    print('Experiment 4: Active-Set Analysis (Strategy C)')
    print('='*60)

    B = u_true.shape[0]
    hfuncs = [build_hfunc(u_true[j], device, nx=101, nt=101, k=5) for j in range(B)]
    ineq = build_entropy_ineq(device, nx=101, nt=101)
    active_log_path = os.path.join(args.results_dir, f'active_set_log{args.result_suffix}.json')

    # Clear any existing log
    if os.path.exists(active_log_path):
        os.remove(active_log_path)

    active_set_log = []
    method_kwargs = {'eps': args.eps, 'solve_eps': args.solve_eps,
                     'newtonsteps': args.newtonsteps}

    try:
        u_pred, tps = run_sampling(
            model, u_true, 'ipcfm_c', args.n_steps,
            hfuncs, method_kwargs, device,
            active_set_log_ref=active_set_log,
            ineq=ineq,
        )
        metrics = compute_metrics(u_pred, u_true, ineq=ineq, hfuncs=hfuncs)
        print(f'\nMetrics: {metrics}')
    except Exception as e:
        print(f'ERROR: {e}')
        save_error_log(args.results_dir, 'exp4', str(e))
        return {}

    # Aggregate log: for each tau, collect mean and std of |A| across batch
    tau_data = {}
    for entry in active_set_log:
        tau_val = entry['tau']
        sizes = entry['sizes']
        if tau_val not in tau_data:
            tau_data[tau_val] = []
        tau_data[tau_val].extend(sizes)

    tau_list = sorted(tau_data.keys())
    means = [float(np.mean(tau_data[t])) for t in tau_list]
    stds = [float(np.std(tau_data[t])) for t in tau_list if len(tau_data[t]) > 1] + [0.0]
    stds = [float(np.std(tau_data[t])) if len(tau_data[t]) > 1 else 0.0 for t in tau_list]

    # Save aggregated data
    exp4_data = {
        'tau': tau_list,
        'mean_active_set_size': means,
        'std_active_set_size': stds,
        'metrics': metrics,
        'raw_log': active_set_log[:200],  # save first 200 entries
    }
    out_path = os.path.join(args.results_dir, f'exp4_active_set{args.result_suffix}.json')
    save_json(exp4_data, out_path)

    # Also save to active_set_log.json (per-step JSONL format)
    for tau_val, sizes in tau_data.items():
        log_active_set_size(tau_val, sizes, log_file=active_log_path)

    # Generate plot
    _plot_exp4_active_set(tau_list, means, stds,
                          os.path.join(args.results_dir, f'exp4_active_set{args.result_suffix}.png'))

    summary = (
        f'\n## Exp 4: Active-Set Analysis\n'
        f'- n_steps={args.n_steps}, n_samples={args.n_samples}, eps={args.eps}\n'
        f'- Final metrics: {metrics}\n'
        f'- Peak mean |A|: {max(means) if means else "N/A":.1f}\n'
    )
    append_summary(args.results_dir, summary)

    return exp4_data


def _plot_exp4_active_set(tau_list, means, stds, out_path):
    """Plot mean |A| vs tau with std shading."""
    fig, ax = plt.subplots(figsize=(8, 5))
    taus = np.array(tau_list)
    means_arr = np.array(means)
    stds_arr = np.array(stds)

    ax.plot(taus, means_arr, 'b-', linewidth=2, label='Mean |A(τ)|')
    ax.fill_between(taus, means_arr - stds_arr, means_arr + stds_arr,
                    alpha=0.3, color='blue', label='±1 std')
    ax.set_xlabel('Flow time τ')
    ax.set_ylabel('Active set size |A|')
    ax.set_title('Active-Set Size vs Flow Time (Strategy C, Exp 4)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f'Saved plot: {out_path}')


# ---------------------------------------------------------------------------
# Helper: build method kwargs from args
# ---------------------------------------------------------------------------

def _build_method_kwargs(args, method_name: str) -> dict:
    """Build the keyword arguments dict for each method from CLI args."""
    base = {'newtonsteps': args.newtonsteps}
    if method_name == 'ipcfm_a':
        return {**base, 'slack_threshold': args.slack_threshold, 'eps': args.solve_eps}
    elif method_name == 'ipcfm_b':
        return {**base, 'mu_0': args.mu_0, 'decay_rate': args.decay_rate,
                'guided_interpolation': True}
    elif method_name == 'ipcfm_c':
        return {**base, 'eps': args.eps, 'solve_eps': args.solve_eps}
    elif method_name == 'soft_penalty':
        return {**base, 'lam_soft': args.lam_soft}
    elif method_name in ('vanilla', 'pcfm_equality'):
        return {}
    return {}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Init wandb
    wb = None
    if not args.no_wandb:
        wb = init_wandb(args)

    # Load config and model
    cfg = load_config(args.config)
    model = load_model(args.ckpt, cfg, device)

    # Load test data
    if args.good_indices_file:
        with open(args.good_indices_file) as f:
            good_indices = json.load(f)['good_indices']
        n_load = max(good_indices) + 1
        u_true_all = load_test_data(args.data, n_load, device)
        u_true = u_true_all[good_indices]
        print(f'Using {len(good_indices)} pre-screened samples from {args.good_indices_file}')
    else:
        u_true = load_test_data(args.data, args.n_samples, device)

    # Dispatch to experiments
    any_exp_run = False

    # ipcfm_a has no hyperparameter to sweep, so run_exp2_sweep handles it directly
    # Trigger exp2 for ipcfm_a even without --exp2_sweep flag
    if args.exp2_sweep or args.method == 'ipcfm_a':
        any_exp_run = True
        try:
            results = run_exp2_sweep(args, model, u_true, device)
            wandb_log(wb, {'exp2_complete': 1})
        except Exception as e:
            save_error_log(args.results_dir, 'exp2', str(e))
            print(f'[ERROR] Exp 2 failed: {e}')

    if args.exp3_timing:
        any_exp_run = True
        try:
            results = run_exp3_timing(args, model, u_true, device)
            wandb_log(wb, {'exp3_complete': 1})
        except Exception as e:
            save_error_log(args.results_dir, 'exp3', str(e))
            print(f'[ERROR] Exp 3 failed: {e}')

    if args.exp4_active_set:
        any_exp_run = True
        try:
            results = run_exp4_active_set(args, model, u_true, device)
            wandb_log(wb, {'exp4_complete': 1})
        except Exception as e:
            save_error_log(args.results_dir, 'exp4', str(e))
            print(f'[ERROR] Exp 4 failed: {e}')

    if args.exp1_main or not any_exp_run:
        try:
            results = run_exp1_main(args, model, u_true, device)
            if wb:
                for method, metrics in results.items():
                    if 'error' not in metrics:
                        wandb_log(wb, metrics, prefix=f'{method}/')
        except Exception as e:
            save_error_log(args.results_dir, 'exp1', str(e))
            print(f'[ERROR] Exp 1 failed: {e}')

    if wb:
        try:
            wb.finish()
        except Exception:
            pass

    print(f'\nAll results saved to: {args.results_dir}/')


if __name__ == '__main__':
    main()
