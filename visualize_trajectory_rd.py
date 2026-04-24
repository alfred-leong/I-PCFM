"""Visualize flow-time trajectory heatmaps at tau = 0, 0.2, 0.4, 0.6, 0.8, 1.0
for each method on the 1-D Fisher--KPP reaction--diffusion task.

Mirrors visualize_trajectory.py (which targets Burgers') but:
  - loads the RD dataset + RD checkpoint
  - uses RD's equality residual (IC + mass-with-reaction) and inequality
    (HeatMaxPrincipleIneq with u in [0, 1])
  - plots with a cmap and range tuned to Fisher--KPP's [0, 1] invariant interval.

Usage:
    CUDA_VISIBLE_DEVICES=3 python visualize_trajectory_rd.py \
        --n_samples 10 --out_dir results/sample_heatmaps_rd
"""
import argparse, json, sys, os, random, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models import get_flow_model
from scripts.training.utils import load_config
from pcfm.pcfm_sampling import make_grid
from pcfm.ipcfm_sampling import (
    ipcfm_a_batched, ipcfm_b_batched, ipcfm_c_batched,
    HeatMaxPrincipleIneq, _combined_newton_project,
)
from pcfm.baselines import vanilla_ffm_batched, pcfm_equality_batched
from datasets.rd1d import RD1DDataset
from evaluate_ipcfm_rd import build_hfunc_rd

N_STEPS = 100
SNAPSHOT_TAUS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
METHODS = ['vanilla', 'pcfm_equality', 'ipcfm_a', 'ipcfm_b', 'ipcfm_c']
METHOD_FN = {
    'vanilla': vanilla_ffm_batched,
    'pcfm_equality': pcfm_equality_batched,
    'ipcfm_a': ipcfm_a_batched,
    'ipcfm_b': ipcfm_b_batched,
    'ipcfm_c': ipcfm_c_batched,
}
CONFIG_PATH = 'configs/rd1d.yml'
CKPT_PATH = 'logs/rd_ic/20000.pt'
DATA_ROOT = 'datasets/data'
DATA_FILE = 'RD_neumann_test_nIC30_nBC30.h5'


def load_model_from_ckpt(ckpt_path, cfg, device):
    model = get_flow_model(cfg.model, cfg.encoder).to(device)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = {k: v for k, v in ck['model'].items() if k != '_metadata'}
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'Loaded {ckpt_path}')
    return model


def run_with_snapshots(method_name, model, u_true_1, hfunc, ineq, device, seed):
    """Run flow integration for one sample, capturing (x,t) snapshots at given taus."""
    nx, nt = u_true_1.shape
    n = nx * nt
    method_fn = METHOD_FN[method_name]

    grid = make_grid((nx, nt), device=device)
    torch.manual_seed(seed)
    with torch.no_grad():
        u0 = model.gp.sample(grid, (nx, nt), n_samples=1)
    u0 = u0.to(device)
    u = u0.clone()

    dt = 1.0 / N_STEPS
    ts = torch.linspace(0, 1, N_STEPS + 1, device=device)[:-1]

    snapshot_steps = {int(round(tau * N_STEPS)): tau for tau in SNAPSHOT_TAUS if tau < 1.0}
    snapshots = {}
    if 0 in snapshot_steps:
        snapshots[0.0] = u[0].cpu().numpy().copy()

    method_kwargs = {}
    if method_name == 'ipcfm_a':
        method_kwargs = {'slack_threshold': 0.05, 'eps': 1e-4, 'newtonsteps': 1}
    elif method_name == 'ipcfm_b':
        method_kwargs = {'mu_0': 1e-3, 'decay_rate': 3.0,
                         'guided_interpolation': True, 'newtonsteps': 1}
    elif method_name == 'ipcfm_c':
        method_kwargs = {'eps': 1e-3, 'solve_eps': 1e-4, 'newtonsteps': 1}

    for step_idx, tau in enumerate(tqdm(ts, desc=f'{method_name}', leave=False)):
        with torch.no_grad():
            vf = model(tau, u)
        if method_name == 'vanilla':
            with torch.no_grad():
                v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=[hfunc])
        elif method_name in ('ipcfm_a', 'ipcfm_b', 'ipcfm_c'):
            v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=[hfunc],
                               ineq=ineq, **method_kwargs)
        else:
            v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt, hfunc=[hfunc],
                               **method_kwargs)
        u = u + dt * v_proj
        next_step = step_idx + 1
        if next_step in snapshot_steps:
            snapshots[snapshot_steps[next_step]] = u[0].cpu().numpy().copy()

    # Post-loop cleanup for ipcfm_b (barrier method) — force final feasibility
    # via a combined projection step, matching the Burgers' trajectory script.
    if method_name == 'ipcfm_b' and ineq is not None:
        u_flat = u.view(1, n)
        try:
            u_corr = _combined_newton_project(
                u_flat[0].detach().clone(), hfunc, ineq,
                eps=1e-3, reg=1e-4,
            )
            u = u_corr.view(1, nx, nt)
        except Exception:
            pass

    snapshots[1.0] = u[0].detach().cpu().numpy().copy()
    return snapshots


PRETTY = {
    'vanilla': 'Vanilla FFM',
    'pcfm_equality': 'PCFM',
    'ipcfm_a': 'I-PCFM-A',
    'ipcfm_b': 'I-PCFM-B',
    'ipcfm_c': 'I-PCFM-C',
}


def render_one(sample_idx, model, u_true_all, ineq, device, out_path):
    u_true_1 = u_true_all[sample_idx]
    nx, nt = u_true_1.shape
    hfunc = build_hfunc_rd(u_true_1, device, nx=nx, nt=nt)
    seed = 1000 + sample_idx

    n_methods = len(METHODS)
    n_snaps = len(SNAPSHOT_TAUS)
    fig, axes = plt.subplots(n_methods + 1, n_snaps,
                             figsize=(3.5 * n_snaps, 3.0 * (n_methods + 1)))

    # Fisher-KPP invariant range; pad slightly so violations are visible.
    vmin, vmax = -0.2, 1.2
    cmap = 'RdBu_r'
    extent = [0, 1, 0, 1]

    # Row 0: ground truth
    gt = u_true_1.cpu().numpy()
    for col in range(n_snaps):
        ax = axes[0, col]
        im = ax.imshow(gt.T, origin='lower', aspect='auto',
                       extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        if col == 0:
            ax.set_ylabel('Ground Truth\n$t$', fontsize=9)
        ax.set_xlabel('$x$')
        plt.colorbar(im, ax=ax, fraction=0.046)

    for col, tau_val in enumerate(SNAPSHOT_TAUS):
        axes[0, col].set_title(f'τ = {tau_val}', fontsize=10, fontweight='bold')

    # Rows 1+: methods
    for row_idx, method in enumerate(METHODS):
        print(f'  [{method}]')
        try:
            snaps = run_with_snapshots(method, model, u_true_1, hfunc, ineq, device, seed)
        except Exception as e:
            print(f'    ERROR: {e}')
            for col in range(n_snaps):
                ax = axes[row_idx + 1, col]
                ax.text(0.5, 0.5, 'ERROR', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='red')
                if col == 0:
                    ax.set_ylabel(PRETTY[method], fontsize=9)
            continue

        for col, tau_val in enumerate(SNAPSHOT_TAUS):
            ax = axes[row_idx + 1, col]
            if tau_val in snaps:
                sample = snaps[tau_val]
                im = ax.imshow(sample.T, origin='lower', aspect='auto',
                               extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{PRETTY[method]}\n$t$', fontsize=9)
            ax.set_xlabel('$x$')

    plt.suptitle(f'Flow Trajectory (RD, Fisher--KPP): Sample #{sample_idx+1} '
                 f'({nx}×{nt})', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {out_path}')


def load_rd_pool(n_samples, device, seed=42):
    ds = RD1DDataset(root=DATA_ROOT, split='test', data_file=DATA_FILE)
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(ds)), min(n_samples, len(ds))))
    samples = [ds[i] for i in idx]
    return torch.stack(samples).to(device), idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default=CKPT_PATH)
    parser.add_argument('--config', default=CONFIG_PATH)
    parser.add_argument('--out_dir', default='results/sample_heatmaps_rd')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(args.config)
    model = load_model_from_ckpt(args.ckpt, cfg, device)

    u_true_all, picked_idx = load_rd_pool(args.n_samples, device, seed=args.seed)
    nx, nt = u_true_all.shape[1], u_true_all.shape[2]
    ineq = HeatMaxPrincipleIneq(nx, nt, u0_min=0.0, u0_max=1.0)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f'Rendering {len(picked_idx)} RD trajectories -> {args.out_dir}')
    for rank in range(u_true_all.shape[0]):
        out_path = os.path.join(args.out_dir, f'trajectory_{rank+1:03d}.png')
        print(f'\n=== [{rank+1}/{u_true_all.shape[0]}] test-file idx {picked_idx[rank]} ===')
        t0 = time.perf_counter()
        render_one(rank, model, u_true_all, ineq, device, out_path)
        print(f'  elapsed {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
