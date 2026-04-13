"""Visualize flow trajectory heatmaps at tau = 0, 0.2, 0.4, 0.6, 0.8, 1.0 for each method."""
import sys, os, torch, numpy as np, h5py, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_ipcfm import (
    load_model, load_test_data, build_entropy_ineq, build_hfunc,
    METHOD_FN_MAP,
)
from pcfm.ipcfm_sampling import _combined_newton_project
from scripts.training.utils import load_config
from pcfm.pcfm_sampling import make_grid

N_STEPS = 100
SNAPSHOT_TAUS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
METHODS = ['vanilla', 'pcfm_equality', 'ipcfm_a', 'ipcfm_b', 'ipcfm_c']
DATA_PATH = 'datasets/data/burgers_test_nIC30_nBC30.h5'
CKPT_PATH = '/external1/alfred/pcfm_logs/burgers_ic/20000.pt'
CONFIG_PATH = 'configs/burgers1d.yml'
SAMPLE_IDX = 0  # which test sample to use


def run_with_snapshots(method_name, model, u_true_1, hfunc, ineq, device):
    """Run flow integration for a single sample, capturing snapshots at specified taus."""
    nx, nt = u_true_1.shape
    n = nx * nt
    method_fn = METHOD_FN_MAP[method_name]

    # Sample initial noise
    grid = make_grid((nx, nt), device=device)
    torch.manual_seed(42)  # fixed seed for reproducibility across methods
    with torch.no_grad():
        u0 = model.gp.sample(grid, (nx, nt), n_samples=1)
    u0 = u0.to(device)
    u = u0.clone()  # (1, nx, nt)

    dt = 1.0 / N_STEPS
    ts = torch.linspace(0, 1, N_STEPS + 1, device=device)[:-1]

    # Snapshot indices: which Euler steps correspond to our desired taus
    snapshot_steps = {int(round(tau * N_STEPS)): tau for tau in SNAPSHOT_TAUS if tau < 1.0}
    snapshots = {}

    # Capture tau=0
    if 0 in snapshot_steps:
        snapshots[0.0] = u[0].cpu().numpy().copy()

    method_kwargs = {}
    if method_name == 'ipcfm_b':
        method_kwargs = {'mu_0': 0.001}
    elif method_name == 'ipcfm_c':
        method_kwargs = {'eps': 0.001}

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

        # Check if next step corresponds to a snapshot
        next_step = step_idx + 1
        if next_step in snapshot_steps:
            snapshots[snapshot_steps[next_step]] = u[0].cpu().numpy().copy()

    # Post-loop cleanup for ipcfm_b
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

    # Capture tau=1.0
    snapshots[1.0] = u[0].detach().cpu().numpy().copy()

    return snapshots


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(CONFIG_PATH)
    model = load_model(CKPT_PATH, cfg, device)
    u_true = load_test_data(DATA_PATH, max(SAMPLE_IDX + 1, 1), device)
    u_true_1 = u_true[SAMPLE_IDX]  # (nx, nt)
    nx, nt = u_true_1.shape

    hfunc = build_hfunc(u_true_1, device, nx, nt)
    ineq = build_entropy_ineq(device, nx, nt)

    n_methods = len(METHODS)
    n_snaps = len(SNAPSHOT_TAUS)

    fig, axes = plt.subplots(n_methods + 1, n_snaps, figsize=(3.5 * n_snaps, 3.0 * (n_methods + 1)))

    x_extent = [0, 1, 0, 1]
    vmin, vmax = -0.5, 1.5

    # Row 0: ground truth (same across all columns)
    gt = u_true_1.cpu().numpy()
    for col in range(n_snaps):
        ax = axes[0, col]
        im = ax.imshow(gt.T, origin='lower', aspect='auto',
                       extent=x_extent, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        tau_val = SNAPSHOT_TAUS[col]
        ax.set_title(f'Ground Truth' if col == 0 else '', fontsize=9)
        if col == 0:
            ax.set_ylabel('Ground Truth\nt', fontsize=9)
        ax.set_xlabel('x')
        plt.colorbar(im, ax=ax, fraction=0.046)

    # Add tau labels on top
    for col, tau_val in enumerate(SNAPSHOT_TAUS):
        axes[0, col].set_title(f'τ = {tau_val}', fontsize=10, fontweight='bold')

    # Rows 1+: each method
    for row_idx, method in enumerate(METHODS):
        print(f'Running {method}...')
        try:
            snapshots = run_with_snapshots(method, model, u_true_1, hfunc, ineq, device)
        except Exception as e:
            print(f'  ERROR: {e}')
            for col in range(n_snaps):
                ax = axes[row_idx + 1, col]
                ax.text(0.5, 0.5, f'ERROR', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='red')
                if col == 0:
                    ax.set_ylabel(method, fontsize=9)
            continue

        for col, tau_val in enumerate(SNAPSHOT_TAUS):
            ax = axes[row_idx + 1, col]
            if tau_val in snapshots:
                sample = snapshots[tau_val]
                im = ax.imshow(sample.T, origin='lower', aspect='auto',
                               extent=x_extent, cmap='RdBu_r', vmin=vmin, vmax=vmax)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{method}\nt', fontsize=9)
            ax.set_xlabel('x')

    plt.suptitle(f'Flow Trajectory: Sample #{SAMPLE_IDX+1} (101×101 Burgers)', fontsize=13, y=1.01)
    plt.tight_layout()
    out_path = 'results/exp1_trajectory_heatmaps.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
