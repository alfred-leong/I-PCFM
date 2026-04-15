"""Generate and visualize samples per method for visual comparison.
Uses pre-screened good sample indices (no singular matrices for any method)."""
import argparse, sys, os, json, torch, numpy as np, h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_ipcfm import (
    load_model, load_test_data, run_sampling, build_entropy_ineq, build_hfunc,
    METHOD_FN_MAP,
)
from scripts.training.utils import load_config
from pcfm.pcfm_sampling import make_grid

# Pre-screened good indices (0-based) — passed ipcfm_c, ipcfm_b, AND pcfm_equality
DEFAULT_GOOD_INDICES = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 15,
    19, 20, 21, 22, 24, 26, 27, 29, 30, 31, 32, 33, 34,
    36, 37, 38, 40, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 66,
]

N_STEPS = 100
METHODS = ['vanilla', 'pcfm_equality', 'ipcfm_a', 'ipcfm_b', 'ipcfm_c']
CONFIG_PATH = 'configs/burgers1d.yml'


def run_single_sample(method_name, model, u_true_1, hfunc, ineq, device, seed):
    """Run flow integration for one sample with a fixed GP seed.
    Returns the prediction numpy array, or None on failure."""
    nx, nt = u_true_1.shape
    method_fn = METHOD_FN_MAP[method_name]

    grid = make_grid((nx, nt), device=device)
    torch.manual_seed(seed)
    with torch.no_grad():
        u0 = model.gp.sample(grid, (nx, nt), n_samples=1).to(device)
    u = u0.clone()
    dt = 1.0 / N_STEPS
    ts = torch.linspace(0, 1, N_STEPS + 1, device=device)[:-1]

    method_kwargs = {}
    if method_name == 'ipcfm_b':
        method_kwargs = {'mu_0': 0.001}
    elif method_name == 'ipcfm_c':
        method_kwargs = {'eps': 0.001}

    for tau in ts:
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

    # Post-loop cleanup for ipcfm_b
    if method_name == 'ipcfm_b' and ineq is not None:
        from pcfm.ipcfm_sampling import _combined_newton_project
        n = nx * nt
        u_flat = u.view(1, n)
        u_corr = _combined_newton_project(
            u_flat[0].detach().clone(), hfunc, ineq,
            eps=1e-3, reg=1e-4,
        )
        u = u_corr.view(1, nx, nt)

    result = u[0].detach().cpu().numpy()
    # Check for blown-up values
    if not np.isfinite(result).all() or np.abs(result).max() > 100:
        return None
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', default='models/20000.pt')
    parser.add_argument('--data', default='datasets/I-PCFM_data/burgers_test_nIC30_nBC30.h5')
    parser.add_argument('--out_dir', default='results/sample_heatmaps_v4')
    parser.add_argument('--n_samples', type=int, default=20,
                        help='Number of pre-screened samples to visualize (taken from the front)')
    parser.add_argument('--good_indices_file', default=None,
                        help='JSON file with good_indices list; defaults to in-script list')
    args = parser.parse_args()

    if args.good_indices_file and os.path.exists(args.good_indices_file):
        with open(args.good_indices_file) as f:
            good_indices = json.load(f)['good_indices']
    else:
        good_indices = DEFAULT_GOOD_INDICES
    good_indices = list(good_indices)[:args.n_samples]
    print(f'Visualizing {len(good_indices)} samples → {args.out_dir}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config(CONFIG_PATH)
    model = load_model(args.ckpt, cfg, device)

    # Load enough data to cover max index
    n_load = max(good_indices) + 1
    u_true_all = load_test_data(args.data, n_load, device)
    nx, nt = u_true_all.shape[1], u_true_all.shape[2]
    ineq = build_entropy_ineq(device, nx, nt)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Build hfuncs: k=5 for vanilla/pcfm_equality, k=20 for ipcfm methods
    EQUALITY_METHODS = {'vanilla', 'pcfm_equality', 'soft_penalty'}
    print(f'Building hfuncs for {len(good_indices)} pre-screened samples...')
    hfuncs_k5 = {j: build_hfunc(u_true_all[j], device, nx, nt, k=5) for j in good_indices}
    hfuncs_k20 = {j: build_hfunc(u_true_all[j], device, nx, nt, k=20) for j in good_indices}

    # Run all methods on good samples
    method_preds = {m: {} for m in METHODS}
    for method in METHODS:
        hfuncs = hfuncs_k5 if method in EQUALITY_METHODS else hfuncs_k20
        print(f'\n=== Running {method} on {len(good_indices)} samples ===')
        for j in good_indices:
            seed = 1000 + j
            try:
                pred = run_single_sample(method, model, u_true_all[j], hfuncs[j],
                                         ineq, device, seed)
                method_preds[method][j] = pred
                if pred is not None:
                    print(f'  Sample {j+1:3d}: OK')
                else:
                    print(f'  Sample {j+1:3d}: BLOWN UP')
            except Exception as e:
                print(f'  Sample {j+1:3d}: FAIL ({type(e).__name__})')
                method_preds[method][j] = None
            torch.cuda.empty_cache()

    # Save heatmap PNGs
    print(f'\n=== Saving {len(good_indices)} heatmap PNGs ===')
    x_extent = [0, 1, 0, 1]
    n_cols = len(METHODS) + 1

    for rank, j in enumerate(good_indices):
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 3.2))

        # Ground truth
        ax = axes[0]
        gt = u_true_all[j].cpu().numpy()
        im = ax.imshow(gt.T, origin='lower', aspect='auto',
                       extent=x_extent, cmap='RdBu_r', vmin=-0.5, vmax=1.5)
        ax.set_title('Ground Truth', fontsize=9)
        ax.set_ylabel('t')
        ax.set_xlabel('x')
        plt.colorbar(im, ax=ax, fraction=0.046)

        for col, method in enumerate(METHODS):
            ax = axes[col + 1]
            pred = method_preds[method].get(j)
            if pred is not None:
                im = ax.imshow(pred.T, origin='lower', aspect='auto',
                               extent=x_extent, cmap='RdBu_r', vmin=-0.5, vmax=1.5)
                plt.colorbar(im, ax=ax, fraction=0.046)
            else:
                ax.text(0.5, 0.5, 'ERROR', transform=ax.transAxes,
                        ha='center', va='center', fontsize=10, color='red')
            ax.set_title(method, fontsize=9)
            ax.set_xlabel('x')

        plt.suptitle(f'Sample #{j+1} (pre-screened, 101x101 Burgers)', fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f'sample_{rank+1:03d}.png')
        plt.savefig(out_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved {out_path}')

    print(f'\nDone. {len(good_indices)} sample PNGs saved to {out_dir}/')


if __name__ == '__main__':
    main()
