"""Screen samples for hangs across all methods using per-sample timeouts.
Runs each (method, sample) pair in a subprocess with a timeout.
If it hangs, kill it and mark that sample as bad."""
import subprocess, sys, os, json, time

GOOD_INDICES_PHASE1 = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 13, 15,
    19, 20, 21, 22, 24, 26, 27, 29, 30, 31, 32, 33, 34,
    36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49,
    50, 51, 52, 53, 54, 56, 57, 58, 59, 61, 62, 63, 64, 66,
]

METHODS_TO_SCREEN = ['pcfm_equality', 'ipcfm_a', 'ipcfm_b', 'ipcfm_c']
TIMEOUT_SEC = 120  # 2 minutes per sample — if it takes longer, it's hung
GPU_ID = os.environ.get('CUDA_VISIBLE_DEVICES', '4')

WORKER_SCRIPT = """
import sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate_ipcfm import load_model, load_test_data, build_entropy_ineq, build_hfunc, METHOD_FN_MAP
from scripts.training.utils import load_config
from pcfm.pcfm_sampling import make_grid

method_name = sys.argv[1]
sample_idx = int(sys.argv[2])

device = torch.device('cuda')
cfg = load_config('configs/burgers1d.yml')
model = load_model('/external1/alfred/pcfm_logs/burgers_ic/20000.pt', cfg, device)
u_true_all = load_test_data('datasets/data/burgers_test_nIC30_nBC30.h5', sample_idx + 1, device)
u_true = u_true_all[sample_idx]
nx, nt = u_true.shape
ineq = build_entropy_ineq(device, nx, nt)
hfunc = build_hfunc(u_true, device, nx, nt)
method_fn = METHOD_FN_MAP[method_name]

grid = make_grid((nx, nt), device=device)
torch.manual_seed(1000 + sample_idx)
with torch.no_grad():
    u0 = model.gp.sample(grid, (nx, nt), n_samples=1).to(device)
u = u0.clone()
N_STEPS = 100
dt_val = 1.0 / N_STEPS
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
            v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt_val, hfunc=[hfunc])
    elif method_name in ('ipcfm_a', 'ipcfm_b', 'ipcfm_c'):
        v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt_val, hfunc=[hfunc],
                           ineq=ineq, **method_kwargs)
    else:
        v_proj = method_fn(ut=u, vf=vf, t=tau, u0=u0, dt=dt_val, hfunc=[hfunc],
                           **method_kwargs)
    u = u + dt_val * v_proj

import numpy as np
result = u[0].detach().cpu().numpy()
if not np.isfinite(result).all() or np.abs(result).max() > 100:
    print("BLOWN_UP")
    sys.exit(1)
print("OK")
"""

def screen_one(method, sample_idx):
    """Run one (method, sample) in subprocess with timeout. Returns 'OK', 'FAIL', 'TIMEOUT', or 'BLOWN_UP'."""
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = GPU_ID
    try:
        result = subprocess.run(
            ['conda', 'run', '--no-capture-output', '-n', 'i-pcfm',
             'python', '-u', 'screen_worker.py', method, str(sample_idx)],
            capture_output=True, text=True, timeout=TIMEOUT_SEC, env=env,
        )
        stdout = result.stdout.strip()
        if result.returncode == 0 and 'OK' in stdout:
            return 'OK'
        elif 'BLOWN_UP' in stdout:
            return 'BLOWN_UP'
        else:
            return 'FAIL'
    except subprocess.TimeoutExpired:
        return 'TIMEOUT'
    except Exception as e:
        return f'ERROR:{e}'


def main():
    bad_samples = set()
    results = {}

    for method in METHODS_TO_SCREEN:
        print(f'\n=== Screening {method} on {len(GOOD_INDICES_PHASE1)} samples ===')
        results[method] = {}
        for j in GOOD_INDICES_PHASE1:
            if j in bad_samples:
                print(f'  Sample {j+1:3d}: SKIP (already failed)')
                results[method][j] = 'SKIP'
                continue
            t0 = time.time()
            status = screen_one(method, j)
            elapsed = time.time() - t0
            print(f'  Sample {j+1:3d}: {status}  ({elapsed:.1f}s)')
            results[method][j] = status
            if status in ('TIMEOUT', 'FAIL', 'BLOWN_UP') or status.startswith('ERROR'):
                bad_samples.add(j)

    final_good = [j for j in GOOD_INDICES_PHASE1 if j not in bad_samples]
    print(f'\n=== RESULTS ===')
    print(f'Bad samples (0-indexed): {sorted(bad_samples)}')
    print(f'Bad samples (1-indexed): {sorted(j+1 for j in bad_samples)}')
    print(f'Good samples remaining: {len(final_good)} / {len(GOOD_INDICES_PHASE1)}')
    print(f'\nFinal GOOD_INDICES = {final_good}')

    # Save to file
    with open('results/screened_good_indices.json', 'w') as f:
        json.dump({'good_indices': final_good, 'bad_indices': sorted(bad_samples),
                   'details': {m: {str(k): v for k, v in d.items()} for m, d in results.items()}}, f, indent=2)
    print('Saved to results/screened_good_indices.json')


if __name__ == '__main__':
    main()
