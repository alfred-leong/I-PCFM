"""Worker script: run one (method, sample_idx) and print OK/BLOWN_UP or crash."""
import sys, os, torch, numpy as np
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

result = u[0].detach().cpu().numpy()
if not np.isfinite(result).all() or np.abs(result).max() > 100:
    print("BLOWN_UP")
    sys.exit(1)
print("OK")
