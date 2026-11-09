"""Check astgcn_reimpl.py against the unchanged upstream model.   python replication/scripts/test_astgcn_reimpl.py"""
import sys, types
from pathlib import Path
import numpy as np
import torch

MAGN = Path(__file__).resolve().parents[1] / "magn"
sys.path.insert(0, str(MAGN)); sys.path.insert(0, str(Path(__file__).resolve().parent))
for name in ("seaborn", "matplotlib", "matplotlib.pyplot"):          # upstream imports them but only uses them in comments
    sys.modules.setdefault(name, types.ModuleType(name))

from src.models.astgcn import ASTGCN as Upstream
from src.utils.graph_algo import normalize_adj_mx, calculate_cheb_poly
from astgcn_reimpl import ASTGCN as Mine, cheb_polynomials, init_like_upstream

rng = np.random.default_rng(0)

def random_graph(n):
    a = (rng.random((n, n)) < 0.15).astype(float); np.fill_diagonal(a, 1.0); return a

# 1. Chebyshev polynomials: same as upstream's pipeline (main.py)
n = 40; adj = random_graph(n)
m = adj - np.eye(n); u = np.zeros((n, n), dtype=np.float32)
for i in range(n): u[i, np.nonzero(m[i])[0]] = 1
up_polys = [torch.from_numpy(p).float() for p in calculate_cheb_poly(normalize_adj_mx(u, "scalap")[0], 3)]
my_polys = cheb_polynomials(adj, 3)
print("cheb polys max abs diff:", max(float((a - b).abs().max()) for a, b in zip(up_polys, my_polys)))

# 2. same weights -> same outputs (state dicts have the same key order, so they load by position)
f = 15
up = Upstream(node_num=n, input_dim=f, output_dim=1, device="cpu", cheb_poly=up_polys, order=3, nb_block=2,
              nb_chev_filter=64, nb_time_filter=64, time_stride=1)
mine = Mine(my_polys, n, f)
init_like_upstream(up)
mine.load_state_dict({k2: v for (k1, v), k2 in zip(up.state_dict().items(), mine.state_dict().keys())})
x = torch.randn(3, 12, n, f)
with torch.no_grad():
    d = (up(x) - mine(x)).abs().max()
print("output shape", tuple(mine(x).shape), " max abs diff upstream vs mine:", float(d))

# 3. parameter counts for SD (n = 716)
for f in (15, 3):
    pm = Mine([torch.eye(716)] * 3, 716, f)
    print(f"SD, f={f}: {sum(p.numel() for p in pm.parameters()):,} parameters")
