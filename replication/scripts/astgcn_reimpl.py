"""Standalone re-implementation of the ASTGCN used in the MAGN repo (src/models/astgcn.py), written from scratch.

Only torch, numpy and scipy are needed. `test_astgcn_reimpl.py` checks it against the upstream code: same parameter
count (2,164,866 for SD, f=15) and identical outputs for identical weights.

Tensor layout used throughout: x is (b, n, f, T) inside the blocks; the model takes (b, T, n, f) and returns (b, 12, n, 1).
  b batch, n sensors, f channels, T = 12 input steps
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import linalg


def cheb_polynomials(adj, order=3):
    """Fixed graph input. adj: (n, n) ndarray. Returns [T_0, T_1, T_2] as float32 tensors, each (n, n).
    Steps as in experiments/astgcn/main.py + src/utils/graph_algo.py:
      1. remove the diagonal, binarise: A[i, j] = 1 where adj[i, j] != 0
      2. symmetrise: A = max(A, A^T)
      3. normalised Laplacian L = I - D^-1/2 A D^-1/2
      4. scaled Laplacian L~ = 2 L / lambda_max - I   (lambda_max = largest eigenvalue of L)
      5. T_0 = I, T_1 = L~, T_k = 2 L~ T_{k-1} - T_{k-2}
    """
    n = adj.shape[0]
    a = adj - np.eye(n)
    a = (a != 0).astype(np.float64)
    a = np.maximum(a, a.T)
    d = a.sum(1)
    d_inv_sqrt = np.where(d > 0, d, 1.0) ** -0.5
    d_inv_sqrt[d == 0] = 0.0
    lap = np.eye(n) - d_inv_sqrt[:, None] * a * d_inv_sqrt[None, :]
    lam = linalg.eigsh(sp.csr_matrix(lap), 1, which="LM")[0][0]
    l_t = 2.0 / lam * lap - np.eye(n)
    polys = [np.eye(n), l_t]
    for _ in range(2, order):
        polys.append(2 * l_t @ polys[-1] - polys[-2])
    return [torch.tensor(p, dtype=torch.float32) for p in polys[:order]]


class TemporalAttention(nn.Module):
    def __init__(self, f, n, T):
        super().__init__()
        self.U1 = nn.Parameter(torch.empty(n))
        self.U2 = nn.Parameter(torch.empty(f, n))
        self.U3 = nn.Parameter(torch.empty(f))
        self.be = nn.Parameter(torch.empty(1, T, T))
        self.Ve = nn.Parameter(torch.empty(T, T))

    def forward(self, x):                                   # (b, n, f, T) -> (b, T, T)
        lhs = torch.einsum("bnft,n->btf", x, self.U1) @ self.U2          # (b, T, n)
        rhs = torch.einsum("f,bnft->bnt", self.U3, x)                    # (b, n, T)
        e = self.Ve @ torch.sigmoid(lhs @ rhs + self.be)                 # (b, T, T)
        return F.softmax(e, dim=1)                                       # columns sum to 1


class SpatialAttention(nn.Module):
    def __init__(self, f, n, T):
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(T))
        self.W2 = nn.Parameter(torch.empty(f, T))
        self.W3 = nn.Parameter(torch.empty(f))
        self.bs = nn.Parameter(torch.empty(1, n, n))
        self.Vs = nn.Parameter(torch.empty(n, n))

    def forward(self, x):                                   # (b, n, f, T) -> (b, n, n)
        lhs = torch.einsum("bnft,t->bnf", x, self.W1) @ self.W2          # (b, n, T)
        rhs = torch.einsum("f,bnft->bnt", self.W3, x).transpose(-1, -2)  # (b, T, n)
        s = self.Vs @ torch.sigmoid(lhs @ rhs + self.bs)                 # (b, n, n)
        return F.softmax(s, dim=1)


class ChebConvWithSAt(nn.Module):
    def __init__(self, polys, f_in, f_out):
        super().__init__()
        self.polys = polys                                   # fixed (n, n) tensors, not parameters
        self.Theta = nn.ParameterList([nn.Parameter(torch.empty(f_in, f_out)) for _ in polys])

    def forward(self, x, s):                                # x (b, n, f, T), s (b, n, n) -> (b, n, f_out, T)
        out = 0
        for t_k, theta in zip(self.polys, self.Theta):
            a = t_k.to(x.device) * s                                     # (b, n, n): T_k masked by attention
            agg = torch.einsum("bmi,bmft->bift", a, x)                   # sum over sensors m: (T_k * S)^T applied to x
            out = out + torch.einsum("bift,fo->biot", agg, theta)        # per-order channel mixing
        return F.relu(out)


class Block(nn.Module):
    def __init__(self, f_in, n, T, polys, chev=64, tfilt=64):
        super().__init__()
        self.tat = TemporalAttention(f_in, n, T)
        self.sat = SpatialAttention(f_in, n, T)
        self.cheb = ChebConvWithSAt(polys, f_in, chev)
        self.time_conv = nn.Conv2d(chev, tfilt, kernel_size=(1, 3), padding=(0, 1))
        self.res_conv = nn.Conv2d(f_in, tfilt, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(tfilt)

    def forward(self, x):                                   # (b, n, f, T)
        b, n, f, T = x.shape
        e = self.tat(x)                                                  # (b, T, T)
        x_tat = (x.reshape(b, -1, T) @ e).reshape(b, n, f, T)            # time-reweighted copy of x
        s = self.sat(x_tat)                                              # spatial attention from x_tat ...
        g = self.cheb(x, s)                                              # ... but the graph conv is applied to x itself
        y = self.time_conv(g.permute(0, 2, 1, 3))                        # (b, 64, n, T)
        r = self.res_conv(x.permute(0, 2, 1, 3))                         # (b, 64, n, T)
        z = F.relu(r + y).permute(0, 3, 2, 1)                            # (b, T, n, 64)
        return self.ln(z).permute(0, 2, 3, 1)                            # (b, n, 64, T)


class ASTGCN(nn.Module):
    def __init__(self, polys, n, f, T=12, horizon=12, blocks=2, chev=64, tfilt=64):
        super().__init__()
        self.blocks = nn.ModuleList([Block(f, n, T, polys, chev, tfilt)] +
                                    [Block(tfilt, n, T, polys, chev, tfilt) for _ in range(blocks - 1)])
        self.final_conv = nn.Conv2d(T, horizon, kernel_size=(1, tfilt))

    def forward(self, x):                                   # (b, T, n, f) -> (b, horizon, n, 1)
        x = x.permute(0, 2, 3, 1)
        for blk in self.blocks:
            x = blk(x)
        return self.final_conv(x.permute(0, 3, 1, 2))[:, :, :, -1].unsqueeze(-1)


def init_like_upstream(model):
    """ASTGCN_Engine.__init__: every tensor with dim > 1 gets Xavier-uniform, every 0/1-dim tensor (biases, U1, U3, W1, W3,
    LayerNorm weight AND bias) gets U(0, 1)."""
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
