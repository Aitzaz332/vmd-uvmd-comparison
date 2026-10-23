"""Evaluation metrics for one decomposition of one signal.

Reconstruction (no ground truth needed):
    recon_rmse      RMSE(x, sum_k u_k)                 x = the (noisy) input the method saw
    recon_rel_err   ||x - sum u|| / ||x||
    recon_snr_db    10 log10(||x||^2 / ||x - sum u||^2)
    spec_l1         UVMD training loss: sum |sum_k u_hat_k - f_hat_plus| on the one-sided mirrored spectrum
    denoise_rmse    RMSE(clean, sum u)                  (needs ground truth; how much noise was left out)
    energy_ratio    sum_k ||u_k||^2 / ||x||^2
    ortho_index     sum_{i<j} |<u_i,u_j>| / ||x||^2     (0 = modes orthogonal / no shared content)

Mode recovery (needs ground-truth modes m_j, j = 1..J):
    One-to-one matching: Hungarian assignment of J estimated modes to the J true modes maximising |corr|.
    Estimated modes not picked are "spurious"; their energy share is reported.
    mode_rmse_j, mode_nrmse_j (RMSE / std(m_j)), mode_corr_j, mode_snr_db_j = 10 log10(||m_j||^2/||m_j-u||^2)
    omega_err_j     |omega of matched mode - true centre frequency| (cycles/sample)
    Grouped matching (works for any K, e.g. K=13): every estimated mode goes to the true mode it projects
    onto most; the assigned modes are summed. Reported as grouped_mode_snr_db_j / grouped_mean_snr_db.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def _corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def _snr_db(ref, est):
    err = np.sum((ref - est) ** 2)
    return float(10 * np.log10(np.sum(ref ** 2) / err)) if err > 0 else float("inf")


def match_modes(u, true_modes):
    """Return the estimated modes re-ordered to the true-mode order (Hungarian on |corr|), shape (J,T)."""
    T = min(u.shape[1], true_modes.shape[1])
    C = np.array([[abs(_corr(u[k, :T], tm[:T])) for tm in true_modes] for k in range(len(u))])
    rows, cols = linear_sum_assignment(-C)
    out = np.zeros((len(true_modes), T))
    for k, j in zip(rows, cols):
        out[j] = u[k, :T]
    return out


def evaluate(x, u, omega=None, spec_l1=None, clean=None, true_modes=None, true_fc=None):
    x = np.asarray(x, dtype=np.float64)
    T = min(len(x), u.shape[1])
    x, u = x[:T], np.asarray(u, dtype=np.float64)[:, :T]
    rec = u.sum(axis=0)
    ex = np.sum(x ** 2)
    m = {
        "recon_rmse": float(np.sqrt(np.mean((x - rec) ** 2))),
        "recon_rel_err": float(np.linalg.norm(x - rec) / np.linalg.norm(x)),
        "recon_snr_db": _snr_db(x, rec),
        "energy_ratio": float(np.sum(u ** 2) / ex),
    }
    G = u @ u.T
    iu = np.triu_indices(len(u), 1)
    m["ortho_index"] = float(np.sum(np.abs(G[iu])) / ex)
    if spec_l1 is not None:
        m["spec_l1"] = float(spec_l1)
    if clean is not None:
        m["denoise_rmse"] = float(np.sqrt(np.mean((clean[:T] - rec) ** 2)))
    if true_modes is None:
        return m

    tm = np.asarray(true_modes, dtype=np.float64)[:, :T]
    J, K = len(tm), len(u)
    C = np.array([[abs(_corr(u[k], tm[j])) for j in range(J)] for k in range(K)])  # (K, J)
    # one-to-one
    if K >= J:
        rows, cols = linear_sum_assignment(-C)
        match = dict(zip(cols, rows))  # true j -> est k
        per = {"rmse": [], "nrmse": [], "corr": [], "snr": [], "omega_err": []}
        for j in range(J):
            k = match[j]
            e = u[k]
            rmse = float(np.sqrt(np.mean((tm[j] - e) ** 2)))
            per["rmse"].append(rmse)
            per["nrmse"].append(rmse / float(np.std(tm[j])))
            per["corr"].append(_corr(e, tm[j]))
            per["snr"].append(_snr_db(tm[j], e))
            if omega is not None and true_fc is not None:
                per["omega_err"].append(abs(float(omega[k]) - float(true_fc[j])))
        for j in range(J):
            m[f"mode_rmse_{j}"] = per["rmse"][j]
            m[f"mode_nrmse_{j}"] = per["nrmse"][j]
            m[f"mode_corr_{j}"] = per["corr"][j]
            m[f"mode_snr_db_{j}"] = per["snr"][j]
            if per["omega_err"]:
                m[f"omega_err_{j}"] = per["omega_err"][j]
        m["mean_mode_nrmse"] = float(np.mean(per["nrmse"]))
        m["mean_mode_corr"] = float(np.mean(per["corr"]))
        m["mean_mode_snr_db"] = float(np.mean(per["snr"]))
        if per["omega_err"]:
            m["mean_omega_err"] = float(np.mean(per["omega_err"]))
        spurious = [k for k in range(K) if k not in set(rows)]
        m["spurious_energy_share"] = float(sum(np.sum(u[k] ** 2) for k in spurious) / np.sum(u ** 2)) \
            if spurious else 0.0
    # grouped
    P = np.abs(u @ tm.T) / np.linalg.norm(tm, axis=1)  # projection of est k on true j
    owner = P.argmax(axis=1)
    g = []
    for j in range(J):
        est = u[owner == j].sum(axis=0) if np.any(owner == j) else np.zeros(T)
        s = _snr_db(tm[j], est)
        m[f"grouped_mode_snr_db_{j}"] = s
        g.append(s)
    m["grouped_mean_snr_db"] = float(np.mean(g))
    return m
