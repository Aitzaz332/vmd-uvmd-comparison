"""Training and evaluation for the CNN pipelines.

Two kinds of pipeline share ONE training loop (``fit``) and ONE classifier (``e2e/cnn1d.py``):

  frozen features   VMD / VMD-N / recon-trained UVMD: the modes are computed once and are not updated. ``fit`` gets
                    the bare ``Fault1DCNN`` and cached mode tensors (n, K, T).
  end to end        ``E2E(uvmd, cnn)``: ``fit`` gets raw windows (n, T); every step runs window -> UVMD -> modes ->
                    CNN, and the loss reaches alpha_raw and lambda of the UVMD:

                        loss = CE(logits, label) + rec_weight * mean_b( |sum_k u_hat_k - f_hat|_1 / |f_hat|_1 )

                    The reconstruction term is the upstream UVMD loss divided by the L1 norm of the window's
                    one-sided spectrum, so it is dimensionless (0 = perfect reconstruction) and ``rec_weight``
                    means the same for every window length and signal scale.

Model selection is identical for all pipelines: the epoch with the best validation accuracy (ties: lower validation
cross-entropy); the best weights, UVMD parameters included, are restored before testing. The reconstruction term
never enters model selection.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score

from decomp.uvmd_torch import BatchedUVMD
from decomp.vmd_numpy import vmd


class E2E(nn.Module):
    """Window (B, T) -> UVMD modes (B, K, T) -> CNN logits. Returns (logits, rec_rel (B,))."""

    def __init__(self, uvmd: BatchedUVMD, cnn: nn.Module):
        super().__init__()
        self.uvmd, self.cnn = uvmd, cnn

    def forward(self, x):
        modes, _, aux = self.uvmd(x)
        return self.cnn(modes), aux["rec_l1"] / aux["f_l1"].clamp_min(1e-12)


# ------------------------------------------------------------------------------------------------ labels
def make_labels(D, mode="fault10"):
    """fault4: normal / inner / ball / outer.  fault10: normal + (inner, ball, outer) x (0.007, 0.014, 0.021 in).
    Returns y (n,) int64 and the class names."""
    cls, size = D["cls"], D["size"]
    if mode == "fault4":
        names = ["normal", "inner", "ball", "outer"]
        keys = [str(c) for c in cls]
    elif mode == "fault10":
        names = ["normal"] + [f"{c}_{s:.3f}" for c in ("inner", "ball", "outer") for s in (0.007, 0.014, 0.021)]
        keys = ["normal" if c == "normal" else f"{c}_{s:.3f}" for c, s in zip(cls, size)]
    else:
        raise ValueError(mode)
    missing = sorted(set(keys) - set(names))
    if missing:
        raise ValueError(f"labels {missing} are not in the class list")
    return np.array([names.index(k) for k in keys], dtype=np.int64), names


# ------------------------------------------------------------------------------------------------ frozen features
def vmd_modes(X, K, alpha=2000.0, n_iter=None, tol=1e-7, max_iter=499, return_omega=False):
    """Classic ADMM VMD on every window. n_iter=None: run to convergence; n_iter=N: exactly N iterations
    (this equals an untrained N-layer UVMD). Returns (n, K, T) float32 and the mean number of iterations
    (and, with return_omega, the final centre frequencies (n, K) as a third value)."""
    out, its, om = [], [], []
    for f in X:
        r = vmd(f, alpha, 0.0, K, True, 1, -1.0 if n_iter else tol, n_iter if n_iter else max_iter)
        out.append(r["u"])
        its.append(r["n_iter"])
        om.append(r["omega"])
    res = (np.stack(out).astype(np.float32), float(np.mean(its)))
    return res + (np.stack(om),) if return_omega else res


@torch.no_grad()
def uvmd_modes(uvmd: BatchedUVMD, X, device, batch=64):
    """Modes of a (frozen) BatchedUVMD for every window: (n, K, T) float32 on the CPU."""
    uvmd.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        out.append(uvmd(xb)[0].cpu())
    return torch.cat(out).numpy()


# ------------------------------------------------------------------------------------------------ evaluation
@torch.no_grad()
def predict(model, X, device, batch=128):
    """Logits (n, C) as a NumPy array and, for E2E models, the per-window relative reconstruction error."""
    model.eval()
    logits, rec = [], []
    for i in range(0, len(X), batch):
        xb = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        out = model(xb)
        if isinstance(model, E2E):
            out, r = out
            rec.append(r.cpu())
        logits.append(out.cpu())
    return torch.cat(logits).numpy(), (torch.cat(rec).numpy() if rec else None)


def score(y, logits, n_classes):
    pred = logits.argmax(1)
    ce = float(F.cross_entropy(torch.as_tensor(logits), torch.as_tensor(y)).item())
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    recall = np.divide(np.diag(cm), cm.sum(1), out=np.full(n_classes, np.nan), where=cm.sum(1) > 0)
    return {"acc": float((pred == y).mean()), "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "ce": ce, "confusion": cm, "recall": recall, "pred": pred}


# ------------------------------------------------------------------------------------------------ training
def fit(model, Xtr, ytr, Xva, yva, *, n_classes, device, seed, epochs=60, patience=15, batch=32,
        rec_weight=0.1, lr_cnn=1e-3, lr_alpha=2.0, lr_lam=0.05, wd_cnn=1e-4, log=print):
    """Train ``model`` (a Fault1DCNN on cached features, or an E2E model on raw windows).
    Returns a history dict; ``model`` is left at its best-validation weights.

    lr_alpha / lr_lam are the learning rates of alpha_raw and lambda (E2E only; 0 freezes that parameter).
    They are per-parameter-group because the two live on very different scales (alpha ~ 2e3, lambda ~ 1e1)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    e2e = isinstance(model, E2E)
    model.to(device)
    cnn = model.cnn if e2e else model
    groups = [{"params": list(cnn.parameters()), "lr": lr_cnn, "weight_decay": wd_cnn}]
    if e2e:
        for p, lr in ((model.uvmd.alpha_raw, lr_alpha), (model.uvmd.lam, lr_lam)):
            p.requires_grad_(lr > 0)
            if lr > 0:
                groups.append({"params": [p], "lr": lr, "weight_decay": 0.0})   # no decay: it would pull alpha, lambda to 0
    opt = torch.optim.Adam(groups)
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(ytr, device=device)
    hist = {"train_ce": [], "train_rec": [], "val_ce": [], "val_acc": [], "val_rec": [], "alpha": [], "epoch_s": []}
    best, best_state, bad = (-1.0, 0.0), None, 0
    for ep in range(epochs):
        t0 = time.perf_counter()
        model.train()
        ce_sum, rec_sum, n_seen = 0.0, 0.0, 0
        order = rng.permutation(len(Xtr_t))
        for i in range(0, len(order), batch):
            idx = torch.as_tensor(order[i:i + batch], device=device)
            if len(idx) < 2:                      # BatchNorm cannot train on a single window
                continue
            out = model(Xtr_t[idx])
            logits, rec = out if e2e else (out, None)
            ce = F.cross_entropy(logits, ytr_t[idx])
            loss = ce + rec_weight * rec.mean() if e2e else ce
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep + 1} (ce={ce.item()}). "
                                         f"Try a smaller --lr-alpha / --lr-lam.")
            opt.zero_grad()
            loss.backward()
            opt.step()
            ce_sum += ce.item() * len(idx)
            rec_sum += (rec.mean().item() * len(idx)) if e2e else 0.0
            n_seen += len(idx)
        va_logits, va_rec = predict(model, Xva, device)
        sc = score(yva, va_logits, n_classes)
        hist["train_ce"].append(ce_sum / n_seen)
        hist["train_rec"].append(rec_sum / n_seen if e2e else None)
        hist["val_ce"].append(sc["ce"])
        hist["val_acc"].append(sc["acc"])
        hist["val_rec"].append(float(va_rec.mean()) if va_rec is not None else None)
        hist["alpha"].append(model.uvmd.alpha().detach().cpu().tolist() if e2e else None)
        hist["epoch_s"].append(time.perf_counter() - t0)
        key = (sc["acc"], -sc["ce"])
        if key > best:
            best, bad = key, 0
            best_state = copy.deepcopy(model.state_dict())
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
        extra = f"  rec {hist['train_rec'][-1]:.4f}/{hist['val_rec'][-1]:.4f}" if e2e else ""
        log(f"    epoch {ep + 1:>3}  train CE {hist['train_ce'][-1]:.4f}  val CE {sc['ce']:.4f}  "
            f"val acc {sc['acc']:.3f}{extra}")
        if bad >= patience:
            log("    early stopping")
            break
    model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["val_acc"])
    hist["best_val_acc"], hist["best_val_ce"] = best[0], -best[1]
    hist["train_time_s"] = float(sum(hist["epoch_s"]))
    return hist
