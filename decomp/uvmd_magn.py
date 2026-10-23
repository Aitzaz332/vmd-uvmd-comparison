"""UVMD exactly as published in the MAGN repository (third_party/MAGN/uvmd.py).

The upstream file is a script: importing it parses CLI args, loads ``his.npz`` and starts training.
To use the *unchanged* model we parse that file, keep only its ``import`` statements and its three
class definitions (``VMD_innerloop``, ``VMD``, ``Unfolding``) and execute those. No line of the model
is retyped here, so the benchmark always runs the authors' code byte-for-byte.

``train_uvmd`` reproduces the training procedure of the upstream script (Adam, lr=0.1, batch_size=1,
shuffle, 100 epochs, early stopping patience 10 on the validation loss, best weights saved on
improvement). One upstream detail is kept as an option: after the loop, the upstream script saves the
*last* weights over the best ones and reloads them, so the final model is the last-epoch model.
``restore_best=False`` (default) reproduces that; ``restore_best=True`` uses the best-val weights.
"""
from __future__ import annotations

import ast
import copy
import time
from pathlib import Path

import numpy as np

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "MAGN" / "uvmd.py"
MODEL_CLASSES = ("VMD_innerloop", "VMD", "Unfolding")

_ns_cache: dict | None = None


def load_magn_namespace() -> dict:
    """Execute only the imports and model classes of the upstream uvmd.py; return its namespace."""
    global _ns_cache
    if _ns_cache is not None:
        return _ns_cache
    src = UPSTREAM.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(UPSTREAM))
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom))
            or (isinstance(n, ast.ClassDef) and n.name in MODEL_CLASSES)]
    found = {n.name for n in keep if isinstance(n, ast.ClassDef)}
    missing = set(MODEL_CLASSES) - found
    if missing:
        raise RuntimeError(f"upstream uvmd.py no longer defines {missing}")
    module = ast.Module(body=keep, type_ignores=[])
    ns: dict = {"__name__": "magn_uvmd"}
    exec(compile(module, str(UPSTREAM), "exec"), ns)  # noqa: S102  (trusted, hash-checked file)
    _ns_cache = ns
    return ns


def Unfolding(*args, **kwargs):
    """The upstream ``Unfolding`` class (constructor passthrough)."""
    return load_magn_namespace()["Unfolding"](*args, **kwargs)


def make_model(T: int, K: int, layers: int, device="cpu", alpha: float = 2000.0):
    """Build the upstream model for signals of (even) length T, as the upstream script does:
    ``Unfolding(device, K=K, layers=layers, freqs_len=T*2)``."""
    if T % 2:
        raise ValueError("use an even signal length (upstream drops the last sample otherwise)")
    import torch
    dev = torch.device(device)
    return Unfolding(dev, K=K, layers=layers, freqs_len=2 * T, alpha=alpha).to(dev)


def count_params(model) -> dict:
    """Upstream prints numel of trainable params (complex entries count once). Also report real DOF."""
    n_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_real = sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters() if p.requires_grad)
    return {"numel": int(n_numel), "real_dof": int(n_real)}


def _epoch(model, X, optimizer, device, train: bool, rng: np.random.Generator | None):
    import torch
    idx = np.arange(len(X))
    if train:
        rng.shuffle(idx)  # DataLoader(shuffle=True, batch_size=1)
        model.train()
    else:
        model.eval()
    total = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i in idx:
            inputs = torch.as_tensor(X[i], dtype=torch.float32, device=device)
            outputs, omega, loss = model(inputs)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
    return total / len(X)


def train_uvmd(model, X_train, X_val, epochs=100, lr=0.1, patience=10, device="cpu",
               seed=0, restore_best=False, log=print):
    """Upstream training loop (see module docstring). Returns history dict."""
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, counter, best_state = float("inf"), 0, None
    hist = {"train_loss": [], "val_loss": [], "epoch_time_s": []}
    for epoch in range(epochs):
        t0 = time.perf_counter()
        tr = _epoch(model, X_train, optimizer, device, True, rng)
        hist["epoch_time_s"].append(time.perf_counter() - t0)
        va = _epoch(model, X_val, None, device, False, None)
        hist["train_loss"].append(tr)
        hist["val_loss"].append(va)
        log(f"  epoch {epoch + 1:>3}/{epochs}  train {tr:.4f}  val {va:.4f}")
        if va < best_val:
            best_val, counter = va, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            counter += 1
        if counter >= patience:
            log("  early stopping triggered")
            break
    hist["epochs_run"] = len(hist["train_loss"])
    hist["best_val_loss"] = best_val
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)
    return hist


def decompose(model, x, device="cpu"):
    """Run the upstream forward on ONE 1-D signal. Returns modes (K,T) and final centre freqs (K,)."""
    import torch
    with torch.no_grad():
        u, omega_plus, _ = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return u.cpu().numpy().astype(np.float64), omega_plus[-1].cpu().numpy().astype(np.float64)


def export_params(model) -> dict:
    """Learned parameters as NumPy arrays: alpha = softplus(Alpha) (K,), lambda rows actually used."""
    import torch.nn.functional as F
    alpha = F.softplus(model.Alpha.detach()).cpu().numpy().astype(np.float64)
    lam = model.lambda_hat.detach().cpu().numpy().astype(np.complex128)
    return {"alpha": alpha, "lambda": lam[: model.layers - 1], "omega0": model.omega_plus[0].cpu().numpy()}
