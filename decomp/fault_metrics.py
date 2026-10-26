"""Bearing-fault metrics for a decomposition (no ground-truth modes needed).

Standard envelope analysis: a localised bearing defect makes periodic impacts that excite a structural
resonance; the envelope (|Hilbert|) of the band containing that resonance is periodic at the defect's
characteristic frequency (BPFI / BPFO / BSF). A good decomposition isolates that band in one mode.

Per window and per method:
  mode_kurtosis_k       kurtosis of every mode (impulsiveness)
  sel_mode              mode with the highest kurtosis (selection rule usable without labels)
  sel_centre_hz         centre frequency of that mode (omega * fs)
  snr_<fault>_db        harmonic SNR of the envelope spectrum of the selected mode at 1x..H x the fault
                        frequency: 20 log10(peak in +-tol / median of a +-band neighbourhood), averaged over
                        harmonics. Computed for inner, outer, ball.
  fault_snr_db          snr of the TRUE fault type (fault windows only)
  envelope_hit          1 if the true fault type has the largest snr among {inner, outer, ball}
  oracle_fault_snr_db   best true-fault snr over all modes (upper bound of what the modes contain)
The same quantities on the raw (undecomposed) window are the "raw signal" baseline.

``edge`` samples are dropped at both ends before kurtosis / envelope: VMD and UVMD share a boundary
artefact (the mirror extension leaves a spike in the last samples of some modes) that would otherwise win
the max-kurtosis selection on its own. The decompositions themselves are not modified.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

FAULTS = ("inner", "outer", "ball")


def envelope_spectrum(sig, fs):
    env = np.abs(hilbert(sig))
    env = env - env.mean()
    w = np.hanning(len(env))
    spec = np.abs(np.fft.rfft(env * w)) * 2 / w.sum()
    return np.fft.rfftfreq(len(env), 1 / fs), spec


def harmonic_snr_db(freqs, spec, f0, harmonics=3, band_hz=30.0):
    df = freqs[1] - freqs[0]
    vals = []
    for h in range(1, harmonics + 1):
        fc = h * f0
        tol = max(1.5 * df, 0.02 * fc)
        near = np.abs(freqs - fc) <= tol
        ring = (np.abs(freqs - fc) <= tol + band_hz) & ~near
        if not near.any() or not ring.any():
            continue
        noise = np.median(spec[ring])
        vals.append(20 * np.log10(spec[near].max() / max(noise, 1e-12)))
    return float(np.mean(vals)) if vals else float("nan")


def _signal_snrs(sig, fs, ff, harmonics):
    f, s = envelope_spectrum(sig, fs)
    return {c: harmonic_snr_db(f, s, ff[c], harmonics) for c in FAULTS}


def fault_eval(u, x, cls, ff, fs, omega=None, harmonics=3, edge=16):
    """u: (K,T) modes; x: raw window; cls: true class; ff: fault_freqs(rpm) dict."""
    T = min(u.shape[1], len(x))
    u = np.asarray(u)[:, edge:T - edge]
    kurt = kurtosis(u, axis=1, fisher=False)
    sel = int(np.nanargmax(kurt))
    m = {f"mode_kurtosis_{k}": float(v) for k, v in enumerate(kurt)}
    m["sel_mode"] = sel
    m["sel_kurtosis"] = float(kurt[sel])
    if omega is not None:
        m["sel_centre_hz"] = float(omega[sel] * fs)
    snr = _signal_snrs(u[sel], fs, ff, harmonics)
    for c in FAULTS:
        m[f"snr_{c}_db"] = snr[c]
    if cls in FAULTS:
        m["fault_snr_db"] = snr[cls]
        m["envelope_hit"] = float(max(FAULTS, key=lambda c: snr[c]) == cls)
        m["oracle_fault_snr_db"] = float(np.nanmax([_signal_snrs(uk, fs, ff, harmonics)[cls] for uk in u]))
    else:
        m["normal_max_fault_snr_db"] = float(np.nanmax(list(snr.values())))
    return m


def raw_eval(x, cls, ff, fs, harmonics=3, edge=16):
    x = np.asarray(x)[edge:len(x) - edge]
    snr = _signal_snrs(x, fs, ff, harmonics)
    m = {f"snr_{c}_db": snr[c] for c in FAULTS}
    m["sel_kurtosis"] = float(kurtosis(x, fisher=False))
    if cls in FAULTS:
        m["fault_snr_db"] = snr[cls]
        m["envelope_hit"] = float(max(FAULTS, key=lambda c: snr[c]) == cls)
    else:
        m["normal_max_fault_snr_db"] = float(np.nanmax(list(snr.values())))
    return m
