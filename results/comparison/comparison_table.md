# VMD vs UVMD across the datasets

Mean over test signals / windows / nodes, with the standard deviation where the run stored it. Bold = best of the rows of that dataset. VMD-N is also UVMD-N before training. A dash means the metric does not exist for that dataset.

| Dataset | Method | Rel. recon. error ↓ | Orthogonality index ↓ | Energy ratio | Mode SNR, grouped (dB) ↑ | Centre-freq. error ↓ | Best-mode fault SNR (dB) ↑ | Fault hit rate ↑ | Mode overlap ↓ | Bandwidth (cycles/day) ↓ |
|---|---|---|---|---|---|---|---|---|---|---|
| Synthetic (K=13) | VMD (converged) | 0.0294 ± 0.0041 | **0.0431 ± 0.021** | 0.894 ± 0.046 | **20.7 ± 1.2** | 0.00106 ± 0.00025 | - | - | - | - |
| Synthetic (K=13) | VMD-1 | 0.0499 ± 0.012 | 0.129 ± 0.019 | 0.672 ± 0.048 | 12.4 ± 0.76 | 0.0012 ± 0.00023 | - | - | - | - |
| Synthetic (K=13) | UVMD-1 | **0.0066 ± 0.0014** | 0.173 ± 0.025 | 0.645 ± 0.051 | 8.54 ± 0.55 | 0.00252 ± 0.00077 | - | - | - | - |
| Synthetic (K=13) | VMD-2 | 0.0379 ± 0.0059 | 0.0824 ± 0.017 | 0.788 ± 0.038 | 18.8 ± 0.96 | **0.0009 ± 0.00021** | - | - | - | - |
| Synthetic (K=13) | UVMD-2 | 0.0142 ± 0.0023 | 0.089 ± 0.029 | 0.928 ± 0.055 | 7.33 ± 0.97 | 0.00207 ± 0.00054 | - | - | - | - |
| CWRU (K=13) | VMD (converged) | 0.076 | **0.0885** | 0.76 | - | - | 10.8 | **0.685** | - | - |
| CWRU (K=13) | VMD-1 | 0.0662 | 0.259 | 0.38 | - | - | **11.5** | 0.656 | - | - |
| CWRU (K=13) | UVMD-1 | 0.0004 | 0.273 | 0.455 | - | - | 11.3 | 0.681 | - | - |
| CWRU (K=13) | VMD-2 | 0.107 | 0.179 | 0.495 | - | - | 11.2 | 0.685 | - | - |
| CWRU (K=13) | UVMD-2 | **0.0004** | 0.454 | 1.02 | - | - | 10.3 | 0.667 | - | - |
| LargeST traffic (K=13) | VMD (converged) | 0.0303 ± 0.013 | **0.0137 ± 0.0097** | 0.964 ± 0.027 | - | - | - | - | **0.00404 ± 0.00083** | **0.102 ± 0.044** |
| LargeST traffic (K=13) | VMD-1 | 0.0729 ± 0.02 | 0.0691 ± 0.029 | 0.747 ± 0.1 | - | - | - | - | 0.0358 ± 0.012 | 0.486 ± 0.15 |
| LargeST traffic (K=13) | UVMD-1 | **0.0205 ± 0.0042** | 0.0709 ± 0.03 | 0.835 ± 0.072 | - | - | - | - | 0.0579 ± 0.016 | 0.524 ± 0.15 |
| LargeST traffic (K=13) | VMD-2 | 0.0838 ± 0.026 | 0.0853 ± 0.035 | 0.698 ± 0.12 | - | - | - | - | 0.0163 ± 0.0071 | 0.447 ± 0.13 |
| LargeST traffic (K=13) | UVMD-2 | 0.0372 ± 0.011 | 0.0806 ± 0.034 | 0.786 ± 0.092 | - | - | - | - | 0.0289 ± 0.0092 | 0.499 ± 0.13 |

Columns shared by all datasets need no ground truth. The last columns are each dataset's own separation measure: mode SNR against the known true modes (synthetic), the fault-line SNR in the best mode (CWRU), spectral overlap and bandwidth of the modes (traffic, no ground truth exists).

## Learned bandwidth penalty alpha_k of the trained UVMD (initial value 2000)

| dataset | model | min | median | max | alphas below 5 |
|---|---|---|---|---|---|
| Synthetic (K=13) | UVMD-1 | 580 | 674 | 821 | 0 |
| Synthetic (K=13) | UVMD-2 | 571 | 663 | 1844 | 0 |
| CWRU (K=13) | UVMD-1 | 0 | 226 | 484 | 1 |
| CWRU (K=13) | UVMD-2 | 0 | 276 | 1434 | 1 |
| LargeST traffic (K=13) | UVMD-1 | 1028 | 1080 | 1173 | 0 |
| LargeST traffic (K=13) | UVMD-2 | 1012 | 1042 | 1130 | 0 |

An alpha near 0 turns a mode into a pass-everything residual: the reconstruction error drops but that mode no longer separates anything.

