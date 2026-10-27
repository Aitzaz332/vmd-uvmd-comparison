# Centre-frequency movement across the datasets

Distance of the final centre frequencies ω_k from the uniform start grid, in grid spacings (0.5/K cycles/sample), mean over the test signals and modes 1..K-1 (mode 0 is the DC mode and does not move). A mode counts as moved when its mean shift exceeds 0.25 spacing. 'Movement ÷ converged VMD' = mean shift divided by that of converged VMD, which is computed on fewer signals. Moving more is not the same as moving to the right place: on the synthetic data compare with the true-frequency error in results/comparison. VMD-N is also UVMD-N before training.

| Dataset | Method | Mean shift (grid spacings) | Largest single-mode shift | Modes moved > 0.25 spacing | Movement ÷ converged VMD | Signals |
|---|---|---|---|---|---|---|
| Synthetic (K=13) | VMD (converged) | 0.461 | 1.085 | 9 of 12 | 1.00× | 10 |
| Synthetic (K=13) | VMD-1 | 0.373 | 0.992 | 4 of 12 | 0.81× | 30 |
| Synthetic (K=13) | UVMD-1 | 0.686 | 1.916 | 10 of 12 | 1.49× | 30 |
| Synthetic (K=13) | VMD-2 | 0.402 | 1.240 | 4 of 12 | 0.87× | 30 |
| Synthetic (K=13) | UVMD-2 | 0.460 | 0.986 | 9 of 12 | 1.00× | 30 |
| CWRU (K=13) | VMD (converged) | 0.621 | 1.197 | 9 of 12 | 1.00× | 10 |
| CWRU (K=13) | VMD-1 | 0.368 | 1.177 | 7 of 12 | 0.59× | 30 |
| CWRU (K=13) | UVMD-1 | 1.859 | 3.392 | 11 of 12 | 2.99× | 30 |
| CWRU (K=13) | VMD-2 | 0.376 | 1.312 | 7 of 12 | 0.61× | 30 |
| CWRU (K=13) | UVMD-2 | 2.045 | 8.085 | 11 of 12 | 3.29× | 30 |
| LargeST traffic (K=13) | VMD (converged) | 1.611 | 2.771 | 12 of 12 | 1.00× | 10 |
| LargeST traffic (K=13) | VMD-1 | 0.186 | 0.279 | 1 of 12 | 0.12× | 30 |
| LargeST traffic (K=13) | UVMD-1 | 0.381 | 0.512 | 10 of 12 | 0.24× | 30 |
| LargeST traffic (K=13) | VMD-2 | 0.190 | 0.335 | 3 of 12 | 0.12× | 30 |
| LargeST traffic (K=13) | UVMD-2 | 0.283 | 0.443 | 8 of 12 | 0.18× | 30 |

ω_k is not a learned parameter of UVMD: each layer recomputes it from the mode spectrum, so UVMD can move it only through the learned alpha_k and lambda.

