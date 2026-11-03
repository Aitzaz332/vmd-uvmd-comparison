# Mode comparison: end-to-end UVMD vs VMD

300 unseen test windows (split 'load'), T=2048, K=4, N=5 layers, alpha=500, e2e seeds [0, 1, 2]. No recon-trained UVMD was given (`--early-params`). For the end-to-end UVMD, numbers are means over seeds.

## Learned parameters

| model                  |   alpha_0 |   alpha_1 |   alpha_2 |   alpha_3 |   lambda_norm |
|:-----------------------|----------:|----------:|----------:|----------:|--------------:|
| UVMD end-to-end seed 0 |   509.073 |   534.255 |   368.889 |   229.381 |       125.761 |
| UVMD end-to-end seed 1 |   571.108 |   576.944 |   431.758 |   347.668 |        96.799 |
| UVMD end-to-end seed 2 |   555     |   569.508 |   361.171 |   293.002 |       121.447 |
| start (untrained UVMD) |   500     |   500     |   500     |   500     |         0     |

`alpha_k` is the bandwidth penalty of mode k: the filter that carves mode k out of the spectrum is 1 / (1 + alpha_k (f - omega_k)^2), so a larger alpha_k gives a narrower mode and alpha_k near 0 lets everything through. Every model starts at alpha = 500 and lambda = 0.

## Centre frequencies of the final layer (mean over test windows) and distance from the start grid

| model                  | family          |   omega_0_Hz |   omega_1_Hz |   omega_2_Hz |   omega_3_Hz |   mean_shift_grid |
|:-----------------------|:----------------|-------------:|-------------:|-------------:|-------------:|------------------:|
| VMD converged          | VMD converged   |            0 |      1563.57 |      3021.33 |      3517.41 |             0.426 |
| UVMD end-to-end seed 0 | UVMD end-to-end |            0 |      1573.4  |      3193.71 |      3609.24 |             0.35  |
| UVMD end-to-end seed 1 | UVMD end-to-end |            0 |      1571.06 |      3194.32 |      3664.59 |             0.338 |
| UVMD end-to-end seed 2 | UVMD end-to-end |            0 |      1560.97 |      3194.4  |      3665.69 |             0.336 |

## Similarity of the modes (mode k vs mode k; Hungarian-matched |correlation|)

| A               | B             |   corr_mode_0 |   corr_mode_1 |   corr_mode_2 |   corr_mode_3 |   aligned_mean |   matched_mean_abs |   matched_spread |
|:----------------|:--------------|--------------:|--------------:|--------------:|--------------:|---------------:|-------------------:|-----------------:|
| UVMD end-to-end | VMD converged |         0.973 |         0.894 |         0.807 |         0.816 |          0.872 |              0.882 |            0.002 |

## Energy share per mode

|        |   VMD converged |   UVMD end-to-end |
|:-------|----------------:|------------------:|
| mode 0 |           0.021 |             0.036 |
| mode 1 |           0.207 |             0.161 |
| mode 2 |           0.43  |             0.613 |
| mode 3 |           0.342 |             0.19  |

## Mode quality and fault content (mean over test windows; end-to-end: mean over seeds)

| family          |   recon_rel_err |   energy_ratio |   ortho_index |   fault_snr_db |   envelope_hit |   oracle_fault_snr_db |
|:----------------|----------------:|---------------:|--------------:|---------------:|---------------:|----------------------:|
| VMD converged   |          0.1669 |         0.7546 |        0.0501 |         7.4096 |         0.6963 |               10.3325 |
| UVMD end-to-end |          0.1822 |         0.6209 |        0.1002 |         8.6752 |         0.7123 |               10.5283 |

Standard deviation across the end-to-end seeds (of the per-seed means):

| family          |   recon_rel_err |   energy_ratio |   ortho_index |   fault_snr_db |   envelope_hit |   oracle_fault_snr_db |
|:----------------|----------------:|---------------:|--------------:|---------------:|---------------:|----------------------:|
| UVMD end-to-end |          0.0113 |         0.0041 |        0.0056 |         0.0047 |         0.0077 |                0.0123 |

## Fault content by class (normal windows have no fault frequency, hence no entry)

| method          | cls   |   fault_snr_db |   envelope_hit |   oracle_fault_snr_db |
|:----------------|:------|---------------:|---------------:|----------------------:|
| VMD converged   | inner |          8.657 |          0.944 |                11.646 |
| VMD converged   | ball  |          3.433 |          0.356 |                 6.262 |
| VMD converged   | outer |         10.139 |          0.789 |                13.09  |
| UVMD end-to-end | inner |         11.251 |          1     |                12.275 |
| UVMD end-to-end | ball  |          3.844 |          0.341 |                 6.401 |
| UVMD end-to-end | outer |         10.93  |          0.796 |                12.909 |
