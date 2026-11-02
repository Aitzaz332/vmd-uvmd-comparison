# CWRU benchmark - VMD vs UVMD

40 recordings, 1199 windows x 2048 samples at 12 kHz, standardised=True, split 'load' (test = 300 windows), K=4, alpha=500.0, tau=0.0.

## Overall (test windows)

| method | recon_rmse | recon_rel_err | recon_snr_db | spec_l1 | energy_ratio | ortho_index | fault_snr_db | envelope_hit | oracle_fault_snr_db | sel_kurtosis | normal_max_fault_snr_db |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw signal | nan | nan | nan | nan | nan | nan | 8.896 | 0.6963 | nan | 7.987 | 6.374 |
| VMD (converged) | 0.1669 | 0.1669 | 15.92 | 1.177e+04 | 0.7546 | 0.05008 | 7.41 | 0.6963 | 10.33 | 8.005 | 5.951 |
| VMD-5 | 0.2022 | 0.2022 | 14.14 | 1.314e+04 | 0.6211 | 0.09032 | 8.59 | 0.7185 | 10.39 | 8.385 | 5.251 |
| UVMD-5 untrained | 0.2022 | 0.2022 | 14.14 | 1.314e+04 | 0.6211 | 0.09032 | 8.59 | 0.7185 | 10.39 | 8.385 | 5.251 |
| UVMD-5 | 0.0004001 | 0.0004001 | 78.72 | 9.558 | 0.6408 | 0.18 | 8.573 | 0.7148 | 10.55 | 8.242 | 4.897 |

## Fault metrics by class

|                                |   fault_snr_db |   envelope_hit |   oracle_fault_snr_db |   sel_kurtosis |   recon_rel_err |
|:-------------------------------|---------------:|---------------:|----------------------:|---------------:|----------------:|
| ('raw signal', 'ball')         |           3.42 |          0.3   |                nan    |           4.7  |      nan        |
| ('raw signal', 'inner')        |          11.7  |          1     |                nan    |          10.4  |      nan        |
| ('raw signal', 'normal')       |         nan    |        nan     |                nan    |           3.02 |      nan        |
| ('raw signal', 'outer')        |          11.6  |          0.789 |                nan    |          10.6  |      nan        |
| ('VMD (converged)', 'ball')    |           3.43 |          0.356 |                  6.26 |           5.68 |        0.154    |
| ('VMD (converged)', 'inner')   |           8.66 |          0.944 |                 11.6  |          10.4  |        0.137    |
| ('VMD (converged)', 'normal')  |         nan    |        nan     |                nan    |           3.28 |        0.27     |
| ('VMD (converged)', 'outer')   |          10.1  |          0.789 |                 13.1  |           9.49 |        0.176    |
| ('VMD-5', 'ball')              |           3.86 |          0.344 |                  6.36 |           6.64 |        0.186    |
| ('VMD-5', 'inner')             |          11    |          0.989 |                 12.1  |          10.6  |        0.204    |
| ('VMD-5', 'normal')            |         nan    |        nan     |                nan    |           3.36 |        0.268    |
| ('VMD-5', 'outer')             |          10.9  |          0.822 |                 12.7  |           9.57 |        0.194    |
| ('UVMD-5 untrained', 'ball')   |           3.86 |          0.344 |                  6.36 |           6.64 |        0.186    |
| ('UVMD-5 untrained', 'inner')  |          11    |          0.989 |                 12.1  |          10.6  |        0.204    |
| ('UVMD-5 untrained', 'normal') |         nan    |        nan     |                nan    |           3.36 |        0.268    |
| ('UVMD-5 untrained', 'outer')  |          10.9  |          0.822 |                 12.7  |           9.57 |        0.194    |
| ('UVMD-5', 'ball')             |           3.66 |          0.378 |                  6.19 |           5.19 |        8.34e-05 |
| ('UVMD-5', 'inner')            |          10.5  |          0.978 |                 12.9  |          11.2  |        8.32e-05 |
| ('UVMD-5', 'normal')           |         nan    |        nan     |                nan    |           3.42 |        0.00324  |
| ('UVMD-5', 'outer')            |          11.6  |          0.789 |                 12.6  |           9.97 |        8.78e-05 |

## Cost (per window)

| method           |   iterations_or_layers |   trainable_params |   flops_per_window |   median_ms_per_window |   train_time_s |   train_flops |
|:-----------------|-----------------------:|-------------------:|-------------------:|-----------------------:|---------------:|--------------:|
| VMD (converged)  |                   98.1 |                  0 |          4.302e+07 |                 27.21  |              0 |     0         |
| VMD-5            |                    5   |                  0 |          3.359e+06 |                  2.226 |              0 |     0         |
| UVMD-5 untrained |                    5   |              49156 |          2.785e+06 |                  5.526 |              0 |     0         |
| UVMD-5           |                    5   |              49156 |          2.785e+06 |                  5.564 |           1007 |     2.402e+11 |

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'numpy': '2.5.4', 'torch': '2.14.1+cu130', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060'}


## Log
```
CWRU: 40 files, 1199 windows of 2048 samples (171 ms) | split 'load': train/val/test = 599/300/300 | K=4

[VMD converged]
  mean iters 98.1 (99% converged), rel err 0.1669, fault SNR 7.41 dB, hit 0.70, 27.2 ms

[VMD-n sweep]
  n=  5 (300 win): rel err 0.2022  fault SNR   8.59 dB  hit 0.72  2.23 ms

[UVMD N=5] upstream Unfolding, K=4, device cuda:0
  UVMD-5 untrained: rel err 0.2022  fault SNR 8.59 dB  hit 0.72  5.53 ms  alpha [500.0, 500.0, 500.0, 500.0]
  epoch   1/100  train 12160.9509  val 12104.2199
  epoch   2/100  train 11467.5824  val 11351.4388
  epoch   3/100  train 10715.6166  val 10505.7314
  epoch   4/100  train 9861.7810  val 9524.4250
  epoch   5/100  train 8830.5486  val 8294.4974
  epoch   6/100  train 7403.6281  val 6438.8914
  epoch   7/100  train 3165.3933  val 19.2530
  epoch   8/100  train 16.5522  val 15.1281
  epoch   9/100  train 14.6216  val 14.0836
  epoch  10/100  train 13.7040  val 13.4286
  epoch  11/100  train 13.3015  val 13.1465
  epoch  12/100  train 12.7425  val 12.5568
  epoch  13/100  train 12.1969  val 12.0176
  epoch  14/100  train 11.7503  val 11.5700
  epoch  15/100  train 11.4109  val 10.9702
  epoch  16/100  train 11.0942  val 11.0908
  epoch  17/100  train 10.7840  val 10.7572
  epoch  18/100  train 10.4197  val 10.2960
  epoch  19/100  train 10.3263  val 10.2116
  epoch  20/100  train 10.1474  val 9.8826
  epoch  21/100  train 9.9629  val 9.8958
  epoch  22/100  train 9.7437  val 9.6025
  epoch  23/100  train 9.7155  val 9.6390
  epoch  24/100  train 9.6001  val 9.5974
  epoch  25/100  train 9.5477  val 9.4295
  epoch  26/100  train 9.5421  val 9.6327
  epoch  27/100  train 9.4963  val 9.4288
  epoch  28/100  train 9.4933  val 9.3760
  epoch  29/100  train 9.3863  val 9.4849
  epoch  30/100  train 9.3309  val 9.4598
  epoch  31/100  train 9.4209  val 9.5727
  epoch  32/100  train 9.4024  val 9.6057
  epoch  33/100  train 9.4336  val 9.5259
  epoch  34/100  train 9.4428  val 9.4964
  epoch  35/100  train 9.4739  val 9.5531
  epoch  36/100  train 9.4727  val 9.6766
  epoch  37/100  train 9.5067  val 9.4715
  epoch  38/100  train 9.3834  val 9.3747
  epoch  39/100  train 9.3529  val 9.4937
  epoch  40/100  train 9.3828  val 9.4424
  epoch  41/100  train 9.5292  val 9.5716
  epoch  42/100  train 9.5104  val 9.5917
  epoch  43/100  train 9.5544  val 9.4709
  epoch  44/100  train 9.5210  val 9.4955
  epoch  45/100  train 9.5363  val 9.4693
  epoch  46/100  train 9.4852  val 9.5109
  epoch  47/100  train 9.4172  val 9.4872
  epoch  48/100  train 9.4382  val 9.5578
  early stopping triggered
  training: 48 epochs, 1007.2 s
  UVMD-5: rel err 0.0004  fault SNR 8.57 dB  hit 0.71  5.56 ms  alpha [85.0, 233.0, 268.0, 0.0]
```
