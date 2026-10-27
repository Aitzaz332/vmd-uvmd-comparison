# CWRU benchmark - VMD vs UVMD

40 recordings, 1199 windows x 2048 samples at 12 kHz, standardised=True, split 'load' (test = 300 windows), K=13, alpha=2000.0, tau=0.0.

## Overall (test windows)

| method | recon_rmse | recon_rel_err | recon_snr_db | spec_l1 | energy_ratio | ortho_index | fault_snr_db | envelope_hit | oracle_fault_snr_db | sel_kurtosis | normal_max_fault_snr_db |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw signal | nan | nan | nan | nan | nan | nan | 8.896 | 0.6963 | nan | 7.987 | 6.374 |
| VMD (converged) | 0.07596 | 0.07596 | 22.74 | 5486 | 0.7601 | 0.08849 | 6.299 | 0.6852 | 10.83 | 7.59 | 5.141 |
| VMD-1 | 0.06616 | 0.06616 | 23.64 | 3443 | 0.3802 | 0.2593 | 6.552 | 0.6556 | 11.49 | 10.26 | 4.249 |
| VMD-2 | 0.1068 | 0.1068 | 19.48 | 7387 | 0.495 | 0.1788 | 6.867 | 0.6852 | 11.24 | 9.807 | 4.543 |
| UVMD-1 untrained | 0.06616 | 0.06616 | 23.64 | 3443 | 0.3802 | 0.2593 | 6.552 | 0.6556 | 11.49 | 10.26 | 4.249 |
| UVMD-1 | 0.0004003 | 0.0004003 | 78.7 | 9.658 | 0.4546 | 0.2727 | 7.26 | 0.6815 | 11.3 | 11.08 | 5.085 |
| UVMD-2 untrained | 0.1068 | 0.1068 | 19.48 | 7387 | 0.495 | 0.1788 | 6.867 | 0.6852 | 11.24 | 9.807 | 4.543 |
| UVMD-2 | 0.0004 | 0.0004 | 78.72 | 9.452 | 1.025 | 0.4543 | 6.946 | 0.6667 | 10.32 | 9.783 | 4.61 |

## Fault metrics by class

|                                |   fault_snr_db |   envelope_hit |   oracle_fault_snr_db |   sel_kurtosis |   recon_rel_err |
|:-------------------------------|---------------:|---------------:|----------------------:|---------------:|----------------:|
| ('raw signal', 'ball')         |           3.42 |          0.3   |                nan    |           4.7  |      nan        |
| ('raw signal', 'inner')        |          11.7  |          1     |                nan    |          10.4  |      nan        |
| ('raw signal', 'normal')       |         nan    |        nan     |                nan    |           3.02 |      nan        |
| ('raw signal', 'outer')        |          11.6  |          0.789 |                nan    |          10.6  |      nan        |
| ('VMD (converged)', 'ball')    |           3.14 |          0.4   |                  7.22 |           6.07 |        0.0756   |
| ('VMD (converged)', 'inner')   |           8.11 |          0.911 |                 12.7  |           7.59 |        0.0636   |
| ('VMD (converged)', 'normal')  |         nan    |        nan     |                nan    |           3.46 |        0.127    |
| ('VMD (converged)', 'outer')   |           7.65 |          0.744 |                 12.6  |          10.5  |        0.0718   |
| ('VMD-1', 'ball')              |           3.03 |          0.3   |                  7.33 |           8.47 |        0.0599   |
| ('VMD-1', 'inner')             |           9.53 |          0.989 |                 13.6  |          11.2  |        0.0657   |
| ('VMD-1', 'normal')            |         nan    |        nan     |                nan    |           3.6  |        0.0675   |
| ('VMD-1', 'outer')             |           7.1  |          0.678 |                 13.5  |          13.3  |        0.0725   |
| ('VMD-2', 'ball')              |           3.17 |          0.389 |                  7.27 |           8.14 |        0.104    |
| ('VMD-2', 'inner')             |           9.29 |          0.933 |                 13    |          10.4  |        0.103    |
| ('VMD-2', 'normal')            |         nan    |        nan     |                nan    |           3.64 |        0.134    |
| ('VMD-2', 'outer')             |           8.15 |          0.733 |                 13.4  |          12.9  |        0.104    |
| ('UVMD-1 untrained', 'ball')   |           3.03 |          0.3   |                  7.33 |           8.47 |        0.0599   |
| ('UVMD-1 untrained', 'inner')  |           9.53 |          0.989 |                 13.6  |          11.2  |        0.0657   |
| ('UVMD-1 untrained', 'normal') |         nan    |        nan     |                nan    |           3.6  |        0.0675   |
| ('UVMD-1 untrained', 'outer')  |           7.1  |          0.678 |                 13.5  |          13.3  |        0.0725   |
| ('UVMD-1', 'ball')             |           2.78 |          0.311 |                  6.33 |           7.23 |        8.36e-05 |
| ('UVMD-1', 'inner')            |          11.1  |          0.989 |                 13.8  |          13.3  |        8.35e-05 |
| ('UVMD-1', 'normal')           |         nan    |        nan     |                nan    |           3.23 |        0.00324  |
| ('UVMD-1', 'outer')            |           7.95 |          0.744 |                 13.8  |          15.4  |        8.8e-05  |
| ('UVMD-2 untrained', 'ball')   |           3.17 |          0.389 |                  7.27 |           8.14 |        0.104    |
| ('UVMD-2 untrained', 'inner')  |           9.29 |          0.933 |                 13    |          10.4  |        0.103    |
| ('UVMD-2 untrained', 'normal') |         nan    |        nan     |                nan    |           3.64 |        0.134    |
| ('UVMD-2 untrained', 'outer')  |           8.15 |          0.733 |                 13.4  |          12.9  |        0.104    |
| ('UVMD-2', 'ball')             |           2.55 |          0.289 |                  6.37 |           6.25 |        8.34e-05 |
| ('UVMD-2', 'inner')            |          10.6  |          0.967 |                 12.4  |          11.2  |        8.32e-05 |
| ('UVMD-2', 'normal')           |         nan    |        nan     |                nan    |           3.22 |        0.00324  |
| ('UVMD-2', 'outer')            |           7.69 |          0.744 |                 12.2  |          14.1  |        8.77e-05 |

## Cost (per window)

| method           |   iterations_or_layers |   trainable_params |   flops_per_window |   median_ms_per_window |   train_time_s |   train_flops |
|:-----------------|-----------------------:|-------------------:|-------------------:|-----------------------:|---------------:|--------------:|
| VMD (converged)  |                  366.6 |                  0 |          5.11e+08  |                420.3   |            0   |     0         |
| VMD-1            |                    1   |                  0 |          4.825e+06 |                  1.705 |            0   |     0         |
| VMD-2            |                    2   |                  0 |          6.21e+06  |                  2.643 |            0   |     0         |
| VMD-3            |                    3   |                  0 |          7.594e+06 |                  3.577 |            0   |     0         |
| VMD-5            |                    5   |                  0 |          1.036e+07 |                  5.486 |            0   |     0         |
| VMD-10           |                   10   |                  0 |          1.729e+07 |                 10.21  |            0   |     0         |
| VMD-20           |                   20   |                  0 |          3.113e+07 |                 19.51  |            0   |     0         |
| VMD-50           |                   50   |                  0 |          7.266e+07 |                 47.5   |            0   |     0         |
| VMD-100          |                  100   |                  0 |          1.419e+08 |                 94.6   |            0   |     0         |
| VMD-200          |                  200   |                  0 |          2.803e+08 |                188.9   |            0   |     0         |
| UVMD-1 untrained |                    1   |              16397 |          4.452e+06 |                  4.598 |            0   |     0         |
| UVMD-1           |                    1   |              16397 |          4.452e+06 |                  4.534 |          425.8 |     4e+11     |
| UVMD-2 untrained |                    2   |              24589 |          5.464e+06 |                  7.541 |            0   |     0         |
| UVMD-2           |                    2   |              24589 |          5.464e+06 |                  7.505 |         1149   |     6.873e+11 |

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'numpy': '2.1.3', 'torch': '2.7.0', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060'}


## Log
```
CWRU: 40 files, 1199 windows of 2048 samples (171 ms) | split 'load': train/val/test = 599/300/300 | K=13

[VMD converged]
  mean iters 366.6 (59% converged), rel err 0.0760, fault SNR 6.30 dB, hit 0.69, 420.3 ms

[VMD-n sweep]
  n=  1 (300 win): rel err 0.0662  fault SNR   6.55 dB  hit 0.66  1.71 ms
  n=  2 (300 win): rel err 0.1068  fault SNR   6.87 dB  hit 0.69  2.64 ms
  n=  3 (100 win): rel err 0.1007  fault SNR   6.15 dB  hit 0.55  3.58 ms
  n=  5 (100 win): rel err 0.0913  fault SNR   6.08 dB  hit 0.53  5.49 ms
  n= 10 (100 win): rel err 0.0856  fault SNR   6.04 dB  hit 0.59  10.21 ms
  n= 20 (100 win): rel err 0.0822  fault SNR   5.77 dB  hit 0.54  19.51 ms
  n= 50 (100 win): rel err 0.0798  fault SNR   5.69 dB  hit 0.56  47.50 ms
  n=100 (100 win): rel err 0.0793  fault SNR   5.83 dB  hit 0.55  94.60 ms
  n=200 (100 win): rel err 0.0791  fault SNR   6.13 dB  hit 0.57  188.87 ms

[UVMD N=1] upstream Unfolding, K=13, device cuda:0
  UVMD-1 untrained: rel err 0.0662  fault SNR 6.55 dB  hit 0.66  4.60 ms  alpha [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
  epoch   1/100  train 3413.0530  val 3337.1768
  epoch   2/100  train 3253.5042  val 3179.4747
  epoch   3/100  train 3094.7244  val 3018.8266
  epoch   4/100  train 2936.8100  val 2862.4690
  epoch   5/100  train 2781.2339  val 2710.8669
  epoch   6/100  train 2627.3580  val 2553.6200
  epoch   7/100  train 2474.6161  val 2404.3200
  epoch   8/100  train 2324.8713  val 2252.7133
  epoch   9/100  train 2176.2578  val 2107.0976
  epoch  10/100  train 2030.7755  val 1962.2574
  epoch  11/100  train 1887.9618  val 1821.4418
  epoch  12/100  train 1748.2203  val 1682.1922
  epoch  13/100  train 1611.7822  val 1552.4699
  epoch  14/100  train 1479.3157  val 1421.0164
  epoch  15/100  train 1351.1258  val 1290.3604
  epoch  16/100  train 1226.4310  val 1169.5694
  epoch  17/100  train 1107.4426  val 1052.0461
  epoch  18/100  train 992.5541  val 944.7348
  epoch  19/100  train 883.7012  val 834.5613
  epoch  20/100  train 780.4983  val 734.0086
  epoch  21/100  train 683.5743  val 642.6244
  epoch  22/100  train 593.3268  val 555.9601
  epoch  23/100  train 510.1772  val 474.6738
  epoch  24/100  train 433.6182  val 400.7350
  epoch  25/100  train 364.6385  val 333.5598
  epoch  26/100  train 303.0464  val 274.6666
  epoch  27/100  train 248.6527  val 225.0350
  epoch  28/100  train 201.7265  val 182.3144
  epoch  29/100  train 161.9730  val 144.8868
  epoch  30/100  train 128.7039  val 116.4715
  epoch  31/100  train 101.3153  val 90.0084
  epoch  32/100  train 79.4773  val 71.5729
  epoch  33/100  train 62.2909  val 55.5585
  epoch  34/100  train 48.7703  val 42.9719
  epoch  35/100  train 38.4281  val 35.0285
  epoch  36/100  train 30.2808  val 26.9630
  epoch  37/100  train 22.4493  val 13.7258
  epoch  38/100  train 11.7900  val 10.8921
  epoch  39/100  train 10.9134  val 10.9805
  epoch  40/100  train 9.9072  val 9.0230
  epoch  41/100  train 9.2766  val 10.0134
  epoch  42/100  train 9.9385  val 9.2710
  epoch  43/100  train 9.1941  val 9.7589
  epoch  44/100  train 9.8221  val 9.7186
  epoch  45/100  train 9.2953  val 9.3172
  epoch  46/100  train 9.5752  val 9.7778
  epoch  47/100  train 9.5140  val 9.2368
  epoch  48/100  train 9.4332  val 9.6451
  epoch  49/100  train 9.5860  val 9.4268
  epoch  50/100  train 9.4553  val 9.6577
  early stopping triggered
  training: 50 epochs, 425.8 s
  UVMD-1: rel err 0.0004  fault SNR 7.26 dB  hit 0.68  4.53 ms  alpha [276.0, 287.0, 215.0, 190.0, 0.0, 165.0, 226.0, 237.0, 249.0, 150.0, 164.0, 286.0, 484.0]

[UVMD N=2] upstream Unfolding, K=13, device cuda:0
  UVMD-2 untrained: rel err 0.1068  fault SNR 6.87 dB  hit 0.69  7.54 ms  alpha [2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0]
  epoch   1/100  train 7531.5469  val 7320.2330
  epoch   2/100  train 7286.5544  val 7005.0820
  epoch   3/100  train 6923.0755  val 6623.2578
  epoch   4/100  train 6510.3088  val 6254.1018
  epoch   5/100  train 6149.6596  val 5938.9105
  epoch   6/100  train 5858.4623  val 5683.8331
  epoch   7/100  train 5618.6444  val 5468.4713
  epoch   8/100  train 5403.4456  val 5263.3247
  epoch   9/100  train 5193.3624  val 5061.3938
  epoch  10/100  train 4983.7910  val 4858.4932
  epoch  11/100  train 4772.4470  val 4654.0705
  epoch  12/100  train 4564.5308  val 4450.7708
  epoch  13/100  train 4359.3466  val 4254.0530
  epoch  14/100  train 4158.8101  val 4060.9061
  epoch  15/100  train 3964.0882  val 3873.2804
  epoch  16/100  train 3773.2629  val 3685.5272
  epoch  17/100  train 3588.3630  val 3505.0656
  epoch  18/100  train 3407.1718  val 3333.9201
  epoch  19/100  train 3230.7569  val 3157.5514
  epoch  20/100  train 3058.0527  val 2995.4100
  epoch  21/100  train 2889.3461  val 2830.8212
  epoch  22/100  train 2722.6179  val 2668.4859
  epoch  23/100  train 2558.8857  val 2508.7741
  epoch  24/100  train 2395.3189  val 2344.1628
  epoch  25/100  train 2231.6945  val 2187.5880
  epoch  26/100  train 2067.2149  val 2020.7303
  epoch  27/100  train 1898.5223  val 1851.4302
  epoch  28/100  train 1724.4300  val 1666.5287
  epoch  29/100  train 1542.5256  val 1477.5297
  epoch  30/100  train 1352.0413  val 1279.0977
  epoch  31/100  train 1151.3218  val 1062.8518
  epoch  32/100  train 927.0361  val 814.2550
  epoch  33/100  train 597.0634  val 279.0898
  epoch  34/100  train 49.4062  val 12.9596
  epoch  35/100  train 12.2393  val 12.1739
  epoch  36/100  train 12.0709  val 11.4574
  epoch  37/100  train 11.0790  val 11.1504
  epoch  38/100  train 11.3296  val 11.3540
  epoch  39/100  train 10.7954  val 10.6925
  epoch  40/100  train 10.7172  val 11.0021
  epoch  41/100  train 10.6579  val 10.2738
  epoch  42/100  train 10.3467  val 10.4203
  epoch  43/100  train 10.2893  val 10.1468
  epoch  44/100  train 10.0654  val 9.9845
  epoch  45/100  train 10.1329  val 10.0839
  epoch  46/100  train 9.8485  val 9.6201
  epoch  47/100  train 9.7616  val 9.7892
  epoch  48/100  train 9.7885  val 9.4726
  epoch  49/100  train 9.6826  val 9.7172
  epoch  50/100  train 9.6266  val 9.4124
  epoch  51/100  train 9.4241  val 9.3666
  epoch  52/100  train 9.4702  val 9.5030
  epoch  53/100  train 9.4905  val 9.5095
  epoch  54/100  train 9.4916  val 9.5342
  epoch  55/100  train 9.4396  val 9.5461
  epoch  56/100  train 9.5072  val 9.4254
  epoch  57/100  train 9.4977  val 9.5536
  epoch  58/100  train 9.4505  val 9.5640
  epoch  59/100  train 9.3889  val 9.3847
  epoch  60/100  train 9.4521  val 9.2844
  epoch  61/100  train 9.3967  val 9.4894
  epoch  62/100  train 9.3584  val 9.3837
  epoch  63/100  train 9.3034  val 9.3861
  epoch  64/100  train 9.3975  val 9.4743
  epoch  65/100  train 9.5288  val 9.5261
  epoch  66/100  train 9.4587  val 9.2935
  epoch  67/100  train 9.5152  val 9.6181
  epoch  68/100  train 9.4397  val 9.4432
  epoch  69/100  train 9.4085  val 9.4851
  epoch  70/100  train 9.3980  val 9.4520
  early stopping triggered
  training: 70 epochs, 1149.1 s
  UVMD-2: rel err 0.0004  fault SNR 6.95 dB  hit 0.67  7.50 ms  alpha [510.0, 511.0, 941.0, 1434.0, 1173.0, 62.0, 316.0, 209.0, 112.0, 276.0, 260.0, 0.0, 159.0]
```
