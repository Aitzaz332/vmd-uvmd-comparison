# CWRU benchmark - VMD vs UVMD

40 recordings, 1199 windows x 2048 samples at 12 kHz, standardised=True, split 'load' (test = 300 windows), K=4, alpha=2000.0, tau=0.0.

## Overall (test windows)

| method | recon_rmse | recon_rel_err | recon_snr_db | spec_l1 | energy_ratio | ortho_index | fault_snr_db | envelope_hit | oracle_fault_snr_db | sel_kurtosis | normal_max_fault_snr_db |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw signal | nan | nan | nan | nan | nan | nan | 8.896 | 0.6963 | nan | 7.987 | 6.374 |
| VMD (converged) | 0.324 | 0.324 | 10.09 | 2.275e+04 | 0.6301 | 0.03378 | 7.478 | 0.6852 | 10.3 | 6.988 | 5.524 |
| VMD-1 | 0.5543 | 0.5543 | 5.195 | 3.176e+04 | 0.2426 | 0.02422 | 8.469 | 0.7074 | 10.81 | 8.784 | 4.536 |
| VMD-2 | 0.4794 | 0.4794 | 6.458 | 2.903e+04 | 0.3375 | 0.02573 | 8.678 | 0.6778 | 10.62 | 9.257 | 4.712 |
| UVMD-1 untrained | 0.5543 | 0.5543 | 5.195 | 3.176e+04 | 0.2426 | 0.02422 | 8.469 | 0.7074 | 10.81 | 8.784 | 4.536 |
| UVMD-1 | 0.0003982 | 0.0003982 | 78.92 | 9.561 | 0.8331 | 0.08345 | 8.819 | 0.7 | 10.09 | 8.586 | 4.746 |
| UVMD-2 untrained | 0.4794 | 0.4794 | 6.458 | 2.903e+04 | 0.3375 | 0.02573 | 8.678 | 0.6778 | 10.62 | 9.257 | 4.712 |
| UVMD-2 | 0.000399 | 0.000399 | 78.83 | 9.481 | 1.207 | 0.3804 | 6.78 | 0.6667 | 8.591 | 6.898 | 4.075 |

## Fault metrics by class

|                                |   fault_snr_db |   envelope_hit |   oracle_fault_snr_db |   sel_kurtosis |   recon_rel_err |
|:-------------------------------|---------------:|---------------:|----------------------:|---------------:|----------------:|
| ('raw signal', 'ball')         |           3.42 |          0.3   |                nan    |           4.7  |      nan        |
| ('raw signal', 'inner')        |          11.7  |          1     |                nan    |          10.4  |      nan        |
| ('raw signal', 'normal')       |         nan    |        nan     |                nan    |           3.02 |      nan        |
| ('raw signal', 'outer')        |          11.6  |          0.789 |                nan    |          10.6  |      nan        |
| ('VMD (converged)', 'ball')    |           3.22 |          0.322 |                  6.35 |           5.33 |        0.314    |
| ('VMD (converged)', 'inner')   |           9.24 |          0.956 |                 11.4  |           8.92 |        0.304    |
| ('VMD (converged)', 'normal')  |         nan    |        nan     |                nan    |           3.23 |        0.458    |
| ('VMD (converged)', 'outer')   |           9.98 |          0.778 |                 13.2  |           7.97 |        0.31     |
| ('VMD-1', 'ball')              |           3.26 |          0.411 |                  6.19 |           5.29 |        0.499    |
| ('VMD-1', 'inner')             |          11.4  |          1     |                 13.6  |          10.6  |        0.546    |
| ('VMD-1', 'normal')            |         nan    |        nan     |                nan    |           3.39 |        0.589    |
| ('VMD-1', 'outer')             |          10.7  |          0.711 |                 12.6  |          12.3  |        0.607    |
| ('VMD-2', 'ball')              |           3.59 |          0.311 |                  6.41 |           5.1  |        0.421    |
| ('VMD-2', 'inner')             |          12    |          1     |                 13    |          13    |        0.476    |
| ('VMD-2', 'normal')            |         nan    |        nan     |                nan    |           3.39 |        0.54     |
| ('VMD-2', 'outer')             |          10.5  |          0.722 |                 12.4  |          11.6  |        0.52     |
| ('UVMD-1 untrained', 'ball')   |           3.26 |          0.411 |                  6.19 |           5.29 |        0.499    |
| ('UVMD-1 untrained', 'inner')  |          11.4  |          1     |                 13.6  |          10.6  |        0.546    |
| ('UVMD-1 untrained', 'normal') |         nan    |        nan     |                nan    |           3.39 |        0.589    |
| ('UVMD-1 untrained', 'outer')  |          10.7  |          0.711 |                 12.6  |          12.3  |        0.607    |
| ('UVMD-1', 'ball')             |           3.29 |          0.311 |                  5.76 |           5.39 |        8.14e-05 |
| ('UVMD-1', 'inner')            |          11.7  |          1     |                 12    |          11.3  |        8.12e-05 |
| ('UVMD-1', 'normal')           |         nan    |        nan     |                nan    |           3.06 |        0.00324  |
| ('UVMD-1', 'outer')            |          11.5  |          0.789 |                 12.6  |          11    |        8.57e-05 |
| ('UVMD-2 untrained', 'ball')   |           3.59 |          0.311 |                  6.41 |           5.1  |        0.421    |
| ('UVMD-2 untrained', 'inner')  |          12    |          1     |                 13    |          13    |        0.476    |
| ('UVMD-2 untrained', 'normal') |         nan    |        nan     |                nan    |           3.39 |        0.54     |
| ('UVMD-2 untrained', 'outer')  |          10.5  |          0.722 |                 12.4  |          11.6  |        0.52     |
| ('UVMD-2', 'ball')             |           1.84 |          0.178 |                  4.21 |           4.86 |        8.23e-05 |
| ('UVMD-2', 'inner')            |          10.4  |          1     |                 10.9  |           8.37 |        8.21e-05 |
| ('UVMD-2', 'normal')           |         nan    |        nan     |                nan    |           3.11 |        0.00324  |
| ('UVMD-2', 'outer')            |           8.09 |          0.822 |                 10.7  |           8.72 |        8.66e-05 |

## Cost (per window)

| method           |   iterations_or_layers |   trainable_params |   flops_per_window |   median_ms_per_window |   train_time_s |   train_flops |
|:-----------------|-----------------------:|-------------------:|-------------------:|-----------------------:|---------------:|--------------:|
| VMD (converged)  |                   82.4 |                  0 |          3.634e+07 |                22.74   |            0   |     0         |
| VMD-1            |                    1   |                  0 |          1.655e+06 |                 0.6378 |            0   |     0         |
| VMD-2            |                    2   |                  0 |          2.081e+06 |                 1.008  |            0   |     0         |
| VMD-3            |                    3   |                  0 |          2.507e+06 |                 1.327  |            0   |     0         |
| VMD-5            |                    5   |                  0 |          3.359e+06 |                 1.983  |            0   |     0         |
| VMD-10           |                   10   |                  0 |          5.489e+06 |                 3.62   |            0   |     0         |
| VMD-20           |                   20   |                  0 |          9.748e+06 |                 6.831  |            0   |     0         |
| VMD-50           |                   50   |                  0 |          2.253e+07 |                16.43   |            0   |     0         |
| VMD-100          |                  100   |                  0 |          4.383e+07 |                32.44   |            0   |     0         |
| VMD-200          |                  200   |                  0 |          8.643e+07 |                64.47   |            0   |     0         |
| UVMD-1 untrained |                    1   |              16388 |          1.54e+06  |                 1.659  |            0   |     0         |
| UVMD-1           |                    1   |              16388 |          1.54e+06  |                 1.675  |          226.8 |     2.159e+11 |
| UVMD-2 untrained |                    2   |              24580 |          1.851e+06 |                 2.598  |            0   |     0         |
| UVMD-2           |                    2   |              24580 |          1.851e+06 |                 2.579  |          576.6 |     2.362e+11 |

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'numpy': '2.1.3', 'torch': '2.7.0', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060'}


## Log
```
CWRU: 40 files, 1199 windows of 2048 samples (171 ms) | split 'load': train/val/test = 599/300/300 | K=4

[VMD converged]
  mean iters 82.4 (100% converged), rel err 0.3240, fault SNR 7.48 dB, hit 0.69, 22.7 ms

[VMD-n sweep]
  n=  1 (300 win): rel err 0.5543  fault SNR   8.47 dB  hit 0.71  0.64 ms
  n=  2 (300 win): rel err 0.4794  fault SNR   8.68 dB  hit 0.68  1.01 ms
  n=  3 (100 win): rel err 0.4124  fault SNR   8.44 dB  hit 0.68  1.33 ms
  n=  5 (100 win): rel err 0.3888  fault SNR   8.18 dB  hit 0.67  1.98 ms
  n= 10 (100 win): rel err 0.3601  fault SNR   8.09 dB  hit 0.72  3.62 ms
  n= 20 (100 win): rel err 0.3475  fault SNR   7.75 dB  hit 0.71  6.83 ms
  n= 50 (100 win): rel err 0.3313  fault SNR   7.27 dB  hit 0.70  16.43 ms
  n=100 (100 win): rel err 0.3259  fault SNR   7.36 dB  hit 0.68  32.44 ms
  n=200 (100 win): rel err 0.3240  fault SNR   7.38 dB  hit 0.68  64.47 ms

[UVMD N=1] upstream Unfolding, K=4, device cuda:0
  UVMD-1 untrained: rel err 0.5543  fault SNR 8.47 dB  hit 0.71  1.66 ms  alpha [2000.0, 2000.0, 2000.0, 2000.0]
  epoch   1/100  train 31140.8265  val 31163.5474
  epoch   2/100  train 30763.6357  val 30790.9449
  epoch   3/100  train 30372.8714  val 30398.9108
  epoch   4/100  train 29964.6831  val 29985.6441
  epoch   5/100  train 29537.6982  val 29552.9766
  epoch   6/100  train 29091.1371  val 29096.7061
  epoch   7/100  train 28622.6984  val 28619.1757
  epoch   8/100  train 28132.3899  val 28117.4655
  epoch   9/100  train 27617.3540  val 27587.7636
  epoch  10/100  train 27076.6833  val 27033.7281
  epoch  11/100  train 26507.2534  val 26447.5254
  epoch  12/100  train 25907.7367  val 25830.9595
  epoch  13/100  train 25274.8199  val 25177.5923
  epoch  14/100  train 24604.8619  val 24487.6371
  epoch  15/100  train 23895.2734  val 23750.9097
  epoch  16/100  train 23141.3207  val 22970.2559
  epoch  17/100  train 22338.1591  val 22135.5931
  epoch  18/100  train 21479.8418  val 21243.5257
  epoch  19/100  train 20560.2525  val 20285.7518
  epoch  20/100  train 19570.8978  val 19250.1258
  epoch  21/100  train 18505.0919  val 18134.5777
  epoch  22/100  train 17352.1410  val 16929.3531
  epoch  23/100  train 16099.6017  val 15609.9109
  epoch  24/100  train 14732.8663  val 14166.8596
  epoch  25/100  train 13236.4801  val 12585.3735
  epoch  26/100  train 11592.0615  val 10847.9267
  epoch  27/100  train 9785.0297  val 8937.8970
  epoch  28/100  train 7798.8976  val 6839.5209
  epoch  29/100  train 5633.9236  val 4554.5660
  epoch  30/100  train 3179.0685  val 1496.9884
  epoch  31/100  train 123.7970  val 13.5885
  epoch  32/100  train 13.5030  val 13.4091
  epoch  33/100  train 12.9320  val 12.3547
  epoch  34/100  train 11.9746  val 11.7104
  epoch  35/100  train 11.7704  val 11.7720
  epoch  36/100  train 11.3503  val 11.1876
  epoch  37/100  train 10.9905  val 10.9137
  epoch  38/100  train 10.8803  val 10.6629
  epoch  39/100  train 10.7687  val 10.9255
  epoch  40/100  train 10.6426  val 10.4907
  epoch  41/100  train 10.4273  val 10.4307
  epoch  42/100  train 10.3175  val 10.1184
  epoch  43/100  train 10.0926  val 10.0611
  epoch  44/100  train 10.0544  val 10.0138
  epoch  45/100  train 9.9112  val 9.7889
  epoch  46/100  train 9.7393  val 9.5747
  epoch  47/100  train 9.7055  val 9.5667
  epoch  48/100  train 9.5600  val 9.4262
  epoch  49/100  train 9.5476  val 9.4418
  epoch  50/100  train 9.4903  val 9.4233
  epoch  51/100  train 9.5007  val 9.5592
  epoch  52/100  train 9.4346  val 9.3025
  epoch  53/100  train 9.4136  val 9.5876
  epoch  54/100  train 9.5139  val 9.3541
  epoch  55/100  train 9.4141  val 9.4800
  epoch  56/100  train 9.3147  val 9.2711
  epoch  57/100  train 9.3632  val 9.4368
  epoch  58/100  train 9.4631  val 9.5399
  epoch  59/100  train 9.4502  val 9.4400
  epoch  60/100  train 9.4561  val 9.4024
  epoch  61/100  train 9.4948  val 9.4864
  epoch  62/100  train 9.3669  val 9.2606
  epoch  63/100  train 9.5486  val 9.3745
  epoch  64/100  train 9.5117  val 9.3430
  epoch  65/100  train 9.3725  val 9.5302
  epoch  66/100  train 9.4778  val 9.4011
  epoch  67/100  train 9.3495  val 9.3877
  epoch  68/100  train 9.3527  val 9.1463
  epoch  69/100  train 9.3244  val 9.4547
  epoch  70/100  train 9.3980  val 9.3668
  epoch  71/100  train 9.4567  val 9.4772
  epoch  72/100  train 9.4955  val 9.5635
  epoch  73/100  train 9.4439  val 9.3292
  epoch  74/100  train 9.4409  val 9.6273
  epoch  75/100  train 9.4248  val 9.3750
  epoch  76/100  train 9.5466  val 9.6939
  epoch  77/100  train 9.5223  val 9.5171
  epoch  78/100  train 9.4500  val 9.5607
  early stopping triggered
  training: 78 epochs, 226.8 s
  UVMD-1: rel err 0.0004  fault SNR 8.82 dB  hit 0.70  1.67 ms  alpha [220.0, 0.0, 127.0, 146.0]

[UVMD N=2] upstream Unfolding, K=4, device cuda:0
  UVMD-2 untrained: rel err 0.4794  fault SNR 8.68 dB  hit 0.68  2.60 ms  alpha [2000.0, 2000.0, 2000.0, 2000.0]
  epoch   1/100  train 28636.7197  val 28580.6922
  epoch   2/100  train 28242.0531  val 28176.4338
  epoch   3/100  train 27775.6503  val 27629.9111
  epoch   4/100  train 27169.1949  val 26975.3468
  epoch   5/100  train 26546.8735  val 26359.2803
  epoch   6/100  train 25947.1015  val 25755.2469
  epoch   7/100  train 25353.6316  val 25153.1425
  epoch   8/100  train 24761.8763  val 24550.4576
  epoch   9/100  train 24166.9488  val 23943.3338
  epoch  10/100  train 23571.6615  val 23337.6341
  epoch  11/100  train 22977.5525  val 22731.8264
  epoch  12/100  train 22385.3888  val 22131.2301
  epoch  13/100  train 21793.4845  val 21529.3174
  epoch  14/100  train 21197.7334  val 20930.4913
  epoch  15/100  train 20594.5288  val 20322.6002
  epoch  16/100  train 19973.8938  val 19695.4190
  epoch  17/100  train 19328.9372  val 19042.2052
  epoch  18/100  train 18650.6441  val 18355.1804
  epoch  19/100  train 17933.2019  val 17622.9060
  epoch  20/100  train 17172.5515  val 16846.0878
  epoch  21/100  train 16368.7324  val 16024.3811
  epoch  22/100  train 15518.2801  val 15154.8551
  epoch  23/100  train 14622.0005  val 14235.9185
  epoch  24/100  train 13677.6116  val 13267.1233
  epoch  25/100  train 12683.6767  val 12251.5008
  epoch  26/100  train 11634.9179  val 11179.6460
  epoch  27/100  train 10523.1026  val 10042.5638
  epoch  28/100  train 9338.5400  val 8821.6614
  epoch  29/100  train 8059.1913  val 7487.8114
  epoch  30/100  train 6631.8024  val 5942.1771
  epoch  31/100  train 4859.4166  val 3728.2268
  epoch  32/100  train 1935.1214  val 604.5751
  epoch  33/100  train 156.1149  val 20.4952
  epoch  34/100  train 17.9635  val 15.7356
  epoch  35/100  train 15.4792  val 14.8464
  epoch  36/100  train 14.2583  val 13.7416
  epoch  37/100  train 13.7385  val 13.1264
  epoch  38/100  train 13.1280  val 12.9103
  epoch  39/100  train 12.7639  val 12.2381
  epoch  40/100  train 12.3642  val 12.5319
  epoch  41/100  train 12.1569  val 11.6454
  epoch  42/100  train 11.7142  val 11.3431
  epoch  43/100  train 11.3830  val 11.3271
  epoch  44/100  train 11.2380  val 11.0573
  epoch  45/100  train 10.9656  val 10.7370
  epoch  46/100  train 10.7094  val 10.5104
  epoch  47/100  train 10.4979  val 10.4153
  epoch  48/100  train 10.3497  val 10.3153
  epoch  49/100  train 10.1559  val 10.0221
  epoch  50/100  train 9.9765  val 9.9395
  epoch  51/100  train 9.8359  val 9.6774
  epoch  52/100  train 9.7621  val 9.7277
  epoch  53/100  train 9.6381  val 9.6561
  epoch  54/100  train 9.6638  val 9.6127
  epoch  55/100  train 9.5573  val 9.4225
  epoch  56/100  train 9.5046  val 9.5586
  epoch  57/100  train 9.5549  val 9.4308
  epoch  58/100  train 9.5099  val 9.4779
  epoch  59/100  train 9.4243  val 9.3744
  epoch  60/100  train 9.3527  val 9.2589
  epoch  61/100  train 9.3516  val 9.2502
  epoch  62/100  train 9.3281  val 9.3977
  epoch  63/100  train 9.4327  val 9.5454
  epoch  64/100  train 9.4375  val 9.4284
  epoch  65/100  train 9.3928  val 9.4951
  epoch  66/100  train 9.3647  val 9.2987
  epoch  67/100  train 9.3788  val 9.5504
  epoch  68/100  train 9.4958  val 9.4036
  epoch  69/100  train 9.4393  val 9.4776
  epoch  70/100  train 9.3922  val 9.4075
  epoch  71/100  train 9.3402  val 9.4812
  early stopping triggered
  training: 71 epochs, 576.6 s
  UVMD-2: rel err 0.0004  fault SNR 6.78 dB  hit 0.67  2.58 ms  alpha [257.0, 76.0, 0.0, 168.0]
```
