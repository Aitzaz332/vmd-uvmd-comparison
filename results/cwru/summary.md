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
| UVMD-1 | 0.0003972 | 0.0003972 | 79.03 | 9.304 | 0.8331 | 0.08346 | 8.639 | 0.6963 | 10.07 | 8.727 | 4.905 |
| UVMD-2 untrained | 0.4794 | 0.4794 | 6.458 | 2.903e+04 | 0.3375 | 0.02573 | 8.678 | 0.6778 | 10.62 | 9.257 | 4.712 |
| UVMD-2 | 0.0003985 | 0.0003985 | 78.89 | 9.572 | 1.207 | 0.3807 | 7.575 | 0.6852 | 8.621 | 6.651 | 4.162 |

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
| ('UVMD-1', 'ball')             |           3.23 |          0.289 |                  5.49 |           5.6  |        8.02e-05 |
| ('UVMD-1', 'inner')            |          11.6  |          1     |                 12.2  |          11.3  |        8.01e-05 |
| ('UVMD-1', 'normal')           |         nan    |        nan     |                nan    |           3.06 |        0.00324  |
| ('UVMD-1', 'outer')            |          11    |          0.8   |                 12.6  |          11.1  |        8.46e-05 |
| ('UVMD-2 untrained', 'ball')   |           3.59 |          0.311 |                  6.41 |           5.1  |        0.421    |
| ('UVMD-2 untrained', 'inner')  |          12    |          1     |                 13    |          13    |        0.476    |
| ('UVMD-2 untrained', 'normal') |         nan    |        nan     |                nan    |           3.39 |        0.54     |
| ('UVMD-2 untrained', 'outer')  |          10.5  |          0.722 |                 12.4  |          11.6  |        0.52     |
| ('UVMD-2', 'ball')             |           2.09 |          0.222 |                  4.44 |           4.66 |        8.16e-05 |
| ('UVMD-2', 'inner')            |          10.4  |          1     |                 10.6  |           8.37 |        8.15e-05 |
| ('UVMD-2', 'normal')           |         nan    |        nan     |                nan    |           3.11 |        0.00324  |
| ('UVMD-2', 'outer')            |          10.2  |          0.833 |                 10.8  |           8.09 |        8.6e-05  |

## Cost (per window)

| method           |   iterations_or_layers |   trainable_params |   flops_per_window |   median_ms_per_window |   train_time_s |   train_flops |
|:-----------------|-----------------------:|-------------------:|-------------------:|-----------------------:|---------------:|--------------:|
| VMD (converged)  |                   82.4 |                  0 |          3.634e+07 |                 43.32  |            0   |     0         |
| VMD-1            |                    1   |                  0 |          1.655e+06 |                  1.416 |            0   |     0         |
| VMD-2            |                    2   |                  0 |          2.081e+06 |                  2.017 |            0   |     0         |
| VMD-3            |                    3   |                  0 |          2.507e+06 |                  2.626 |            0   |     0         |
| VMD-5            |                    5   |                  0 |          3.359e+06 |                  3.768 |            0   |     0         |
| VMD-10           |                   10   |                  0 |          5.489e+06 |                  6.635 |            0   |     0         |
| VMD-20           |                   20   |                  0 |          9.748e+06 |                 13.18  |            0   |     0         |
| VMD-50           |                   50   |                  0 |          2.253e+07 |                 31.09  |            0   |     0         |
| VMD-100          |                  100   |                  0 |          4.383e+07 |                 62.16  |            0   |     0         |
| VMD-200          |                  200   |                  0 |          8.643e+07 |                127.2   |            0   |     0         |
| UVMD-1 untrained |                    1   |              16388 |          1.54e+06  |                  2.696 |            0   |     0         |
| UVMD-1           |                    1   |              16388 |          1.54e+06  |                  2.857 |          274.8 |     1.827e+11 |
| UVMD-2 untrained |                    2   |              24580 |          1.851e+06 |                  3.972 |            0   |     0         |
| UVMD-2           |                    2   |              24580 |          1.851e+06 |                  4.05  |          511.5 |     2.528e+11 |

Environment: {'python': '3.13.17', 'platform': 'Linux-6.18.44-fc-v153-x86_64-with-glibc2.39', 'numpy': '2.5.3', 'torch': '2.14.1+cpu', 'cuda': False, 'gpu': None}


## Log
```
CWRU: 40 files, 1199 windows of 2048 samples (171 ms) | split 'load': train/val/test = 599/300/300 | K=4

[VMD converged]
  mean iters 82.4 (100% converged), rel err 0.3240, fault SNR 7.48 dB, hit 0.69, 43.3 ms

[VMD-n sweep]
  n=  1 (300 win): rel err 0.5543  fault SNR   8.47 dB  hit 0.71  1.42 ms
  n=  2 (300 win): rel err 0.4794  fault SNR   8.68 dB  hit 0.68  2.02 ms
  n=  3 (100 win): rel err 0.4124  fault SNR   8.44 dB  hit 0.68  2.63 ms
  n=  5 (100 win): rel err 0.3888  fault SNR   8.18 dB  hit 0.67  3.77 ms
  n= 10 (100 win): rel err 0.3601  fault SNR   8.09 dB  hit 0.72  6.64 ms
  n= 20 (100 win): rel err 0.3475  fault SNR   7.75 dB  hit 0.71  13.18 ms
  n= 50 (100 win): rel err 0.3313  fault SNR   7.27 dB  hit 0.70  31.09 ms
  n=100 (100 win): rel err 0.3259  fault SNR   7.36 dB  hit 0.68  62.16 ms
  n=200 (100 win): rel err 0.3240  fault SNR   7.38 dB  hit 0.68  127.19 ms

[UVMD N=1] upstream Unfolding, K=4, device cpu
  UVMD-1 untrained: rel err 0.5543  fault SNR 8.47 dB  hit 0.71  2.70 ms  alpha [2000.0, 2000.0, 2000.0, 2000.0]
  epoch   1/100  train 31140.8267  val 31163.5877
  epoch   2/100  train 30763.6403  val 30790.9531
  epoch   3/100  train 30372.8842  val 30398.8962
  epoch   4/100  train 29964.6649  val 29985.3943
  epoch   5/100  train 29537.6838  val 29553.0396
  epoch   6/100  train 29091.1253  val 29096.8413
  epoch   7/100  train 28622.6846  val 28619.2077
  epoch   8/100  train 28132.3667  val 28117.3757
  epoch   9/100  train 27617.3428  val 27587.5917
  epoch  10/100  train 27076.6751  val 27033.7829
  epoch  11/100  train 26507.2400  val 26447.4685
  epoch  12/100  train 25907.7396  val 25830.9148
  epoch  13/100  train 25274.8018  val 25177.6439
  epoch  14/100  train 24604.8498  val 24487.4481
  epoch  15/100  train 23895.2907  val 23750.8627
  epoch  16/100  train 23141.3231  val 22970.4187
  epoch  17/100  train 22338.1666  val 22135.6251
  epoch  18/100  train 21479.8248  val 21243.5577
  epoch  19/100  train 20560.2013  val 20285.5178
  epoch  20/100  train 19570.8783  val 19250.0344
  epoch  21/100  train 18505.0808  val 18134.6311
  epoch  22/100  train 17352.1424  val 16929.5117
  epoch  23/100  train 16099.5974  val 15609.7723
  epoch  24/100  train 14732.8321  val 14166.8331
  epoch  25/100  train 13236.4766  val 12585.4344
  epoch  26/100  train 11592.0220  val 10847.8787
  epoch  27/100  train 9784.9850  val 8937.8096
  epoch  28/100  train 7798.8719  val 6839.5719
  epoch  29/100  train 5633.9048  val 4554.4733
  epoch  30/100  train 3179.1050  val 1497.2180
  epoch  31/100  train 123.9259  val 13.8124
  epoch  32/100  train 13.5247  val 13.4772
  epoch  33/100  train 12.9218  val 12.1411
  epoch  34/100  train 11.9206  val 11.6688
  epoch  35/100  train 11.6250  val 11.6129
  epoch  36/100  train 11.2481  val 10.9864
  epoch  37/100  train 11.1030  val 11.1489
  epoch  38/100  train 10.9596  val 10.7812
  epoch  39/100  train 10.6202  val 10.9205
  epoch  40/100  train 10.5358  val 10.6710
  epoch  41/100  train 10.4013  val 10.2598
  epoch  42/100  train 10.3524  val 10.2266
  epoch  43/100  train 10.1767  val 10.0695
  epoch  44/100  train 10.0149  val 9.8650
  epoch  45/100  train 9.9147  val 9.9999
  epoch  46/100  train 9.7898  val 9.7106
  epoch  47/100  train 9.6636  val 9.5191
  epoch  48/100  train 9.6809  val 9.6112
  epoch  49/100  train 9.5865  val 9.4514
  epoch  50/100  train 9.4854  val 9.4804
  epoch  51/100  train 9.4418  val 9.4632
  epoch  52/100  train 9.3809  val 9.3368
  epoch  53/100  train 9.3108  val 9.5005
  epoch  54/100  train 9.5026  val 9.4159
  epoch  55/100  train 9.4733  val 9.4701
  epoch  56/100  train 9.4035  val 9.2783
  epoch  57/100  train 9.4339  val 9.4497
  epoch  58/100  train 9.3982  val 9.4994
  epoch  59/100  train 9.4376  val 9.4813
  epoch  60/100  train 9.4464  val 9.3961
  epoch  61/100  train 9.5088  val 9.4939
  epoch  62/100  train 9.5169  val 9.5803
  epoch  63/100  train 9.4240  val 9.4336
  epoch  64/100  train 9.3849  val 9.5059
  epoch  65/100  train 9.4818  val 9.5299
  epoch  66/100  train 9.4459  val 9.3043
  early stopping triggered
  training: 66 epochs, 274.8 s
  UVMD-1: rel err 0.0004  fault SNR 8.64 dB  hit 0.70  2.86 ms  alpha [220.0, 0.0, 127.0, 146.0]

[UVMD N=2] upstream Unfolding, K=4, device cpu
  UVMD-2 untrained: rel err 0.4794  fault SNR 8.68 dB  hit 0.68  3.97 ms  alpha [2000.0, 2000.0, 2000.0, 2000.0]
  epoch   1/100  train 28636.7187  val 28580.7094
  epoch   2/100  train 28242.0572  val 28176.4367
  epoch   3/100  train 27775.6523  val 27629.9893
  epoch   4/100  train 27169.1876  val 26975.3487
  epoch   5/100  train 26546.8638  val 26359.3366
  epoch   6/100  train 25947.1071  val 25755.2852
  epoch   7/100  train 25353.6570  val 25153.1229
  epoch   8/100  train 24761.8879  val 24550.5500
  epoch   9/100  train 24166.9538  val 23943.3861
  epoch  10/100  train 23571.6706  val 23337.6288
  epoch  11/100  train 22977.5319  val 22731.9407
  epoch  12/100  train 22385.4143  val 22131.2686
  epoch  13/100  train 21793.4857  val 21529.3429
  epoch  14/100  train 21197.7316  val 20930.4661
  epoch  15/100  train 20594.5217  val 20322.7174
  epoch  16/100  train 19973.8876  val 19695.5289
  epoch  17/100  train 19328.9481  val 19042.1237
  epoch  18/100  train 18650.6864  val 18355.1114
  epoch  19/100  train 17933.2047  val 17622.8556
  epoch  20/100  train 17172.5635  val 16846.2356
  epoch  21/100  train 16368.7685  val 16024.5700
  epoch  22/100  train 15518.3292  val 15154.7750
  epoch  23/100  train 14622.0155  val 14235.8319
  epoch  24/100  train 13677.6280  val 13267.1702
  epoch  25/100  train 12683.6733  val 12251.5999
  epoch  26/100  train 11634.8759  val 11179.6014
  epoch  27/100  train 10523.1016  val 10042.5037
  epoch  28/100  train 9338.5322  val 8821.5680
  epoch  29/100  train 8059.1898  val 7487.6794
  epoch  30/100  train 6631.7973  val 5942.1849
  epoch  31/100  train 4859.4128  val 3728.0396
  epoch  32/100  train 1934.9973  val 604.3913
  epoch  33/100  train 156.1595  val 21.0050
  epoch  34/100  train 17.9291  val 15.8728
  epoch  35/100  train 15.4974  val 14.7438
  epoch  36/100  train 14.2513  val 13.5530
  epoch  37/100  train 13.6391  val 13.2240
  epoch  38/100  train 13.0440  val 13.0727
  epoch  39/100  train 12.7504  val 12.3908
  epoch  40/100  train 12.3581  val 12.4509
  epoch  41/100  train 12.1186  val 11.7132
  epoch  42/100  train 11.8572  val 11.6682
  epoch  43/100  train 11.3822  val 11.3822
  epoch  44/100  train 11.2154  val 10.9200
  epoch  45/100  train 10.8384  val 10.6851
  epoch  46/100  train 10.7066  val 10.5625
  epoch  47/100  train 10.4340  val 10.5335
  epoch  48/100  train 10.1386  val 10.0994
  epoch  49/100  train 10.0471  val 9.9445
  epoch  50/100  train 9.9447  val 9.9084
  epoch  51/100  train 9.8714  val 9.7621
  epoch  52/100  train 9.7992  val 9.8475
  epoch  53/100  train 9.6969  val 9.6889
  epoch  54/100  train 9.6916  val 9.7796
  epoch  55/100  train 9.7313  val 9.7162
  epoch  56/100  train 9.6128  val 9.4101
  epoch  57/100  train 9.4548  val 9.3008
  epoch  58/100  train 9.3951  val 9.3719
  epoch  59/100  train 9.3642  val 9.4839
  epoch  60/100  train 9.4133  val 9.5843
  epoch  61/100  train 9.3973  val 9.3640
  epoch  62/100  train 9.3669  val 9.2521
  epoch  63/100  train 9.4083  val 9.4492
  epoch  64/100  train 9.4338  val 9.4299
  epoch  65/100  train 9.5049  val 9.4871
  epoch  66/100  train 9.3876  val 9.1725
  epoch  67/100  train 9.2915  val 9.2669
  epoch  68/100  train 9.4017  val 9.4463
  epoch  69/100  train 9.4417  val 9.4609
  epoch  70/100  train 9.4967  val 9.3223
  epoch  71/100  train 9.5237  val 9.6071
  epoch  72/100  train 9.5027  val 9.3849
  epoch  73/100  train 9.4895  val 9.4603
  epoch  74/100  train 9.3868  val 9.4669
  epoch  75/100  train 9.4389  val 9.5192
  epoch  76/100  train 9.5288  val 9.5722
  early stopping triggered
  training: 76 epochs, 511.5 s
  UVMD-2: rel err 0.0004  fault SNR 7.58 dB  hit 0.69  4.05 ms  alpha [257.0, 75.0, 0.0, 168.0]
```
