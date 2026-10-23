# Synthetic benchmark - VMD vs UVMD

200 signals, T=1024, 4 true modes, noise std 0.1, K=13, alpha=2000.0, tau=0.0. Test set: 30 unseen signals. Mean over test signals.

## Accuracy

| method | recon_rmse | recon_rel_err | recon_snr_db | spec_l1 | denoise_rmse | mean_mode_nrmse | mean_mode_corr | mean_mode_snr_db | grouped_mean_snr_db | mean_omega_err | ortho_index |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VMD (converged) | 0.03714 | 0.02943 | 30.71 | 1056 | 0.08915 | 0.2938 | 0.9275 | 15.44 | 20.68 | 0.001061 | 0.0431 |
| VMD-1 | 0.06294 | 0.04989 | 26.3 | 779.6 | 0.1121 | 0.368 | 0.9874 | 9.524 | 12.36 | 0.001205 | 0.1293 |
| VMD-2 | 0.04788 | 0.03792 | 28.53 | 1264 | 0.0922 | 0.2617 | 0.9948 | 14.47 | 18.84 | 0.0009002 | 0.08237 |
| UVMD-1 untrained | 0.06294 | 0.04989 | 26.3 | 779.6 | 0.1121 | 0.368 | 0.9874 | 9.524 | 12.36 | 0.001205 | 0.1293 |
| UVMD-1 | 0.008308 | 0.006605 | 43.79 | 117 | 0.09972 | 0.5182 | 0.9634 | 5.935 | 8.537 | 0.002523 | 0.1732 |
| UVMD-2 untrained | 0.04788 | 0.03792 | 28.53 | 1264 | 0.0922 | 0.2617 | 0.9948 | 14.47 | 18.84 | 0.0009002 | 0.08237 |
| UVMD-2 | 0.01794 | 0.01424 | 37.04 | 385.1 | 0.09743 | 0.5137 | 0.899 | 5.944 | 7.329 | 0.002072 | 0.08899 |

## Cost (per signal unless stated)

| method                    |   iterations_or_layers |   trainable_params |   flops_per_signal |   median_ms_per_signal |   train_time_s |   train_flops |
|:--------------------------|-----------------------:|-------------------:|-------------------:|-----------------------:|---------------:|--------------:|
| VMD (converged)           |                  497.7 |                  0 |          3.461e+08 |                449.8   |            0   |     0         |
| vmdpy package (converged) |                  497.7 |                  0 |          3.461e+08 |                607     |            0   |     0         |
| VMD-1                     |                    1   |                  0 |          2.269e+06 |                  2.152 |            0   |     0         |
| VMD-2                     |                    2   |                  0 |          2.961e+06 |                  3.278 |            0   |     0         |
| VMD-3                     |                    3   |                  0 |          3.654e+06 |                  2.871 |            0   |     0         |
| VMD-5                     |                    5   |                  0 |          5.038e+06 |                  5.139 |            0   |     0         |
| VMD-10                    |                   10   |                  0 |          8.499e+06 |                 10.71  |            0   |     0         |
| VMD-20                    |                   20   |                  0 |          1.542e+07 |                 15.41  |            0   |     0         |
| VMD-50                    |                   50   |                  0 |          3.619e+07 |                 47.03  |            0   |     0         |
| VMD-100                   |                  100   |                  0 |          7.08e+07  |                 95.15  |            0   |     0         |
| VMD-200                   |                  200   |                  0 |          1.4e+08   |                185.6   |            0   |     0         |
| VMD-499                   |                  499   |                  0 |          3.47e+08  |                455.7   |            0   |     0         |
| UVMD-1 untrained          |                    1   |               8205 |          2.083e+06 |                  4.037 |            0   |     0         |
| UVMD-1                    |                    1   |               8205 |          2.083e+06 |                  3.787 |          130   |     8.748e+10 |
| UVMD-2 untrained          |                    2   |              12301 |          2.589e+06 |                  5.223 |            0   |     0         |
| UVMD-2                    |                    2   |              12301 |          2.589e+06 |                  5.119 |          230.6 |     1.087e+11 |

Environment: {'python': '3.13.17', 'platform': 'Linux-6.18.44-fc-v153-x86_64-with-glibc2.39', 'numpy': '2.5.3', 'processor': 'x86_64', 'torch': '2.14.1+cpu', 'torch_threads': 2}


## Log
```
synthetic: 200 signals, T=1024, noise std=0.1, K=13 | train/val/test = 140/30/30

[VMD converged]
  mean iterations 497.7, recon_rel_err 0.0294, mode SNR 15.44 dB, median 449.80 ms
  vmdpy package median 607.05 ms (stores all 500 iterates)

[VMD-n sweep]
  n=  1: recon_rel_err 0.0499  mode SNR   9.52 dB  omega err 0.0012  2.15 ms
  n=  2: recon_rel_err 0.0379  mode SNR  14.47 dB  omega err 0.0009  3.28 ms
  n=  3: recon_rel_err 0.0321  mode SNR  15.82 dB  omega err 0.0009  2.87 ms
  n=  5: recon_rel_err 0.0304  mode SNR  16.12 dB  omega err 0.0009  5.14 ms
  n= 10: recon_rel_err 0.0300  mode SNR  16.18 dB  omega err 0.0009  10.71 ms
  n= 20: recon_rel_err 0.0297  mode SNR  16.05 dB  omega err 0.0009  15.41 ms
  n= 50: recon_rel_err 0.0295  mode SNR  15.83 dB  omega err 0.0010  47.03 ms
  n=100: recon_rel_err 0.0295  mode SNR  15.59 dB  omega err 0.0010  95.15 ms
  n=200: recon_rel_err 0.0295  mode SNR  15.47 dB  omega err 0.0011  185.57 ms
  n=499: recon_rel_err 0.0294  mode SNR  15.44 dB  omega err 0.0011  455.67 ms

[UVMD N=1]  upstream Unfolding, K=13
  trainable params: 4109 entries (8205 real DOF)
  UVMD-1 untrained: recon_rel_err 0.0499  mode SNR 9.52 dB  median 4.04 ms  (numpy replica dev 2.7e-07)
  epoch   1/100  train 771.0062  val 753.7401
  epoch   2/100  train 761.7769  val 743.5227
  epoch   3/100  train 754.0113  val 734.8417
  epoch   4/100  train 746.5481  val 731.0637
  epoch   5/100  train 738.8574  val 721.4925
  epoch   6/100  train 730.7345  val 713.8444
  epoch   7/100  train 724.1491  val 706.2886
  epoch   8/100  train 715.5388  val 699.3931
  epoch   9/100  train 707.8159  val 689.6817
  epoch  10/100  train 699.4183  val 684.0954
  epoch  11/100  train 692.3700  val 679.8069
  epoch  12/100  train 684.1603  val 669.3831
  epoch  13/100  train 676.4968  val 661.7307
  epoch  14/100  train 669.9243  val 654.3060
  epoch  15/100  train 662.0097  val 646.8204
  epoch  16/100  train 654.3551  val 639.9747
  epoch  17/100  train 646.8792  val 631.8580
  epoch  18/100  train 638.5450  val 624.8204
  epoch  19/100  train 632.3064  val 615.7334
  epoch  20/100  train 624.2587  val 610.5091
  epoch  21/100  train 615.7297  val 601.5866
  epoch  22/100  train 608.2972  val 593.8021
  epoch  23/100  train 601.6623  val 589.1986
  epoch  24/100  train 593.1932  val 579.8385
  epoch  25/100  train 587.6670  val 573.3071
  epoch  26/100  train 579.0635  val 566.9227
  epoch  27/100  train 571.3992  val 560.6153
  epoch  28/100  train 564.6628  val 552.2906
  epoch  29/100  train 556.5884  val 543.8277
  epoch  30/100  train 549.6134  val 536.6759
  epoch  31/100  train 542.4868  val 528.0118
  epoch  32/100  train 535.0367  val 521.9039
  epoch  33/100  train 527.0560  val 513.6058
  epoch  34/100  train 519.9651  val 507.0739
  epoch  35/100  train 512.8522  val 500.0620
  epoch  36/100  train 505.3460  val 492.4385
  epoch  37/100  train 498.1976  val 485.4988
  epoch  38/100  train 490.5697  val 476.7913
  epoch  39/100  train 483.9711  val 472.8071
  epoch  40/100  train 476.7133  val 466.0871
  epoch  41/100  train 469.9963  val 461.4692
  epoch  42/100  train 462.9244  val 449.9619
  epoch  43/100  train 456.3764  val 443.4339
  epoch  44/100  train 448.0923  val 435.9866
  epoch  45/100  train 441.6347  val 433.1336
  epoch  46/100  train 433.9911  val 424.4497
  epoch  47/100  train 427.2547  val 417.3666
  epoch  48/100  train 419.8980  val 407.9829
  epoch  49/100  train 413.1136  val 401.5080
  epoch  50/100  train 406.1628  val 398.5514
  epoch  51/100  train 399.2193  val 390.2425
  epoch  52/100  train 391.9678  val 381.0891
  epoch  53/100  train 385.4274  val 375.0839
  epoch  54/100  train 378.3925  val 370.8884
  epoch  55/100  train 372.2432  val 364.8373
  epoch  56/100  train 365.1912  val 356.7661
  epoch  57/100  train 358.0512  val 349.0632
  epoch  58/100  train 351.2700  val 348.5963
  epoch  59/100  train 345.9931  val 337.0727
  epoch  60/100  train 338.0795  val 328.6709
  epoch  61/100  train 331.7747  val 321.8297
  epoch  62/100  train 326.0088  val 318.5312
  epoch  63/100  train 319.3478  val 310.4536
  epoch  64/100  train 312.7337  val 305.7335
  epoch  65/100  train 306.7062  val 297.5841
  epoch  66/100  train 300.1923  val 292.5476
  epoch  67/100  train 294.2529  val 283.4534
  epoch  68/100  train 288.2997  val 280.1047
  epoch  69/100  train 281.8817  val 274.0671
  epoch  70/100  train 275.3379  val 270.7622
  epoch  71/100  train 269.5078  val 261.9322
  epoch  72/100  train 263.0905  val 255.3037
  epoch  73/100  train 257.2697  val 249.5237
  epoch  74/100  train 250.9204  val 244.6157
  epoch  75/100  train 245.6261  val 242.0137
  epoch  76/100  train 239.2656  val 232.8791
  epoch  77/100  train 233.3449  val 227.4991
  epoch  78/100  train 227.8656  val 221.6516
  epoch  79/100  train 222.4394  val 214.3701
  epoch  80/100  train 217.1644  val 210.7406
  epoch  81/100  train 211.1497  val 204.9551
  epoch  82/100  train 205.8810  val 199.9076
  epoch  83/100  train 199.9603  val 195.1746
  epoch  84/100  train 194.6722  val 187.0615
  epoch  85/100  train 188.7830  val 183.5349
  epoch  86/100  train 183.2822  val 177.1022
  epoch  87/100  train 178.4431  val 172.8832
  epoch  88/100  train 173.4312  val 168.9497
  epoch  89/100  train 168.5958  val 162.0301
  epoch  90/100  train 163.4437  val 158.2559
  epoch  91/100  train 158.6613  val 153.3449
  epoch  92/100  train 153.7702  val 148.7072
  epoch  93/100  train 148.7583  val 148.2720
  epoch  94/100  train 144.4513  val 139.1995
  epoch  95/100  train 139.6759  val 136.3104
  epoch  96/100  train 135.3515  val 130.2486
  epoch  97/100  train 130.6083  val 126.6673
  epoch  98/100  train 126.0316  val 120.9947
  epoch  99/100  train 121.7736  val 117.2121
  epoch 100/100  train 117.4393  val 112.6326
  training: 100 epochs, 130.0 s
  UVMD-1: recon_rel_err 0.0066  mode SNR 5.94 dB  median 3.79 ms  (numpy replica dev 2.9e-07)

[UVMD N=2]  upstream Unfolding, K=13
  trainable params: 6157 entries (12301 real DOF)
  UVMD-2 untrained: recon_rel_err 0.0379  mode SNR 14.47 dB  median 5.22 ms  (numpy replica dev 4.9e-07)
  epoch   1/100  train 1261.6394  val 1233.1273
  epoch   2/100  train 1210.6214  val 1165.8968
  epoch   3/100  train 1160.5401  val 1127.5726
  epoch   4/100  train 1135.6534  val 1117.6166
  epoch   5/100  train 1119.3473  val 1100.0256
  epoch   6/100  train 1105.9730  val 1085.5178
  epoch   7/100  train 1095.2753  val 1072.2743
  epoch   8/100  train 1081.8539  val 1064.0248
  epoch   9/100  train 1067.5268  val 1043.6537
  epoch  10/100  train 1050.8384  val 1036.4931
  epoch  11/100  train 1037.3336  val 1027.8033
  epoch  12/100  train 1025.2021  val 1009.6418
  epoch  13/100  train 1014.9580  val 999.3207
  epoch  14/100  train 1006.5675  val 990.5087
  epoch  15/100  train 996.1230  val 982.1030
  epoch  16/100  train 987.6772  val 975.2714
  epoch  17/100  train 977.6411  val 965.0379
  epoch  18/100  train 968.0227  val 957.9771
  epoch  19/100  train 959.5316  val 943.2840
  epoch  20/100  train 949.5425  val 937.0811
  epoch  21/100  train 939.1243  val 927.3687
  epoch  22/100  train 930.2591  val 916.7701
  epoch  23/100  train 922.2035  val 914.1126
  epoch  24/100  train 913.2394  val 900.5040
  epoch  25/100  train 906.5423  val 892.5225
  epoch  26/100  train 896.7977  val 886.6524
  epoch  27/100  train 889.5443  val 882.9367
  epoch  28/100  train 882.9023  val 875.6212
  epoch  29/100  train 874.7486  val 864.0564
  epoch  30/100  train 867.7268  val 856.4836
  epoch  31/100  train 860.2252  val 849.4750
  epoch  32/100  train 853.1745  val 842.6491
  epoch  33/100  train 844.9305  val 836.7775
  epoch  34/100  train 838.7250  val 828.6206
  epoch  35/100  train 831.9441  val 822.8176
  epoch  36/100  train 824.9429  val 816.8456
  epoch  37/100  train 817.7017  val 808.5243
  epoch  38/100  train 810.5372  val 800.2291
  epoch  39/100  train 804.7029  val 798.4896
  epoch  40/100  train 797.2750  val 794.1636
  epoch  41/100  train 791.6585  val 786.0591
  epoch  42/100  train 784.3076  val 775.8980
  epoch  43/100  train 778.7443  val 770.5721
  epoch  44/100  train 770.3712  val 762.1203
  epoch  45/100  train 764.3592  val 761.5322
  epoch  46/100  train 757.2807  val 753.5153
  epoch  47/100  train 751.3611  val 744.3577
  epoch  48/100  train 744.1750  val 737.6840
  epoch  49/100  train 737.2572  val 729.2444
  epoch  50/100  train 731.0286  val 728.9815
  epoch  51/100  train 723.6464  val 719.5397
  epoch  52/100  train 716.6834  val 709.9502
  epoch  53/100  train 710.1692  val 703.9198
  epoch  54/100  train 703.6033  val 702.8446
  epoch  55/100  train 697.3542  val 697.3864
  epoch  56/100  train 689.6448  val 688.2344
  epoch  57/100  train 682.3507  val 678.1892
  epoch  58/100  train 675.9894  val 679.7849
  epoch  59/100  train 671.1614  val 667.2414
  epoch  60/100  train 662.5828  val 655.5191
  epoch  61/100  train 655.5964  val 648.8755
  epoch  62/100  train 649.6622  val 648.9013
  epoch  63/100  train 643.0802  val 639.5487
  epoch  64/100  train 635.1374  val 632.2707
  epoch  65/100  train 628.2744  val 629.2460
  epoch  66/100  train 621.4463  val 621.7776
  epoch  67/100  train 615.8661  val 608.7401
  epoch  68/100  train 607.1524  val 603.9235
  epoch  69/100  train 600.8079  val 597.0035
  epoch  70/100  train 592.9270  val 593.0810
  epoch  71/100  train 586.4985  val 583.2623
  epoch  72/100  train 578.5027  val 575.6504
  epoch  73/100  train 571.3168  val 567.4862
  epoch  74/100  train 564.2335  val 561.0646
  epoch  75/100  train 557.7569  val 558.7603
  epoch  76/100  train 549.7935  val 547.0889
  epoch  77/100  train 542.7236  val 541.0295
  epoch  78/100  train 535.5759  val 535.2048
  epoch  79/100  train 529.0220  val 524.3927
  epoch  80/100  train 522.9543  val 518.9244
  epoch  81/100  train 514.7145  val 509.2947
  epoch  82/100  train 508.4683  val 505.3136
  epoch  83/100  train 500.5243  val 498.9425
  epoch  84/100  train 493.7699  val 489.2944
  epoch  85/100  train 485.6463  val 481.8531
  epoch  86/100  train 478.3571  val 476.5797
  epoch  87/100  train 471.4078  val 469.3554
  epoch  88/100  train 465.2057  val 464.9511
  epoch  89/100  train 458.5114  val 452.6582
  epoch  90/100  train 452.0677  val 446.2477
  epoch  91/100  train 444.0340  val 437.7111
  epoch  92/100  train 436.5419  val 434.8008
  epoch  93/100  train 430.2217  val 431.4624
  epoch  94/100  train 424.1666  val 418.4512
  epoch  95/100  train 415.6747  val 413.3654
  epoch  96/100  train 410.1583  val 409.8809
  epoch  97/100  train 402.0666  val 403.9472
  epoch  98/100  train 395.8178  val 388.8400
  epoch  99/100  train 388.5186  val 384.4170
  epoch 100/100  train 382.4559  val 378.0438
  training: 100 epochs, 230.6 s
  UVMD-2: recon_rel_err 0.0142  mode SNR 5.94 dB  median 5.12 ms  (numpy replica dev 3.6e-07)
```
