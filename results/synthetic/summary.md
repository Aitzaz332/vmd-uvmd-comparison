# Synthetic benchmark - VMD vs UVMD

200 signals, T=1024, 4 true modes, noise std 0.1, K=4, alpha=2000.0, tau=0.0. Test set: 30 unseen signals. Mean over test signals.

## Accuracy

| method | recon_rmse | recon_rel_err | recon_snr_db | spec_l1 | denoise_rmse | mean_mode_nrmse | mean_mode_corr | mean_mode_snr_db | grouped_mean_snr_db | mean_omega_err | ortho_index |
|---|---|---|---|---|---|---|---|---|---|---|---|
| VMD (converged) | 0.2733 | 0.2188 | 15.96 | 3839 | 0.2623 | 0.261 | 0.8554 | 16.55 | 17.48 | 0.03601 | 0.00698 |
| VMD-1 | 0.5917 | 0.4686 | 6.622 | 6405 | 0.5899 | 0.5337 | 0.7551 | 7.217 | 7.219 | 0.08229 | 0.01289 |
| VMD-2 | 0.48 | 0.3795 | 8.49 | 4693 | 0.4778 | 0.3676 | 0.7613 | 12.33 | 12.33 | 0.08177 | 0.01393 |
| UVMD-1 untrained | 0.5917 | 0.4686 | 6.622 | 6405 | 0.5899 | 0.5337 | 0.7551 | 7.217 | 7.219 | 0.08229 | 0.01289 |
| UVMD-1 | 0.2788 | 0.2206 | 13.18 | 2973 | 0.2854 | 0.5417 | 0.7501 | 6.353 | 6.348 | 0.08213 | 0.05567 |
| UVMD-2 untrained | 0.48 | 0.3795 | 8.49 | 4693 | 0.4778 | 0.3676 | 0.7613 | 12.33 | 12.33 | 0.08177 | 0.01393 |
| UVMD-2 | 0.2223 | 0.1756 | 15.18 | 2411 | 0.2295 | 0.4667 | 0.7406 | 8.916 | 8.916 | 0.08648 | 0.04695 |

## Cost (per signal unless stated)

| method                    |   iterations_or_layers |   trainable_params |   flops_per_signal |   median_ms_per_signal |   train_time_s |   train_flops |
|:--------------------------|-----------------------:|-------------------:|-------------------:|-----------------------:|---------------:|--------------:|
| VMD (converged)           |                  117.3 |                  0 |          2.555e+07 |                35.59   |            0   |     0         |
| vmdpy package (converged) |                  117.3 |                  0 |          2.555e+07 |                51.39   |            0   |     0         |
| VMD-1                     |                    1   |                  0 |          7.762e+05 |                 0.6248 |            0   |     0         |
| VMD-2                     |                    2   |                  0 |          9.892e+05 |                 0.9271 |            0   |     0         |
| VMD-3                     |                    3   |                  0 |          1.202e+06 |                 1.228  |            0   |     0         |
| VMD-5                     |                    5   |                  0 |          1.628e+06 |                 1.84   |            0   |     0         |
| VMD-10                    |                   10   |                  0 |          2.693e+06 |                 3.386  |            0   |     0         |
| VMD-20                    |                   20   |                  0 |          4.823e+06 |                14.41   |            0   |     0         |
| VMD-50                    |                   50   |                  0 |          1.121e+07 |                31.23   |            0   |     0         |
| VMD-100                   |                  100   |                  0 |          2.186e+07 |                60.39   |            0   |     0         |
| VMD-200                   |                  200   |                  0 |          4.316e+07 |               119.9    |            0   |     0         |
| VMD-499                   |                  499   |                  0 |          1.068e+08 |               295      |            0   |     0         |
| UVMD-1 untrained          |                    1   |               8196 |          7.188e+05 |                27.87   |            0   |     0         |
| UVMD-1                    |                    1   |               8196 |          7.188e+05 |                12.79   |          497.8 |     3.019e+10 |
| UVMD-2 untrained          |                    2   |              12292 |          8.745e+05 |                12.82   |            0   |     0         |
| UVMD-2                    |                    2   |              12292 |          8.745e+05 |                 2.471  |          181.6 |     3.673e+10 |

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'numpy': '2.1.3', 'processor': 'x86_64', 'torch': '2.7.0', 'torch_threads': 24}


## Log
```
synthetic: 200 signals, T=1024, noise std=0.1, K=4 | train/val/test = 140/30/30

[VMD converged]
  mean iterations 117.3, recon_rel_err 0.2188, mode SNR 16.55 dB, median 35.59 ms
  vmdpy package median 51.39 ms (stores all 500 iterates)

[VMD-n sweep]
  n=  1: recon_rel_err 0.4686  mode SNR   7.22 dB  omega err 0.0823  0.62 ms
  n=  2: recon_rel_err 0.3795  mode SNR  12.33 dB  omega err 0.0818  0.93 ms
  n=  3: recon_rel_err 0.3822  mode SNR  12.58 dB  omega err 0.0813  1.23 ms
  n=  5: recon_rel_err 0.3823  mode SNR  12.59 dB  omega err 0.0802  1.84 ms
  n= 10: recon_rel_err 0.3821  mode SNR  12.59 dB  omega err 0.0776  3.39 ms
  n= 20: recon_rel_err 0.3819  mode SNR  12.59 dB  omega err 0.0724  14.41 ms
  n= 50: recon_rel_err 0.2508  mode SNR  16.00 dB  omega err 0.0430  31.23 ms
  n=100: recon_rel_err 0.2188  mode SNR  16.76 dB  omega err 0.0373  60.39 ms
  n=200: recon_rel_err 0.2188  mode SNR  16.67 dB  omega err 0.0361  119.88 ms
  n=499: recon_rel_err 0.2188  mode SNR  16.55 dB  omega err 0.0360  294.96 ms

[UVMD N=1]  upstream Unfolding, K=4
  trainable params: 4100 entries (8196 real DOF)
  UVMD-1 untrained: recon_rel_err 0.4686  mode SNR 7.22 dB  median 27.87 ms  (numpy replica dev 4.9e-07)
  epoch   1/100  train 6193.9655  val 6129.4467
  epoch   2/100  train 6169.6292  val 6115.3706
  epoch   3/100  train 6150.0178  val 6099.1222
  epoch   4/100  train 6131.5447  val 6085.6350
  epoch   5/100  train 6112.0058  val 6068.3264
  epoch   6/100  train 6093.7082  val 6051.2518
  epoch   7/100  train 6075.2649  val 6032.8954
  epoch   8/100  train 6056.3602  val 6016.8065
  epoch   9/100  train 6037.6984  val 5998.4863
  epoch  10/100  train 6018.7961  val 5981.1443
  epoch  11/100  train 5999.8597  val 5965.6225
  epoch  12/100  train 5980.6828  val 5944.7681
  epoch  13/100  train 5961.0847  val 5926.9279
  epoch  14/100  train 5942.2203  val 5907.4770
  epoch  15/100  train 5922.4191  val 5885.8767
  epoch  16/100  train 5902.3881  val 5868.4759
  epoch  17/100  train 5882.3641  val 5849.4896
  epoch  18/100  train 5861.5767  val 5829.6060
  epoch  19/100  train 5841.8048  val 5809.3662
  epoch  20/100  train 5820.4909  val 5787.7029
  epoch  21/100  train 5799.5484  val 5767.3164
  epoch  22/100  train 5778.5076  val 5745.6645
  epoch  23/100  train 5757.4471  val 5727.6369
  epoch  24/100  train 5735.6017  val 5703.8201
  epoch  25/100  train 5714.4609  val 5681.0376
  epoch  26/100  train 5691.3953  val 5659.7943
  epoch  27/100  train 5669.6109  val 5639.7264
  epoch  28/100  train 5647.2632  val 5617.1638
  epoch  29/100  train 5623.9397  val 5593.6715
  epoch  30/100  train 5601.4368  val 5569.2263
  epoch  31/100  train 5577.9626  val 5546.0745
  epoch  32/100  train 5554.3660  val 5523.9720
  epoch  33/100  train 5530.2435  val 5499.3256
  epoch  34/100  train 5506.4533  val 5475.2925
  epoch  35/100  train 5482.3099  val 5450.7112
  epoch  36/100  train 5458.0311  val 5428.1092
  epoch  37/100  train 5432.2806  val 5400.8041
  epoch  38/100  train 5407.3059  val 5375.3945
  epoch  39/100  train 5382.2438  val 5350.6631
  epoch  40/100  train 5356.4313  val 5324.6725
  epoch  41/100  train 5330.6334  val 5298.8649
  epoch  42/100  train 5304.5187  val 5272.7836
  epoch  43/100  train 5278.1687  val 5244.5941
  epoch  44/100  train 5250.5857  val 5218.6976
  epoch  45/100  train 5223.7241  val 5191.5966
  epoch  46/100  train 5195.9657  val 5163.9105
  epoch  47/100  train 5168.2516  val 5134.4721
  epoch  48/100  train 5139.6164  val 5107.5264
  epoch  49/100  train 5110.8365  val 5077.5039
  epoch  50/100  train 5081.8896  val 5051.6344
  epoch  51/100  train 5052.1685  val 5020.3247
  epoch  52/100  train 5022.6205  val 4989.6455
  epoch  53/100  train 4992.4881  val 4959.7566
  epoch  54/100  train 4961.8576  val 4930.5834
  epoch  55/100  train 4931.6517  val 4899.7011
  epoch  56/100  train 4899.8068  val 4867.8213
  epoch  57/100  train 4868.7243  val 4834.6042
  epoch  58/100  train 4835.9071  val 4802.7889
  epoch  59/100  train 4803.7386  val 4772.3573
  epoch  60/100  train 4770.5910  val 4735.7035
  epoch  61/100  train 4737.1451  val 4701.3703
  epoch  62/100  train 4703.0220  val 4669.4268
  epoch  63/100  train 4669.1798  val 4634.1506
  epoch  64/100  train 4633.2056  val 4598.8899
  epoch  65/100  train 4598.7648  val 4564.8601
  epoch  66/100  train 4562.4187  val 4529.8414
  epoch  67/100  train 4526.8497  val 4490.7191
  epoch  68/100  train 4489.3454  val 4452.8155
  epoch  69/100  train 4451.3286  val 4414.7364
  epoch  70/100  train 4412.8920  val 4379.4056
  epoch  71/100  train 4374.7645  val 4338.6461
  epoch  72/100  train 4334.6670  val 4297.3686
  epoch  73/100  train 4295.0212  val 4257.4064
  epoch  74/100  train 4254.7077  val 4215.5918
  epoch  75/100  train 4213.1383  val 4176.1387
  epoch  76/100  train 4171.0605  val 4134.3832
  epoch  77/100  train 4128.6229  val 4090.2707
  epoch  78/100  train 4084.5689  val 4048.1902
  epoch  79/100  train 4041.0024  val 4003.2409
  epoch  80/100  train 3996.6302  val 3957.1575
  epoch  81/100  train 3950.8580  val 3911.3964
  epoch  82/100  train 3904.4612  val 3864.3888
  epoch  83/100  train 3856.8801  val 3814.8943
  epoch  84/100  train 3808.6281  val 3767.9133
  epoch  85/100  train 3759.6300  val 3717.7095
  epoch  86/100  train 3710.1276  val 3668.9348
  epoch  87/100  train 3658.5772  val 3619.1460
  epoch  88/100  train 3606.4304  val 3566.1783
  epoch  89/100  train 3554.9505  val 3509.1955
  epoch  90/100  train 3501.5588  val 3456.3781
  epoch  91/100  train 3445.5621  val 3400.9272
  epoch  92/100  train 3388.8458  val 3347.1779
  epoch  93/100  train 3332.4644  val 3291.7995
  epoch  94/100  train 3274.2908  val 3229.9479
  epoch  95/100  train 3214.7869  val 3171.5942
  epoch  96/100  train 3154.2703  val 3109.1716
  epoch  97/100  train 3091.4456  val 3049.0700
  epoch  98/100  train 3028.5262  val 2982.1700
  epoch  99/100  train 2963.0736  val 2915.5933
  epoch 100/100  train 2897.5403  val 2848.5194
  training: 100 epochs, 497.8 s
  UVMD-1: recon_rel_err 0.2206  mode SNR 6.35 dB  median 12.79 ms  (numpy replica dev 4.2e-07)

[UVMD N=2]  upstream Unfolding, K=4
  trainable params: 6148 entries (12292 real DOF)
  UVMD-2 untrained: recon_rel_err 0.3795  mode SNR 12.33 dB  median 12.82 ms  (numpy replica dev 5.9e-07)
  epoch   1/100  train 4603.9979  val 4634.4768
  epoch   2/100  train 4585.5225  val 4626.4254
  epoch   3/100  train 4572.6302  val 4615.8059
  epoch   4/100  train 4560.4253  val 4610.0619
  epoch   5/100  train 4547.6347  val 4599.1709
  epoch   6/100  train 4535.6661  val 4586.9501
  epoch   7/100  train 4523.8145  val 4574.5589
  epoch   8/100  train 4511.1548  val 4564.7174
  epoch   9/100  train 4499.3892  val 4553.8491
  epoch  10/100  train 4486.6439  val 4545.3244
  epoch  11/100  train 4474.5206  val 4535.3872
  epoch  12/100  train 4462.1047  val 4518.3270
  epoch  13/100  train 4449.5370  val 4506.3718
  epoch  14/100  train 4437.2717  val 4494.4789
  epoch  15/100  train 4424.2929  val 4479.9865
  epoch  16/100  train 4411.6599  val 4470.9290
  epoch  17/100  train 4398.3291  val 4455.6192
  epoch  18/100  train 4384.5845  val 4445.2440
  epoch  19/100  train 4371.9308  val 4428.9402
  epoch  20/100  train 4357.4882  val 4414.8722
  epoch  21/100  train 4343.9468  val 4402.7900
  epoch  22/100  train 4330.0131  val 4385.8176
  epoch  23/100  train 4316.1529  val 4374.9637
  epoch  24/100  train 4301.5221  val 4358.2510
  epoch  25/100  train 4287.9906  val 4343.2973
  epoch  26/100  train 4271.7900  val 4327.9877
  epoch  27/100  train 4258.3125  val 4313.9180
  epoch  28/100  train 4242.7956  val 4301.7142
  epoch  29/100  train 4227.4095  val 4283.7694
  epoch  30/100  train 4212.7820  val 4267.2582
  epoch  31/100  train 4196.7887  val 4250.0620
  epoch  32/100  train 4181.0108  val 4236.9991
  epoch  33/100  train 4164.3852  val 4220.0609
  epoch  34/100  train 4149.3737  val 4203.5615
  epoch  35/100  train 4133.0897  val 4186.6271
  epoch  36/100  train 4117.1037  val 4173.2581
  epoch  37/100  train 4100.1403  val 4153.0133
  epoch  38/100  train 4083.9246  val 4135.1827
  epoch  39/100  train 4067.4153  val 4120.9074
  epoch  40/100  train 4050.3142  val 4103.0156
  epoch  41/100  train 4033.6811  val 4084.6206
  epoch  42/100  train 4016.9145  val 4067.0022
  epoch  43/100  train 3999.2587  val 4049.3546
  epoch  44/100  train 3981.1611  val 4031.5187
  epoch  45/100  train 3963.7367  val 4012.8488
  epoch  46/100  train 3945.0249  val 3997.1092
  epoch  47/100  train 3927.7866  val 3974.3308
  epoch  48/100  train 3908.8344  val 3956.4364
  epoch  49/100  train 3889.3676  val 3935.9201
  epoch  50/100  train 3870.4528  val 3921.1049
  epoch  51/100  train 3850.9432  val 3897.6508
  epoch  52/100  train 3831.7686  val 3876.6980
  epoch  53/100  train 3811.6172  val 3859.0499
  epoch  54/100  train 3791.9419  val 3836.5949
  epoch  55/100  train 3772.3566  val 3819.3204
  epoch  56/100  train 3750.8627  val 3798.0043
  epoch  57/100  train 3731.0359  val 3773.1618
  epoch  58/100  train 3709.0232  val 3752.5015
  epoch  59/100  train 3688.2417  val 3733.8281
  epoch  60/100  train 3666.0071  val 3705.5626
  epoch  61/100  train 3643.9869  val 3681.9819
  epoch  62/100  train 3621.6905  val 3664.7652
  epoch  63/100  train 3599.4432  val 3640.0788
  epoch  64/100  train 3575.4585  val 3615.6466
  epoch  65/100  train 3552.4221  val 3594.0524
  epoch  66/100  train 3528.8238  val 3571.2644
  epoch  67/100  train 3506.0115  val 3540.3607
  epoch  68/100  train 3480.5883  val 3513.6505
  epoch  69/100  train 3454.6278  val 3488.9138
  epoch  70/100  train 3429.3232  val 3470.5503
  epoch  71/100  train 3404.0101  val 3437.9110
  epoch  72/100  train 3376.9941  val 3408.7256
  epoch  73/100  train 3350.8775  val 3383.4763
  epoch  74/100  train 3323.8411  val 3351.5137
  epoch  75/100  train 3296.2482  val 3329.9510
  epoch  76/100  train 3267.9447  val 3300.0155
  epoch  77/100  train 3239.6477  val 3267.9223
  epoch  78/100  train 3209.5215  val 3242.2188
  epoch  79/100  train 3181.1368  val 3208.8219
  epoch  80/100  train 3151.6036  val 3175.6487
  epoch  81/100  train 3120.5355  val 3145.4258
  epoch  82/100  train 3089.6852  val 3114.7237
  epoch  83/100  train 3056.9960  val 3078.9787
  epoch  84/100  train 3024.5258  val 3046.7881
  epoch  85/100  train 2990.9592  val 3011.4830
  epoch  86/100  train 2958.3298  val 2978.9160
  epoch  87/100  train 2923.1133  val 2943.0937
  epoch  88/100  train 2887.4700  val 2908.0174
  epoch  89/100  train 2852.5295  val 2865.4059
  epoch  90/100  train 2817.2616  val 2828.0781
  epoch  91/100  train 2778.4331  val 2791.6529
  epoch  92/100  train 2738.9155  val 2756.4500
  epoch  93/100  train 2700.4392  val 2717.7037
  epoch  94/100  train 2661.1806  val 2672.0186
  epoch  95/100  train 2620.0247  val 2631.3898
  epoch  96/100  train 2578.4290  val 2588.9684
  epoch  97/100  train 2534.4470  val 2546.8882
  epoch  98/100  train 2491.5627  val 2494.9382
  epoch  99/100  train 2445.2837  val 2447.2081
  epoch 100/100  train 2400.2811  val 2401.3870
  training: 100 epochs, 181.6 s
  UVMD-2: recon_rel_err 0.1756  mode SNR 8.92 dB  median 2.47 ms  (numpy replica dev 4.4e-07)
```
