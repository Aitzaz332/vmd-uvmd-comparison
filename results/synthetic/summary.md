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
| UVMD-2 | 0.2223 | 0.1756 | 15.18 | 2411 | 0.2295 | 0.4667 | 0.7406 | 8.916 | 8.916 | 0.08649 | 0.04694 |

## Cost (per signal unless stated)

| method                    |   iterations_or_layers |   trainable_params |   flops_per_signal |   median_ms_per_signal |   train_time_s |   train_flops |
|:--------------------------|-----------------------:|-------------------:|-------------------:|-----------------------:|---------------:|--------------:|
| VMD (converged)           |                  117.3 |                  0 |          2.555e+07 |                16.36   |           0    |     0         |
| vmdpy package (converged) |                  117.3 |                  0 |          2.555e+07 |                18.68   |           0    |     0         |
| VMD-1                     |                    1   |                  0 |          7.762e+05 |                 0.6411 |           0    |     0         |
| VMD-2                     |                    2   |                  0 |          9.892e+05 |                 0.8765 |           0    |     0         |
| VMD-3                     |                    3   |                  0 |          1.202e+06 |                 1.12   |           0    |     0         |
| VMD-5                     |                    5   |                  0 |          1.628e+06 |                 1.67   |           0    |     0         |
| VMD-10                    |                   10   |                  0 |          2.693e+06 |                 2.962  |           0    |     0         |
| VMD-20                    |                   20   |                  0 |          4.823e+06 |                 5.689  |           0    |     0         |
| VMD-50                    |                   50   |                  0 |          1.121e+07 |                13.2    |           0    |     0         |
| VMD-100                   |                  100   |                  0 |          2.186e+07 |                23.84   |           0    |     0         |
| VMD-200                   |                  200   |                  0 |          4.316e+07 |                52.85   |           0    |     0         |
| VMD-499                   |                  499   |                  0 |          1.068e+08 |               136.4    |           0    |     0         |
| UVMD-1 untrained          |                    1   |               8196 |          7.188e+05 |                 1.39   |           0    |     0         |
| UVMD-1                    |                    1   |               8196 |          7.188e+05 |                 1.535  |          48.6  |     3.019e+10 |
| UVMD-2 untrained          |                    2   |              12292 |          8.745e+05 |                 2.027  |           0    |     0         |
| UVMD-2                    |                    2   |              12292 |          8.745e+05 |                 1.608  |          71.78 |     3.673e+10 |

Environment: {'python': '3.13.17', 'platform': 'Linux-6.18.44-fc-v153-x86_64-with-glibc2.39', 'numpy': '2.5.3', 'processor': 'x86_64', 'torch': '2.14.1+cpu', 'torch_threads': 2}


## Log
```
synthetic: 200 signals, T=1024, noise std=0.1, K=4 | train/val/test = 140/30/30

[VMD converged]
  mean iterations 117.3, recon_rel_err 0.2188, mode SNR 16.55 dB, median 16.36 ms
  vmdpy package median 18.68 ms (stores all 500 iterates)

[VMD-n sweep]
  n=  1: recon_rel_err 0.4686  mode SNR   7.22 dB  omega err 0.0823  0.64 ms
  n=  2: recon_rel_err 0.3795  mode SNR  12.33 dB  omega err 0.0818  0.88 ms
  n=  3: recon_rel_err 0.3822  mode SNR  12.58 dB  omega err 0.0813  1.12 ms
  n=  5: recon_rel_err 0.3823  mode SNR  12.59 dB  omega err 0.0802  1.67 ms
  n= 10: recon_rel_err 0.3821  mode SNR  12.59 dB  omega err 0.0776  2.96 ms
  n= 20: recon_rel_err 0.3819  mode SNR  12.59 dB  omega err 0.0724  5.69 ms
  n= 50: recon_rel_err 0.2508  mode SNR  16.00 dB  omega err 0.0430  13.20 ms
  n=100: recon_rel_err 0.2188  mode SNR  16.76 dB  omega err 0.0373  23.84 ms
  n=200: recon_rel_err 0.2188  mode SNR  16.67 dB  omega err 0.0361  52.85 ms
  n=499: recon_rel_err 0.2188  mode SNR  16.55 dB  omega err 0.0360  136.43 ms

[UVMD N=1]  upstream Unfolding, K=4
  trainable params: 4100 entries (8196 real DOF)
  UVMD-1 untrained: recon_rel_err 0.4686  mode SNR 7.22 dB  median 1.39 ms  (numpy replica dev 2.7e-07)
  epoch   1/100  train 6193.9678  val 6129.4602
  epoch   2/100  train 6169.6082  val 6115.4146
  epoch   3/100  train 6150.0251  val 6099.1465
  epoch   4/100  train 6131.5476  val 6085.6418
  epoch   5/100  train 6112.0014  val 6068.2664
  epoch   6/100  train 6093.7010  val 6051.1898
  epoch   7/100  train 6075.3298  val 6032.8272
  epoch   8/100  train 6056.3613  val 6016.6739
  epoch   9/100  train 6037.7029  val 5998.5441
  epoch  10/100  train 6018.7887  val 5980.9961
  epoch  11/100  train 5999.8520  val 5965.5309
  epoch  12/100  train 5980.6565  val 5944.7329
  epoch  13/100  train 5961.0544  val 5926.9100
  epoch  14/100  train 5942.1947  val 5907.4550
  epoch  15/100  train 5922.4060  val 5885.8097
  epoch  16/100  train 5902.3882  val 5868.3307
  epoch  17/100  train 5882.3352  val 5849.4929
  epoch  18/100  train 5861.5861  val 5829.7822
  epoch  19/100  train 5841.8174  val 5809.3222
  epoch  20/100  train 5820.5264  val 5787.6445
  epoch  21/100  train 5799.5029  val 5767.2540
  epoch  22/100  train 5778.5231  val 5745.6791
  epoch  23/100  train 5757.4149  val 5727.6574
  epoch  24/100  train 5735.6236  val 5703.9021
  epoch  25/100  train 5714.4931  val 5681.0954
  epoch  26/100  train 5691.4231  val 5659.8000
  epoch  27/100  train 5669.6307  val 5639.6275
  epoch  28/100  train 5647.2540  val 5617.1929
  epoch  29/100  train 5623.9571  val 5593.5747
  epoch  30/100  train 5601.4173  val 5569.2878
  epoch  31/100  train 5577.9096  val 5545.9847
  epoch  32/100  train 5554.3679  val 5523.8628
  epoch  33/100  train 5530.2404  val 5499.2915
  epoch  34/100  train 5506.4135  val 5475.3772
  epoch  35/100  train 5482.3344  val 5450.7357
  epoch  36/100  train 5458.0834  val 5427.9771
  epoch  37/100  train 5432.3052  val 5400.9063
  epoch  38/100  train 5407.3106  val 5375.4990
  epoch  39/100  train 5382.2497  val 5350.6705
  epoch  40/100  train 5356.4349  val 5324.5880
  epoch  41/100  train 5330.6257  val 5298.9911
  epoch  42/100  train 5304.5097  val 5272.8233
  epoch  43/100  train 5278.1454  val 5244.5000
  epoch  44/100  train 5250.5852  val 5218.7012
  epoch  45/100  train 5223.7700  val 5191.5277
  epoch  46/100  train 5195.9800  val 5164.1041
  epoch  47/100  train 5168.2962  val 5134.4298
  epoch  48/100  train 5139.6085  val 5107.5286
  epoch  49/100  train 5110.7981  val 5077.5282
  epoch  50/100  train 5081.9219  val 5051.6612
  epoch  51/100  train 5052.1841  val 5020.1669
  epoch  52/100  train 5022.6400  val 4989.4803
  epoch  53/100  train 4992.4740  val 4959.7178
  epoch  54/100  train 4961.8576  val 4930.5486
  epoch  55/100  train 4931.6341  val 4899.7509
  epoch  56/100  train 4899.8103  val 4867.8173
  epoch  57/100  train 4868.7500  val 4834.5626
  epoch  58/100  train 4835.8686  val 4802.7242
  epoch  59/100  train 4803.7296  val 4772.4604
  epoch  60/100  train 4770.5736  val 4735.6321
  epoch  61/100  train 4737.1300  val 4701.2003
  epoch  62/100  train 4703.0617  val 4669.6191
  epoch  63/100  train 4669.1471  val 4634.0776
  epoch  64/100  train 4633.2212  val 4599.0061
  epoch  65/100  train 4598.7546  val 4564.7392
  epoch  66/100  train 4562.4210  val 4529.7540
  epoch  67/100  train 4526.8498  val 4490.6827
  epoch  68/100  train 4489.3333  val 4452.7374
  epoch  69/100  train 4451.2976  val 4414.7604
  epoch  70/100  train 4412.8754  val 4379.4890
  epoch  71/100  train 4374.7615  val 4338.6781
  epoch  72/100  train 4334.6374  val 4297.4184
  epoch  73/100  train 4295.0579  val 4257.3028
  epoch  74/100  train 4254.7362  val 4215.4638
  epoch  75/100  train 4213.1445  val 4176.0033
  epoch  76/100  train 4171.0459  val 4134.3552
  epoch  77/100  train 4128.5386  val 4090.1241
  epoch  78/100  train 4084.5658  val 4048.3254
  epoch  79/100  train 4040.9590  val 4003.1956
  epoch  80/100  train 3996.6490  val 3957.1685
  epoch  81/100  train 3950.8567  val 3911.3489
  epoch  82/100  train 3904.4954  val 3864.4540
  epoch  83/100  train 3856.8534  val 3814.8758
  epoch  84/100  train 3808.6116  val 3768.0147
  epoch  85/100  train 3759.5995  val 3717.7603
  epoch  86/100  train 3710.1472  val 3669.0107
  epoch  87/100  train 3658.5561  val 3619.0947
  epoch  88/100  train 3606.3993  val 3566.2905
  epoch  89/100  train 3554.9356  val 3509.3973
  epoch  90/100  train 3501.5361  val 3456.1902
  epoch  91/100  train 3445.5837  val 3400.8272
  epoch  92/100  train 3388.8339  val 3347.1496
  epoch  93/100  train 3332.4536  val 3291.6486
  epoch  94/100  train 3274.2859  val 3230.1112
  epoch  95/100  train 3214.7850  val 3171.3885
  epoch  96/100  train 3154.2300  val 3109.0636
  epoch  97/100  train 3091.4446  val 3048.9592
  epoch  98/100  train 3028.5944  val 2981.9291
  epoch  99/100  train 2963.0978  val 2915.6104
  epoch 100/100  train 2897.5507  val 2848.2992
  training: 100 epochs, 48.6 s
  UVMD-1: recon_rel_err 0.2206  mode SNR 6.35 dB  median 1.53 ms  (numpy replica dev 3.2e-07)

[UVMD N=2]  upstream Unfolding, K=4
  trainable params: 6148 entries (12292 real DOF)
  UVMD-2 untrained: recon_rel_err 0.3795  mode SNR 12.33 dB  median 2.03 ms  (numpy replica dev 7.8e-07)
  epoch   1/100  train 4603.9984  val 4634.4684
  epoch   2/100  train 4585.5219  val 4626.4411
  epoch   3/100  train 4572.6429  val 4615.8129
  epoch   4/100  train 4560.4274  val 4610.0670
  epoch   5/100  train 4547.6380  val 4599.2238
  epoch   6/100  train 4535.6512  val 4586.9176
  epoch   7/100  train 4523.8114  val 4574.7131
  epoch   8/100  train 4511.1758  val 4564.7133
  epoch   9/100  train 4499.3817  val 4553.8406
  epoch  10/100  train 4486.6583  val 4545.3382
  epoch  11/100  train 4474.5050  val 4535.4424
  epoch  12/100  train 4462.1500  val 4518.3244
  epoch  13/100  train 4449.5286  val 4506.3643
  epoch  14/100  train 4437.2507  val 4494.5588
  epoch  15/100  train 4424.2562  val 4480.0232
  epoch  16/100  train 4411.6627  val 4470.8965
  epoch  17/100  train 4398.3107  val 4455.4711
  epoch  18/100  train 4384.5813  val 4445.2901
  epoch  19/100  train 4371.9179  val 4428.9682
  epoch  20/100  train 4357.4876  val 4414.8499
  epoch  21/100  train 4343.9642  val 4402.7457
  epoch  22/100  train 4330.0198  val 4385.9237
  epoch  23/100  train 4316.1687  val 4374.9764
  epoch  24/100  train 4301.5153  val 4358.2518
  epoch  25/100  train 4288.0058  val 4343.2837
  epoch  26/100  train 4271.7774  val 4328.0276
  epoch  27/100  train 4258.3032  val 4313.8629
  epoch  28/100  train 4242.7927  val 4301.8081
  epoch  29/100  train 4227.4376  val 4283.7822
  epoch  30/100  train 4212.8101  val 4267.2939
  epoch  31/100  train 4196.7713  val 4250.0502
  epoch  32/100  train 4181.0038  val 4237.0415
  epoch  33/100  train 4164.4220  val 4220.0963
  epoch  34/100  train 4149.3733  val 4203.5401
  epoch  35/100  train 4133.0871  val 4186.7225
  epoch  36/100  train 4117.1127  val 4173.3520
  epoch  37/100  train 4100.1120  val 4152.9828
  epoch  38/100  train 4083.9071  val 4135.1912
  epoch  39/100  train 4067.4776  val 4120.9996
  epoch  40/100  train 4050.3181  val 4103.0819
  epoch  41/100  train 4033.6772  val 4084.6446
  epoch  42/100  train 4016.9211  val 4066.9766
  epoch  43/100  train 3999.2452  val 4049.3197
  epoch  44/100  train 3981.1635  val 4031.4543
  epoch  45/100  train 3963.7590  val 4012.8770
  epoch  46/100  train 3945.0056  val 3997.1849
  epoch  47/100  train 3927.8210  val 3974.2605
  epoch  48/100  train 3908.8304  val 3956.4543
  epoch  49/100  train 3889.3372  val 3935.9200
  epoch  50/100  train 3870.4699  val 3921.0698
  epoch  51/100  train 3850.9605  val 3897.6949
  epoch  52/100  train 3831.7864  val 3876.6964
  epoch  53/100  train 3811.5940  val 3859.0076
  epoch  54/100  train 3791.9751  val 3836.5742
  epoch  55/100  train 3772.3470  val 3819.3318
  epoch  56/100  train 3750.8689  val 3797.9481
  epoch  57/100  train 3731.0233  val 3773.0813
  epoch  58/100  train 3709.0087  val 3752.6148
  epoch  59/100  train 3688.1769  val 3733.8768
  epoch  60/100  train 3666.0109  val 3705.6884
  epoch  61/100  train 3643.9901  val 3682.0342
  epoch  62/100  train 3621.6869  val 3664.8643
  epoch  63/100  train 3599.4762  val 3640.2126
  epoch  64/100  train 3575.4510  val 3615.5937
  epoch  65/100  train 3552.4500  val 3594.0568
  epoch  66/100  train 3528.7941  val 3571.3597
  epoch  67/100  train 3506.0243  val 3540.3121
  epoch  68/100  train 3480.5625  val 3513.7851
  epoch  69/100  train 3454.6375  val 3488.9396
  epoch  70/100  train 3429.3016  val 3470.4577
  epoch  71/100  train 3404.0325  val 3437.8774
  epoch  72/100  train 3377.0049  val 3408.6589
  epoch  73/100  train 3350.8762  val 3383.4318
  epoch  74/100  train 3323.8196  val 3351.5949
  epoch  75/100  train 3296.2580  val 3329.9021
  epoch  76/100  train 3267.9312  val 3300.1282
  epoch  77/100  train 3239.6577  val 3267.7999
  epoch  78/100  train 3209.5044  val 3242.2100
  epoch  79/100  train 3181.1546  val 3208.6689
  epoch  80/100  train 3151.6065  val 3175.6638
  epoch  81/100  train 3120.5518  val 3145.4034
  epoch  82/100  train 3089.6268  val 3114.7846
  epoch  83/100  train 3056.9795  val 3079.2011
  epoch  84/100  train 3024.5039  val 3046.9627
  epoch  85/100  train 2990.9644  val 3011.5465
  epoch  86/100  train 2958.3985  val 2978.7967
  epoch  87/100  train 2923.1056  val 2943.1800
  epoch  88/100  train 2887.4581  val 2907.8837
  epoch  89/100  train 2852.5521  val 2865.2881
  epoch  90/100  train 2817.2526  val 2828.0335
  epoch  91/100  train 2778.4304  val 2791.6983
  epoch  92/100  train 2738.9013  val 2756.4280
  epoch  93/100  train 2700.4407  val 2717.6647
  epoch  94/100  train 2661.1788  val 2671.9958
  epoch  95/100  train 2620.0458  val 2631.3519
  epoch  96/100  train 2578.4568  val 2588.9257
  epoch  97/100  train 2534.4780  val 2546.8289
  epoch  98/100  train 2491.5745  val 2494.8510
  epoch  99/100  train 2445.2810  val 2447.2769
  epoch 100/100  train 2400.2847  val 2401.5212
  training: 100 epochs, 71.8 s
  UVMD-2: recon_rel_err 0.1756  mode SNR 8.92 dB  median 1.61 ms  (numpy replica dev 2.8e-07)
```
