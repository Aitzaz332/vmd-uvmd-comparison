# Traffic benchmark - VMD vs UVMD (sd)

716 nodes x 35040 samples (365 days at 15 min), standardised flow. Node split train/val/test = 100/20/30 (test nodes scored). K=13, alpha=2000.0, tau=0.0, UVMD device cuda:0.

## Accuracy and mode structure (mean over test nodes)

| method | recon_rel_err | recon_snr_db | spec_l1 | spec_l2 | energy_ratio | ortho_index | mean_bandwidth_cpd | mode_overlap | n_active_modes | daily_ownership | daily_capture | weekly_ownership | weekly_capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VMD (converged) | 0.03027 | 31.24 | 1.644e+05 | 1475 | 0.9636 | 0.01372 | 0.1021 | 0.004042 | 3.833 | 1 | 0.999 | 0.9991 | 0.9216 |
| VMD-1 | 0.07286 | 23.27 | 9.838e+04 | 3399 | 0.7471 | 0.06915 | 0.4862 | 0.03578 | 1.867 | 0.9927 | 0.6802 | 1 | 0.9912 |
| VMD-2 | 0.08383 | 22.14 | 1.57e+05 | 3996 | 0.6981 | 0.08533 | 0.4467 | 0.01628 | 2.2 | 0.9714 | 0.5738 | 1 | 0.9877 |
| UVMD-1 untrained | 0.07286 | 23.27 | 9.838e+04 | 3399 | 0.7471 | 0.06915 | 0.4862 | 0.03578 | 1.867 | 0.9927 | 0.6802 | 1 | 0.9912 |
| UVMD-1 | 0.02048 | 33.96 | 3.476e+04 | 950.1 | 0.8354 | 0.0709 | 0.5245 | 0.05786 | 1.733 | 0.9955 | 0.8268 | 1 | 0.9964 |
| UVMD-2 untrained | 0.08384 | 22.14 | 1.57e+05 | 3996 | 0.6981 | 0.08533 | 0.4467 | 0.01628 | 2.2 | 0.9714 | 0.5738 | 1 | 0.9877 |
| UVMD-2 | 0.03723 | 29.01 | 7.989e+04 | 1768 | 0.7864 | 0.08062 | 0.4989 | 0.02886 | 2.033 | 0.9898 | 0.7306 | 1 | 0.995 |

## Paired comparison, trained UVMD-N vs VMD-N (share of nodes where UVMD is better)

|                 |   recon_rel_err |   mode_overlap |   mean_bandwidth_cpd |
|:----------------|----------------:|---------------:|---------------------:|
| UVMD-1 vs VMD-1 |            1.00 |           0.00 |                 0.00 |
| UVMD-2 vs VMD-2 |            1.00 |           0.03 |                 0.00 |

## Cost

| method           |   iterations_or_layers |   trainable_params |   flops_per_node |   mean_s_per_node |   median_s_per_node | device   |   train_time_s |   train_flops |   est_min_region |   est_min_6902_sensors |   speedup_vs_VMD_converged |
|:-----------------|-----------------------:|-------------------:|-----------------:|------------------:|--------------------:|:---------|---------------:|--------------:|-----------------:|-----------------------:|---------------------------:|
| VMD (converged)  |                  465.3 |                  0 |        1.11e+10  |          12.8     |           13.23     | cpu      |            0   |     0         |         152.8    |               1473     |                      1     |
| VMD-1            |                    1   |                  0 |        1.027e+08 |           0.06084 |            0.06025  | cpu      |            0   |     0         |           0.7261 |                  6.999 |                    210.4   |
| VMD-2            |                    2   |                  0 |        1.263e+08 |           0.08816 |            0.08821  | cpu      |            0   |     0         |           1.052  |                 10.14  |                    145.2   |
| VMD-5            |                    5   |                  0 |        1.974e+08 |           0.168   |            0.1666   | cpu      |            0   |     0         |           2.005  |                 19.33  |                     76.19  |
| VMD-10           |                   10   |                  0 |        3.158e+08 |           0.2983  |            0.2986   | cpu      |            0   |     0         |           3.56   |                 34.32  |                     42.91  |
| VMD-20           |                   20   |                  0 |        5.527e+08 |           0.5661  |            0.5635   | cpu      |            0   |     0         |           6.755  |                 65.11  |                     22.61  |
| VMD-50           |                   50   |                  0 |        1.263e+09 |           1.357   |            1.355    | cpu      |            0   |     0         |          16.19   |                156.1   |                      9.436 |
| VMD-100          |                  100   |                  0 |        2.448e+09 |           3.011   |            2.684    | cpu      |            0   |     0         |          35.93   |                346.3   |                      4.252 |
| VMD-200          |                  200   |                  0 |        4.816e+09 |           5.437   |            5.328    | cpu      |            0   |     0         |          64.89   |                625.5   |                      2.354 |
| UVMD-1 untrained |                    1   |             280333 |        9.627e+07 |           0.0671  |            0.07588  | cuda:0   |            0   |     0         |           0.8008 |                  7.719 |                    190.8   |
| UVMD-1           |                    1   |             280333 |        9.627e+07 |           0.02768 |            0.008113 | cuda:0   |          181.9 |     2.888e+12 |           0.3304 |                  3.185 |                    462.4   |
| UVMD-1 (CPU)     |                    1   |             280333 |        9.627e+07 |           0.05677 |            0.04066  | cpu      |            0   |     0         |           0.6775 |                  6.531 |                    225.5   |
| UVMD-2 untrained |                    2   |             420493 |        1.136e+08 |           0.06582 |            0.07431  | cuda:0   |            0   |     0         |           0.7854 |                  7.571 |                    194.5   |
| UVMD-2           |                    2   |             420493 |        1.136e+08 |           0.05292 |            0.06587  | cuda:0   |          479.1 |     3.408e+12 |           0.6315 |                  6.087 |                    241.9   |
| UVMD-2 (CPU)     |                    2   |             420493 |        1.136e+08 |           0.08291 |            0.0667   | cpu      |            0   |     0         |           0.9894 |                  9.537 |                    154.4   |

`est_min_*` = mean seconds per node x number of nodes (an extrapolation, not a measurement). 6902 = GBA+GLA+SD sensors of the paper. VMD is NumPy on CPU; UVMD runs on the device above.

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'numpy': '2.1.3', 'torch': '2.7.0', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060'}


## Log
```
traffic [sd_his_2019.h5]: 716 nodes x 35040 samples (365 days) | node split train/val/test = 100/20/30 | K=13, alpha=2000.0, device cuda:0

[VMD converged] (NumPy, CPU)
  5/30 nodes, 11.89 s/node, mean 446 iterations
  10/30 nodes, 13.75 s/node, mean 473 iterations
  15/30 nodes, 13.43 s/node, mean 474 iterations
  20/30 nodes, 12.77 s/node, mean 457 iterations
  25/30 nodes, 12.87 s/node, mean 464 iterations
  30/30 nodes, 12.80 s/node, mean 465 iterations
  rel err 0.0303, mode overlap 0.004, daily-line ownership 1.00, 30% converged

[VMD-n sweep]
  n=  1 (30 nodes): rel err 0.0729  overlap 0.036  0.061 s/node
  n=  2 (30 nodes): rel err 0.0838  overlap 0.016  0.088 s/node
  n=  5 (10 nodes): rel err 0.0577  overlap 0.012  0.168 s/node
  n= 10 (10 nodes): rel err 0.0481  overlap 0.010  0.298 s/node
  n= 20 (10 nodes): rel err 0.0403  overlap 0.007  0.566 s/node
  n= 50 (10 nodes): rel err 0.0332  overlap 0.005  1.357 s/node
  n=100 (10 nodes): rel err 0.0326  overlap 0.005  3.011 s/node
  n=200 (10 nodes): rel err 0.0299  overlap 0.004  5.437 s/node

[UVMD N=1] upstream Unfolding, K=13, device cuda:0
  trainable parameters: 140173 entries (280333 real DOF)
  UVMD-1 untrained: rel err 0.0729  overlap 0.036  daily-line ownership 0.99  0.067 s/node
  epoch   1/100  train 79964.2249  val 81388.2183
  epoch   2/100  train 76813.1992  val 79149.2864
  epoch   3/100  train 74986.6888  val 77414.5271
  epoch   4/100  train 73595.2875  val 76094.4584
  epoch   5/100  train 72510.5280  val 75031.3031
  epoch   6/100  train 71614.7436  val 73896.0037
  epoch   7/100  train 70766.1980  val 73071.0654
  epoch   8/100  train 70007.9466  val 72211.3945
  epoch   9/100  train 69327.6525  val 71459.0237
  epoch  10/100  train 68696.4116  val 70751.1057
  epoch  11/100  train 68004.3691  val 70158.6318
  epoch  12/100  train 67414.6248  val 69523.7407
  epoch  13/100  train 66827.4461  val 68866.4295
  epoch  14/100  train 66245.8868  val 68230.6326
  epoch  15/100  train 65669.4819  val 67617.6151
  epoch  16/100  train 65105.1858  val 67005.2136
  epoch  17/100  train 64552.1703  val 66502.0835
  epoch  18/100  train 64012.1526  val 66002.3428
  epoch  19/100  train 63472.5604  val 65332.2848
  epoch  20/100  train 62929.5581  val 64773.0785
  epoch  21/100  train 62398.8297  val 64198.6761
  epoch  22/100  train 61881.7176  val 63653.6605
  epoch  23/100  train 61367.9576  val 63176.4743
  epoch  24/100  train 60865.8809  val 62566.5932
  epoch  25/100  train 60346.5025  val 61999.0127
  epoch  26/100  train 59882.4228  val 61503.8941
  epoch  27/100  train 59351.7498  val 60923.6103
  epoch  28/100  train 58851.1535  val 60437.6904
  epoch  29/100  train 58354.4363  val 59941.8579
  epoch  30/100  train 57854.9473  val 59415.7328
  epoch  31/100  train 57400.0404  val 58911.4697
  epoch  32/100  train 56874.0741  val 58458.4912
  epoch  33/100  train 56385.3743  val 57894.6449
  epoch  34/100  train 55934.7007  val 57407.6464
  epoch  35/100  train 55416.8035  val 56876.5656
  epoch  36/100  train 54938.4717  val 56460.8568
  epoch  37/100  train 54497.4154  val 55854.4326
  epoch  38/100  train 54004.7463  val 55376.1012
  epoch  39/100  train 53533.2956  val 54938.0172
  epoch  40/100  train 53044.2286  val 54403.5271
  epoch  41/100  train 52574.1440  val 53900.7826
  epoch  42/100  train 52129.4304  val 53495.7727
  epoch  43/100  train 51675.5251  val 53069.2281
  epoch  44/100  train 51207.7443  val 52448.8402
  epoch  45/100  train 50734.2292  val 52040.1221
  epoch  46/100  train 50282.4375  val 51512.7910
  epoch  47/100  train 49836.2409  val 51017.0495
  epoch  48/100  train 49357.2982  val 50517.2303
  epoch  49/100  train 48914.5485  val 50118.1572
  epoch  50/100  train 48444.6110  val 49663.9679
  epoch  51/100  train 47999.2249  val 49160.4482
  epoch  52/100  train 47562.6220  val 48720.0792
  epoch  53/100  train 47123.2915  val 48181.2516
  epoch  54/100  train 46640.4331  val 47855.2481
  epoch  55/100  train 46205.4222  val 47287.3413
  epoch  56/100  train 45770.9195  val 46775.7929
  epoch  57/100  train 45300.8698  val 46307.4384
  epoch  58/100  train 44836.1991  val 45959.9812
  epoch  59/100  train 44427.7114  val 45408.7953
  epoch  60/100  train 43991.3079  val 44984.5434
  epoch  61/100  train 43558.0191  val 44549.7643
  epoch  62/100  train 43117.4377  val 44103.1277
  epoch  63/100  train 42704.9762  val 43656.7322
  epoch  64/100  train 42237.4817  val 43168.0781
  epoch  65/100  train 41804.8418  val 42735.8561
  epoch  66/100  train 41346.3354  val 42333.6120
  epoch  67/100  train 40945.0984  val 41827.9107
  epoch  68/100  train 40516.3005  val 41419.1524
  epoch  69/100  train 40096.7032  val 41029.3462
  epoch  70/100  train 39655.5345  val 40539.7680
  epoch  71/100  train 39233.4755  val 40170.3227
  epoch  72/100  train 38806.5239  val 39689.6679
  epoch  73/100  train 38394.0975  val 39211.8688
  epoch  74/100  train 37960.4232  val 38819.1083
  epoch  75/100  train 37564.1239  val 38375.8639
  epoch  76/100  train 37140.0002  val 37927.7148
  epoch  77/100  train 36699.3490  val 37531.1410
  epoch  78/100  train 36289.4758  val 37014.6544
  epoch  79/100  train 35891.3211  val 36573.2774
  epoch  80/100  train 35450.1550  val 36151.6484
  epoch  81/100  train 35043.6766  val 35872.6306
  epoch  82/100  train 34619.9994  val 35414.2499
  epoch  83/100  train 34219.8764  val 34954.5683
  epoch  84/100  train 33816.8693  val 34494.1129
  epoch  85/100  train 33429.3718  val 34098.5436
  epoch  86/100  train 33019.5832  val 33692.5733
  epoch  87/100  train 32611.2485  val 33361.7688
  epoch  88/100  train 32230.5234  val 32845.5077
  epoch  89/100  train 31822.2177  val 32446.9763
  epoch  90/100  train 31446.3669  val 32029.2222
  epoch  91/100  train 31042.5241  val 31640.1657
  epoch  92/100  train 30653.8319  val 31277.9950
  epoch  93/100  train 30275.6962  val 30804.6851
  epoch  94/100  train 29885.1683  val 30453.1398
  epoch  95/100  train 29490.8966  val 30083.7794
  epoch  96/100  train 29112.6928  val 29714.5517
  epoch  97/100  train 28720.7284  val 29303.1445
  epoch  98/100  train 28338.6488  val 28877.5644
  epoch  99/100  train 27959.5185  val 28475.8731
  epoch 100/100  train 27583.5305  val 28031.8988
  training: 100 epochs, 181.9 s
  UVMD-1: rel err 0.0205  overlap 0.058  daily-line ownership 1.00  0.028 s/node
  learned alpha_k: [1173, 1171, 1115, 1093, 1084, 1080, 1080, 1077, 1069, 1064, 1055, 1042, 1028]
  same trained model on CPU: 0.057 s/node (3 nodes)

[UVMD N=2] upstream Unfolding, K=13, device cuda:0
  trainable parameters: 210253 entries (420493 real DOF)
  UVMD-2 untrained: rel err 0.0838  overlap 0.016  daily-line ownership 0.97  0.066 s/node
  epoch   1/100  train 121836.3246  val 124111.6391
  epoch   2/100  train 117894.9900  val 121444.7723
  epoch   3/100  train 115686.1652  val 119326.9578
  epoch   4/100  train 113929.6398  val 117609.5896
  epoch   5/100  train 112489.1870  val 116199.0832
  epoch   6/100  train 111250.0952  val 114722.6676
  epoch   7/100  train 110083.7402  val 113591.0309
  epoch   8/100  train 109038.9944  val 112449.3109
  epoch   9/100  train 108088.3018  val 111424.6906
  epoch  10/100  train 107189.6960  val 110525.8082
  epoch  11/100  train 106253.4216  val 109656.2010
  epoch  12/100  train 105424.0542  val 108810.5113
  epoch  13/100  train 104614.8755  val 107932.2168
  epoch  14/100  train 103826.9673  val 107093.0469
  epoch  15/100  train 103045.9158  val 106297.8193
  epoch  16/100  train 102303.8907  val 105530.7182
  epoch  17/100  train 101571.4375  val 104863.0453
  epoch  18/100  train 100870.4816  val 104212.4189
  epoch  19/100  train 100181.8472  val 103386.0773
  epoch  20/100  train 99490.4871  val 102703.6434
  epoch  21/100  train 98823.8407  val 101975.0404
  epoch  22/100  train 98171.5482  val 101324.8031
  epoch  23/100  train 97537.5355  val 100728.9264
  epoch  24/100  train 96925.0187  val 100018.7699
  epoch  25/100  train 96295.3761  val 99336.9150
  epoch  26/100  train 95726.9715  val 98808.3697
  epoch  27/100  train 95090.1865  val 98101.7207
  epoch  28/100  train 94490.5301  val 97512.5787
  epoch  29/100  train 93901.4501  val 96931.7152
  epoch  30/100  train 93309.0697  val 96323.1750
  epoch  31/100  train 92768.8128  val 95740.1539
  epoch  32/100  train 92152.7459  val 95211.4922
  epoch  33/100  train 91575.8830  val 94556.3227
  epoch  34/100  train 91057.0134  val 93987.6775
  epoch  35/100  train 90457.6720  val 93365.5959
  epoch  36/100  train 89906.7777  val 92875.1266
  epoch  37/100  train 89380.6796  val 92182.4889
  epoch  38/100  train 88815.2472  val 91644.4748
  epoch  39/100  train 88274.3106  val 91127.5168
  epoch  40/100  train 87718.6341  val 90526.3340
  epoch  41/100  train 87178.9588  val 89949.2664
  epoch  42/100  train 86677.4085  val 89521.8881
  epoch  43/100  train 86162.3274  val 88995.3961
  epoch  44/100  train 85618.5953  val 88314.4355
  epoch  45/100  train 85092.4159  val 87818.1004
  epoch  46/100  train 84574.5272  val 87212.3223
  epoch  47/100  train 84067.5373  val 86687.5672
  epoch  48/100  train 83535.1306  val 86081.7531
  epoch  49/100  train 83027.3968  val 85605.2711
  epoch  50/100  train 82486.2538  val 85108.6730
  epoch  51/100  train 81984.9186  val 84570.4518
  epoch  52/100  train 81494.0780  val 83988.7914
  epoch  53/100  train 80988.7604  val 83435.8133
  epoch  54/100  train 80450.8855  val 83041.3943
  epoch  55/100  train 79966.3661  val 82400.6803
  epoch  56/100  train 79462.5833  val 81784.0648
  epoch  57/100  train 78928.7717  val 81286.3299
  epoch  58/100  train 78397.7063  val 80875.1467
  epoch  59/100  train 77922.3118  val 80252.5437
  epoch  60/100  train 77428.2061  val 79751.6195
  epoch  61/100  train 76940.3625  val 79275.5627
  epoch  62/100  train 76429.9843  val 78709.1314
  epoch  63/100  train 75962.6944  val 78221.2719
  epoch  64/100  train 75423.5521  val 77681.8955
  epoch  65/100  train 74934.1604  val 77185.8529
  epoch  66/100  train 74396.9760  val 76683.2572
  epoch  67/100  train 73954.4602  val 76149.5523
  epoch  68/100  train 73464.7675  val 75637.9428
  epoch  69/100  train 72980.2313  val 75287.9850
  epoch  70/100  train 72463.3289  val 74641.4803
  epoch  71/100  train 71989.9602  val 74225.9350
  epoch  72/100  train 71488.7910  val 73642.2869
  epoch  73/100  train 71007.4902  val 73105.7350
  epoch  74/100  train 70520.4186  val 72660.2936
  epoch  75/100  train 70067.1889  val 72159.1264
  epoch  76/100  train 69566.5066  val 71565.7092
  epoch  77/100  train 69054.6517  val 71185.8119
  epoch  78/100  train 68560.1523  val 70518.1008
  epoch  79/100  train 68101.7339  val 70037.0326
  epoch  80/100  train 67584.1146  val 69506.7258
  epoch  81/100  train 67094.2415  val 69203.7607
  epoch  82/100  train 66614.3655  val 68628.7898
  epoch  83/100  train 66115.8909  val 68124.4010
  epoch  84/100  train 65639.5698  val 67586.9787
  epoch  85/100  train 65195.4605  val 67088.6280
  epoch  86/100  train 64692.8164  val 66603.1504
  epoch  87/100  train 64211.5887  val 66200.0771
  epoch  88/100  train 63751.3374  val 65623.2455
  epoch  89/100  train 63263.5842  val 65171.4581
  epoch  90/100  train 62820.1364  val 64649.2051
  epoch  91/100  train 62326.5830  val 64142.8139
  epoch  92/100  train 61862.2173  val 63671.1950
  epoch  93/100  train 61415.1250  val 63145.8428
  epoch  94/100  train 60940.8695  val 62699.7697
  epoch  95/100  train 60458.3378  val 62233.4683
  epoch  96/100  train 59995.0026  val 61811.4764
  epoch  97/100  train 59509.3822  val 61263.9554
  epoch  98/100  train 59025.9668  val 60729.4195
  epoch  99/100  train 58578.7578  val 60313.9909
  epoch 100/100  train 58097.9035  val 59731.9365
  training: 100 epochs, 479.1 s
  UVMD-2: rel err 0.0372  overlap 0.029  daily-line ownership 0.99  0.053 s/node
  learned alpha_k: [1086, 1130, 1110, 1083, 1063, 1042, 1041, 1049, 1031, 1023, 1016, 1012, 1031]
  same trained model on CPU: 0.083 s/node (3 nodes)
```
