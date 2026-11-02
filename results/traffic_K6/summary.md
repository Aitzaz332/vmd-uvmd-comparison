# Traffic benchmark - VMD vs UVMD (sd)

716 nodes x 35040 samples (365 days at 15 min), standardised flow. Node split train/val/test = 100/107/30 (test nodes scored). K=6, alpha=500.0, tau=0.0, UVMD device cuda:0.

## Accuracy and mode structure (mean over test nodes)

| method | recon_rel_err | recon_snr_db | spec_l1 | spec_l2 | energy_ratio | ortho_index | mean_bandwidth_cpd | mode_overlap | n_active_modes | daily_ownership | daily_capture | weekly_ownership | weekly_capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VMD (converged) | 0.0455 | 27.7 | 2.526e+05 | 2237 | 0.943 | 0.02243 | 0.1879 | 0.00606 | 3.333 | 1 | 0.9979 | 0.9992 | 0.938 |
| VMD-5 | 0.04433 | 27.85 | 1.766e+05 | 2164 | 0.8311 | 0.05797 | 0.4941 | 0.01779 | 1.9 | 0.9935 | 0.8066 | 1 | 0.9959 |
| UVMD-5 untrained | 0.04433 | 27.85 | 1.766e+05 | 2164 | 0.8311 | 0.05797 | 0.4941 | 0.01779 | 1.9 | 0.9935 | 0.8066 | 1 | 0.9959 |
| UVMD-5 | 0.004688 | 48.41 | 184.1 | 1.136 | 0.8438 | 0.0782 | 0.7617 | 0.1334 | 2.133 | 0.9951 | 0.8274 | 1 | 0.996 |

## Paired comparison, trained UVMD-N vs VMD-N (share of nodes where UVMD is better)

|                 |   recon_rel_err |   mode_overlap |   mean_bandwidth_cpd |
|:----------------|----------------:|---------------:|---------------------:|
| UVMD-5 vs VMD-5 |            1.00 |           0.00 |                 0.00 |

## Cost

| method           |   iterations_or_layers |   trainable_params |   flops_per_node |   mean_s_per_node |   median_s_per_node | device   |   train_time_s |   train_flops |   est_min_region |   est_min_6902_sensors |   speedup_vs_VMD_converged |
|:-----------------|-----------------------:|-------------------:|-----------------:|------------------:|--------------------:|:---------|---------------:|--------------:|-----------------:|-----------------------:|---------------------------:|
| VMD (converged)  |                  411.6 |                  0 |        4.539e+09 |           5.745   |             5.071   | cpu      |            0   |     0         |          68.56   |                660.9   |                       1    |
| VMD-5            |                    5   |                  0 |        9.414e+07 |           0.1846  |             0.1515  | cpu      |            0   |     0         |           2.203  |                 21.24  |                      31.12 |
| UVMD-5 untrained |                    5   |             840966 |        7.943e+07 |           0.05418 |             0.06198 | cuda:0   |            0   |     0         |           0.6466 |                  6.233 |                     106    |
| UVMD-5           |                    5   |             840966 |        7.943e+07 |           0.035   |             0.01276 | cuda:0   |          572.7 |     2.383e+12 |           0.4176 |                  4.026 |                     164.2  |
| UVMD-5 (CPU)     |                    5   |             840966 |        7.943e+07 |           0.07075 |             0.04988 | cpu      |            0   |     0         |           0.8443 |                  8.139 |                      81.21 |

`est_min_*` = mean seconds per node x number of nodes (an extrapolation, not a measurement). 6902 = GBA+GLA+SD sensors of the paper. VMD is NumPy on CPU; UVMD runs on the device above.

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'numpy': '2.5.4', 'torch': '2.14.1+cu130', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060'}


## Log
```
traffic [sd_his_2019.h5]: 716 nodes x 35040 samples (365 days) | node split train/val/test = 100/107/30 | K=6, alpha=500.0, device cuda:0

[VMD converged] (NumPy, CPU)
  5/30 nodes, 4.79 s/node, mean 412 iterations
  10/30 nodes, 4.85 s/node, mean 428 iterations
  15/30 nodes, 4.85 s/node, mean 432 iterations
  20/30 nodes, 5.94 s/node, mean 422 iterations
  25/30 nodes, 6.02 s/node, mean 412 iterations
  30/30 nodes, 5.75 s/node, mean 412 iterations
  rel err 0.0455, mode overlap 0.006, daily-line ownership 1.00, 77% converged

[VMD-n sweep]
  n=  5 (30 nodes): rel err 0.0443  overlap 0.018  0.185 s/node

[UVMD N=5] upstream Unfolding, K=6, device cuda:0
  trainable parameters: 420486 entries (840966 real DOF)
  UVMD-5 untrained: rel err 0.0443  overlap 0.018  daily-line ownership 0.99  0.054 s/node
  epoch   1/100  train 137143.8683  val 134075.6875
  epoch   2/100  train 132173.7978  val 130577.0928
  epoch   3/100  train 128577.6527  val 127343.9027
  epoch   4/100  train 125236.2784  val 124177.9525
  epoch   5/100  train 122060.4303  val 121123.8224
  epoch   6/100  train 118984.5134  val 118074.2043
  epoch   7/100  train 115928.5139  val 115153.1026
  epoch   8/100  train 112981.7348  val 112242.7957
  epoch   9/100  train 110136.0952  val 109507.0170
  epoch  10/100  train 107372.6045  val 106727.1476
  epoch  11/100  train 104557.3868  val 104062.4584
  epoch  12/100  train 101864.8662  val 101483.4775
  epoch  13/100  train 99209.3300  val 98793.8131
  epoch  14/100  train 96578.4226  val 96165.8549
  epoch  15/100  train 93953.6515  val 93544.1733
  epoch  16/100  train 91373.1320  val 90987.0027
  epoch  17/100  train 88787.2587  val 88428.6465
  epoch  18/100  train 86238.7867  val 85903.3680
  epoch  19/100  train 83683.8301  val 83345.3085
  epoch  20/100  train 81137.6833  val 80788.3865
  epoch  21/100  train 78614.7041  val 78305.0670
  epoch  22/100  train 76107.2342  val 75878.2069
  epoch  23/100  train 73622.8460  val 73418.0857
  epoch  24/100  train 71169.5787  val 70903.0264
  epoch  25/100  train 68708.4823  val 68459.9767
  epoch  26/100  train 66324.2570  val 65952.2876
  epoch  27/100  train 63899.6051  val 63633.9021
  epoch  28/100  train 61558.3244  val 61289.8354
  epoch  29/100  train 59224.1241  val 59011.3833
  epoch  30/100  train 56938.1097  val 56721.6876
  epoch  31/100  train 54753.5488  val 54642.2790
  epoch  32/100  train 52533.6972  val 52417.6788
  epoch  33/100  train 50442.5423  val 50361.1279
  epoch  34/100  train 48477.9743  val 48410.4171
  epoch  35/100  train 46470.2682  val 46537.9697
  epoch  36/100  train 44584.4969  val 44712.7020
  epoch  37/100  train 42795.6680  val 42878.0600
  epoch  38/100  train 41016.7742  val 41164.9825
  epoch  39/100  train 39299.1058  val 39592.8331
  epoch  40/100  train 37546.7639  val 37700.2984
  epoch  41/100  train 35886.7500  val 35951.0451
  epoch  42/100  train 34235.8849  val 34339.0854
  epoch  43/100  train 32566.4589  val 32666.6745
  epoch  44/100  train 30833.4775  val 30848.6052
  epoch  45/100  train 29028.9568  val 28981.6519
  epoch  46/100  train 26977.4774  val 26624.3320
  epoch  47/100  train 23929.1227  val 21776.2068
  epoch  48/100  train 15255.6545  val 8829.5807
  epoch  49/100  train 4559.2067  val 2896.4027
  epoch  50/100  train 2555.9758  val 2276.9731
  epoch  51/100  train 2092.9522  val 1940.5257
  epoch  52/100  train 1818.2134  val 1709.4250
  epoch  53/100  train 1611.9450  val 1518.5810
  epoch  54/100  train 1444.7829  val 1370.5027
  epoch  55/100  train 1305.5904  val 1246.4603
  epoch  56/100  train 1191.6136  val 1139.7669
  epoch  57/100  train 1098.0980  val 1056.9040
  epoch  58/100  train 1018.3636  val 983.0249
  epoch  59/100  train 946.7901  val 913.7146
  epoch  60/100  train 880.8512  val 848.3303
  epoch  61/100  train 820.8835  val 791.3352
  epoch  62/100  train 768.5443  val 745.6641
  epoch  63/100  train 722.7235  val 702.1390
  epoch  64/100  train 679.7188  val 659.0323
  epoch  65/100  train 637.9655  val 616.7884
  epoch  66/100  train 596.9525  val 576.8080
  epoch  67/100  train 563.7889  val 550.4676
  epoch  68/100  train 536.9582  val 524.9305
  epoch  69/100  train 510.7472  val 496.3783
  epoch  70/100  train 483.5697  val 471.6448
  epoch  71/100  train 461.0356  val 450.1499
  epoch  72/100  train 440.4675  val 430.7353
  epoch  73/100  train 420.8920  val 412.1615
  epoch  74/100  train 403.0511  val 396.0698
  epoch  75/100  train 387.3857  val 379.3562
  epoch  76/100  train 371.3998  val 363.7801
  epoch  77/100  train 355.2558  val 346.9828
  epoch  78/100  train 339.3794  val 329.6904
  epoch  79/100  train 323.0213  val 314.3439
  epoch  80/100  train 307.3201  val 298.0389
  epoch  81/100  train 291.6400  val 283.1956
  epoch  82/100  train 275.6019  val 267.3414
  epoch  83/100  train 260.6507  val 251.9870
  epoch  84/100  train 244.8430  val 239.2575
  epoch  85/100  train 237.1616  val 234.5700
  epoch  86/100  train 230.8048  val 228.5779
  epoch  87/100  train 224.5335  val 221.0901
  epoch  88/100  train 218.0968  val 215.1239
  epoch  89/100  train 212.3895  val 209.7332
  epoch  90/100  train 205.6144  val 201.8330
  epoch  91/100  train 199.3878  val 196.1707
  epoch  92/100  train 193.5550  val 192.5132
  epoch  93/100  train 192.4388  val 191.7422
  epoch  94/100  train 190.3222  val 190.5352
  epoch  95/100  train 189.7030  val 190.2812
  epoch  96/100  train 189.1501  val 188.9161
  epoch  97/100  train 188.0264  val 187.9338
  epoch  98/100  train 188.1410  val 188.1779
  epoch  99/100  train 186.5100  val 185.4848
  epoch 100/100  train 185.5861  val 183.8118
  training: 100 epochs, 572.7 s
  UVMD-5: rel err 0.0047  overlap 0.133  daily-line ownership 1.00  0.035 s/node
  learned alpha_k: [181, 72, 164, 133, 0, 63]
  same trained model on CPU: 0.071 s/node (3 nodes)
```
