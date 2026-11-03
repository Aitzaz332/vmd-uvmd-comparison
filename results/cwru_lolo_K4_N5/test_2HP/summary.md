# CWRU: one 1D-CNN, five input representations

1199 windows x 2048 samples (12 kHz), split 'lolo' (train/val/test = 599/300/300), 10 classes (fault10), K=4, N=5, seeds [0, 1, 2]. Same CNN, training loop, split and model selection (best validation accuracy) in every row; only the CNN input differs. uvmd_e2e: loss = CE + 0.1 x relative spectral L1 residual, lr(alpha)=2.0, lr(lambda)=0.05, start from 'untrained' UVMD.

## Test results (3 HP recordings, unseen load)

|                            | test acc (mean ± std over seeds)   | macro-F1        |   test CE |   best val acc |   recon rel. L1 (test) |   CNN params |   UVMD params |   train s |
|:---------------------------|:-----------------------------------|:----------------|----------:|---------------:|-----------------------:|-------------:|--------------:|----------:|
| VMD (converged) -> CNN     | 0.9689 ± 0.0453                    | 0.9631 ± 0.0554 |    0.2053 |         0.8656 |               nan      |       112818 |             0 |         2 |
| UVMD-5 end-to-end with CNN | 0.9722 ± 0.0201                    | 0.9711 ± 0.0214 |    0.1769 |         0.8478 |                 0.1877 |       112818 |         40964 |        14 |

## Per-class recall (summed over seeds)

|             |   VMD (converged) -> CNN |   UVMD-5 end-to-end with CNN |
|:------------|-------------------------:|-----------------------------:|
| normal      |                    1     |                        1     |
| inner_0.007 |                    1     |                        1     |
| inner_0.014 |                    0.722 |                        0.722 |
| inner_0.021 |                    1     |                        1     |
| ball_0.007  |                    0.978 |                        1     |
| ball_0.014  |                    1     |                        1     |
| ball_0.021  |                    0.989 |                        1     |
| outer_0.007 |                    1     |                        1     |
| outer_0.014 |                    1     |                        1     |
| outer_0.021 |                    1     |                        1     |

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'torch': '2.14.1+cu130', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060', 'commit': '2c45abff0517a11ed951c2cda391e403e72bb1e9'}


## Log
```
CWRU: 40 files, 1199 windows x 2048 samples | split 'lolo' (test load 2 HP): train/val/test = 599/300/300 | 10 classes (fault10) | K=4, N=5 layers | seeds [0, 1, 2] | device cuda:0
  train windows per class: normal:59, inner_0.007:60, inner_0.014:60, inner_0.021:60, ball_0.007:60, ball_0.014:60, ball_0.021:60, outer_0.007:60, outer_0.014:60, outer_0.021:60
    val windows per class: normal:30, inner_0.007:30, inner_0.014:30, inner_0.021:30, ball_0.007:30, ball_0.014:30, ball_0.021:30, outer_0.007:30, outer_0.014:30, outer_0.021:30
   test windows per class: normal:30, inner_0.007:30, inner_0.014:30, inner_0.021:30, ball_0.007:30, ball_0.014:30, ball_0.021:30, outer_0.007:30, outer_0.014:30, outer_0.021:30
[vmd] converged VMD (alpha 500) on 1199 windows: mean 145.5 iterations, 0 s

[VMD (converged) -> CNN] seed 0
    epoch   1  train CE 1.0989  val CE 1.4461  val acc 0.660
    epoch   2  train CE 0.3575  val CE 0.6299  val acc 0.853
    epoch   3  train CE 0.1752  val CE 0.5922  val acc 0.850
    epoch   4  train CE 0.0898  val CE 0.5761  val acc 0.853
    epoch   5  train CE 0.0704  val CE 0.8184  val acc 0.793
    epoch   6  train CE 0.0452  val CE 0.8831  val acc 0.793
    epoch   7  train CE 0.0317  val CE 0.8315  val acc 0.793
    epoch   8  train CE 0.0272  val CE 1.0896  val acc 0.783
    epoch   9  train CE 0.0243  val CE 1.1516  val acc 0.793
    epoch  10  train CE 0.0212  val CE 1.1445  val acc 0.793
    epoch  11  train CE 0.0167  val CE 1.1463  val acc 0.793
    epoch  12  train CE 0.0133  val CE 1.2151  val acc 0.793
    epoch  13  train CE 0.0115  val CE 1.3416  val acc 0.793
    epoch  14  train CE 0.0147  val CE 0.7854  val acc 0.797
    epoch  15  train CE 0.0104  val CE 0.7893  val acc 0.853
    epoch  16  train CE 0.0092  val CE 1.1407  val acc 0.793
    epoch  17  train CE 0.0080  val CE 1.1192  val acc 0.793
    epoch  18  train CE 0.0066  val CE 1.2446  val acc 0.793
    epoch  19  train CE 0.0068  val CE 1.1665  val acc 0.793
    early stopping
  -> test acc 0.9967  macro-F1 0.9967  CE 0.0629  (best val acc 0.8533 @ epoch 4, 2 s)

[VMD (converged) -> CNN] seed 1
    epoch   1  train CE 1.1134  val CE 1.4262  val acc 0.683
    epoch   2  train CE 0.3544  val CE 0.6684  val acc 0.783
    epoch   3  train CE 0.1758  val CE 0.4534  val acc 0.887
    epoch   4  train CE 0.1170  val CE 0.7803  val acc 0.810
    epoch   5  train CE 0.0712  val CE 0.6532  val acc 0.807
    epoch   6  train CE 0.0437  val CE 0.8363  val acc 0.793
    epoch   7  train CE 0.0355  val CE 1.2315  val acc 0.787
    epoch   8  train CE 0.0296  val CE 0.9598  val acc 0.793
    epoch   9  train CE 0.0218  val CE 0.9991  val acc 0.793
    epoch  10  train CE 0.0201  val CE 0.8197  val acc 0.793
    epoch  11  train CE 0.0166  val CE 0.8354  val acc 0.793
    epoch  12  train CE 0.0152  val CE 0.9792  val acc 0.793
    epoch  13  train CE 0.0134  val CE 0.7968  val acc 0.793
    epoch  14  train CE 0.0135  val CE 1.3410  val acc 0.767
    epoch  15  train CE 0.0096  val CE 0.8234  val acc 0.793
    epoch  16  train CE 0.0111  val CE 1.3439  val acc 0.793
    epoch  17  train CE 0.0076  val CE 0.8718  val acc 0.793
    epoch  18  train CE 0.0064  val CE 0.9976  val acc 0.793
    early stopping
  -> test acc 0.9933  macro-F1 0.9933  CE 0.1582  (best val acc 0.8867 @ epoch 3, 2 s)

[VMD (converged) -> CNN] seed 2
    epoch   1  train CE 1.1898  val CE 1.5410  val acc 0.687
    epoch   2  train CE 0.4508  val CE 0.7299  val acc 0.857
    epoch   3  train CE 0.1945  val CE 0.6575  val acc 0.830
    epoch   4  train CE 0.1157  val CE 1.0432  val acc 0.760
    epoch   5  train CE 0.0763  val CE 0.6993  val acc 0.837
    epoch   6  train CE 0.0552  val CE 0.7118  val acc 0.803
    epoch   7  train CE 0.0435  val CE 0.8440  val acc 0.800
    epoch   8  train CE 0.0343  val CE 1.0201  val acc 0.793
    epoch   9  train CE 0.0267  val CE 0.8252  val acc 0.793
    epoch  10  train CE 0.0205  val CE 1.2726  val acc 0.793
    epoch  11  train CE 0.0162  val CE 0.7505  val acc 0.810
    epoch  12  train CE 0.0158  val CE 1.1689  val acc 0.793
    epoch  13  train CE 0.0117  val CE 1.0173  val acc 0.793
    epoch  14  train CE 0.0128  val CE 1.6554  val acc 0.793
    epoch  15  train CE 0.0107  val CE 1.0658  val acc 0.793
    epoch  16  train CE 0.0093  val CE 1.2850  val acc 0.793
    epoch  17  train CE 0.0090  val CE 1.5994  val acc 0.700
    early stopping
  -> test acc 0.9167  macro-F1 0.8992  CE 0.3948  (best val acc 0.8567 @ epoch 2, 2 s)

[UVMD-5 end-to-end with CNN] seed 0
    epoch   1  train CE 1.1253  val CE 1.5248  val acc 0.480  rec 0.2063/0.2144
    epoch   2  train CE 0.3849  val CE 0.6820  val acc 0.827  rec 0.2071/0.2149
    epoch   3  train CE 0.1857  val CE 0.7702  val acc 0.800  rec 0.2070/0.2142
    epoch   4  train CE 0.0932  val CE 0.5660  val acc 0.837  rec 0.2060/0.2128
    epoch   5  train CE 0.0705  val CE 0.7751  val acc 0.800  rec 0.2048/0.2112
    epoch   6  train CE 0.0480  val CE 0.9111  val acc 0.800  rec 0.2033/0.2094
    epoch   7  train CE 0.0344  val CE 0.7669  val acc 0.800  rec 0.2016/0.2072
    epoch   8  train CE 0.0285  val CE 1.1875  val acc 0.800  rec 0.1997/0.2049
    epoch   9  train CE 0.0221  val CE 1.0163  val acc 0.800  rec 0.1975/0.2023
    epoch  10  train CE 0.0212  val CE 0.7411  val acc 0.803  rec 0.1953/0.1998
    epoch  11  train CE 0.0176  val CE 1.4922  val acc 0.800  rec 0.1931/0.1970
    epoch  12  train CE 0.0135  val CE 1.0157  val acc 0.800  rec 0.1905/0.1939
    epoch  13  train CE 0.0125  val CE 1.0618  val acc 0.800  rec 0.1878/0.1904
    epoch  14  train CE 0.0121  val CE 0.7532  val acc 0.800  rec 0.1847/0.1869
    epoch  15  train CE 0.0093  val CE 0.9680  val acc 0.800  rec 0.1813/0.1827
    epoch  16  train CE 0.0084  val CE 0.9046  val acc 0.800  rec 0.1785/0.1793
    epoch  17  train CE 0.0084  val CE 1.1049  val acc 0.800  rec 0.1749/0.1750
    epoch  18  train CE 0.0076  val CE 1.6992  val acc 0.800  rec 0.1713/0.1701
    epoch  19  train CE 0.0094  val CE 0.7723  val acc 0.863  rec 0.1669/0.1665
    epoch  20  train CE 0.0098  val CE 0.9853  val acc 0.800  rec 0.1656/0.1648
    epoch  21  train CE 0.0071  val CE 1.3421  val acc 0.800  rec 0.1632/0.1614
    epoch  22  train CE 0.0054  val CE 1.0246  val acc 0.803  rec 0.1599/0.1574
    epoch  23  train CE 0.0069  val CE 2.6965  val acc 0.593  rec 0.1560/0.1523
    epoch  24  train CE 0.0081  val CE 2.4469  val acc 0.603  rec 0.1517/0.1475
    epoch  25  train CE 0.0049  val CE 0.5980  val acc 0.900  rec 0.1467/0.1416
    epoch  26  train CE 0.0048  val CE 1.4529  val acc 0.800  rec 0.1409/0.1338
    epoch  27  train CE 0.0057  val CE 1.5800  val acc 0.800  rec 0.1354/0.1321
    epoch  28  train CE 0.0131  val CE 1.3687  val acc 0.707  rec 0.1365/0.1327
    epoch  29  train CE 0.0097  val CE 1.7569  val acc 0.800  rec 0.1360/0.1302
    epoch  30  train CE 0.0057  val CE 1.6939  val acc 0.800  rec 0.1321/0.1259
    epoch  31  train CE 0.0031  val CE 1.3004  val acc 0.800  rec 0.1276/0.1205
    epoch  32  train CE 0.0031  val CE 1.5031  val acc 0.800  rec 0.1218/0.1135
    epoch  33  train CE 0.0040  val CE 1.3464  val acc 0.800  rec 0.1145/0.1063
    epoch  34  train CE 0.0032  val CE 1.1570  val acc 0.800  rec 0.1102/0.1020
    epoch  35  train CE 0.0027  val CE 1.4746  val acc 0.800  rec 0.1019/0.0901
    epoch  36  train CE 0.0030  val CE 1.4808  val acc 0.800  rec 0.0914/0.0862
    epoch  37  train CE 0.0025  val CE 1.7327  val acc 0.800  rec 0.0853/0.0725
    epoch  38  train CE 0.0032  val CE 2.0835  val acc 0.800  rec 0.0736/0.0633
    epoch  39  train CE 0.0028  val CE 1.6397  val acc 0.800  rec 0.0711/0.0670
    epoch  40  train CE 0.0035  val CE 1.3852  val acc 0.800  rec 0.0667/0.0683
    early stopping
  -> test acc 0.9533  macro-F1 0.9506  CE 0.0924  (best val acc 0.9000 @ epoch 25, 23 s)  test recon rel. L1 0.1422  alpha [412.47, 512.16, 262.42, 120.17]

[UVMD-5 end-to-end with CNN] seed 1
    epoch   1  train CE 1.1395  val CE 1.6916  val acc 0.477  rec 0.2064/0.2140
    epoch   2  train CE 0.3743  val CE 0.5995  val acc 0.817  rec 0.2065/0.2139
    epoch   3  train CE 0.1713  val CE 0.6289  val acc 0.800  rec 0.2060/0.2129
    epoch   4  train CE 0.1125  val CE 0.8512  val acc 0.800  rec 0.2049/0.2116
    epoch   5  train CE 0.0711  val CE 0.8406  val acc 0.800  rec 0.2038/0.2100
    epoch   6  train CE 0.0454  val CE 0.7551  val acc 0.800  rec 0.2025/0.2085
    epoch   7  train CE 0.0346  val CE 1.3978  val acc 0.800  rec 0.2013/0.2071
    epoch   8  train CE 0.0274  val CE 1.0835  val acc 0.800  rec 0.1999/0.2053
    epoch   9  train CE 0.0210  val CE 0.9107  val acc 0.800  rec 0.1984/0.2035
    epoch  10  train CE 0.0193  val CE 0.8886  val acc 0.800  rec 0.1968/0.2017
    epoch  11  train CE 0.0167  val CE 0.9238  val acc 0.800  rec 0.1952/0.1995
    epoch  12  train CE 0.0147  val CE 1.1484  val acc 0.800  rec 0.1932/0.1970
    epoch  13  train CE 0.0125  val CE 0.9983  val acc 0.800  rec 0.1911/0.1947
    epoch  14  train CE 0.0095  val CE 0.8281  val acc 0.800  rec 0.1889/0.1919
    epoch  15  train CE 0.0089  val CE 1.0272  val acc 0.800  rec 0.1866/0.1892
    epoch  16  train CE 0.0108  val CE 1.3268  val acc 0.800  rec 0.1843/0.1862
    epoch  17  train CE 0.0075  val CE 0.9474  val acc 0.800  rec 0.1816/0.1830
    early stopping
  -> test acc 0.9700  macro-F1 0.9693  CE 0.2891  (best val acc 0.8167 @ epoch 2, 9 s)  test recon rel. L1 0.2111  alpha [558.18, 559.75, 487.44, 428.24]

[UVMD-5 end-to-end with CNN] seed 2
    epoch   1  train CE 1.1751  val CE 1.5875  val acc 0.567  rec 0.2065/0.2141
    epoch   2  train CE 0.4140  val CE 0.7646  val acc 0.800  rec 0.2067/0.2139
    epoch   3  train CE 0.1761  val CE 0.5447  val acc 0.827  rec 0.2061/0.2126
    epoch   4  train CE 0.1066  val CE 0.7497  val acc 0.793  rec 0.2050/0.2112
    epoch   5  train CE 0.0701  val CE 0.6801  val acc 0.800  rec 0.2037/0.2095
    epoch   6  train CE 0.0507  val CE 0.6069  val acc 0.810  rec 0.2023/0.2078
    epoch   7  train CE 0.0425  val CE 1.0213  val acc 0.800  rec 0.2007/0.2058
    epoch   8  train CE 0.0327  val CE 0.9012  val acc 0.800  rec 0.1990/0.2037
    epoch   9  train CE 0.0252  val CE 0.8629  val acc 0.800  rec 0.1971/0.2013
    epoch  10  train CE 0.0202  val CE 0.8986  val acc 0.800  rec 0.1949/0.1987
    epoch  11  train CE 0.0156  val CE 0.8995  val acc 0.800  rec 0.1927/0.1959
    epoch  12  train CE 0.0153  val CE 1.1718  val acc 0.800  rec 0.1902/0.1928
    epoch  13  train CE 0.0122  val CE 0.8541  val acc 0.800  rec 0.1877/0.1900
    epoch  14  train CE 0.0113  val CE 0.9200  val acc 0.800  rec 0.1850/0.1863
    epoch  15  train CE 0.0087  val CE 1.0975  val acc 0.800  rec 0.1817/0.1822
    epoch  16  train CE 0.0092  val CE 1.1322  val acc 0.800  rec 0.1780/0.1776
    epoch  17  train CE 0.0082  val CE 1.0395  val acc 0.800  rec 0.1740/0.1728
    epoch  18  train CE 0.0076  val CE 1.1728  val acc 0.800  rec 0.1698/0.1673
    early stopping
  -> test acc 0.9933  macro-F1 0.9933  CE 0.1492  (best val acc 0.8267 @ epoch 3, 10 s)  test recon rel. L1 0.2097  alpha [589.68, 576.62, 454.64, 421.41]
```
