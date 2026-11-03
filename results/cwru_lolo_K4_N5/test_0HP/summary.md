# CWRU: one 1D-CNN, five input representations

1199 windows x 2048 samples (12 kHz), split 'lolo' (train/val/test = 600/300/299), 10 classes (fault10), K=4, N=5, seeds [0, 1, 2]. Same CNN, training loop, split and model selection (best validation accuracy) in every row; only the CNN input differs. uvmd_e2e: loss = CE + 0.1 x relative spectral L1 residual, lr(alpha)=2.0, lr(lambda)=0.05, start from 'untrained' UVMD.

## Test results (3 HP recordings, unseen load)

|                            | test acc (mean ± std over seeds)   | macro-F1        |   test CE |   best val acc |   recon rel. L1 (test) |   CNN params |   UVMD params |   train s |
|:---------------------------|:-----------------------------------|:----------------|----------:|---------------:|-----------------------:|-------------:|--------------:|----------:|
| VMD (converged) -> CNN     | 0.9866 ± 0.0058                    | 0.9864 ± 0.0061 |    0.0793 |            0.9 |               nan      |       112818 |             0 |         2 |
| UVMD-5 end-to-end with CNN | 0.9911 ± 0.0070                    | 0.9910 ± 0.0071 |    0.0416 |            0.9 |                 0.1808 |       112818 |         40964 |        15 |

## Per-class recall (summed over seeds)

|             |   VMD (converged) -> CNN |   UVMD-5 end-to-end with CNN |
|:------------|-------------------------:|-----------------------------:|
| normal      |                    1     |                        1     |
| inner_0.007 |                    1     |                        1     |
| inner_0.014 |                    1     |                        1     |
| inner_0.021 |                    1     |                        1     |
| ball_0.007  |                    1     |                        1     |
| ball_0.014  |                    0.867 |                        0.911 |
| ball_0.021  |                    1     |                        1     |
| outer_0.007 |                    1     |                        1     |
| outer_0.014 |                    1     |                        1     |
| outer_0.021 |                    1     |                        1     |

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'torch': '2.14.1+cu130', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060', 'commit': '2c45abff0517a11ed951c2cda391e403e72bb1e9'}


## Log
```
CWRU: 40 files, 1199 windows x 2048 samples | split 'lolo' (test load 0 HP): train/val/test = 600/300/299 | 10 classes (fault10) | K=4, N=5 layers | seeds [0, 1, 2] | device cuda:0
  train windows per class: normal:60, inner_0.007:60, inner_0.014:60, inner_0.021:60, ball_0.007:60, ball_0.014:60, ball_0.021:60, outer_0.007:60, outer_0.014:60, outer_0.021:60
    val windows per class: normal:30, inner_0.007:30, inner_0.014:30, inner_0.021:30, ball_0.007:30, ball_0.014:30, ball_0.021:30, outer_0.007:30, outer_0.014:30, outer_0.021:30
   test windows per class: normal:29, inner_0.007:30, inner_0.014:30, inner_0.021:30, ball_0.007:30, ball_0.014:30, ball_0.021:30, outer_0.007:30, outer_0.014:30, outer_0.021:30
[vmd] converged VMD (alpha 500) on 1199 windows: mean 145.5 iterations, 61 s

[VMD (converged) -> CNN] seed 0
    epoch   1  train CE 1.0274  val CE 1.3535  val acc 0.773
    epoch   2  train CE 0.3214  val CE 0.6590  val acc 0.793
    epoch   3  train CE 0.1531  val CE 0.5206  val acc 0.900
    epoch   4  train CE 0.0836  val CE 0.4456  val acc 0.900
    epoch   5  train CE 0.0591  val CE 0.4430  val acc 0.893
    epoch   6  train CE 0.0470  val CE 0.4355  val acc 0.893
    epoch   7  train CE 0.0357  val CE 0.4489  val acc 0.900
    epoch   8  train CE 0.0353  val CE 0.4070  val acc 0.893
    epoch   9  train CE 0.0235  val CE 0.5226  val acc 0.900
    epoch  10  train CE 0.0175  val CE 0.4257  val acc 0.900
    epoch  11  train CE 0.0154  val CE 0.4584  val acc 0.893
    epoch  12  train CE 0.0106  val CE 0.4833  val acc 0.893
    epoch  13  train CE 0.0136  val CE 0.4855  val acc 0.900
    epoch  14  train CE 0.0140  val CE 0.4741  val acc 0.897
    epoch  15  train CE 0.0160  val CE 0.5330  val acc 0.893
    epoch  16  train CE 0.0102  val CE 0.4796  val acc 0.883
    epoch  17  train CE 0.0087  val CE 0.5131  val acc 0.900
    epoch  18  train CE 0.0071  val CE 0.4696  val acc 0.900
    epoch  19  train CE 0.0051  val CE 0.5092  val acc 0.893
    epoch  20  train CE 0.0060  val CE 0.5151  val acc 0.893
    epoch  21  train CE 0.0054  val CE 0.4987  val acc 0.893
    epoch  22  train CE 0.0045  val CE 0.5000  val acc 0.897
    epoch  23  train CE 0.0052  val CE 0.4863  val acc 0.897
    epoch  24  train CE 0.0040  val CE 0.4892  val acc 0.893
    epoch  25  train CE 0.0037  val CE 0.4792  val acc 0.900
    early stopping
  -> test acc 0.9900  macro-F1 0.9899  CE 0.0758  (best val acc 0.9000 @ epoch 10, 3 s)

[VMD (converged) -> CNN] seed 1
    epoch   1  train CE 1.0441  val CE 1.3960  val acc 0.700
    epoch   2  train CE 0.3247  val CE 0.6069  val acc 0.887
    epoch   3  train CE 0.1449  val CE 0.4420  val acc 0.900
    epoch   4  train CE 0.0851  val CE 0.4550  val acc 0.900
    epoch   5  train CE 0.0563  val CE 0.4303  val acc 0.897
    epoch   6  train CE 0.0465  val CE 0.4777  val acc 0.897
    epoch   7  train CE 0.0337  val CE 0.4227  val acc 0.900
    epoch   8  train CE 0.0278  val CE 0.4745  val acc 0.897
    epoch   9  train CE 0.0219  val CE 0.4760  val acc 0.893
    epoch  10  train CE 0.0205  val CE 0.5150  val acc 0.900
    epoch  11  train CE 0.0136  val CE 0.4692  val acc 0.900
    epoch  12  train CE 0.0146  val CE 0.4834  val acc 0.893
    epoch  13  train CE 0.0110  val CE 0.5210  val acc 0.900
    epoch  14  train CE 0.0115  val CE 0.4371  val acc 0.897
    epoch  15  train CE 0.0091  val CE 0.5074  val acc 0.900
    epoch  16  train CE 0.0082  val CE 0.5140  val acc 0.897
    epoch  17  train CE 0.0071  val CE 0.5702  val acc 0.897
    epoch  18  train CE 0.0059  val CE 0.4801  val acc 0.897
    epoch  19  train CE 0.0075  val CE 0.5203  val acc 0.897
    epoch  20  train CE 0.0054  val CE 0.5389  val acc 0.893
    epoch  21  train CE 0.0054  val CE 0.5586  val acc 0.897
    epoch  22  train CE 0.0050  val CE 0.4453  val acc 0.897
    early stopping
  -> test acc 0.9900  macro-F1 0.9899  CE 0.0598  (best val acc 0.9000 @ epoch 7, 2 s)

[VMD (converged) -> CNN] seed 2
    epoch   1  train CE 1.1386  val CE 1.5003  val acc 0.693
    epoch   2  train CE 0.3873  val CE 0.7640  val acc 0.857
    epoch   3  train CE 0.1836  val CE 0.5541  val acc 0.900
    epoch   4  train CE 0.1067  val CE 0.5412  val acc 0.900
    epoch   5  train CE 0.0690  val CE 0.6126  val acc 0.897
    epoch   6  train CE 0.0530  val CE 0.5262  val acc 0.900
    epoch   7  train CE 0.0393  val CE 0.6172  val acc 0.900
    epoch   8  train CE 0.0287  val CE 0.4913  val acc 0.900
    epoch   9  train CE 0.0265  val CE 0.5945  val acc 0.897
    epoch  10  train CE 0.0195  val CE 0.5925  val acc 0.900
    epoch  11  train CE 0.0167  val CE 0.5793  val acc 0.900
    epoch  12  train CE 0.0155  val CE 0.5858  val acc 0.900
    epoch  13  train CE 0.0130  val CE 0.4369  val acc 0.900
    epoch  14  train CE 0.0121  val CE 0.6129  val acc 0.897
    epoch  15  train CE 0.0116  val CE 0.5608  val acc 0.900
    epoch  16  train CE 0.0110  val CE 0.6377  val acc 0.900
    epoch  17  train CE 0.0117  val CE 0.4426  val acc 0.900
    epoch  18  train CE 0.0118  val CE 0.7211  val acc 0.870
    epoch  19  train CE 0.0111  val CE 0.6256  val acc 0.900
    epoch  20  train CE 0.0078  val CE 0.5328  val acc 0.900
    epoch  21  train CE 0.0076  val CE 0.6821  val acc 0.897
    epoch  22  train CE 0.0062  val CE 0.5179  val acc 0.897
    epoch  23  train CE 0.0074  val CE 0.5525  val acc 0.900
    epoch  24  train CE 0.0063  val CE 0.6702  val acc 0.897
    epoch  25  train CE 0.0056  val CE 0.7564  val acc 0.900
    epoch  26  train CE 0.0041  val CE 0.5446  val acc 0.900
    epoch  27  train CE 0.0036  val CE 0.6120  val acc 0.900
    epoch  28  train CE 0.0033  val CE 0.6365  val acc 0.900
    early stopping
  -> test acc 0.9799  macro-F1 0.9794  CE 0.1024  (best val acc 0.9000 @ epoch 13, 2 s)

[UVMD-5 end-to-end with CNN] seed 0
    epoch   1  train CE 1.0829  val CE 1.4118  val acc 0.540  rec 0.2083/0.2135
    epoch   2  train CE 0.3492  val CE 0.5584  val acc 0.800  rec 0.2079/0.2120
    epoch   3  train CE 0.1572  val CE 0.4034  val acc 0.897  rec 0.2064/0.2101
    epoch   4  train CE 0.0807  val CE 0.3335  val acc 0.900  rec 0.2047/0.2082
    epoch   5  train CE 0.0565  val CE 0.3607  val acc 0.900  rec 0.2030/0.2062
    epoch   6  train CE 0.0456  val CE 0.3710  val acc 0.900  rec 0.2013/0.2044
    epoch   7  train CE 0.0323  val CE 0.2788  val acc 0.900  rec 0.1997/0.2023
    epoch   8  train CE 0.0292  val CE 0.3500  val acc 0.900  rec 0.1979/0.2001
    epoch   9  train CE 0.0205  val CE 0.3504  val acc 0.900  rec 0.1959/0.1977
    epoch  10  train CE 0.0174  val CE 0.2900  val acc 0.900  rec 0.1939/0.1953
    epoch  11  train CE 0.0156  val CE 0.3587  val acc 0.900  rec 0.1915/0.1924
    epoch  12  train CE 0.0102  val CE 0.2801  val acc 0.900  rec 0.1890/0.1894
    epoch  13  train CE 0.0121  val CE 0.3476  val acc 0.900  rec 0.1862/0.1859
    epoch  14  train CE 0.0109  val CE 0.2580  val acc 0.900  rec 0.1833/0.1824
    epoch  15  train CE 0.0100  val CE 0.4239  val acc 0.900  rec 0.1798/0.1779
    epoch  16  train CE 0.0084  val CE 0.2435  val acc 0.900  rec 0.1762/0.1741
    epoch  17  train CE 0.0074  val CE 0.3558  val acc 0.900  rec 0.1728/0.1702
    epoch  18  train CE 0.0066  val CE 0.3050  val acc 0.900  rec 0.1693/0.1655
    epoch  19  train CE 0.0054  val CE 0.3300  val acc 0.900  rec 0.1645/0.1602
    epoch  20  train CE 0.0063  val CE 0.3742  val acc 0.900  rec 0.1603/0.1540
    epoch  21  train CE 0.0058  val CE 0.2958  val acc 0.900  rec 0.1554/0.1507
    epoch  22  train CE 0.0048  val CE 0.3711  val acc 0.900  rec 0.1512/0.1440
    epoch  23  train CE 0.0052  val CE 0.3322  val acc 0.900  rec 0.1457/0.1387
    epoch  24  train CE 0.0043  val CE 0.3537  val acc 0.900  rec 0.1390/0.1297
    epoch  25  train CE 0.0056  val CE 0.3252  val acc 0.900  rec 0.1310/0.1227
    epoch  26  train CE 0.0058  val CE 0.5083  val acc 0.877  rec 0.1265/0.1199
    epoch  27  train CE 0.0050  val CE 0.3334  val acc 0.900  rec 0.1227/0.1140
    epoch  28  train CE 0.0045  val CE 0.3423  val acc 0.900  rec 0.1159/0.1029
    epoch  29  train CE 0.0155  val CE 1.9087  val acc 0.600  rec 0.1087/0.1074
    epoch  30  train CE 0.0094  val CE 0.7663  val acc 0.820  rec 0.1142/0.1078
    epoch  31  train CE 0.0047  val CE 0.5564  val acc 0.900  rec 0.1120/0.1044
    early stopping
  -> test acc 0.9967  macro-F1 0.9967  CE 0.0299  (best val acc 0.9000 @ epoch 16, 16 s)  test recon rel. L1 0.1712  alpha [433.99, 408.11, 429.87, 224.18]

[UVMD-5 end-to-end with CNN] seed 1
    epoch   1  train CE 1.0884  val CE 1.6445  val acc 0.577  rec 0.2091/0.2142
    epoch   2  train CE 0.3546  val CE 0.6792  val acc 0.797  rec 0.2087/0.2132
    epoch   3  train CE 0.1607  val CE 0.3902  val acc 0.900  rec 0.2075/0.2118
    epoch   4  train CE 0.0873  val CE 0.3260  val acc 0.900  rec 0.2063/0.2102
    epoch   5  train CE 0.0541  val CE 0.3111  val acc 0.900  rec 0.2048/0.2087
    epoch   6  train CE 0.0432  val CE 0.3288  val acc 0.900  rec 0.2034/0.2070
    epoch   7  train CE 0.0326  val CE 0.3264  val acc 0.900  rec 0.2021/0.2054
    epoch   8  train CE 0.0259  val CE 0.3677  val acc 0.900  rec 0.2005/0.2035
    epoch   9  train CE 0.0219  val CE 0.3181  val acc 0.900  rec 0.1989/0.2014
    epoch  10  train CE 0.0228  val CE 0.3294  val acc 0.900  rec 0.1970/0.1995
    epoch  11  train CE 0.0145  val CE 0.3067  val acc 0.900  rec 0.1955/0.1977
    epoch  12  train CE 0.0146  val CE 0.3402  val acc 0.900  rec 0.1937/0.1952
    epoch  13  train CE 0.0100  val CE 0.3201  val acc 0.900  rec 0.1913/0.1923
    epoch  14  train CE 0.0096  val CE 0.2978  val acc 0.900  rec 0.1888/0.1894
    epoch  15  train CE 0.0088  val CE 0.3369  val acc 0.900  rec 0.1862/0.1862
    epoch  16  train CE 0.0086  val CE 0.2733  val acc 0.900  rec 0.1832/0.1824
    epoch  17  train CE 0.0079  val CE 0.3260  val acc 0.900  rec 0.1799/0.1787
    epoch  18  train CE 0.0059  val CE 0.3343  val acc 0.900  rec 0.1765/0.1742
    epoch  19  train CE 0.0078  val CE 0.3105  val acc 0.900  rec 0.1722/0.1694
    epoch  20  train CE 0.0057  val CE 0.3612  val acc 0.900  rec 0.1692/0.1660
    epoch  21  train CE 0.0056  val CE 0.3392  val acc 0.900  rec 0.1647/0.1602
    epoch  22  train CE 0.0050  val CE 0.2812  val acc 0.900  rec 0.1609/0.1562
    epoch  23  train CE 0.0040  val CE 0.4246  val acc 0.900  rec 0.1557/0.1494
    epoch  24  train CE 0.0056  val CE 0.3832  val acc 0.900  rec 0.1504/0.1464
    epoch  25  train CE 0.0052  val CE 0.3102  val acc 0.900  rec 0.1484/0.1427
    epoch  26  train CE 0.0066  val CE 0.3128  val acc 0.900  rec 0.1445/0.1377
    epoch  27  train CE 0.0044  val CE 0.4254  val acc 0.900  rec 0.1392/0.1312
    epoch  28  train CE 0.0032  val CE 0.3947  val acc 0.900  rec 0.1325/0.1224
    epoch  29  train CE 0.0049  val CE 0.5297  val acc 0.900  rec 0.1244/0.1162
    epoch  30  train CE 0.0051  val CE 0.3766  val acc 0.900  rec 0.1198/0.1120
    epoch  31  train CE 0.0030  val CE 0.3364  val acc 0.900  rec 0.1131/0.0992
    early stopping
  -> test acc 0.9933  macro-F1 0.9933  CE 0.0282  (best val acc 0.9000 @ epoch 16, 17 s)  test recon rel. L1 0.1782  alpha [446.3, 437.28, 443.15, 259.56]

[UVMD-5 end-to-end with CNN] seed 2
    epoch   1  train CE 1.1413  val CE 1.5271  val acc 0.520  rec 0.2086/0.2133
    epoch   2  train CE 0.3753  val CE 0.6552  val acc 0.900  rec 0.2077/0.2115
    epoch   3  train CE 0.1725  val CE 0.4657  val acc 0.900  rec 0.2060/0.2096
    epoch   4  train CE 0.1002  val CE 0.4135  val acc 0.900  rec 0.2044/0.2078
    epoch   5  train CE 0.0661  val CE 0.4670  val acc 0.900  rec 0.2027/0.2057
    epoch   6  train CE 0.0499  val CE 0.4719  val acc 0.900  rec 0.2009/0.2037
    epoch   7  train CE 0.0353  val CE 0.4162  val acc 0.900  rec 0.1991/0.2014
    epoch   8  train CE 0.0257  val CE 0.3073  val acc 0.900  rec 0.1970/0.1988
    epoch   9  train CE 0.0244  val CE 0.4180  val acc 0.900  rec 0.1947/0.1962
    epoch  10  train CE 0.0191  val CE 0.4429  val acc 0.900  rec 0.1923/0.1934
    epoch  11  train CE 0.0162  val CE 0.4088  val acc 0.900  rec 0.1898/0.1900
    epoch  12  train CE 0.0156  val CE 0.4586  val acc 0.900  rec 0.1866/0.1861
    epoch  13  train CE 0.0134  val CE 0.3406  val acc 0.900  rec 0.1837/0.1834
    epoch  14  train CE 0.0116  val CE 0.4192  val acc 0.900  rec 0.1809/0.1797
    epoch  15  train CE 0.0108  val CE 0.3852  val acc 0.900  rec 0.1777/0.1755
    epoch  16  train CE 0.0096  val CE 0.4257  val acc 0.900  rec 0.1737/0.1708
    epoch  17  train CE 0.0092  val CE 0.4283  val acc 0.900  rec 0.1694/0.1652
    epoch  18  train CE 0.0069  val CE 0.4308  val acc 0.900  rec 0.1639/0.1584
    epoch  19  train CE 0.0073  val CE 0.4957  val acc 0.900  rec 0.1573/0.1519
    epoch  20  train CE 0.0071  val CE 0.3955  val acc 0.900  rec 0.1516/0.1440
    epoch  21  train CE 0.0062  val CE 0.4615  val acc 0.900  rec 0.1450/0.1373
    epoch  22  train CE 0.0062  val CE 0.4361  val acc 0.900  rec 0.1371/0.1273
    epoch  23  train CE 0.0067  val CE 0.3409  val acc 0.900  rec 0.1334/0.1265
    early stopping
  -> test acc 0.9833  macro-F1 0.9830  CE 0.0666  (best val acc 0.9000 @ epoch 8, 12 s)  test recon rel. L1 0.1929  alpha [512.15, 460.72, 490.63, 338.94]
```
