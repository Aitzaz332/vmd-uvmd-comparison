# CWRU: one 1D-CNN, five input representations

1199 windows x 2048 samples (12 kHz), split 'lolo' (train/val/test = 599/300/300), 10 classes (fault10), K=4, N=5, seeds [0, 1, 2]. Same CNN, training loop, split and model selection (best validation accuracy) in every row; only the CNN input differs. uvmd_e2e: loss = CE + 0.1 x relative spectral L1 residual, lr(alpha)=2.0, lr(lambda)=0.05, start from 'untrained' UVMD.

## Test results (3 HP recordings, unseen load)

|                            | test acc (mean ± std over seeds)   | macro-F1        |   test CE |   best val acc |   recon rel. L1 (test) |   CNN params |   UVMD params |   train s |
|:---------------------------|:-----------------------------------|:----------------|----------:|---------------:|-----------------------:|-------------:|--------------:|----------:|
| VMD (converged) -> CNN     | 1.0000 ± 0.0000                    | 1.0000 ± 0.0000 |    0.0119 |         0.9989 |               nan      |       112818 |             0 |         3 |
| UVMD-5 end-to-end with CNN | 0.9989 ± 0.0019                    | 0.9989 ± 0.0019 |    0.011  |         1      |                 0.1652 |       112818 |         40964 |        18 |

## Per-class recall (summed over seeds)

|             |   VMD (converged) -> CNN |   UVMD-5 end-to-end with CNN |
|:------------|-------------------------:|-----------------------------:|
| normal      |                        1 |                        1     |
| inner_0.007 |                        1 |                        1     |
| inner_0.014 |                        1 |                        1     |
| inner_0.021 |                        1 |                        1     |
| ball_0.007  |                        1 |                        1     |
| ball_0.014  |                        1 |                        1     |
| ball_0.021  |                        1 |                        0.989 |
| outer_0.007 |                        1 |                        1     |
| outer_0.014 |                        1 |                        1     |
| outer_0.021 |                        1 |                        1     |

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'torch': '2.14.1+cu130', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060', 'commit': '2c45abff0517a11ed951c2cda391e403e72bb1e9'}


## Log
```
CWRU: 40 files, 1199 windows x 2048 samples | split 'lolo' (test load 1 HP): train/val/test = 599/300/300 | 10 classes (fault10) | K=4, N=5 layers | seeds [0, 1, 2] | device cuda:0
  train windows per class: normal:59, inner_0.007:60, inner_0.014:60, inner_0.021:60, ball_0.007:60, ball_0.014:60, ball_0.021:60, outer_0.007:60, outer_0.014:60, outer_0.021:60
    val windows per class: normal:30, inner_0.007:30, inner_0.014:30, inner_0.021:30, ball_0.007:30, ball_0.014:30, ball_0.021:30, outer_0.007:30, outer_0.014:30, outer_0.021:30
   test windows per class: normal:30, inner_0.007:30, inner_0.014:30, inner_0.021:30, ball_0.007:30, ball_0.014:30, ball_0.021:30, outer_0.007:30, outer_0.014:30, outer_0.021:30
[vmd] converged VMD (alpha 500) on 1199 windows: mean 145.5 iterations, 0 s

[VMD (converged) -> CNN] seed 0
    epoch   1  train CE 1.0956  val CE 1.4110  val acc 0.660
    epoch   2  train CE 0.3945  val CE 0.3562  val acc 0.997
    epoch   3  train CE 0.2052  val CE 0.1687  val acc 0.997
    epoch   4  train CE 0.1082  val CE 0.1156  val acc 0.993
    epoch   5  train CE 0.0943  val CE 0.1406  val acc 0.997
    epoch   6  train CE 0.0618  val CE 0.0709  val acc 0.997
    epoch   7  train CE 0.0408  val CE 0.0411  val acc 0.997
    epoch   8  train CE 0.0340  val CE 0.0498  val acc 0.997
    epoch   9  train CE 0.0307  val CE 0.0580  val acc 0.997
    epoch  10  train CE 0.0233  val CE 0.0342  val acc 0.997
    epoch  11  train CE 0.0187  val CE 0.0160  val acc 0.997
    epoch  12  train CE 0.0200  val CE 0.0175  val acc 1.000
    epoch  13  train CE 0.0161  val CE 0.0318  val acc 0.997
    epoch  14  train CE 0.0129  val CE 0.0154  val acc 0.997
    epoch  15  train CE 0.0094  val CE 0.0115  val acc 0.997
    epoch  16  train CE 0.0087  val CE 0.0142  val acc 0.997
    epoch  17  train CE 0.0081  val CE 0.0114  val acc 0.997
    epoch  18  train CE 0.0070  val CE 0.0151  val acc 0.997
    epoch  19  train CE 0.0075  val CE 0.0120  val acc 0.997
    epoch  20  train CE 0.0066  val CE 0.0123  val acc 0.997
    epoch  21  train CE 0.0068  val CE 0.0183  val acc 0.997
    epoch  22  train CE 0.0065  val CE 0.0148  val acc 0.997
    epoch  23  train CE 0.0066  val CE 0.0166  val acc 0.997
    epoch  24  train CE 0.0060  val CE 0.0133  val acc 0.997
    epoch  25  train CE 0.0043  val CE 0.0114  val acc 0.997
    epoch  26  train CE 0.0046  val CE 0.0149  val acc 0.997
    epoch  27  train CE 0.0045  val CE 0.0053  val acc 0.997
    early stopping
  -> test acc 1.0000  macro-F1 1.0000  CE 0.0161  (best val acc 1.0000 @ epoch 12, 2 s)

[VMD (converged) -> CNN] seed 1
    epoch   1  train CE 1.1267  val CE 1.4647  val acc 0.427
    epoch   2  train CE 0.3955  val CE 0.3317  val acc 0.993
    epoch   3  train CE 0.2114  val CE 0.2493  val acc 0.980
    epoch   4  train CE 0.1377  val CE 0.1267  val acc 0.997
    epoch   5  train CE 0.0933  val CE 0.0920  val acc 0.997
    epoch   6  train CE 0.0569  val CE 0.0462  val acc 0.997
    epoch   7  train CE 0.0436  val CE 0.1870  val acc 0.913
    epoch   8  train CE 0.0343  val CE 0.0307  val acc 0.997
    epoch   9  train CE 0.0247  val CE 0.0264  val acc 0.997
    epoch  10  train CE 0.0228  val CE 0.0250  val acc 0.997
    epoch  11  train CE 0.0184  val CE 0.0166  val acc 0.997
    epoch  12  train CE 0.0199  val CE 0.0252  val acc 0.993
    epoch  13  train CE 0.0247  val CE 0.0419  val acc 0.997
    epoch  14  train CE 0.0211  val CE 0.0304  val acc 0.997
    epoch  15  train CE 0.0120  val CE 0.0166  val acc 0.997
    epoch  16  train CE 0.0126  val CE 0.0278  val acc 0.997
    epoch  17  train CE 0.0083  val CE 0.0189  val acc 0.997
    epoch  18  train CE 0.0071  val CE 0.0138  val acc 0.997
    epoch  19  train CE 0.0082  val CE 0.0199  val acc 0.997
    epoch  20  train CE 0.0060  val CE 0.0185  val acc 0.997
    epoch  21  train CE 0.0056  val CE 0.0105  val acc 0.997
    epoch  22  train CE 0.0058  val CE 0.0167  val acc 0.997
    epoch  23  train CE 0.0042  val CE 0.0170  val acc 0.997
    epoch  24  train CE 0.0048  val CE 0.0098  val acc 0.997
    epoch  25  train CE 0.0045  val CE 0.0152  val acc 0.997
    epoch  26  train CE 0.0068  val CE 0.0089  val acc 1.000
    epoch  27  train CE 0.0053  val CE 0.0126  val acc 0.997
    epoch  28  train CE 0.0044  val CE 0.0133  val acc 0.997
    epoch  29  train CE 0.0039  val CE 0.0131  val acc 0.997
    epoch  30  train CE 0.0050  val CE 0.0097  val acc 0.997
    epoch  31  train CE 0.0035  val CE 0.0081  val acc 0.997
    epoch  32  train CE 0.0035  val CE 0.0090  val acc 0.997
    epoch  33  train CE 0.0033  val CE 0.0131  val acc 0.997
    epoch  34  train CE 0.0025  val CE 0.0129  val acc 0.997
    epoch  35  train CE 0.0022  val CE 0.0085  val acc 0.997
    epoch  36  train CE 0.0032  val CE 0.0277  val acc 0.997
    epoch  37  train CE 0.0032  val CE 0.0090  val acc 0.997
    epoch  38  train CE 0.0033  val CE 0.0076  val acc 0.997
    epoch  39  train CE 0.0024  val CE 0.0095  val acc 0.997
    epoch  40  train CE 0.0020  val CE 0.0122  val acc 0.997
    epoch  41  train CE 0.0018  val CE 0.0091  val acc 0.997
    early stopping
  -> test acc 1.0000  macro-F1 1.0000  CE 0.0098  (best val acc 1.0000 @ epoch 26, 3 s)

[VMD (converged) -> CNN] seed 2
    epoch   1  train CE 1.2397  val CE 1.5485  val acc 0.597
    epoch   2  train CE 0.5041  val CE 0.4493  val acc 0.877
    epoch   3  train CE 0.2386  val CE 0.2177  val acc 0.983
    epoch   4  train CE 0.1430  val CE 0.2260  val acc 0.990
    epoch   5  train CE 0.0926  val CE 0.1602  val acc 0.973
    epoch   6  train CE 0.0714  val CE 0.0834  val acc 0.997
    epoch   7  train CE 0.0551  val CE 0.0502  val acc 0.997
    epoch   8  train CE 0.0392  val CE 0.0693  val acc 0.997
    epoch   9  train CE 0.0306  val CE 0.0256  val acc 0.997
    epoch  10  train CE 0.0236  val CE 0.0312  val acc 0.997
    epoch  11  train CE 0.0173  val CE 0.0263  val acc 0.997
    epoch  12  train CE 0.0172  val CE 0.0285  val acc 0.993
    epoch  13  train CE 0.0120  val CE 0.0187  val acc 0.997
    epoch  14  train CE 0.0142  val CE 0.0203  val acc 0.997
    epoch  15  train CE 0.0128  val CE 0.0714  val acc 0.993
    epoch  16  train CE 0.0115  val CE 0.0230  val acc 0.997
    epoch  17  train CE 0.0115  val CE 0.0812  val acc 0.977
    epoch  18  train CE 0.0084  val CE 0.0184  val acc 0.997
    epoch  19  train CE 0.0089  val CE 0.0229  val acc 0.997
    epoch  20  train CE 0.0069  val CE 0.0170  val acc 0.993
    epoch  21  train CE 0.0068  val CE 0.0192  val acc 0.997
    epoch  22  train CE 0.0070  val CE 0.0188  val acc 0.993
    epoch  23  train CE 0.0057  val CE 0.0180  val acc 0.997
    epoch  24  train CE 0.0050  val CE 0.0199  val acc 0.997
    epoch  25  train CE 0.0054  val CE 0.0149  val acc 0.997
    epoch  26  train CE 0.0050  val CE 0.0206  val acc 0.997
    epoch  27  train CE 0.0041  val CE 0.0142  val acc 0.997
    epoch  28  train CE 0.0034  val CE 0.0199  val acc 0.993
    epoch  29  train CE 0.0039  val CE 0.0204  val acc 0.997
    epoch  30  train CE 0.0037  val CE 0.0145  val acc 0.997
    epoch  31  train CE 0.0036  val CE 0.0268  val acc 0.993
    epoch  32  train CE 0.0030  val CE 0.0151  val acc 0.997
    epoch  33  train CE 0.0041  val CE 0.0186  val acc 0.993
    epoch  34  train CE 0.0031  val CE 0.0251  val acc 0.997
    epoch  35  train CE 0.0031  val CE 0.0184  val acc 0.993
    epoch  36  train CE 0.0024  val CE 0.0178  val acc 0.993
    epoch  37  train CE 0.0024  val CE 0.0197  val acc 0.993
    epoch  38  train CE 0.0021  val CE 0.0189  val acc 0.993
    epoch  39  train CE 0.0028  val CE 0.0162  val acc 0.993
    epoch  40  train CE 0.0026  val CE 0.0216  val acc 0.993
    epoch  41  train CE 0.0022  val CE 0.0161  val acc 0.993
    epoch  42  train CE 0.0024  val CE 0.0201  val acc 0.997
    early stopping
  -> test acc 1.0000  macro-F1 1.0000  CE 0.0099  (best val acc 0.9967 @ epoch 27, 3 s)

[UVMD-5 end-to-end with CNN] seed 0
    epoch   1  train CE 1.1333  val CE 1.4745  val acc 0.370  rec 0.2107/0.2125
    epoch   2  train CE 0.4113  val CE 0.4286  val acc 0.980  rec 0.2114/0.2122
    epoch   3  train CE 0.1985  val CE 0.1071  val acc 1.000  rec 0.2104/0.2109
    epoch   4  train CE 0.1085  val CE 0.0646  val acc 1.000  rec 0.2088/0.2088
    epoch   5  train CE 0.0803  val CE 0.1156  val acc 0.987  rec 0.2067/0.2065
    epoch   6  train CE 0.0559  val CE 0.0395  val acc 1.000  rec 0.2044/0.2040
    epoch   7  train CE 0.0414  val CE 0.0142  val acc 1.000  rec 0.2018/0.2013
    epoch   8  train CE 0.0304  val CE 0.0397  val acc 1.000  rec 0.1990/0.1982
    epoch   9  train CE 0.0265  val CE 0.0580  val acc 0.987  rec 0.1957/0.1950
    epoch  10  train CE 0.0222  val CE 0.0115  val acc 1.000  rec 0.1924/0.1915
    epoch  11  train CE 0.0173  val CE 0.0130  val acc 1.000  rec 0.1889/0.1879
    epoch  12  train CE 0.0135  val CE 0.0070  val acc 1.000  rec 0.1850/0.1837
    epoch  13  train CE 0.0124  val CE 0.0213  val acc 1.000  rec 0.1808/0.1789
    epoch  14  train CE 0.0116  val CE 0.0073  val acc 1.000  rec 0.1756/0.1735
    epoch  15  train CE 0.0093  val CE 0.0068  val acc 1.000  rec 0.1700/0.1672
    epoch  16  train CE 0.0083  val CE 0.0046  val acc 1.000  rec 0.1637/0.1604
    epoch  17  train CE 0.0089  val CE 0.0088  val acc 1.000  rec 0.1559/0.1520
    epoch  18  train CE 0.0095  val CE 0.0146  val acc 1.000  rec 0.1514/0.1491
    epoch  19  train CE 0.0120  val CE 0.0106  val acc 1.000  rec 0.1464/0.1449
    epoch  20  train CE 0.0167  val CE 0.0539  val acc 0.993  rec 0.1432/0.1408
    epoch  21  train CE 0.0105  val CE 0.0177  val acc 1.000  rec 0.1402/0.1409
    epoch  22  train CE 0.0061  val CE 0.0130  val acc 1.000  rec 0.1370/0.1349
    epoch  23  train CE 0.0065  val CE 0.0048  val acc 1.000  rec 0.1286/0.1245
    epoch  24  train CE 0.0065  val CE 0.0109  val acc 1.000  rec 0.1178/0.1157
    epoch  25  train CE 0.0052  val CE 0.0508  val acc 0.980  rec 0.1078/0.1051
    epoch  26  train CE 0.0060  val CE 0.0756  val acc 0.987  rec 0.0966/0.0971
    epoch  27  train CE 0.0060  val CE 0.0170  val acc 1.000  rec 0.0967/0.1027
    epoch  28  train CE 0.0053  val CE 0.0107  val acc 1.000  rec 0.0954/0.0922
    epoch  29  train CE 0.0038  val CE 0.0167  val acc 1.000  rec 0.0795/0.0663
    epoch  30  train CE 0.0053  val CE 0.0767  val acc 0.970  rec 0.0389/0.0041
    epoch  31  train CE 0.0050  val CE 0.0613  val acc 0.993  rec 0.0032/0.0020
    early stopping
  -> test acc 0.9967  macro-F1 0.9967  CE 0.0207  (best val acc 1.0000 @ epoch 16, 17 s)  test recon rel. L1 0.1605  alpha [422.61, 521.31, 334.86, 163.08]

[UVMD-5 end-to-end with CNN] seed 1
    epoch   1  train CE 1.1415  val CE 1.5779  val acc 0.307  rec 0.2111/0.2125
    epoch   2  train CE 0.3926  val CE 0.2808  val acc 0.967  rec 0.2113/0.2122
    epoch   3  train CE 0.1879  val CE 0.1192  val acc 1.000  rec 0.2105/0.2110
    epoch   4  train CE 0.1270  val CE 0.0627  val acc 1.000  rec 0.2091/0.2094
    epoch   5  train CE 0.0798  val CE 0.0628  val acc 0.997  rec 0.2078/0.2082
    epoch   6  train CE 0.0525  val CE 0.0380  val acc 1.000  rec 0.2064/0.2068
    epoch   7  train CE 0.0431  val CE 0.0429  val acc 1.000  rec 0.2051/0.2054
    epoch   8  train CE 0.0339  val CE 0.0190  val acc 1.000  rec 0.2036/0.2039
    epoch   9  train CE 0.0251  val CE 0.0167  val acc 1.000  rec 0.2019/0.2022
    epoch  10  train CE 0.0236  val CE 0.0296  val acc 1.000  rec 0.2004/0.2006
    epoch  11  train CE 0.0192  val CE 0.0250  val acc 1.000  rec 0.1985/0.1987
    epoch  12  train CE 0.0171  val CE 0.0107  val acc 1.000  rec 0.1966/0.1965
    epoch  13  train CE 0.0144  val CE 0.0155  val acc 1.000  rec 0.1945/0.1947
    epoch  14  train CE 0.0103  val CE 0.0082  val acc 1.000  rec 0.1926/0.1925
    epoch  15  train CE 0.0091  val CE 0.0136  val acc 1.000  rec 0.1901/0.1900
    epoch  16  train CE 0.0118  val CE 0.0046  val acc 1.000  rec 0.1878/0.1877
    epoch  17  train CE 0.0075  val CE 0.0304  val acc 0.997  rec 0.1852/0.1848
    epoch  18  train CE 0.0067  val CE 0.0090  val acc 1.000  rec 0.1823/0.1819
    epoch  19  train CE 0.0079  val CE 0.0084  val acc 1.000  rec 0.1794/0.1789
    epoch  20  train CE 0.0059  val CE 0.0103  val acc 1.000  rec 0.1762/0.1752
    epoch  21  train CE 0.0059  val CE 0.0074  val acc 1.000  rec 0.1722/0.1713
    epoch  22  train CE 0.0060  val CE 0.0083  val acc 1.000  rec 0.1683/0.1671
    epoch  23  train CE 0.0049  val CE 0.0260  val acc 0.997  rec 0.1635/0.1620
    epoch  24  train CE 0.0053  val CE 0.0032  val acc 1.000  rec 0.1587/0.1569
    epoch  25  train CE 0.0050  val CE 0.0319  val acc 0.993  rec 0.1528/0.1509
    epoch  26  train CE 0.0074  val CE 0.0064  val acc 1.000  rec 0.1464/0.1432
    epoch  27  train CE 0.0057  val CE 0.0055  val acc 1.000  rec 0.1386/0.1363
    epoch  28  train CE 0.0047  val CE 0.0406  val acc 0.983  rec 0.1292/0.1259
    epoch  29  train CE 0.0045  val CE 0.0222  val acc 1.000  rec 0.1172/0.1104
    epoch  30  train CE 0.0601  val CE 0.8749  val acc 0.777  rec 0.1126/0.1265
    epoch  31  train CE 0.0326  val CE 0.1016  val acc 0.960  rec 0.1287/0.1340
    epoch  32  train CE 0.0171  val CE 0.1222  val acc 0.937  rec 0.1311/0.1320
    epoch  33  train CE 0.0084  val CE 0.0247  val acc 1.000  rec 0.1280/0.1280
    epoch  34  train CE 0.0040  val CE 0.0133  val acc 1.000  rec 0.1234/0.1234
    epoch  35  train CE 0.0034  val CE 0.0146  val acc 0.997  rec 0.1184/0.1182
    epoch  36  train CE 0.0035  val CE 0.0273  val acc 0.993  rec 0.1128/0.1123
    epoch  37  train CE 0.0033  val CE 0.0079  val acc 1.000  rec 0.1066/0.1058
    epoch  38  train CE 0.0035  val CE 0.0178  val acc 0.997  rec 0.0998/0.0988
    epoch  39  train CE 0.0030  val CE 0.0122  val acc 1.000  rec 0.0921/0.0897
    early stopping
  -> test acc 1.0000  macro-F1 1.0000  CE 0.0047  (best val acc 1.0000 @ epoch 24, 21 s)  test recon rel. L1 0.1566  alpha [343.57, 526.58, 319.16, 164.3]

[UVMD-5 end-to-end with CNN] seed 2
    epoch   1  train CE 1.2148  val CE 1.5714  val acc 0.400  rec 0.2106/0.2118
    epoch   2  train CE 0.4736  val CE 0.4667  val acc 0.893  rec 0.2107/0.2117
    epoch   3  train CE 0.2146  val CE 0.1209  val acc 1.000  rec 0.2100/0.2102
    epoch   4  train CE 0.1299  val CE 0.0564  val acc 1.000  rec 0.2086/0.2088
    epoch   5  train CE 0.0832  val CE 0.2426  val acc 0.900  rec 0.2072/0.2070
    epoch   6  train CE 0.0602  val CE 0.0635  val acc 1.000  rec 0.2053/0.2052
    epoch   7  train CE 0.0527  val CE 0.0320  val acc 1.000  rec 0.2036/0.2033
    epoch   8  train CE 0.0358  val CE 0.0443  val acc 1.000  rec 0.2018/0.2013
    epoch   9  train CE 0.0268  val CE 0.0144  val acc 1.000  rec 0.1998/0.1988
    epoch  10  train CE 0.0217  val CE 0.0148  val acc 1.000  rec 0.1973/0.1963
    epoch  11  train CE 0.0164  val CE 0.0176  val acc 1.000  rec 0.1949/0.1935
    epoch  12  train CE 0.0159  val CE 0.0126  val acc 1.000  rec 0.1918/0.1902
    epoch  13  train CE 0.0118  val CE 0.0101  val acc 1.000  rec 0.1889/0.1872
    epoch  14  train CE 0.0122  val CE 0.0102  val acc 1.000  rec 0.1857/0.1836
    epoch  15  train CE 0.0094  val CE 0.0080  val acc 1.000  rec 0.1820/0.1793
    epoch  16  train CE 0.0102  val CE 0.0337  val acc 0.993  rec 0.1780/0.1749
    epoch  17  train CE 0.0083  val CE 0.0147  val acc 1.000  rec 0.1732/0.1697
    epoch  18  train CE 0.0077  val CE 0.0096  val acc 1.000  rec 0.1683/0.1643
    epoch  19  train CE 0.0092  val CE 0.0138  val acc 1.000  rec 0.1627/0.1578
    epoch  20  train CE 0.0065  val CE 0.0109  val acc 1.000  rec 0.1558/0.1498
    epoch  21  train CE 0.0077  val CE 0.0253  val acc 1.000  rec 0.1469/0.1418
    epoch  22  train CE 0.0082  val CE 0.0242  val acc 1.000  rec 0.1405/0.1355
    epoch  23  train CE 0.0074  val CE 0.0110  val acc 1.000  rec 0.1316/0.1231
    epoch  24  train CE 0.0064  val CE 0.0129  val acc 1.000  rec 0.1209/0.1125
    epoch  25  train CE 0.0152  val CE 0.9805  val acc 0.663  rec 0.1105/0.1139
    epoch  26  train CE 0.0246  val CE 0.1258  val acc 0.973  rec 0.1200/0.1233
    epoch  27  train CE 0.0120  val CE 0.0470  val acc 0.980  rec 0.1217/0.1189
    epoch  28  train CE 0.0055  val CE 0.0108  val acc 1.000  rec 0.1160/0.1131
    epoch  29  train CE 0.0049  val CE 0.0162  val acc 1.000  rec 0.1083/0.1036
    epoch  30  train CE 0.0042  val CE 0.0132  val acc 1.000  rec 0.0968/0.0911
    early stopping
  -> test acc 1.0000  macro-F1 1.0000  CE 0.0075  (best val acc 1.0000 @ epoch 15, 17 s)  test recon rel. L1 0.1785  alpha [495.72, 555.27, 335.97, 259.13]
```
