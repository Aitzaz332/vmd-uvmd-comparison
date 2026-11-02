Synthetic, K=5, 30 unseen test signals. Error = RMSE between a recovered mode and the true mode, as % of the true mode's standard deviation (lower is better). Each estimated mode is matched one-to-one to the true mode it correlates with most; the remaining mode is the extra one.

| method   | true mode                         |   error (% of mode size) |   error std |   SNR (dB) |   correlation |
|:---------|:----------------------------------|-------------------------:|------------:|-----------:|--------------:|
| VMD      | trend                             |                  5.14    |    2.98     |  26.7      |        0.998  |
| VMD      | tone                              |                 10.9     |   16.8      |  21.6      |        0.966  |
| VMD      | AM tone                           |                  9.94    |    5.24     |  20.6      |        0.994  |
| VMD      | chirp                             |                 15.4     |   12.9      |  17.6      |        0.979  |
| VMD      | all modes (mean)                  |                 10.3     |  nan        |  21.6      |        0.984  |
| VMD      | sum of modes vs clean signal      |                  6.43    |    1.93     | nan        |      nan      |
| VMD      | extra mode energy share (K=5 > 4) |                  0.453   |    1.38     | nan        |      nan      |
| UVMD-5   | trend                             |                 47.4     |    9.8      |   6.73     |        0.902  |
| UVMD-5   | tone                              |                 98.7     |    4.81     |   0.121    |        0.0577 |
| UVMD-5   | AM tone                           |                 91.7     |   15.8      |   0.871    |        0.668  |
| UVMD-5   | chirp                             |                100       |    0.000248 |   4.88e-05 |        0.443  |
| UVMD-5   | all modes (mean)                  |                 84.5     |  nan        |   1.93     |        0.518  |
| UVMD-5   | sum of modes vs clean signal      |                  8.08    |    0.92     | nan        |      nan      |
| UVMD-5   | extra mode energy share (K=5 > 4) |                  1.2e-09 |    2.28e-10 | nan        |      nan      |
