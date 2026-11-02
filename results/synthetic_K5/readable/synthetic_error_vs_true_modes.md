Synthetic, K=5, 30 unseen test signals. Error = RMSE between a recovered mode and the true mode, as % of the true mode's standard deviation (lower is better). Each estimated mode is matched one-to-one to the true mode it correlates with most; the remaining mode is the extra one.

| method   | true mode                         |   error (% of mode size) |   error std |   SNR (dB) |   correlation |
|:---------|:----------------------------------|-------------------------:|------------:|-----------:|--------------:|
| VMD      | trend                             |                   8.41   |      5.52   |    23.3    |         0.995 |
| VMD      | tone                              |                  41.8    |     45.6    |    14.2    |         0.651 |
| VMD      | AM tone                           |                  11.5    |      4.11   |    19.3    |         0.993 |
| VMD      | chirp                             |                  36.6    |     28.7    |    12.2    |         0.883 |
| VMD      | all modes (mean)                  |                  24.6    |    nan      |    17.3    |         0.88  |
| VMD      | sum of modes vs clean signal      |                  16.3    |     14.8    |   nan      |       nan     |
| VMD      | extra mode energy share (K=5 > 4) |                   2.16   |      2.64   |   nan      |       nan     |
| UVMD-3   | trend                             |                  33.5    |      7.1    |     9.75   |         0.947 |
| UVMD-3   | tone                              |                  99.5    |      0.0847 |     0.0459 |         0.117 |
| UVMD-3   | AM tone                           |                  37.6    |     11.8    |     8.88   |         0.929 |
| UVMD-3   | chirp                             |                  15.9    |      3.12   |    16.1    |         0.987 |
| UVMD-3   | all modes (mean)                  |                  46.6    |    nan      |     8.7    |         0.745 |
| UVMD-3   | sum of modes vs clean signal      |                  19.5    |      2.26   |   nan      |       nan     |
| UVMD-3   | extra mode energy share (K=5 > 4) |                   0.0973 |      0.0256 |   nan      |       nan     |
| UVMD-5   | trend                             |                  34.2    |      7.25   |     9.57   |         0.944 |
| UVMD-5   | tone                              |                  99.4    |      0.0795 |     0.0525 |         0.144 |
| UVMD-5   | AM tone                           |                  37.2    |     11.5    |     8.96   |         0.929 |
| UVMD-5   | chirp                             |                  13.4    |      2.71   |    17.6    |         0.991 |
| UVMD-5   | all modes (mean)                  |                  46      |    nan      |     9.06   |         0.752 |
| UVMD-5   | sum of modes vs clean signal      |                  19.3    |      2.26   |   nan      |       nan     |
| UVMD-5   | extra mode energy share (K=5 > 4) |                   0.0971 |      0.0256 |   nan      |       nan     |
