# omega shift (sd, 30 test sensors, K=13); units: cycles/day (cpd)

Initial grid: omega_k = 0.5/K * k cycles/sample = 3.69 cpd apart, 0.0 .. 44.3 cpd. Mode 0 is the DC mode and does not move.

Check that the rebuilt UVMD models are the trained ones: mean relative reconstruction error (bounded run reported 0.0205 for UVMD-1 and 0.0372 for UVMD-2):

* VMD-1: 0.0729
* VMD-2: 0.0838
* VMD converged: 0.0307
* UVMD-1: 0.0205
* UVMD-2: 0.0372

## Final omega_k, mean over sensors (cpd)

|    |   init |   VMD-1 |   VMD-2 |   VMD converged |   UVMD-1 |   UVMD-2 |
|---:|-------:|--------:|--------:|----------------:|---------:|---------:|
|  0 |   0.00 |    0.00 |    0.00 |            0.00 |     0.00 |     0.00 |
|  1 |   3.69 |    2.96 |    2.42 |            1.00 |     3.13 |     2.97 |
|  2 |   7.38 |    7.23 |    6.80 |            2.10 |     7.72 |     7.29 |
|  3 |  11.08 |   11.37 |   11.06 |            3.42 |    12.11 |    11.64 |
|  4 |  14.77 |   15.37 |   15.13 |            5.41 |    16.28 |    15.66 |
|  5 |  18.46 |   19.30 |   19.11 |            7.96 |    20.31 |    19.64 |
|  6 |  22.15 |   23.13 |   23.14 |           11.42 |    23.85 |    23.40 |
|  7 |  25.85 |   26.33 |   26.18 |           18.80 |    27.21 |    26.79 |
|  8 |  29.54 |   30.32 |   30.16 |           23.44 |    31.27 |    30.51 |
|  9 |  33.23 |   34.07 |   33.95 |           27.93 |    35.04 |    34.26 |
| 10 |  36.92 |   37.78 |   37.69 |           33.48 |    38.77 |    38.05 |
| 11 |  40.62 |   41.47 |   41.46 |           39.45 |    42.43 |    42.03 |
| 12 |  44.31 |   45.03 |   45.33 |           45.50 |    45.79 |    46.02 |

## How far omega_k moved from the start (modes 1..K-1)

|               |   mean abs shift (cpd) |   max abs shift (cpd) |   mean abs shift per grid spacing |   modes that moved > 0.1 cpd |
|:--------------|-----------------------:|----------------------:|----------------------------------:|-----------------------------:|
| VMD-1         |                  0.687 |                 1.314 |                             0.186 |                       12.000 |
| VMD-2         |                  0.698 |                 1.897 |                             0.189 |                       12.000 |
| VMD converged |                  5.933 |                13.581 |                             1.607 |                       12.000 |
| UVMD-1        |                  1.423 |                 2.481 |                             0.385 |                       12.000 |
| UVMD-2        |                  1.012 |                 2.240 |                             0.274 |                       12.000 |
