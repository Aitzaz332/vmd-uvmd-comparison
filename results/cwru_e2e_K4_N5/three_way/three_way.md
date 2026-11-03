# Three-way mode comparison (300 CWRU test windows, K=4, N=5)

Correlation between the modes of two methods, mode k with mode k, averaged over the test windows.

| pair                              |   matched_mean_abs_corr |   mode_1_corr |   mode_2_corr |   mode_3_corr |   mode_4_corr |   aligned_mean_all_modes |
|:----------------------------------|------------------------:|--------------:|--------------:|--------------:|--------------:|-------------------------:|
| end to end vs VMD                 |                   0.885 |         0.973 |         0.895 |         0.807 |         0.823 |                    0.875 |
| reconstruction-only vs VMD        |                   0.776 |         0.835 |         0.667 |         0.702 |         0.595 |                    0.7   |
| end to end vs reconstruction-only |                   0.831 |         0.866 |         0.715 |         0.962 |         0.762 |                    0.827 |

| method                    |   mode |   alpha |   centre_Hz |   centre_std_Hz |
|:--------------------------|-------:|--------:|------------:|----------------:|
| VMD (converged)           |      1 |   500   |         0   |             0   |
| VMD (converged)           |      2 |   500   |      1563.6 |           742.1 |
| VMD (converged)           |      3 |   500   |      3021.3 |           308.2 |
| VMD (converged)           |      4 |   500   |      3517.4 |           497.2 |
| UVMD, reconstruction only |      1 |    85.4 |         0   |             0   |
| UVMD, reconstruction only |      2 |   233.3 |      1969.6 |           587.5 |
| UVMD, reconstruction only |      3 |   268.1 |      3267.5 |           145   |
| UVMD, reconstruction only |      4 |     0   |      2767.7 |           344.7 |
| UVMD, end to end          |      1 |   509.1 |         0   |             0   |
| UVMD, end to end          |      2 |   534.3 |      1573.4 |           574.3 |
| UVMD, end to end          |      3 |   368.9 |      3193.7 |           190.5 |
| UVMD, end to end          |      4 |   229.4 |      3609.2 |           370.5 |