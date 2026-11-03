# Prediction comparison: VMD (converged) → CNN  vs  UVMD-5 end-to-end → CNN

Same CNN, training loop and model selection; test = 299 unseen windows (split 'lolo', 10 classes, fault10), seeds [0, 1, 2]. K=4, alpha=500, N=5 layers. A = VMD (converged) (modes not updated by any loss), B = UVMD-5 end-to-end (UVMD updated by CE + 0.1 x reconstruction term).

## Loss and accuracy on the test windows (mean ± std over seeds)

|                   | test_ce         | test_acc        | macro_f1        | balanced_acc    | mean_p_true     | n_wrong   | best_val_acc    |
|:------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------|:----------------|
| VMD (converged)   | 0.0793 ± 0.0216 | 0.9866 ± 0.0058 | 0.9864 ± 0.0061 | 0.9867 ± 0.0058 | 0.9477 ± 0.0044 | 4.0 ± 1.7 | 0.9000 ± 0.0000 |
| UVMD-5 end-to-end | 0.0416 ± 0.0217 | 0.9911 ± 0.0070 | 0.9910 ± 0.0071 | 0.9911 ± 0.0069 | 0.9670 ± 0.0166 | 2.7 ± 2.1 | 0.9000 ± 0.0000 |

Lower is better for `test_ce` and `n_wrong`; higher for the rest. `test_ce` is the mean cross-entropy of the test windows, `mean_p_true` the mean probability given to the true class.

## B minus A, per seed

|              |   B - A (mean over seeds) |   std over seeds |   seeds where B is better |
|:-------------|--------------------------:|-----------------:|--------------------------:|
| test_ce      |                   -0.0377 |           0.0074 |                         3 |
| test_acc     |                    0.0045 |           0.0019 |                         3 |
| macro_f1     |                    0.0046 |           0.0019 |                         3 |
| balanced_acc |                    0.0044 |           0.0019 |                         3 |
| mean_p_true  |                    0.0193 |           0.0129 |                         3 |
| n_wrong      |                   -1.3333 |           0.5774 |                         3 |
| best_val_acc |                    0      |           0      |                         0 |

## Windows right or wrong in both, per seed

|   seed |   both right |   only A right |   only B right |   both wrong |   McNemar exact p |   mean CE A - CE B per window |
|-------:|-------------:|---------------:|---------------:|-------------:|------------------:|------------------------------:|
|      0 |          296 |              0 |              2 |            1 |               0.5 |                        0.0459 |
|      1 |          296 |              0 |              1 |            2 |               1   |                        0.0315 |
|      2 |          290 |              3 |              4 |            2 |               1   |                        0.0358 |

A window is 'right' when the predicted class equals the true class. The McNemar p-value looks only at the windows where exactly one pipeline is right; with a few hundred windows from a handful of recordings, treat it as a rough guide.

## Per-class precision / recall / F1 (pooled over seeds)

| class       |   P VMD (converged) |   R VMD (converged) |   F1 VMD (converged) |   support |   P UVMD-5 end-to-end |   R UVMD-5 end-to-end |   F1 UVMD-5 end-to-end |   F1 B-A |
|:------------|--------------------:|--------------------:|---------------------:|----------:|----------------------:|----------------------:|-----------------------:|---------:|
| normal      |               1     |               1     |                1     |        87 |                 1     |                 1     |                  1     |    0     |
| inner_0.007 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| inner_0.014 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| inner_0.021 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| ball_0.007  |               0.947 |               1     |                0.973 |        90 |                 0.968 |                 1     |                  0.984 |    0.011 |
| ball_0.014  |               1     |               0.867 |                0.929 |        90 |                 1     |                 0.911 |                  0.953 |    0.025 |
| ball_0.021  |               0.928 |               1     |                0.963 |        90 |                 0.947 |                 1     |                  0.973 |    0.01  |
| outer_0.007 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| outer_0.014 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| outer_0.021 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |

## Most frequent confusions (true → predicted: count, pooled over seeds)

| VMD (converged)            | UVMD-5 end-to-end          |
|:---------------------------|:---------------------------|
| ball_0.014 → ball_0.021: 7 | ball_0.014 → ball_0.021: 5 |
| ball_0.014 → ball_0.007: 5 | ball_0.014 → ball_0.007: 3 |
|                            |                            |
|                            |                            |
|                            |                            |
|                            |                            |

Confusion matrices: `figures/confusion_matrices.png`, `confusion_<pipeline>_pooled.csv`. Curves: `figures/loss_and_accuracy.png`.
