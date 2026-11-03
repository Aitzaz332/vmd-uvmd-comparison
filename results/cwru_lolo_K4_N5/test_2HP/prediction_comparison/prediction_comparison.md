# Prediction comparison: VMD (converged) → CNN  vs  UVMD-5 end-to-end → CNN

Same CNN, training loop and model selection; test = 300 unseen windows (split 'lolo', 10 classes, fault10), seeds [0, 1, 2]. K=4, alpha=500, N=5 layers. A = VMD (converged) (modes not updated by any loss), B = UVMD-5 end-to-end (UVMD updated by CE + 0.1 x reconstruction term).

## Loss and accuracy on the test windows (mean ± std over seeds)

|                   | test_ce         | test_acc        | macro_f1        | balanced_acc    | mean_p_true     | n_wrong    | best_val_acc    |
|:------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-----------|:----------------|
| VMD (converged)   | 0.2053 ± 0.1709 | 0.9689 ± 0.0453 | 0.9631 ± 0.0554 | 0.9689 ± 0.0453 | 0.8424 ± 0.1215 | 9.3 ± 13.6 | 0.8656 ± 0.0184 |
| UVMD-5 end-to-end | 0.1769 ± 0.1012 | 0.9722 ± 0.0201 | 0.9711 ± 0.0214 | 0.9722 ± 0.0201 | 0.8674 ± 0.0906 | 8.3 ± 6.0  | 0.8478 ± 0.0455 |

Lower is better for `test_ce` and `n_wrong`; higher for the rest. `test_ce` is the mean cross-entropy of the test windows, `mean_p_true` the mean probability given to the true class.

## B minus A, per seed

|              |   B - A (mean over seeds) |   std over seeds |   seeds where B is better |
|:-------------|--------------------------:|-----------------:|--------------------------:|
| test_ce      |                   -0.0284 |           0.1948 |                         1 |
| test_acc     |                    0.0033 |           0.0643 |                         1 |
| macro_f1     |                    0.008  |           0.0754 |                         1 |
| balanced_acc |                    0.0033 |           0.0643 |                         1 |
| mean_p_true  |                    0.025  |           0.1419 |                         2 |
| n_wrong      |                   -1      |          19.2873 |                         1 |
| best_val_acc |                   -0.0178 |           0.0593 |                         1 |

## Windows right or wrong in both, per seed

|   seed |   both right |   only A right |   only B right |   both wrong |   McNemar exact p |   mean CE A - CE B per window |
|-------:|-------------:|---------------:|---------------:|-------------:|------------------:|------------------------------:|
|      0 |          285 |             14 |              1 |            0 |            0.001  |                       -0.0295 |
|      1 |          289 |              9 |              2 |            0 |            0.0654 |                       -0.1309 |
|      2 |          275 |              0 |             23 |            2 |            0      |                        0.2455 |

A window is 'right' when the predicted class equals the true class. The McNemar p-value looks only at the windows where exactly one pipeline is right; with a few hundred windows from a handful of recordings, treat it as a rough guide.

## Per-class precision / recall / F1 (pooled over seeds)

| class       |   P VMD (converged) |   R VMD (converged) |   F1 VMD (converged) |   support |   P UVMD-5 end-to-end |   R UVMD-5 end-to-end |   F1 UVMD-5 end-to-end |   F1 B-A |
|:------------|--------------------:|--------------------:|---------------------:|----------:|----------------------:|----------------------:|-----------------------:|---------:|
| normal      |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| inner_0.007 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| inner_0.014 |               1     |               0.722 |                0.839 |        90 |                 1     |                 0.722 |                  0.839 |    0     |
| inner_0.021 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| ball_0.007  |               0.989 |               0.978 |                0.983 |        90 |                 1     |                 1     |                  1     |    0.017 |
| ball_0.014  |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| ball_0.021  |               0.978 |               0.989 |                0.983 |        90 |                 1     |                 1     |                  1     |    0.017 |
| outer_0.007 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| outer_0.014 |               1     |               1     |                1     |        90 |                 1     |                 1     |                  1     |    0     |
| outer_0.021 |               0.783 |               1     |                0.878 |        90 |                 0.783 |                 1     |                  0.878 |    0     |

## Most frequent confusions (true → predicted: count, pooled over seeds)

| VMD (converged)               | UVMD-5 end-to-end             |
|:------------------------------|:------------------------------|
| inner_0.014 → outer_0.021: 25 | inner_0.014 → outer_0.021: 25 |
| ball_0.007 → ball_0.021: 2    |                               |
| ball_0.021 → ball_0.007: 1    |                               |
|                               |                               |
|                               |                               |
|                               |                               |

Confusion matrices: `figures/confusion_matrices.png`, `confusion_<pipeline>_pooled.csv`. Curves: `figures/loss_and_accuracy.png`.
