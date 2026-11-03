# Prediction comparison: VMD (converged) → CNN  vs  UVMD-5 end-to-end → CNN

Same CNN, training loop and model selection; test = 300 unseen windows (split 'lolo', 10 classes, fault10), seeds [0, 1, 2]. K=4, alpha=500, N=5 layers. A = VMD (converged) (modes not updated by any loss), B = UVMD-5 end-to-end (UVMD updated by CE + 0.1 x reconstruction term).

## Loss and accuracy on the test windows (mean ± std over seeds)

|                   | test_ce         | test_acc        | macro_f1        | balanced_acc    | mean_p_true     | n_wrong    | best_val_acc    |
|:------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-----------|:----------------|
| VMD (converged)   | 1.1397 ± 0.0829 | 0.7933 ± 0.0000 | 0.7433 ± 0.0000 | 0.7933 ± 0.0000 | 0.7899 ± 0.0023 | 62.0 ± 0.0 | 0.9967 ± 0.0000 |
| UVMD-5 end-to-end | 1.0469 ± 0.2869 | 0.8000 ± 0.0000 | 0.7500 ± 0.0000 | 0.8000 ± 0.0000 | 0.7891 ± 0.0048 | 60.0 ± 0.0 | 1.0000 ± 0.0000 |

Lower is better for `test_ce` and `n_wrong`; higher for the rest. `test_ce` is the mean cross-entropy of the test windows, `mean_p_true` the mean probability given to the true class.

## B minus A, per seed

|              |   B - A (mean over seeds) |   std over seeds |   seeds where B is better |
|:-------------|--------------------------:|-----------------:|--------------------------:|
| test_ce      |                   -0.0928 |           0.2889 |                         2 |
| test_acc     |                    0.0067 |           0      |                         3 |
| macro_f1     |                    0.0067 |           0      |                         3 |
| balanced_acc |                    0.0067 |           0      |                         3 |
| mean_p_true  |                   -0.0009 |           0.0063 |                         2 |
| n_wrong      |                   -2      |           0      |                         3 |
| best_val_acc |                    0.0033 |           0      |                         3 |

## Windows right or wrong in both, per seed

|   seed |   both right |   only A right |   only B right |   both wrong |   McNemar exact p |   mean CE A - CE B per window |
|-------:|-------------:|---------------:|---------------:|-------------:|------------------:|------------------------------:|
|      0 |          238 |              0 |              2 |           60 |               0.5 |                       -0.2132 |
|      1 |          238 |              0 |              2 |           60 |               0.5 |                        0.1309 |
|      2 |          238 |              0 |              2 |           60 |               0.5 |                        0.3608 |

A window is 'right' when the predicted class equals the true class. The McNemar p-value looks only at the windows where exactly one pipeline is right; with a few hundred windows from a handful of recordings, treat it as a rough guide.

## Per-class precision / recall / F1 (pooled over seeds)

| class       |   P VMD (converged) |   R VMD (converged) |   F1 VMD (converged) |   support |   P UVMD-5 end-to-end |   R UVMD-5 end-to-end |   F1 UVMD-5 end-to-end |   F1 B-A |
|:------------|--------------------:|--------------------:|---------------------:|----------:|----------------------:|----------------------:|-----------------------:|---------:|
| normal      |               1     |               1     |                1     |        90 |                 1     |                     1 |                    1   |    0     |
| inner_0.007 |               1     |               1     |                1     |        90 |                 1     |                     1 |                    1   |    0     |
| inner_0.014 |               0     |               0     |                0     |        90 |                 0     |                     0 |                    0   |    0     |
| inner_0.021 |               1     |               1     |                1     |        90 |                 1     |                     1 |                    1   |    0     |
| ball_0.007  |               0.938 |               1     |                0.968 |        90 |                 1     |                     1 |                    1   |    0.032 |
| ball_0.014  |               0.333 |               1     |                0.5   |        90 |                 0.333 |                     1 |                    0.5 |    0     |
| ball_0.021  |               0     |               0     |                0     |        90 |                 0     |                     0 |                    0   |    0     |
| outer_0.007 |               1     |               1     |                1     |        90 |                 1     |                     1 |                    1   |    0     |
| outer_0.014 |               1     |               0.933 |                0.966 |        90 |                 1     |                     1 |                    1   |    0.034 |
| outer_0.021 |               1     |               1     |                1     |        90 |                 1     |                     1 |                    1   |    0     |

## Most frequent confusions (true → predicted: count, pooled over seeds)

| VMD (converged)              | UVMD-5 end-to-end            |
|:-----------------------------|:-----------------------------|
| inner_0.014 → ball_0.014: 90 | inner_0.014 → ball_0.014: 90 |
| ball_0.021 → ball_0.014: 90  | ball_0.021 → ball_0.014: 90  |
| outer_0.014 → ball_0.007: 6  |                              |
|                              |                              |
|                              |                              |
|                              |                              |

Confusion matrices: `figures/confusion_matrices.png`, `confusion_<pipeline>_pooled.csv`. Curves: `figures/loss_and_accuracy.png`.
