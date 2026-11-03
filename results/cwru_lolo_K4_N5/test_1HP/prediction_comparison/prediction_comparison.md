# Prediction comparison: VMD (converged) → CNN  vs  UVMD-5 end-to-end → CNN

Same CNN, training loop and model selection; test = 300 unseen windows (split 'lolo', 10 classes, fault10), seeds [0, 1, 2]. K=4, alpha=500, N=5 layers. A = VMD (converged) (modes not updated by any loss), B = UVMD-5 end-to-end (UVMD updated by CE + 0.1 x reconstruction term).

## Loss and accuracy on the test windows (mean ± std over seeds)

|                   | test_ce         | test_acc        | macro_f1        | balanced_acc    | mean_p_true     | n_wrong   | best_val_acc    |
|:------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------|:----------------|
| VMD (converged)   | 0.0119 ± 0.0036 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9885 ± 0.0035 | 0.0 ± 0.0 | 0.9989 ± 0.0019 |
| UVMD-5 end-to-end | 0.0110 ± 0.0085 | 0.9989 ± 0.0019 | 0.9989 ± 0.0019 | 0.9989 ± 0.0019 | 0.9904 ± 0.0064 | 0.3 ± 0.6 | 1.0000 ± 0.0000 |

Lower is better for `test_ce` and `n_wrong`; higher for the rest. `test_ce` is the mean cross-entropy of the test windows, `mean_p_true` the mean probability given to the true class.

## B minus A, per seed

|              |   B - A (mean over seeds) |   std over seeds |   seeds where B is better |
|:-------------|--------------------------:|-----------------:|--------------------------:|
| test_ce      |                   -0.001  |           0.005  |                         2 |
| test_acc     |                   -0.0011 |           0.0019 |                         0 |
| macro_f1     |                   -0.0011 |           0.0019 |                         0 |
| balanced_acc |                   -0.0011 |           0.0019 |                         0 |
| mean_p_true  |                    0.0018 |           0.0031 |                         2 |
| n_wrong      |                    0.3333 |           0.5774 |                         0 |
| best_val_acc |                    0.0011 |           0.0019 |                         1 |

## Windows right or wrong in both, per seed

|   seed |   both right |   only A right |   only B right |   both wrong |   McNemar exact p |   mean CE A - CE B per window |
|-------:|-------------:|---------------:|---------------:|-------------:|------------------:|------------------------------:|
|      0 |          299 |              1 |              0 |            0 |                 1 |                       -0.0046 |
|      1 |          300 |              0 |              0 |            0 |                 1 |                        0.0051 |
|      2 |          300 |              0 |              0 |            0 |                 1 |                        0.0024 |

A window is 'right' when the predicted class equals the true class. The McNemar p-value looks only at the windows where exactly one pipeline is right; with a few hundred windows from a handful of recordings, treat it as a rough guide.

## Per-class precision / recall / F1 (pooled over seeds)

| class       |   P VMD (converged) |   R VMD (converged) |   F1 VMD (converged) |   support |   P UVMD-5 end-to-end |   R UVMD-5 end-to-end |   F1 UVMD-5 end-to-end |   F1 B-A |
|:------------|--------------------:|--------------------:|---------------------:|----------:|----------------------:|----------------------:|-----------------------:|---------:|
| normal      |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |
| inner_0.007 |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |
| inner_0.014 |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |
| inner_0.021 |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |
| ball_0.007  |                   1 |                   1 |                    1 |        90 |                 0.989 |                 1     |                  0.994 |   -0.006 |
| ball_0.014  |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |
| ball_0.021  |                   1 |                   1 |                    1 |        90 |                 1     |                 0.989 |                  0.994 |   -0.006 |
| outer_0.007 |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |
| outer_0.014 |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |
| outer_0.021 |                   1 |                   1 |                    1 |        90 |                 1     |                 1     |                  1     |    0     |

## Most frequent confusions (true → predicted: count, pooled over seeds)

| VMD (converged)   | UVMD-5 end-to-end          |
|:------------------|:---------------------------|
|                   | ball_0.021 → ball_0.007: 1 |
|                   |                            |
|                   |                            |
|                   |                            |
|                   |                            |
|                   |                            |

Confusion matrices: `figures/confusion_matrices.png`, `confusion_<pipeline>_pooled.csv`. Curves: `figures/loss_and_accuracy.png`.
