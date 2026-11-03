# Leave-one-load-out: VMD → CNN vs end-to-end UVMD → CNN

Every motor load (0-3 HP) is the test set once; train and validation use the other loads (see `datasets/cwru.py::LOLO_FOLDS`). Same CNN, K, N, alpha, seeds and model selection in every fold.

|   test_load_HP |   acc vmd |   acc uvmd_e2e |   ce vmd |   ce uvmd_e2e |   acc diff (uvmd - vmd) |   ce diff (uvmd - vmd) |
|---------------:|----------:|---------------:|---------:|--------------:|------------------------:|-----------------------:|
|              0 |    0.9866 |         0.9911 |   0.0793 |        0.0416 |                  0.0045 |                -0.0377 |
|              1 |    1      |         0.9989 |   0.0119 |        0.011  |                 -0.0011 |                -0.001  |
|              2 |    0.9689 |         0.9722 |   0.2053 |        0.1769 |                  0.0033 |                -0.0284 |
|              3 |    0.7933 |         0.8    |   1.1397 |        1.0469 |                  0.0067 |                -0.0928 |

Mean over folds: accuracy VMD 0.9372, UVMD end-to-end 0.9405; test CE VMD 0.3591, UVMD end-to-end 0.3191.

UVMD end-to-end has the higher accuracy in 3 of 4 folds (tied in 0) and the lower test loss in 4 of 4. With four folds, treat differences as indications, not as significance tests.