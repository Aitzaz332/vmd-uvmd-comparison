# traffic: zero start vs uniform start (K=6, alpha=500, 5 layers, 30 test items)

| method        |   reconstruction error % |   ortho_index |   mode_overlap |   daily_ownership |
|:--------------|-------------------------:|--------------:|---------------:|------------------:|
| VMD uniform   |                    4.55  |        0.0224 |        0.00606 |             1     |
| VMD zero      |                    4.74  |        0.0182 |        0.00599 |             1     |
| VMD-5 uniform |                    4.43  |        0.058  |        0.0178  |             0.994 |
| VMD-5 zero    |                    5.55  |        0.114  |        0.0646  |             0.952 |
| UVMD uniform  |                    0.469 |        0.0782 |        0.133   |             0.995 |
| UVMD zero     |                    0.479 |        0.098  |        0.145   |             0.99  |

Mean final centre frequencies (cycles/day, modes sorted by frequency):

- VMD uniform: [0.0, 1.01, 2.63, 9.52, 22.56, 38.36]
- VMD zero: [0.0, 1.0, 2.42, 5.94, 13.68, 27.65]
- VMD-5 uniform: [0.0, 3.58, 14.31, 24.05, 32.74, 42.57]
- VMD-5 zero: [0.0, 2.05, 5.86, 9.88, 14.9, 19.66]
- UVMD uniform: [0.0, 6.3, 16.33, 19.04, 26.11, 43.04]
- UVMD zero: [0.0, 4.2, 11.97, 19.68, 22.2, 34.41]

Learned alpha, UVMD zero start: [0.0004, 149.8631, 217.8551, 230.3468, 233.7469, 253.3242]