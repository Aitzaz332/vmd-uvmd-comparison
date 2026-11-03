# cwru: zero start vs uniform start (K=4, alpha=500, 5 layers, 300 test items)

| method        |   reconstruction error % |   ortho_index |   fault_snr_db |   envelope_hit |
|:--------------|-------------------------:|--------------:|---------------:|---------------:|
| VMD uniform   |                  16.7    |        0.0501 |           7.41 |          0.696 |
| VMD zero      |                  16.8    |        0.0451 |           7.42 |          0.674 |
| VMD-5 uniform |                  20.2    |        0.0903 |           8.59 |          0.719 |
| VMD-5 zero    |                  24.1    |        0.154  |           8.7  |          0.7   |
| UVMD uniform  |                   0.04   |        0.18   |           8.57 |          0.715 |
| UVMD zero     |                   0.0397 |        0.0696 |           8.55 |          0.711 |

Mean final centre frequencies (Hz, modes sorted by frequency):

- VMD uniform: [0.0, 1563.57, 2933.56, 3605.18]
- VMD zero: [0.0, 1385.15, 2602.63, 3497.8]
- VMD-5 uniform: [0.0, 1638.82, 3195.17, 3721.87]
- VMD-5 zero: [0.0, 1523.3, 2712.29, 3198.41]
- UVMD uniform: [0.0, 1943.12, 2769.35, 3292.32]
- UVMD zero: [0.0, 971.59, 2726.96, 3885.78]

Learned alpha, UVMD zero start: [0.0, 145.4349, 323.1838, 325.1473]