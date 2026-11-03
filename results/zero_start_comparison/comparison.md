# Uniform start vs zero start, alpha = 500, UVMD 5 layers (test items of each dataset)

Reconstruction error = ||x - sum of modes|| / ||x||. Second column per dataset: synthetic = error against the TRUE modes; CWRU = orthogonality index; traffic = mode overlap (lower = more separated).


## Synthetic (K=5)

| method       |   reconstruction error % |   error vs true modes % | mode centre frequencies (cycles/sample)   |
|:-------------|-------------------------:|------------------------:|:------------------------------------------|
| VMD uniform  |                    3.71  |                    10.3 | [0.0, 0.047, 0.105, 0.234, 0.405]         |
| VMD zero     |                    4.74  |                    13.2 | [0.0, 0.045, 0.098, 0.104, 0.234]         |
| UVMD uniform |                    0.147 |                    84.5 | [0.0, 0.125, 0.317, 0.444, 0.486]         |
| UVMD zero    |                    0.147 |                    58.5 | [0.0, 0.045, 0.109, 0.159, 0.204]         |

Actual frequencies: [0.003, 0.045, 0.101, 0.234]

## CWRU (K=4)

| method       |   reconstruction error % |   orthogonality index | mode centre frequencies (cycles/sample)   |
|:-------------|-------------------------:|----------------------:|:------------------------------------------|
| VMD uniform  |                  16.7    |                0.0501 | [0.0, 0.13, 0.244, 0.3]                   |
| VMD zero     |                  16.8    |                0.0451 | [0.0, 0.115, 0.217, 0.291]                |
| UVMD uniform |                   0.04   |                0.18   | [0.0, 0.162, 0.231, 0.274]                |
| UVMD zero    |                   0.0397 |                0.0696 | [0.0, 0.081, 0.227, 0.324]                |

## Traffic (K=6)

| method       |   reconstruction error % |   mode overlap | mode centre frequencies (cycles/sample)   |
|:-------------|-------------------------:|---------------:|:------------------------------------------|
| VMD uniform  |                    4.55  |        0.00606 | [0.0, 0.011, 0.027, 0.099, 0.235, 0.4]    |
| VMD zero     |                    4.74  |        0.00599 | [0.0, 0.01, 0.025, 0.062, 0.143, 0.288]   |
| UVMD uniform |                    0.469 |        0.133   | [0.0, 0.066, 0.17, 0.198, 0.272, 0.448]   |
| UVMD zero    |                    0.479 |        0.145   | [0.0, 0.044, 0.125, 0.205, 0.231, 0.358]  |
