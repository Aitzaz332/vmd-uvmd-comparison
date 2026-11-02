# Starting all centre frequencies at zero (synthetic, K=5, alpha=500)

| method        | start   |   reconstruction error % |   error vs true modes % (mean of 4) |   trend % |   tone % |   AM tone % |   chirp % |
|:--------------|:--------|-------------------------:|------------------------------------:|----------:|---------:|------------:|----------:|
| VMD converged | uniform |                    3.71  |                                10.3 |      5.14 |    10.9  |        9.94 |      15.4 |
| VMD converged | zero    |                    4.74  |                                13.2 |      4.55 |     8.56 |       27.7  |      11.9 |
| VMD-5         | uniform |                   17.2   |                                45.8 |     28.7  |    94.6  |       44.4  |      15.4 |
| VMD-5         | zero    |                   12.5   |                                35   |      9.87 |    25.2  |       41.3  |      63.5 |
| UVMD-5        | uniform |                    0.147 |                                84.5 |     47.4  |    98.7  |       91.7  |     100   |
| UVMD-5        | zero    |                    0.147 |                                58.5 |     10.2  |    30.1  |       66.6  |     127   |

Mean final centre frequencies (cycles/sample, modes sorted), zero start:

- VMD converged: [0.    0.045 0.098 0.104 0.234]
- VMD-5: [0.    0.047 0.101 0.105 0.178]
- UVMD-5: [0.    0.045 0.109 0.159 0.204]

Learned alpha, UVMD zero start: [1.9751e-04 3.2985e+02 3.8066e+02 3.9562e+02 4.1333e+02]
