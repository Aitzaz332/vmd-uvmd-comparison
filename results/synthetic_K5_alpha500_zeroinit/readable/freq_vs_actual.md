# Distance from each true frequency to the nearest recovered centre frequency (cycles/sample, mean over 30 test signals)

True: trend 0.003, tone 0.045, AM tone 0.101, chirp 0.234

| method       |   trend |   tone |   AM tone |   chirp |   mean |
|:-------------|--------:|-------:|----------:|--------:|-------:|
| VMD uniform  |   0.003 |  0.001 |     0     |    0    |  0.001 |
| VMD zero     |   0.003 |  0     |     0     |    0    |  0.001 |
| UVMD uniform |   0.003 |  0.045 |     0.024 |    0.08 |  0.038 |
| UVMD zero    |   0.003 |  0     |     0.008 |    0.03 |  0.01  |
