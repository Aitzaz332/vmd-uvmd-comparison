# Centre frequencies, start -> end (cycles/sample, mean over 30 test signals, modes sorted)

True bands (mean): trend 0.003, tone 0.045, AM tone 0.101, chirp 0.234

| method                       | mode 1         | mode 2         | mode 3         | mode 4         | mode 5         |   total movement |
|:-----------------------------|:---------------|:---------------|:---------------|:---------------|:---------------|-----------------:|
| UVMD uniform start           | 0.000 -> 0.000 | 0.100 -> 0.125 | 0.200 -> 0.317 | 0.300 -> 0.444 | 0.400 -> 0.486 |            0.372 |
| UVMD zero start              | 0.000 -> 0.000 | 0.000 -> 0.045 | 0.000 -> 0.109 | 0.000 -> 0.159 | 0.000 -> 0.204 |            0.518 |
| VMD converged, uniform start | 0.000 -> 0.000 | 0.100 -> 0.047 | 0.200 -> 0.105 | 0.300 -> 0.234 | 0.400 -> 0.405 |            0.219 |
| VMD converged, zero start    | 0.000 -> 0.000 | 0.000 -> 0.045 | 0.000 -> 0.098 | 0.000 -> 0.104 | 0.000 -> 0.234 |            0.482 |
