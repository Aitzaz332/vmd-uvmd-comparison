# Synthetic data: original modes vs VMD vs UVMD

Setting: K = 5 modes (the signals contain 4 true modes, so one mode is extra), alpha = 500 for both methods,
UVMD with 5 layers trained with the original reconstruction loss. 30 unseen test signals.

| file | what it shows |
|---|---|
| `synthetic_side_by_side.png` | one test signal: original modes, VMD and UVMD in three columns (the signal with the median UVMD error) |
| `recon_error_trend.png` | reconstruction error: VMD over its ADMM iterations, UVMD over its training epochs |
| `mode_error_trend.png` | the same two curves, but the error against the true modes (shows what the reconstruction error hides) |
| `synthetic_error_vs_true_modes.md` / `.csv` | the same numbers as a table (mean and standard deviation over the test signals) |
| `synthetic_per_signal.csv` | the numbers for every test signal |

Error = RMSE between a recovered mode and the true mode, as a percentage of the true mode's standard deviation
(lower is better). Each recovered mode is matched one-to-one to the true mode it correlates with most.

Made with `python experiments/mode_figures.py synthetic --dir results/synthetic_K5_alpha500 --depths 5` (modes, table)
and `python experiments/trend_figures.py synthetic --dir results/synthetic_K5_alpha500_L5` (trends).
Training run: `python experiments/run_synthetic.py --K 5 --alpha 500 --layers 5 --vmd-iters 5`
(output in `results/synthetic_K5_alpha500_L5`; the same model as in `results/synthetic_K5_alpha500`, re-run to log the per-epoch error).
