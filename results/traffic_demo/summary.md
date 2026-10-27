# Traffic benchmark - VMD vs UVMD (demo (generated stand-in))

60 nodes x 2688 samples (28 days at 15 min), standardised flow. Node split train/val/test = 6/3/4 (test nodes scored). K=13, alpha=2000.0, tau=0.0, UVMD device cpu.
**Generated stand-in data - not real traffic.**

## Accuracy and mode structure (mean over test nodes)

| method | recon_rel_err | recon_snr_db | spec_l1 | spec_l2 | energy_ratio | ortho_index | mean_bandwidth_cpd | mode_overlap | n_active_modes | daily_ownership | daily_capture | weekly_ownership | weekly_capture |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VMD (converged) | 0.05807 | 24.73 | 7190 | 243.4 | 0.7701 | 0.08812 | 0.3994 | 0.006486 | 4 | 0.7332 | 0.5002 | 0.9998 | 0.9602 |
| VMD-1 | 0.09708 | 20.28 | 5298 | 406.5 | 0.5646 | 0.1404 | 0.8796 | 0.03418 | 2 | 0.9927 | 0.6802 | 1 | 0.9911 |
| VMD-2 | 0.09531 | 20.42 | 7773 | 402 | 0.5883 | 0.1327 | 0.684 | 0.01221 | 2 | 0.9709 | 0.5742 | 1 | 0.9876 |
| UVMD-1 untrained | 0.09708 | 20.28 | 5298 | 406.5 | 0.5646 | 0.1404 | 0.8796 | 0.03418 | 2 | 0.9927 | 0.6802 | 1 | 0.9911 |
| UVMD-1 | 0.09571 | 20.4 | 5265 | 401.9 | 0.5663 | 0.141 | 0.8801 | 0.03422 | 2 | 0.9927 | 0.6827 | 1 | 0.9967 |
| UVMD-2 untrained | 0.09531 | 20.42 | 7773 | 402 | 0.5883 | 0.1327 | 0.684 | 0.01221 | 2 | 0.9709 | 0.5742 | 1 | 0.9876 |
| UVMD-2 | 0.09382 | 20.56 | 7737 | 397 | 0.5902 | 0.1334 | 0.6849 | 0.01223 | 2 | 0.9709 | 0.5766 | 1 | 0.994 |

## Paired comparison, trained UVMD-N vs VMD-N (share of nodes where UVMD is better)

|                 |   recon_rel_err |   mode_overlap |   mean_bandwidth_cpd |
|:----------------|----------------:|---------------:|---------------------:|
| UVMD-1 vs VMD-1 |            1.00 |           0.00 |                 0.00 |
| UVMD-2 vs VMD-2 |            1.00 |           0.00 |                 0.00 |

## Cost

| method           |   iterations_or_layers |   trainable_params |   flops_per_node |   mean_s_per_node |   median_s_per_node | device   |   train_time_s |   train_flops |   est_min_region |   est_min_6902_sensors |   speedup_vs_VMD_converged |
|:-----------------|-----------------------:|-------------------:|-----------------:|------------------:|--------------------:|:---------|---------------:|--------------:|-----------------:|-----------------------:|---------------------------:|
| VMD (converged)  |                    499 |                  0 |        9.114e+08 |          0.747    |            0.7479   | cpu      |           0    |     0         |         0.747    |                85.93   |                      1     |
| VMD-1            |                      1 |                  0 |        6.481e+06 |          0.002937 |            0.002916 | cpu      |           0    |     0         |         0.002937 |                 0.3378 |                    254.4   |
| VMD-2            |                      2 |                  0 |        8.298e+06 |          0.004548 |            0.004518 | cpu      |           0    |     0         |         0.004548 |                 0.5232 |                    164.2   |
| VMD-20           |                     20 |                  0 |        4.101e+07 |          0.03089  |            0.03152  | cpu      |           0    |     0         |         0.03089  |                 3.553  |                     24.19  |
| UVMD-1 untrained |                      1 |              21517 |        5.991e+06 |          0.2516   |            0.2476   | cpu      |           0    |     0         |         0.2516   |                28.94   |                      2.969 |
| UVMD-1           |                      1 |              21517 |        5.991e+06 |          0.2811   |            0.3077   | cpu      |          18.36 |     3.235e+08 |         0.2811   |                32.33   |                      2.658 |
| UVMD-2 untrained |                      2 |              32269 |        7.319e+06 |          0.4893   |            0.4972   | cpu      |           0    |     0         |         0.4893   |                56.29   |                      1.527 |
| UVMD-2           |                      2 |              32269 |        7.319e+06 |          0.4398   |            0.4415   | cpu      |          37.42 |     3.952e+08 |         0.4398   |                50.59   |                      1.698 |

`est_min_*` = mean seconds per node x number of nodes (an extrapolation, not a measurement). 6902 = GBA+GLA+SD sensors of the paper. VMD is NumPy on CPU; UVMD runs on the device above.

Environment: {'python': '3.13.12', 'platform': 'Linux-5.15.0-190-generic-x86_64-with-glibc2.35', 'numpy': '2.1.3', 'torch': '2.7.0', 'cuda': True, 'gpu': 'NVIDIA GeForce RTX 3060'}


## Log
```
traffic [demo (generated stand-in)]: 60 nodes x 2688 samples (28 days) | node split train/val/test = 6/3/4 | K=13, alpha=2000.0, device cpu
  NOTE: generated stand-in data, only for checking the installation. Results are NOT real traffic.

[VMD converged] (NumPy, CPU)
  4/4 nodes, 0.75 s/node, mean 499 iterations
  rel err 0.0581, mode overlap 0.006, daily-line ownership 0.73, 0% converged

[VMD-n sweep]
  n=  1 (4 nodes): rel err 0.0971  overlap 0.034  0.003 s/node
  n=  2 (4 nodes): rel err 0.0953  overlap 0.012  0.005 s/node
  n= 20 (3 nodes): rel err 0.0735  overlap 0.009  0.031 s/node

[UVMD N=1] upstream Unfolding, K=13, device cpu
  trainable parameters: 10765 entries (21517 real DOF)
  UVMD-1 untrained: rel err 0.0971  overlap 0.034  daily-line ownership 0.99  0.252 s/node
  epoch   1/3  train 3968.9164  val 5056.1252
  epoch   2/3  train 3855.8337  val 5049.2516
  epoch   3/3  train 3787.3765  val 5040.2593
  training: 3 epochs, 18.4 s
  UVMD-1: rel err 0.0957  overlap 0.034  daily-line ownership 0.99  0.281 s/node
  learned alpha_k: [1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998]

[UVMD N=2] upstream Unfolding, K=13, device cpu
  trainable parameters: 16141 entries (32269 real DOF)
  UVMD-2 untrained: rel err 0.0953  overlap 0.012  daily-line ownership 0.97  0.489 s/node
  epoch   1/3  train 5858.4317  val 7407.4119
  epoch   2/3  train 5698.1461  val 7402.2104
  epoch   3/3  train 5593.0598  val 7399.6147
  training: 3 epochs, 37.4 s
  UVMD-2: rel err 0.0938  overlap 0.012  daily-line ownership 0.97  0.440 s/node
  learned alpha_k: [1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998, 1998]
```
