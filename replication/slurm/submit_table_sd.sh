#!/usr/bin/env bash
# Submit the SD runs of Table I (K=13,N=1) and Table II (K=3,6,9,15 with N=1; K=6,13 with N=2) as separate SLURM jobs.
# Two RTX 4090 nodes exist, so two jobs run at a time and the rest wait in the queue.
#   bash replication/slurm/submit_table_sd.sh            # all seven
#   bash replication/slurm/submit_table_sd.sh "13 1"     # only the listed "K N" pairs
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p replication/slurm_logs
CFGS=("$@"); [ ${#CFGS[@]} -gt 0 ] || CFGS=("13 1" "3 1" "6 1" "9 1" "15 1" "6 2" "13 2")
for cfg in "${CFGS[@]}"; do
  set -- $cfg
  sbatch --job-name="magn_sd_K$1_N$2" --export=ALL,REG=sd,K=$1,LAYERS=$2 replication/slurm/run_config.sbatch
done
squeue -u "$(id -un)"
