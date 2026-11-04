#!/usr/bin/env bash
# Table II / Fig. 3 of the paper on SD with the UNCHANGED upstream code: number of modes K and layers N.
# (K=13, N=1 is the Table I run; do that one first with run_region.sh.)
#   bash replication/scripts/run_table2_sd.sh [device]
# Case I (shared alpha) and Case II (shorter windows) need code the upstream repo does not contain; see README.
set -euo pipefail
DEV=${1:-cuda:0}
HERE="$(cd "$(dirname "$0")" && pwd)"
for cfg in "3 1" "6 1" "9 1" "15 1" "6 2" "13 2"; do
  set -- $cfg
  echo "=== SD  K=$1  N=$2 ==="
  K=$1 LAYERS=$2 bash "$HERE/run_region.sh" sd "$DEV"
done
