#!/usr/bin/env bash
# Run several (K,N) configurations one after another on ONE GPU, each in its own private folder (run_isolated.sh).
#   bash replication/scripts/run_queue_isolated.sh <region> <device> K,N [K,N ...]
# Example, Table II on SD, first half on the second GPU:
#   bash replication/scripts/run_queue_isolated.sh sd cuda:1 3,1 6,1 9,1
# A failed configuration is reported and the queue goes on with the next one.
set -uo pipefail
REG=${1:?region}; DEV=${2:?device, e.g. cuda:1}; shift 2
[ $# -ge 1 ] || { echo "give at least one K,N"; exit 1; }
HERE="$(cd "$(dirname "$0")" && pwd)"
for cfg in "$@"; do
  K=${cfg%,*}; N=${cfg#*,}
  echo "=== $(date '+%F %T')  $REG  K=$K  N=$N  on $DEV ==="
  K=$K LAYERS=$N bash "$HERE/run_isolated.sh" "$REG" "$DEV" || echo "!!! K=$K N=$N failed or skipped (see message above)"
done
echo "queue finished: $(date '+%F %T')"
