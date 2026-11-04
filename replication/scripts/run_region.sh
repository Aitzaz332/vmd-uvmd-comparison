#!/usr/bin/env bash
# One region, the paper's pipeline, exactly as in the upstream README:
#   1. UVMD:  python uvmd.py --K 13 --layers 1     (reconstruction loss only, trained on 70% of the sensors)
#   2. replace his.npz with the unfolded one (the original is kept as his_raw.npz)
#   3. ASTGCN train + test with --input_dim 15
#   bash replication/scripts/run_region.sh <gba|gla|sd> [device]
# Env: LAYERS=1 (paper main result is N=1)   SEEDS="2023"   PUSH=1 (commit+push replication/results after each stage)
#      BS=<override batch size>   (default: paper Sec. IV, 4 for GBA/GLA and 48 for SD)
set -euo pipefail
REG=${1:?gba, gla or sd}; DEV=${2:-cuda:0}
LAYERS=${LAYERS:-1}; SEEDS=${SEEDS:-2023}; PUSH=${PUSH:-1}
UP=$(echo "$REG" | tr a-z A-Z)
case "$REG" in sd) BS_DEFAULT=48;; gba|gla) BS_DEFAULT=4;; *) echo "bad region"; exit 1;; esac
BS=${BS:-$BS_DEFAULT}

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$REPO/replication/magn"
OUT="$REPO/replication/results/${REG}_N${LAYERS}"
D="data/$REG/2019"
cd "$ROOT"
mkdir -p "$OUT"
[ -f "$D/his.npz" ] || { echo "missing $ROOT/$D/his.npz, run prepare_data.sh first"; exit 1; }

save() {
  cd "$REPO"; git add replication/results
  if git diff --cached --quiet; then echo "[save] nothing new: $1"; cd "$ROOT"; return 0; fi
  git commit -q -m "$1"
  if [ "$PUSH" = "1" ]; then git pull --rebase -q && git push -q && echo "[save] pushed: $1" || echo "[save] PUSH FAILED (committed locally): $1"; fi
  cd "$ROOT"
}

{ date; hostname; nvidia-smi -L 2>/dev/null || true; python -c "import torch,numpy,pandas;print('torch',torch.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
  git -C "$REPO" rev-parse HEAD; cat "$REPO/replication/UPSTREAM_COMMIT.txt"; } > "$OUT/environment.txt" 2>&1

# ---- 1 + 2: UVMD ----------------------------------------------------------------------------
if [ ! -f "$D/.uvmd_done_N$LAYERS" ]; then
  [ -f "$D/his_raw.npz" ] || cp "$D/his.npz" "$D/his_raw.npz"
  UD="$D/unfold_N$LAYERS"; mkdir -p "$UD"
  /usr/bin/time -v python uvmd.py --K 13 --layers "$LAYERS" --input "$D/his_raw.npz" --output "$UD/his.npz" --dir "$UD" \
      2>&1 | tee "$OUT/uvmd_stdout.log" | tail -n 5
  cp "$UD/training.log" "$UD/model.pth" "$OUT/"
  cp "$UD/his.npz" "$D/his.npz"                       # the file ASTGCN reads
  touch "$D/.uvmd_done_N$LAYERS"
  save "replication: $REG UVMD (K=13, N=$LAYERS) trained"
else
  echo "UVMD for $REG N=$LAYERS already done (delete $D/.uvmd_done_N$LAYERS to redo)"
fi

# ---- 3: ASTGCN -------------------------------------------------------------------------------
for S in $SEEDS; do
  LOG="experiments/astgcn/$UP/record_s$S.log"
  python experiments/astgcn/main.py --device "$DEV" --model_name astgcn --dataset "$UP" --years 2019 \
         --bs "$BS" --input_dim 15 --seed "$S"
  python experiments/astgcn/main.py --device "$DEV" --model_name astgcn --dataset "$UP" --years 2019 \
         --bs "$BS" --input_dim 15 --seed "$S" --mode test
  cp "$LOG" "$OUT/record_s$S.log"
  save "replication: $REG ASTGCN seed $S (bs=$BS, input_dim=15)"
done

python "$REPO/replication/scripts/compare_to_paper.py" --results "$REPO/replication/results" --regions "$REG" \
       --layers "$LAYERS" | tee "$OUT/comparison.md"
save "replication: $REG comparison with paper Table I"
