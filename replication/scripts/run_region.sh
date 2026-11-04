#!/usr/bin/env bash
# One region, the paper's pipeline, exactly as in the upstream README:
#   1. UVMD:  python uvmd.py --K <K> --layers <N>   (reconstruction loss only, trained on 70% of the sensors)
#   2. replace his.npz with the unfolded one (the original is kept as his_raw.npz)
#   3. ASTGCN train + test with --input_dim K+2  (K modes + time-of-day + day-of-week; 15 for K=13)
#   bash replication/scripts/run_region.sh <gba|gla|sd> [device]
# Env: K=13  LAYERS=1 (paper main result is N=1)   SEEDS="2023"   PUSH=1 (commit+push replication/results after each stage)
#      BS=<override batch size>   (default: paper Sec. IV, 4 for GBA/GLA and 48 for SD)
# Output dir: results/<region>_N<N> for K=13, results/<region>_K<K>_N<N> otherwise (see scripts/paper_report.py).
set -euo pipefail
REG=${1:?gba, gla or sd}; DEV=${2:-cuda:0}
K=${K:-13}; LAYERS=${LAYERS:-1}; SEEDS=${SEEDS:-2023}; PUSH=${PUSH:-1}
UP=$(echo "$REG" | tr a-z A-Z)
case "$REG" in sd) BS_DEFAULT=48;; gba|gla) BS_DEFAULT=4;; *) echo "bad region"; exit 1;; esac
BS=${BS:-$BS_DEFAULT}
IN_DIM=$((K + 2))

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="$REPO/replication/magn"
TAG="${REG}_N${LAYERS}"; [ "$K" = "13" ] || TAG="${REG}_K${K}_N${LAYERS}"
OUT="$REPO/replication/results/$TAG"
D="data/$REG/2019"
UD="$D/unfold_K${K}_N${LAYERS}"
cd "$ROOT"
mkdir -p "$OUT"
[ -f "$D/his.npz" ] || [ -f "$D/his_raw.npz" ] || { echo "missing $ROOT/$D/his.npz, run prepare_data.sh first"; exit 1; }

save() {
  cd "$REPO"; git add replication/results
  if git diff --cached --quiet; then echo "[save] nothing new: $1"; cd "$ROOT"; return 0; fi
  git commit -q -m "$1

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
  if [ "$PUSH" = "1" ]; then git pull --rebase -q && git push -q && echo "[save] pushed: $1" || echo "[save] PUSH FAILED (committed locally): $1"; fi
  cd "$ROOT"
}

{ date; hostname; nvidia-smi -L 2>/dev/null || true; python -c "import torch,numpy,pandas;print('torch',torch.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
  git -C "$REPO" rev-parse HEAD; cat "$REPO/replication/UPSTREAM_COMMIT.txt"; echo "K=$K N=$LAYERS bs=$BS input_dim=$IN_DIM seeds=$SEEDS"; } > "$OUT/environment.txt" 2>&1

# ---- 1: UVMD (uvmd.py always reads the ORIGINAL his_raw.npz) ---------------------------------
[ -f "$D/his_raw.npz" ] || cp "$D/his.npz" "$D/his_raw.npz"
if [ ! -f "$UD/his.npz" ]; then
  mkdir -p "$UD"
  /usr/bin/time -v python uvmd.py --K "$K" --layers "$LAYERS" --input "$D/his_raw.npz" --output "$UD/his.npz" --dir "$UD" \
      > "$OUT/uvmd_stdout.log" 2>&1
  tail -n 5 "$OUT/uvmd_stdout.log"
  cp "$UD/training.log" "$UD/model.pth" "$OUT/"
  save "replication: $TAG UVMD trained"
else
  echo "UVMD output for $TAG already exists in $UD (delete it to redo)"
fi

# ---- 2: make it the file ASTGCN reads -------------------------------------------------------
cp "$UD/his.npz" "$D/his.npz"

# ---- 3: ASTGCN -------------------------------------------------------------------------------
for S in $SEEDS; do
  LOG="experiments/astgcn/$UP/record_s$S.log"
  rm -f "$LOG"                                       # one log per run, so the parser never mixes configurations
  python experiments/astgcn/main.py --device "$DEV" --model_name astgcn --dataset "$UP" --years 2019 \
         --bs "$BS" --input_dim "$IN_DIM" --seed "$S"
  python experiments/astgcn/main.py --device "$DEV" --model_name astgcn --dataset "$UP" --years 2019 \
         --bs "$BS" --input_dim "$IN_DIM" --seed "$S" --mode test
  cp "$LOG" "$OUT/record_s$S.log"
  save "replication: $TAG ASTGCN seed $S (bs=$BS, input_dim=$IN_DIM)"
done

python "$REPO/replication/scripts/paper_report.py" --results "$REPO/replication/results" > /dev/null || true
save "replication: $TAG report"
echo "done: $TAG  (report: replication/results/PAPER_STYLE_REPORT.md)"
