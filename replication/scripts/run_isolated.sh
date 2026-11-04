#!/usr/bin/env bash
# Run ONE configuration (region, K, N) in its own private folder, so it can run next to other runs (e.g. on the second GPU)
# without touching their files. Same pipeline and same upstream code as run_region.sh.
#   K=3 LAYERS=1 bash replication/scripts/run_isolated.sh sd cuda:1
#
# What is shared (read-only): replication/magn/data/<region>/2019/{his_raw.npz, idx_*.npy}, <region>_rn_adj.npy
# What is private to the run (~/magn_runs/<tag>/): a copy of the upstream code (hash-checked), the UVMD output,
#   the ASTGCN log and checkpoint. Nothing in replication/magn/ is written. his.npz in the shared data folder is never touched.
# At the end the logs are copied to replication/results/<tag>/ (no git commands here: commit them yourself when no other
# job is committing, or let the final report step pick them up).
# Env: K=13 LAYERS=1 SEEDS="2023" BS=<override>   RUNS=~/magn_runs   DRYRUN=1 (build the folder, print commands, stop)
set -euo pipefail
REG=${1:?gba, gla or sd}; DEV=${2:-cuda:0}
K=${K:-13}; LAYERS=${LAYERS:-1}; SEEDS=${SEEDS:-2023}; RUNS=${RUNS:-$HOME/magn_runs}; DRYRUN=${DRYRUN:-0}
UP=$(echo "$REG" | tr a-z A-Z)
case "$REG" in sd) BS_DEFAULT=48;; gba|gla) BS_DEFAULT=4;; *) echo "bad region"; exit 1;; esac
BS=${BS:-$BS_DEFAULT}; IN_DIM=$((K + 2))
GPU_IDX=${DEV#cuda:}; [ "$GPU_IDX" = "$DEV" ] && GPU_IDX=0          # 'cuda' alone means GPU 0

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$REPO/replication/magn"
TAG="${REG}_N${LAYERS}"; [ "$K" = "13" ] || TAG="${REG}_K${K}_N${LAYERS}"
RUN="$RUNS/$TAG"
OUT="$REPO/replication/results/$TAG"
SD="$SRC/data/$REG"                                              # shared data folder (read-only use)

# --- guards ------------------------------------------------------------------------------------
(cd "$SRC" && sha256sum -c ../UPSTREAM_SHA256.txt --quiet) || { echo "upstream code differs from the recorded hashes, stop"; exit 1; }
for f in "$SD/2019/his_raw.npz" "$SD/2019/idx_train.npy" "$SD/2019/idx_val.npy" "$SD/2019/idx_test.npy" "$SD/${REG}_rn_adj.npy"; do
  [ -f "$f" ] || { echo "missing shared input: $f (run prepare_data.sh, and wait for the first run to create his_raw.npz)"; exit 1; }
done
if [ -d "$OUT" ] && ls "$OUT"/record_s*.log >/dev/null 2>&1; then echo "results for $TAG already exist in $OUT, not overwriting"; exit 1; fi

# --- private folder -----------------------------------------------------------------------------
mkdir -p "$RUN/data/$REG/2019" "$RUN/unfold"
cp -r "$SRC/src" "$SRC/experiments" "$SRC/uvmd.py" "$RUN/" 2>/dev/null || true
find "$RUN" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf "$RUN/experiments/astgcn/$UP"                           # no logs or checkpoints inherited from the shared copy
ln -sfn "$SD/${REG}_rn_adj.npy" "$RUN/data/$REG/${REG}_rn_adj.npy"
for s in train val test; do ln -sfn "$SD/2019/idx_$s.npy" "$RUN/data/$REG/2019/idx_$s.npy"; done
echo "private run folder: $RUN  (gpu $GPU_IDX, K=$K, N=$LAYERS, bs=$BS, input_dim=$IN_DIM)"
if [ "$DRYRUN" = "1" ]; then ls -la "$RUN" "$RUN/data/$REG" "$RUN/data/$REG/2019"; exit 0; fi

cd "$RUN"
export CUDA_VISIBLE_DEVICES="$GPU_IDX"                         # uvmd.py hard-codes 'cuda' = first visible GPU
{ date; hostname; echo "CUDA_VISIBLE_DEVICES=$GPU_IDX"; python -c "import torch,numpy,pandas;print('torch',torch.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
  git -C "$REPO" rev-parse HEAD; cat "$REPO/replication/UPSTREAM_COMMIT.txt"; echo "K=$K N=$LAYERS bs=$BS input_dim=$IN_DIM seeds=$SEEDS isolated"; } > "$RUN/environment.txt" 2>&1

# 1: UVMD on the original data -> private unfold/his.npz
if [ ! -f "$RUN/unfold/his.npz" ]; then
  /usr/bin/time -v python uvmd.py --K "$K" --layers "$LAYERS" --input "$SD/2019/his_raw.npz" \
      --output "$RUN/unfold/his.npz" --dir "$RUN/unfold" > "$RUN/uvmd_stdout.log" 2>&1
  tail -n 5 "$RUN/uvmd_stdout.log"
fi
ln -sfn "$RUN/unfold/his.npz" "$RUN/data/$REG/2019/his.npz"    # the file ASTGCN reads, private to this run

# 2: ASTGCN train + test
for S in $SEEDS; do
  python experiments/astgcn/main.py --device cuda:0 --model_name astgcn --dataset "$UP" --years 2019 \
         --bs "$BS" --input_dim "$IN_DIM" --seed "$S"
  python experiments/astgcn/main.py --device cuda:0 --model_name astgcn --dataset "$UP" --years 2019 \
         --bs "$BS" --input_dim "$IN_DIM" --seed "$S" --mode test
done

# 3: copy results into the repo (no git)
mkdir -p "$OUT"
cp "$RUN/environment.txt" "$RUN/uvmd_stdout.log" "$OUT/"
cp "$RUN/unfold/training.log" "$RUN/unfold/model.pth" "$OUT/"
cp "$RUN"/experiments/astgcn/"$UP"/record_s*.log "$OUT/"
echo "done: $TAG -> $OUT   (not committed; the unfolded data stays in $RUN/unfold)"
