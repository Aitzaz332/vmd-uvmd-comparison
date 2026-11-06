#!/usr/bin/env bash
# Plain ASTGCN (no UVMD) on the same SD data, same upstream code, same settings as the MAGN runs. Only the INPUT differs:
# x = [raw flow, time-of-day, day-of-week] (input_dim 3, the LargeST convention) instead of [K modes, time-of-day, day-of-week].
# Upstream's dataloader takes channel 0 as the target and channels 1.. as input, so the file is built as
# [flow(target), flow, tod, dow] from the untouched his_raw.npz; no upstream file is edited. Same split, bs, lr, epochs, seed.
#   bash replication/scripts/run_plain_astgcn.sh sd cuda:0
set -euo pipefail
REG=${1:?sd, gla or gba}; DEV=${2:-cuda:0}; SEEDS=${SEEDS:-2023}; RUNS=${RUNS:-$HOME/magn_runs}
UP=$(echo "$REG" | tr a-z A-Z)
case "$REG" in sd) BS_DEFAULT=48;; gba|gla) BS_DEFAULT=4;; *) echo "bad region"; exit 1;; esac
BS=${BS:-$BS_DEFAULT}
GPU_IDX=${DEV#cuda:}; [ "$GPU_IDX" = "$DEV" ] && GPU_IDX=0
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; SRC="$REPO/replication/magn"
TAG="${REG}_plain"; RUN="$RUNS/$TAG"; OUT="$REPO/replication/results/$TAG"; SD="$SRC/data/$REG"
(cd "$SRC" && sha256sum -c ../UPSTREAM_SHA256.txt --quiet) || { echo "upstream code differs from recorded hashes, stop"; exit 1; }
for f in "$SD/2019/his_raw.npz" "$SD/2019/idx_train.npy" "$SD/2019/idx_val.npy" "$SD/2019/idx_test.npy" "$SD/${REG}_rn_adj.npy"; do
  [ -f "$f" ] || { echo "missing shared input: $f"; exit 1; }
done
if [ -d "$OUT" ] && ls "$OUT"/record_s*.log >/dev/null 2>&1; then echo "results for $TAG exist, not overwriting"; exit 1; fi
mkdir -p "$RUN/data/$REG/2019"
cp -r "$SRC/src" "$SRC/experiments" "$RUN/"
find "$RUN" -name __pycache__ -type d -prune -exec rm -rf {} +
rm -rf "$RUN/experiments/astgcn/$UP"
ln -sfn "$SD/${REG}_rn_adj.npy" "$RUN/data/$REG/${REG}_rn_adj.npy"
for s in train val test; do ln -sfn "$SD/2019/idx_$s.npy" "$RUN/data/$REG/2019/idx_$s.npy"; done
cd "$RUN"; export CUDA_VISIBLE_DEVICES="$GPU_IDX"
python - "$SD/2019/his_raw.npz" "$RUN/data/$REG/2019/his.npz" <<'PY'
import sys, numpy as np
z = np.load(sys.argv[1]); d = z["data"]
assert d.ndim == 3 and d.shape[2] == 3, d.shape            # (T, N, [flow, tod, dow])
out = np.concatenate([d[..., :1], d], axis=-1).astype(np.float32)   # [target flow, flow, tod, dow]
np.savez(sys.argv[2], data=out, mean=z["mean"], std=z["std"])
print("plain input file", out.shape)
PY
{ date; hostname; python -c "import torch;print('torch',torch.__version__)"; git -C "$REPO" rev-parse HEAD
  echo "plain ASTGCN, input = [flow, tod, dow], input_dim=3, bs=$BS, seeds=$SEEDS"; } > environment.txt 2>&1
for S in $SEEDS; do
  python experiments/astgcn/main.py --device cuda:0 --model_name astgcn --dataset "$UP" --years 2019 --bs "$BS" --input_dim 3 --seed "$S"
  python experiments/astgcn/main.py --device cuda:0 --model_name astgcn --dataset "$UP" --years 2019 --bs "$BS" --input_dim 3 --seed "$S" --mode test
done
mkdir -p "$OUT"; cp environment.txt "$OUT/"; cp experiments/astgcn/"$UP"/record_s*.log "$OUT/"
echo "done: $TAG -> $OUT"
