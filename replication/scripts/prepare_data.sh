#!/usr/bin/env bash
# Build the MAGN training data for one region with the AUTHORS' notebooks and script (nothing re-implemented).
#   bash replication/scripts/prepare_data.sh <dir with the unzipped Kaggle LargeST data> <gba|gla|sd>
# Result: replication/magn/data/<region>/2019/{his.npz, idx_train.npy, idx_val.npy, idx_test.npy}
#         replication/magn/data/<region>/<region>_rn_adj.npy
# Needs about 10 GB RAM for the California file (the notebook reads the whole 5-minute table).
set -euo pipefail
LARGEST=${1:?directory with the unzipped LargeST download}
REG=${2:?gba, gla or sd}
case "$REG" in gba|gla|sd) ;; *) echo "region must be gba, gla or sd"; exit 1;; esac

ROOT="$(cd "$(dirname "$0")/../magn" && pwd)"
DATA="$ROOT/data"
SCRATCH="${TMPDIR:-/tmp}/magn_nb_$$"; mkdir -p "$SCRATCH"; trap 'rm -rf "$SCRATCH"' EXIT

find_one() { find -L "$LARGEST" -name "$1" -print -quit; }
RAW=$(find_one ca_his_raw_2019.h5); META=$(find_one ca_meta.csv); ADJ=$(find_one ca_rn_adj.npy)
for v in RAW META ADJ; do [ -n "${!v}" ] || { echo "missing $v under $LARGEST (need ca_his_raw_2019.h5, ca_meta.csv, ca_rn_adj.npy)"; exit 1; }; done
echo "raw=$RAW"; echo "meta=$META"; echo "adj=$ADJ"

# The notebooks expect:  data/ca_meta.csv   data/ca/ca_rn_adj.npy   data/ca/ca_his_raw_2019.h5
ln -sf "$META" "$DATA/ca_meta.csv"
ln -sf "$ADJ"  "$DATA/ca/ca_rn_adj.npy"
ln -sf "$RAW"  "$DATA/ca/ca_his_raw_2019.h5"

# Run a notebook as a script from its own folder. The region notebooks ship with  year = ''  and say to set 2019.
run_nb() {  # run_nb <folder> <notebook>
  local dir="$1" nb="$2"
  jupyter nbconvert --to script "$DATA/$dir/$nb" --output-dir "$SCRATCH" --output "${nb%.ipynb}" >/dev/null
  sed -E "s/^year = ''/year = '2019'/" "$SCRATCH/${nb%.ipynb}.py" > "$SCRATCH/${nb%.ipynb}_2019.py"
  grep -q "^year = '2019'" "$SCRATCH/${nb%.ipynb}_2019.py" || { echo "year not set in $nb"; exit 1; }
  (cd "$DATA/$dir" && python "$SCRATCH/${nb%.ipynb}_2019.py")
}

[ -f "$DATA/ca/ca_his_2019.h5" ] || run_nb ca process_ca_his.ipynb          # 15-min resample, round, fill NaN with 0
run_nb "$REG" "generate_${REG}_dataset.ipynb"                               # region h5 + adjacency
(cd "$DATA" && python generate_data_for_training.py --dataset "$REG" --years 2019)

python - "$DATA/$REG/2019/his.npz" "$REG" <<'E'
import sys, numpy as np
d = np.load(sys.argv[1]); x = d["data"]
want = {"gba": 2352, "gla": 3834, "sd": 716}[sys.argv[2]]
print("his.npz data shape", x.shape, "(expect T=35040, N=%d, F=3)" % want)
assert x.shape == (35040, want, 3), "shape differs from the paper's setup, stop here"
print("ok")
E
