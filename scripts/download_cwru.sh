#!/usr/bin/env bash
# Download the default CWRU subset (40 files, ~135 MB): normal baseline + 12 kHz drive-end
# inner race / ball / outer race @6:00, fault sizes 0.007/0.014/0.021 in, loads 0-3 HP.
# Usage: bash scripts/download_cwru.sh [target_dir]   (default ~/data/CWRU)
set -euo pipefail
DIR=${1:-$HOME/data/CWRU}
BASE=https://engineering.case.edu/sites/default/files
mkdir -p "$DIR"
FILES="97 98 99 100 \
105 106 107 108 118 119 120 121 130 131 132 133 \
169 170 171 172 185 186 187 188 197 198 199 200 \
209 210 211 212 222 223 224 225 234 235 236 237"
for f in $FILES; do
  if [ ! -s "$DIR/$f.mat" ]; then
    echo "downloading $f.mat"
    curl -fsSL --retry 3 -o "$DIR/$f.mat" "$BASE/$f.mat"
  fi
done
echo "$(ls "$DIR"/*.mat | wc -l) files in $DIR"
