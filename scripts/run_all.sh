#!/usr/bin/env bash
# Reproduce every number in the README. Finished runs are skipped, so the
# script can be interrupted and restarted safely.
#
#   bash scripts/run_all.sh            # full benchmark (~6 h on a 4-core CPU)
#   SEEDS="0" bash scripts/run_all.sh  # one seed only (~2 h)
set -euo pipefail

PY=${PY:-python}
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-2000}
GAN_ITERS=${GAN_ITERS:-3000}
OUT=${OUT:-results}

[ -f data/ham10000_64.npz ] || $PY -m lesionbench prepare

for seed in $SEEDS; do
  [ -f "$OUT/gan/seed$seed/synthetic.npz" ] || \
    $PY -m lesionbench --out "$OUT" gan --seed "$seed" --iters "$GAN_ITERS"

  for method in erm aug weighted focal oversample gan gan_mix synthetic_only; do
    [ -f "$OUT/runs/lesion_${method}_s${seed}.json" ] || \
      $PY -m lesionbench --out "$OUT" classify --split lesion --method "$method" --seed "$seed" --steps "$STEPS"
  done

  # The leak-prone protocol, for the split-level comparison.
  for method in aug oversample; do
    [ -f "$OUT/runs/image_${method}_s${seed}.json" ] || \
      $PY -m lesionbench --out "$OUT" classify --split image --method "$method" --seed "$seed" --steps "$STEPS"
  done
done

$PY -m lesionbench --out "$OUT" report
