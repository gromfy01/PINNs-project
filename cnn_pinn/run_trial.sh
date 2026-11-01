#!/bin/bash
# Пробный прогон: три варианта × три сида, затем графики. Запуск из cnn_pinn/:  ./run_trial.sh <python>
set -eu
PY=${1:-python}
for seed in 0 1 2; do
  for cfg in hard data_only profile1d; do
    [ -f runs/${cfg}_s${seed}/metrics.json ] && continue
    $PY src/train.py --config configs/$cfg.yaml --seed $seed > runs/log_${cfg}_s${seed}.txt 2>&1
  done
  [ $seed = 0 ] && $PY src/plots.py --data data/trial_w2.npz --runs runs/hard_s0 runs/data_only_s0 runs/profile1d_s0 --out runs/figs_s0
done
echo done
