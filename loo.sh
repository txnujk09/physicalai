#!/usr/bin/env bash
# Leave-one-out generalisation check: train B without perturbation type X, test on X only.
set -e
PY=${PY:-.venv/bin/python}
for t in impulse slip target friction noise; do
  $PY train.py --policies B_no_$t
  $PY eval.py --policies B_no_$t --conditions clean $t --out results/loo_$t
done
