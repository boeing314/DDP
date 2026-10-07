#!/usr/bin/env bash
# v14/adaptive: adaptive policy on top of the trained base (needs ../base/base_best.json and ../base/scenarios).
# Rough times with 7 parallel jobs: step 2 ~7 h, step 3 ~25 min, step 4 ~2 min.
# Usage: ./run_pipeline.sh [n_jobs]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/asher747/ddp/Simluation/venv/bin/python}
J=${1:-7}

test -f ../base/base_best.json || { echo "Run v14/base first (../base/base_best.json missing)"; exit 1; }
echo "== Step 2: train the adaptive policy (CMA-ES)";      $PY step2_train_policy.py --n-jobs "$J"
echo "== Step 3: test: trained base vs adaptive";          $PY step3_evaluate.py --n-jobs "$J"
echo "== Step 4: figures";                                 $PY step4_analyze.py --n-jobs "$J"
