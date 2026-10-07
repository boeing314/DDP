#!/usr/bin/env bash
# v13 adaptive DIPF parameters (alternative A). Every step resumes where it stopped.
# Rough times with 7 parallel jobs: step 1 ~2 min, step 2 ~8 h (30 generations), control ~2 h,
# step 3 ~20 min, step 4 ~2 min.
# Usage: ./run_pipeline.sh [n_jobs]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/asher747/ddp/Simluation/venv/bin/python}
J=${1:-7}

echo "== Step 1: scenarios (train / val / test seeds)";        $PY step1_make_scenarios.py --n-jobs "$J"
echo "== Step 2: train the adaptive policy (CMA-ES)";          $PY step2_train_policy.py --n-jobs "$J"
echo "== Control: re-tune fixed values only (bias only)";      $PY step2_train_policy.py --bias-only --generations 15 --n-jobs "$J"
echo "== Step 3: test on fresh scenarios";                     $PY step3_evaluate.py --n-jobs "$J"
echo "== Step 4: figures";                                     $PY step4_analyze.py --n-jobs "$J"
