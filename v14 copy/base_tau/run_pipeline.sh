#!/usr/bin/env bash
# v14/base_tau (base with a different tau per k' region): the best FIXED parameter set (strong baseline). Run this BEFORE v14/adaptive.
# Rough times with 7 parallel jobs: step 1 ~2 min (or copy v13/run1/scenarios), step 2 ~6 h, step 3 ~10 min.
# Usage: ./run_pipeline.sh [n_jobs]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/asher747/ddp/Simluation/venv/bin/python}
J=${1:-7}

echo "== Step 1: scenarios (train / val / test seeds)";   $PY step1_make_scenarios.py --n-jobs "$J"
echo "== Step 2: train the fixed set (CMA-ES)";            $PY step2_train_base.py --n-jobs "$J"
echo "== Step 3: test on fresh scenarios";                 $PY step3_evaluate.py --n-jobs "$J"
