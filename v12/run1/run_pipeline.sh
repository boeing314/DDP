#!/usr/bin/env bash
# v12 parameter-library pipeline (v11 with an Optuna search in step 2). Every step is resumable: rerun the same command after a stop.
# Rough times with 7 parallel jobs: step 1 ~10 min, step 2 4-7 h, step 3 ~3 h, step 4 minutes, step 5 ~20 min.
# Usage: ./run_pipeline.sh [n_jobs]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/asher747/ddp/Simluation/venv/bin/python}
J=${1:-7}

echo "== Step 1: scenarios (saved states at the takeover)";   $PY step1_make_scenarios.py --n-jobs "$J"
echo "== Step 2: parameter library (Optuna search)";            $PY step2_build_library.py --n-jobs "$J"
echo "== Step 3: label the dataset with every set";           $PY step3_label_dataset.py --n-jobs "$J"
echo "== Step 4: train the multi-label model";                $PY step4_train_model.py
echo "== Step 5: closed-loop test on fresh scenarios";        $PY step5_evaluate_policy.py --n-jobs "$J"
