#!/usr/bin/env bash
# v14/base_delta_v_new: best fixed set with U_obs = lambda*(1 + max(dv_approach/v_max, 0))/k'. Uses ../base/scenarios (no step 1).
# Rough times with 7 parallel jobs: step 2 ~6 h, step 3 ~10 min.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/asher747/ddp/Simluation/venv/bin/python}
J=${1:-7}
test -d ../base/scenarios || { echo "../base/scenarios missing -- run v14/base step 1 first"; exit 1; }
echo "== Step 2: train the fixed set (CMA-ES)";   $PY step2_train_base.py --n-jobs "$J"
echo "== Step 3: test on fresh scenarios";        $PY step3_evaluate.py --n-jobs "$J"
