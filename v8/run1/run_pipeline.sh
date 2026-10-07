#!/usr/bin/env bash
# v8 situation-cluster pipeline. Steps 1, 2 and 4 take minutes; step 3 takes hours (resumable).
# Usage: ./run_pipeline.sh [n_jobs]
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-/home/asher747/ddp/Simluation/venv/bin/python}
J=${1:-7}

echo "== Step 1: situation features";            $PY make_situations.py --n-jobs "$J"
echo "== Step 2: clusters (K from config.yaml)";  $PY cluster_situations.py
echo "== Step 3: one DE per cluster";             $PY optimize_clusters.py --n-jobs "$J"
echo "== Step 4a: held-out test episodes";        $PY cluster_controller.py --eval-test --n-jobs "$J"
echo "== Step 4b: full road, 500 fresh seeds";    $PY cluster_controller.py --eval-seeds 1000 1499 --n-jobs "$J"
