#!/bin/bash
# NADiSSP — Full Pipeline Runner
# Usage: ./run.sh [generate|train|ablation|evaluate|tco|network|serve|all]
set -e
cd "$(dirname "$0")"
CMD=${1:-serve}
PY=python3

case "$CMD" in
  generate)
    echo "=== Generating datasets ==="
    $PY data/generate_datasets.py ;;
  pretrain)
    echo "=== SimCLR Pre-Training — Contribution 2 (Stage 2) ==="
    $PY scripts/pretrain_simclr.py --epochs 100 --tau 0.5 --target-only ;;
  train)
    echo "=== Training NADiSSP (43 epochs) ==="
    $PY scripts/train.py ;;
  ablation)
    echo "=== Ablation study (4 configs × 15 epochs) ==="
    $PY scripts/ablation.py ;;
  evaluate)
    echo "=== Generating figures and tables (Ch4) ==="
    $PY scripts/evaluate.py ;;
  tco)
    echo "=== TCO/NPV Monte Carlo (Contribution 4) ==="
    $PY scripts/tco_simulation.py --trials 10000 --horizon 5 --sensitivity --national 5 ;;
  network)
    echo "=== Network resilience test (Table 4.10/4.11) ==="
    $PY scripts/network_test.py ;;
  serve)
    echo "=== Starting API on http://localhost:8000 ==="
    echo "    Dashboard: http://localhost:8000/"
    echo "    API docs:  http://localhost:8000/docs"
    $PY api/main.py ;;
  full)
    echo "=== Full pipeline ==="
    $PY data/generate_datasets.py
    $PY scripts/pretrain_simclr.py --epochs 100 --tau 0.5 --target-only
    $PY scripts/train.py
    $PY scripts/ablation.py
    $PY scripts/evaluate.py
    $PY scripts/tco_simulation.py --trials 10000 --horizon 5 --sensitivity
    $PY scripts/network_test.py
    $PY api/main.py ;;
  *)
    echo "Usage: $0 [generate|train|ablation|evaluate|tco|network|serve|full]"
    exit 1 ;;
esac
