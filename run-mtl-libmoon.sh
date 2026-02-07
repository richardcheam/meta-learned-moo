#!/bin/bash
set -e

DATASET=$1
SOLVER=$2
SEEDS=${3:-5}
EPOCHS=${4:-100}

if [ $# -lt 2 ]; then
  echo "Usage: ./run-mtl-libmoon.sh DATASET SOLVER [SEEDS] [EPOCHS]"
  exit 1
fi

ROOT_DIR=$(pwd)
LIBMOON_DIR="$ROOT_DIR/libmoon-enhanced"

echo "======================================"
echo "Dataset : $DATASET"
echo "Solver  : $SOLVER"
echo "Seeds   : $SEEDS"
echo "Epochs  : $EPOCHS"
echo "======================================"

cd "$LIBMOON_DIR"

python multitask_learning_problem_test/test_moo.py \
  --dataset "$DATASET" \
  --solver "$SOLVER" \
  --seeds "$SEEDS" \
  --epochs "$EPOCHS"