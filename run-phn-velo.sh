#!/bin/bash
set -e

DATASET=$1
OPTIM=$2
SOLVER=$3
SEED=$4
EPOCHS=$5

if [ $# -lt 5 ]; then
  echo "Usage: ./run-phn-velo.sh DATASET OPTIM SOLVER SEED EPOCHS"
  exit 1
fi

ROOT_DIR=$(pwd)
PHN_DIR="$ROOT_DIR/phn-velo"

echo "======================================"
echo "Dataset: $DATASET"
echo "Optim:   $OPTIM"
echo "Solver:  $SOLVER"
echo "Seed:    $SEED"
echo "Epochs:  $EPOCHS"
echo "Start:   $(date)"
echo "======================================"

cd "$PHN_DIR"

# ---------------- Dataset routing ----------------
TABULAR_DATASETS=("adult" "compas" "credit")
MO_MNIST_DATASETS=("mnist" "fashion" "fmnist")
TEMPORAL_DATASETS=("electricity")

if [[ " ${TABULAR_DATASETS[*]} " =~ " ${DATASET} " ]]; then
  TRAINER="experiments/src/trainer_tabular.py"
  TASK="tabular"
elif [[ " ${MO_MNIST_DATASETS[*]} " =~ " ${DATASET} " ]]; then
  TRAINER="experiments/src/trainer_mo-mnist.py"
  TASK="mo-mnist"
elif [[ " ${TEMPORAL_DATASETS[*]} " =~ " ${DATASET} " ]]; then
  TRAINER="experiments/src/trainer_temporal.py"
  TASK="temporal"
else
  echo "❌ Unknown dataset: $DATASET"
  exit 1
fi

OUTDIR="$ROOT_DIR/outputs/$TASK/${DATASET}_${OPTIM}_${SOLVER}_seed${SEED}"
mkdir -p "$OUTDIR"

echo "Running task type: $TASK"
echo "Trainer: $TRAINER"
echo "Output:  $OUTDIR"
echo "--------------------------------------"

python "$TRAINER" \
  --dataset "$DATASET" \
  --optim "$OPTIM" \
  --solver "$SOLVER" \
  --seed "$SEED" \
  --n-epochs "$EPOCHS" \
  --out-dir "$OUTDIR"