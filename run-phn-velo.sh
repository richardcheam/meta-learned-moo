#!/bin/bash
set -e

# ---------------- Arguments ----------------
DATASET=$1
OPTIM=$2
SOLVER=$3
SEEDS=${4:-5}
EPOCHS=${5-50}

if [ $# -lt 5 ]; then
  echo "Usage: ./run_experiment.sh DATASET OPTIM SOLVER SEED EPOCHS"
  exit 1
fi

# ---------------- Environment ----------------
# activate venv
source .venv/bin/activate

ROOT_DIR=$(pwd)
OUTBASE="$ROOT_DIR/outputs"

mkdir -p "$OUTBASE"

echo "======================================"
echo "Dataset: $DATASET"
echo "Optim:   $OPTIM"
echo "Solver:  $SOLVER"
echo "Seed:    $SEED"
echo "Epochs:  $EPOCHS"
echo "Start:   $(date)"
echo "======================================"

# ---------------- Dataset routing ----------------
TABULAR_DATASETS=("adult" "compas" "credit")
MO_MNIST_DATASETS=("mnist" "fashion" "fmnist")
TEMPORAL_DATASETS=("electricity")

if [[ " ${TABULAR_DATASETS[*]} " =~ " ${DATASET} " ]]; then
  TASK="tabular"
  TRAINER="experiments/src/trainer_tabular.py"

elif [[ " ${MO_MNIST_DATASETS[*]} " =~ " ${DATASET} " ]]; then
  TASK="mo-mnist"
  TRAINER="experiments/src/trainer_mo-mnist.py"

elif [[ " ${TEMPORAL_DATASETS[*]} " =~ " ${DATASET} " ]]; then
  TASK="temporal"
  TRAINER="experiments/src/trainer_temporal.py"

else
  echo "Unknown dataset: $DATASET"
  exit 1
fi

OUTDIR="$OUTBASE/$TASK/${DATASET}_${OPTIM}_${SOLVER}_seed${SEED}"
mkdir -p "$OUTDIR"

# ---------------- Run ----------------
echo "Running task type: $TASK"
echo "Trainer: $TRAINER"
echo "Output:  $OUTDIR"
echo "--------------------------------------"

if [ "$TASK" = "temporal" ]; then
  python "$TRAINER" \
    --optim "$OPTIM" \
    --solver "$SOLVER" \
    --seed "$SEED" \
    --n-epochs "$EPOCHS" \
    --out-dir "$OUTDIR"
else
  python "$TRAINER" \
    --dataset "$DATASET" \
    --optim "$OPTIM" \
    --solver "$SOLVER" \
    --seed "$SEED" \
    --n-epochs "$EPOCHS" \
    --out-dir "$OUTDIR"
fi

echo "Finished at $(date)"