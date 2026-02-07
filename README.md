# Meta-Learned Multi-Objective Optimization (VeLO)

This repository contains two complementary components for studying **meta-learned optimization** and **multi-objective / multi-task learning** with **VeLO**:

1. **VeLO in LibMOON** – gradient-based multi-task / multi-objective learning benchmarks  
2. **Pareto Hypernetwork with VeLO (PHN-VeLO)** – Pareto front learning with hypernetworks

---

## Install dependencies

Make sure you have Python ≥ 3.9 and a virtual environment.

```bash
pip3 install virtualenv
```

```bash
git clone https://github.com/richardcheam/meta-learned-moo.git
cd meta-learned-moo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
````

## VeLO in LibMOON

This part runs multi-task / multi-objective learning experiments using LibMOON with VeLO and other gradient-based solvers.

All LibMOON experiments are launched using a single local runner script:

Run an experiment

```bash
./run-mtl-libmoon.sh DATASET OPTIM SOLVER SEED EPOCHS
```

Examples:
```bash
./run-mtl-libmoon.sh mnist agg_SmTche 5 200
```
Outputs

Results are saved in: multitask_learning_problem_test/results/<dataset>/.pkl

## Pareto Hypernetwork with VeLO (PHN-VeLO)

This part runs Pareto Hypernetwork (PHN) experiments with VeLO as the optimizer.

A unified local launcher automatically selects the correct training pipeline (tabular, vision, or temporal) based on the dataset.

Run an experiment

```bash
./run_experiment.sh DATASET OPTIM SOLVER SEED EPOCHS
```

Examples:
```bash
./run-phn-velo.sh adult velo epo 0 100
```

Outputs

Results are saved in: outputs/<task_type>/

