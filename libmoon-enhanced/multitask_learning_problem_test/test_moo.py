#!/usr/bin/env python3
import os
import sys
import argparse
import pickle
import numpy as np
import torch

# ============================================================
# Path setup (portable, monorepo-safe)
# ============================================================
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LIBMOON_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
sys.path.append(LIBMOON_ROOT)

# ============================================================
# Imports
# ============================================================
from libmoon.util.mtl import model_from_dataset, numel, get_mtl_prefs
from libmoon.solver.gradient.methods.epo_solver import EPOCore
from libmoon.solver.gradient.methods.mgda_solver import MGDAUBCore
from libmoon.solver.gradient.methods.pmgda_solver import PMGDACore
from libmoon.solver.gradient.methods.moosvgd_solver import MOOSVGDCore
from libmoon.solver.gradient.methods.gradhv_solver import GradHVCore
from libmoon.solver.gradient.methods.pmtl_solver import PMTLCore
from libmoon.solver.gradient.methods.random_solver import RandomCore
from libmoon.solver.gradient.methods.pcgrad_solver import PCGradCore
from libmoon.solver.gradient.methods.nashtmtl_solver import NashMTLCore
from libmoon.solver.gradient.methods.base_solver import AggCore
from libmoon.solver.gradient.methods.core.core_mtl import (
    GradBaseMTLSolver,
    GradBaseMTLSolverMnist,
    GradBaseMTLSolverTemporal,
)

# ============================================================
# Argument parsing
# ============================================================
parser = argparse.ArgumentParser("LibMOON local runner")

parser.add_argument("--dataset", required=True, help="Dataset name")
parser.add_argument("--solver", required=True, help="Solver name (epo, mgdaub, agg_*, ...)")
parser.add_argument("--variant", default="velo", help="Variant name (for logging only)")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--seeds", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=1024)
parser.add_argument("--step-size", type=float, default=1e-4)
parser.add_argument("--n-prob", type=int, default=10)
parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

args = parser.parse_args()

DATASET = args.dataset.lower()
SOLVER = args.solver
VARIANT = args.variant

# ============================================================
# Dataset normalization
# ============================================================
if DATASET == "electricity":
    PROBLEM_NAME = "electricity_demand"
else:
    PROBLEM_NAME = DATASET

# ============================================================
# Output directory
# ============================================================
RESULTS_DIR = os.path.join(
    LIBMOON_ROOT, "results", DATASET, f"{SOLVER}-{VARIANT}"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# Device
# ============================================================
if args.device in ["cuda", "mps"] and torch.backends.mps.is_available():
    device = torch.device(args.device)
else:
    device = torch.device("cpu")

# ============================================================
# Main loop
# ============================================================
for seed in range(args.seeds):
    print("=" * 60)
    print(f"Dataset : {DATASET}")
    print(f"Solver  : {SOLVER}-{VARIANT}")
    print(f"Seed    : {seed}")
    print(f"Epochs  : {args.epochs}")
    print("=" * 60)

    # --------------------------------------------------------
    # Model & preferences
    # --------------------------------------------------------
    model_args = {"num_classes": 2, "hidden_dim": 128, "seq_length": 96}
    model = model_from_dataset(PROBLEM_NAME, args=model_args)
    n_var = numel(model)

    prefs = get_mtl_prefs(problem_name=PROBLEM_NAME, n_prob=args.n_prob)

    # --------------------------------------------------------
    # Core solver selection
    # --------------------------------------------------------
    if SOLVER == "epo":
        core_solver = EPOCore(n_var=n_var, prefs=prefs)
    elif SOLVER == "mgdaub":
        core_solver = MGDAUBCore()
    elif SOLVER == "random":
        core_solver = RandomCore()
    elif SOLVER == "pmgda":
        core_solver = PMGDACore(n_var=n_var, prefs=prefs)
    elif SOLVER.startswith("agg"):
        agg_name = SOLVER.split("_", 1)[1]
        core_solver = AggCore(prefs=prefs, agg_name=agg_name)
    elif SOLVER == "moosvgd":
        core_solver = MOOSVGDCore(n_var=n_var, prefs=prefs)
    elif SOLVER == "hvgrad":
        core_solver = GradHVCore(n_obj=2, n_var=n_var, problem_name=PROBLEM_NAME)
    elif SOLVER == "pmtl":
        core_solver = PMTLCore(
            n_obj=2, n_var=n_var, n_epoch=args.epochs, prefs=prefs
        )
    elif SOLVER == "pcgrad":
        core_solver = PCGradCore(n_var=n_var, prefs=prefs)
    elif SOLVER == "nashmtl":
        core_solver = NashMTLCore(n_var=n_var, prefs=prefs)
    else:
        raise ValueError(f"Unknown solver: {SOLVER}")

    # --------------------------------------------------------
    # High-level solver selection
    # --------------------------------------------------------
    if PROBLEM_NAME in ["adult", "credit", "compas"]:
        solver = GradBaseMTLSolver(
            problem_name=PROBLEM_NAME,
            step_size=args.step_size,
            epoch=args.epochs,
            core_solver=core_solver,
            batch_size=args.batch_size,
            prefs=prefs,
        )

    elif PROBLEM_NAME in ["mnist", "fashion", "fmnist", "dsprites"]:
        solver = GradBaseMTLSolverMnist(
            problem_name=PROBLEM_NAME,
            step_size=args.step_size,
            epoch=args.epochs,
            core_solver=core_solver,
            batch_size=args.batch_size,
            prefs=prefs,
            device=device,
        )

    elif PROBLEM_NAME in ["electricity_demand"]:
        solver = GradBaseMTLSolverTemporal(
            problem_name=PROBLEM_NAME,
            step_size=args.step_size,
            epoch=args.epochs,
            core_solver=core_solver,
            batch_size=args.batch_size,
            prefs=prefs,
            args=model_args,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported problem: {PROBLEM_NAME}")

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------
    res = solver.solve()
    res["prefs"] = prefs
    res["y"] = res.get("loss")

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out_file = os.path.join(
        RESULTS_DIR, f"res_{DATASET}_{SOLVER}-{VARIANT}_{seed}.pkl"
    )

    with open(out_file, "wb") as f:
        pickle.dump(res, f)

    print(f"Saved results to: {out_file}")

print("\nAll runs finished successfully.")