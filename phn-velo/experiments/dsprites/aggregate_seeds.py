import json
import argparse
from pathlib import Path
import numpy as np

## before plot, agg seed for pareto front evolution ###

# MNIST VERSION and TEMPORAL (since losses are labeled loss0 loss1)
def load_test_results(exp_dir: Path):
    files = list(exp_dir.glob("test_results_*.json"))
    if len(files) != 1:
        raise RuntimeError(f"Expected 1 test_results file in {exp_dir}, found {len(files)}")
    with open(files[0], "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser("Aggregate mean ± std over seeds (per epoch)")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--optim", type=str, required=True)
    parser.add_argument("--solver", type=str, default="epo")
    parser.add_argument("--out", type=str, default="aggregated_results.json")
    args = parser.parse_args()

    root = Path(args.root)
    pattern = f"{args.dataset}_{args.optim}_{args.solver}_seed*_job*"
    exp_dirs = sorted(root.glob(pattern))

    if len(exp_dirs) == 0:
        raise RuntimeError(f"No experiment folders found for pattern: {pattern}")

    print(f"Found {len(exp_dirs)} runs:")
    for d in exp_dirs:
        print("  ", d.name)

    # ---- load all runs ----
    runs = [load_test_results(d) for d in exp_dirs]

    # ---- epochs present ----
    epochs = sorted(runs[0].keys(), key=lambda x: int(x.split("_")[1]))
    print("\nEpochs found:", epochs)

    aggregated = {}

    metrics = ["task0_loss", "task1_loss", "task0_acc", "task1_acc"]
    # metrics = ["task0_loss", "task1_loss"] # for temporal

    for epoch in epochs:
        aggregated[epoch] = {}

        # rays (same across seeds)
        rays = runs[0][epoch]["ray"]
        aggregated[epoch]["ray"] = rays

        for metric in metrics:
            values = np.array([run[epoch][metric] for run in runs])
            aggregated[epoch][metric] = {
                "mean": values.mean(axis=0).tolist(),
                "std": values.std(axis=0).tolist(),
            }

    # ---- save ----
    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nSaved aggregated results to: {out_path.resolve()}")


if __name__ == "__main__":
    main()


# TABULAR VERSION
# import json
# import argparse
# from pathlib import Path
# import numpy as np


# def load_test_results(exp_dir: Path):
#     files = list(exp_dir.glob("test_results_*.json"))
#     if len(files) != 1:
#         raise RuntimeError(f"Expected 1 test_results file in {exp_dir}")
#     with open(files[0], "r") as f:
#         return json.load(f)


# def main():
#     parser = argparse.ArgumentParser(
#         "Aggregate tabular Pareto evolution (mean ± std over seeds)"
#     )
#     parser.add_argument("--root", type=str, required=True)
#     parser.add_argument("--dataset", type=str, required=True)
#     parser.add_argument("--optim", type=str, required=True)
#     parser.add_argument("--solver", type=str, default="epo")
#     parser.add_argument("--out", type=str, required=True)
#     args = parser.parse_args()

#     root = Path(args.root)
#     pattern = f"{args.dataset}_{args.optim}_{args.solver}_seed*_job*"
#     exp_dirs = sorted(root.glob(pattern))

#     if len(exp_dirs) == 0:
#         raise RuntimeError(f"No runs found for {pattern}")

#     runs = [load_test_results(d) for d in exp_dirs]

#     # Epochs are keys like epoch_10, epoch_20, ...
#     epochs = sorted(runs[0].keys(), key=lambda x: int(x.split("_")[1]))
#     print("Epochs:", epochs)

#     aggregated = {}

#     for epoch in epochs:
#         aggregated[epoch] = {}
#         rays = runs[0][epoch]["ray"]
#         aggregated[epoch]["ray"] = rays

#         for metric in ["pred_loss", "fair_loss_ddp", "acc"]:
#             values = np.array([run[epoch][metric] for run in runs])
#             aggregated[epoch][metric] = {
#                 "mean": values.mean(axis=0).tolist(),
#                 "std": values.std(axis=0).tolist(),
#             }

#     out_path = Path(args.out)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     with open(out_path, "w") as f:
#         json.dump(aggregated, f, indent=2)

#     print(f"\n✅ Saved Pareto evolution to {out_path}")


# if __name__ == "__main__":
#     main()