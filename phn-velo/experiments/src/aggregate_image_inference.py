import json
import argparse
from pathlib import Path
import numpy as np

# For mnist 

def load_test_results(exp_dir: Path):
    files = list(exp_dir.glob("test_results.json"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly 1 test_results.json in {exp_dir}, found {len(files)}"
        )
    with open(files[0], "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        "Aggregate inference results (mean ± std over seeds)"
    )
    parser.add_argument("--root", type=str, required=True, default="/Users/macbookpro/Desktop/M2DS/stageMOO/velo-hpn/outputs-mo-mnist")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--optim", type=str, required=True)
    parser.add_argument("--solver", type=str, default="epo")
    parser.add_argument("--out", type=str, required=True)
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

    aggregated = {}

    # rays (shared across all seeds)
    aggregated["ray"] = runs[0]["ray"]

    # metrics = ["task0_acc", "task1_acc", "task0_loss", "task1_loss"]
    metrics = ["task0_loss", "task1_loss"]

    for metric in metrics:
        values = np.array([run[metric] for run in runs])  # shape: (n_seeds, n_rays)
        aggregated[metric] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }

    # ---- save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\nSaved aggregated inference results to:")
    print(out_path.resolve())


if __name__ == "__main__":
    main()
