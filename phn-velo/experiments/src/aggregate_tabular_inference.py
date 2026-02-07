import json
import argparse
from pathlib import Path
import numpy as np


def load_one_run(exp_dir: Path):
    path = exp_dir / "inference" / "test_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser("Aggregate tabular inference (mean ± std)")
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--optim", type=str, required=True)
    parser.add_argument("--solver", type=str, default="epo")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    root = Path(args.root)
    pattern = f"{args.dataset}_{args.optim}_{args.solver}_seed*_job*"
    exp_dirs = sorted(root.glob(pattern))

    if len(exp_dirs) == 0:
        raise RuntimeError(f"No runs found for pattern {pattern}")

    print(f"Found {len(exp_dirs)} runs:")
    for d in exp_dirs:
        print(" ", d.name)

    runs = [load_one_run(d) for d in exp_dirs]

    rays = runs[0]["ray"]
    metrics = ["acc", "pred_loss", "fair_loss_ddp", "fair_loss_deo"]

    aggregated = {"ray": rays}

    for m in metrics:
        values = np.array([r[m] for r in runs])  # (n_seeds, n_rays)
        aggregated[m] = {
            "mean": values.mean(axis=0).tolist(),
            "std": values.std(axis=0).tolist(),
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"\n✅ Aggregated inference saved to {out_path}")


if __name__ == "__main__":
    main()
