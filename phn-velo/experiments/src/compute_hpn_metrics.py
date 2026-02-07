import json
import argparse
import numpy as np
from pathlib import Path
from pymoo.indicators.hv import HV

from experiments.utils import circle_points


# ------------------------------------------------------------
# PHN Uniformity (KL-based, PHN paper)
# ------------------------------------------------------------
def compute_phn_uniformity(objs, prefs, eps=1e-12):
    """
    objs: (K, 2) loss values
    prefs: (K, 2) preference rays
    """
    m = objs.shape[1]
    uniform = np.ones(m) / m
    kl_vals = []

    for l, r in zip(objs, prefs):
        num = r * l
        denom = np.sum(num) + eps
        l_hat = num / denom
        kl = np.sum(l_hat * np.log((l_hat + eps) / uniform))
        kl_vals.append(kl)

    return 1.0 - float(np.mean(kl_vals))


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_last_epoch_test(exp_dir: Path):
    files = list(exp_dir.glob("test_results_*.json"))
    assert len(files) == 1, f"{exp_dir}: expected 1 test_results file"

    data = json.load(open(files[0]))
    last_epoch = sorted(
        data.keys(),
        key=lambda x: int(x.split("_")[1])
    )[-1]
    return data[last_epoch]


def load_runtime_seconds(exp_dir: Path):
    cfgs = [c for c in exp_dir.glob("config_*.json") if c.name != "config.json"]
    assert len(cfgs) == 1, f"{exp_dir}: expected 1 timestamped config"

    cfg = json.load(open(cfgs[0]))
    return cfg["training_time_seconds"]
    # return cfg["training_time"]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("PHN-style metrics + runtime")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--optim", choices=["adam", "velo"], required=True)
    parser.add_argument("--solver", default="epo")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-rays", type=int, default=25)
    args = parser.parse_args()

    root = Path(args.root)
    pattern = f"{args.dataset}_{args.optim}_{args.solver}_seed*_job*"
    runs = sorted(root.glob(pattern))

    if not runs:
        raise RuntimeError(f"No runs found: {pattern}")

    print(f"Found {len(runs)} runs")

    hv_vals = []
    unif_vals = []
    runtimes = []

    # PHN evaluation rays (same as training)
    prefs = circle_points(
        K=args.n_rays,
        n_tasks=2,
        min_angle=0.1,
        max_angle=np.pi / 2 - 0.1
    )

    for r in runs:
        res = load_last_epoch_test(r)
        runtimes.append(load_runtime_seconds(r))

        # objectives
        if args.dataset in ["mnist", "fmnist", "fashion", "electricity"]:
            objs = np.stack(
                [res["task0_loss"], res["task1_loss"]],
                axis=1
            )
        else:  # tabular
            objs = np.stack(
                [res["pred_loss"], res["fair_loss_ddp"]],
                axis=1
            )

        # safety: subsample if needed
        if objs.shape[0] != args.n_rays:
            idx = np.linspace(0, objs.shape[0] - 1, args.n_rays).astype(int)
            objs = objs[idx]

        # ---------------------------
        # PHN metrics
        # ---------------------------
        hv = HV(ref_point=np.ones(2))(objs)
        unif = compute_phn_uniformity(objs, prefs)

        hv_vals.append(hv)
        unif_vals.append(unif)

    # --------------------------------------------------------
    # Aggregate
    # --------------------------------------------------------
    hv_vals = np.array(hv_vals)
    unif_vals = np.array(unif_vals)
    runtimes = np.array(runtimes) / 60.0  # minutes

    out = {
        "HV": {
            "mean": float(hv_vals.mean()),
            "std": float(hv_vals.std(ddof=1)),
        },
        "Uniformity": {
            "mean": float(unif_vals.mean()),
            "std": float(unif_vals.std(ddof=1)),
        },
        "Runtime_min": {
            "mean": float(runtimes.mean()),
            "std": float(runtimes.std(ddof=1)),
        }
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)

    print(f"✅ Saved PHN-style metrics → {out_path}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()