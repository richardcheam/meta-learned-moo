import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Plot style (IEEE friendly)
# ============================================================
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

# ============================================================
# Method naming (internal → paper)
# ============================================================
METHOD_DISPLAY = {
    "adam": "PHN",
    "velo": "VeLO-PHN",
}

METHOD_COLORS = {
    "PHN": "#D1495B",
    "VeLO-PHN": "#0077B6",
}

# ============================================================
# Load one inference json
# ============================================================
def load_one(path):
    with open(path) as f:
        return json.load(f)

# ============================================================
# Aggregate multiple seeds
# ============================================================
def aggregate_runs(files):
    pred, ddp, deo = [], [], []

    for f in files:
        d = load_one(f)
        pred.append(d["pred_loss"])
        ddp.append(d["fair_loss_ddp"])
        deo.append(d["fair_loss_deo"])

    pred = np.array(pred)
    ddp  = np.array(ddp)
    deo  = np.array(deo)

    return {
        "pred_mean": pred.mean(axis=0),
        "ddp_mean":  ddp.mean(axis=0),
        "deo_mean":  deo.mean(axis=0),
    }

# ============================================================
# Collect inference jsons
# ============================================================
def collect_runs(root, dataset, method, filename):
    root = Path(root)
    files = []

    for d in root.iterdir():
        if not d.is_dir():
            continue
        name = d.name.lower()
        if dataset in name and method in name:
            f = d / "inference" / filename
            if f.exists():
                files.append(str(f))

    return sorted(files)

# ============================================================
# Parallel coordinates plot
# ============================================================
def plot_parallel(data_dict, title, out):

    metrics = ["Prediction loss", "DDP loss", "DEO loss"]
    keys = ["pred_mean", "ddp_mean", "deo_mean"]
    X = np.arange(len(metrics))

    # Stack all values to compute global normalization
    all_vals = np.vstack([
        np.vstack([v[k] for k in keys]).T
        for v in data_dict.values()
    ])

    min_v = all_vals.min(axis=0)
    max_v = all_vals.max(axis=0)

    def normalize(v):
        return (v - min_v) / (max_v - min_v + 1e-8)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))

    # --------------------------------------------------------
    # Background lines (all rays)
    # --------------------------------------------------------
    for method, data in data_dict.items():
        Y = np.vstack([data[k] for k in keys]).T
        Y = normalize(Y)
        for y in Y:
            ax.plot(X, y, color="#9AA0A6", alpha=0.35, linewidth=1)

    # --------------------------------------------------------
    # Highlight mean trends
    # --------------------------------------------------------
    for method, data in data_dict.items():
        Y = np.vstack([data[k] for k in keys]).T
        Y = normalize(Y)
        mean_y = Y.mean(axis=0)

        ax.plot(
            X, mean_y,
            color=METHOD_COLORS[method],
            linewidth=3,
            label=method,
        )

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------
    ax.set_xticks(X)
    ax.set_xticklabels(metrics)
    ax.set_xlim(X[0], X[-1])
    ax.set_ylim(0, 1)

    ax.margins(x=0.15)
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(1.02, 1.02),
    )

    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)

    # Clean IEEE look
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved:", out)

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser("Parallel fairness plot")
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["adult", "compas", "credit"])
    parser.add_argument("--file", default="test_results.json")
    parser.add_argument("--out", default="parallel_fairness.png")
    args = parser.parse_args()

    methods = ["adam", "velo"]
    data = {}

    for m in methods:
        files = collect_runs(args.root, args.dataset, m, args.file)
        display_name = METHOD_DISPLAY[m]
        print(f"{display_name} runs:", len(files))
        data[display_name] = aggregate_runs(files)

    plot_parallel(
        data,
        title=f"{args.dataset.capitalize()} — Accuracy–Fairness Trade-off",
        out=args.out,
    )

if __name__ == "__main__":
    main()