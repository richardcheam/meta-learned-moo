import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

# -----------------------------
# Paper-style fonts
# -----------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
})

# -----------------------------
# Helpers
# -----------------------------
def load_one(path):
    with open(path) as f:
        return json.load(f)


def collect_runs(root, dataset, method, filename):
    root = Path(root)
    files = []
    dataset = dataset.lower()

    for d in root.iterdir():
        if not d.is_dir():
            continue

        name_parts = d.name.lower().split("_")

        # exact dataset match → fixes mnist/fmnist bug
        if name_parts[0] != dataset:
            continue

        if method not in d.name.lower():
            continue

        f = d / "inference" / filename
        if f.exists():
            files.append(str(f))

    return sorted(files)


def aggregate_runs(files):
    all_t0, all_t1 = [], []
    rays = None

    for f in files:
        d = load_one(f)

        if rays is None:
            rays = np.array(d["ray"])

        # all_t0.append(d["task0_loss"])
        # all_t1.append(d["task1_loss"])

        all_t0.append(d["pred_loss"])
        all_t1.append(d["fair_loss_ddp"])

    t0 = np.array(all_t0)
    t1 = np.array(all_t1)

    return {
        "rays": rays,
        "t0_mean": t0.mean(axis=0),
        "t0_std": t0.std(axis=0, ddof=1),
        "t1_mean": t1.mean(axis=0),
        "t1_std": t1.std(axis=0, ddof=1),
    }


# -----------------------------
# PHN-style colored curve
# -----------------------------
def colored_curve(ax, x, y, cmap, lw, ls):
    pts = np.array([x, y]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

    lc = LineCollection(
        segs,
        cmap=cmap,
        norm=plt.Normalize(0, len(x)-1),
        linewidth=lw,
        linestyles=ls,
        zorder=3,
    )
    lc.set_array(np.arange(len(x)))
    ax.add_collection(lc)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--file", default="test_50epochs_results_10rays.json")
    parser.add_argument("--out", default="pareto_final.png")
    parser.add_argument("--title", default="Test Pareto Front")

    args = parser.parse_args()

    hpn_files = collect_runs(args.root, args.dataset, "adam", args.file)
    mml_files = collect_runs(args.root, args.dataset, "velo", args.file)

    print("HPN runs:", len(hpn_files))
    print("MML runs:", len(mml_files))

    hpn = aggregate_runs(hpn_files)
    mml = aggregate_runs(mml_files)

    # -----------------------------
    # sort by task0 loss
    # -----------------------------
    order = np.argsort(hpn["t0_mean"])

    for k in hpn:
        if k != "rays":
            hpn[k] = hpn[k][order]

    for k in mml:
        if k != "rays":
            mml[k] = mml[k][order]

    rays = hpn["rays"][order]
    n = len(rays)

    cmap = plt.cm.turbo
    colors = cmap(np.linspace(0.05, 0.95, n))

    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    # -----------------------------
    # Std bands
    # -----------------------------
    ax.fill_between(
        hpn["t0_mean"],
        hpn["t1_mean"] - hpn["t1_std"],
        hpn["t1_mean"] + hpn["t1_std"],
        color="black",
        alpha=0.07,
        zorder=1,
    )

    ax.fill_between(
        mml["t0_mean"],
        mml["t1_mean"] - mml["t1_std"],
        mml["t1_mean"] + mml["t1_std"],
        color="black",
        alpha=0.07,
        zorder=1,
    )

    # -----------------------------
    # Colored Pareto curves
    # -----------------------------
    colored_curve(ax, hpn["t0_mean"], hpn["t1_mean"], cmap, 2.2, "--")
    colored_curve(ax, mml["t0_mean"], mml["t1_mean"], cmap, 2.2, "-")

    # -----------------------------
    # Preference rays
    # -----------------------------
    rho = max(
        np.max(hpn["t0_mean"]),
        np.max(hpn["t1_mean"]),
        np.max(mml["t0_mean"]),
        np.max(mml["t1_mean"]),
    ) * 1.15

    for i, r in enumerate(rays):
        ax.plot(
            [0, rho*r[0]],
            [0, rho*r[1]],
            linestyle="--",
            color=colors[i],
            linewidth=1.0,
            alpha=0.9,
            zorder=0,
        )

    # -----------------------------
    # Legend (clean)
    # -----------------------------
    legend_handles = [
        Line2D([0],[0], linestyle="--", color="black", label="PHN (ours)"),
        Line2D([0],[0], linestyle="-", color="black", label="MML (ours)"),
    ]

    ax.legend(handles=legend_handles, frameon=False)

    # -----------------------------
    # Labels
    # -----------------------------
    ax.set_xlabel(r"$\ell_1$")
    ax.set_ylabel(r"$\ell_2$")
    ax.set_title(args.title)

    ax.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")

    print("Saved:", args.out)


if __name__ == "__main__":
    main()