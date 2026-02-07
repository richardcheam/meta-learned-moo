import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.lines import Line2D

# ------------------------------------------------------------
# Paper-style fonts
# ------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,

    # 🔒 lock visual layout consistency
    "figure.figsize": (6.2, 4.8),
    "figure.dpi": 120,
    "savefig.dpi": 300,

    "axes.linewidth": 1.0,
})
# ------------------------------------------------------------
# Load json
# ------------------------------------------------------------
def load_one(path):
    with open(path) as f:
        return json.load(f)

# ------------------------------------------------------------
# Aggregate runs
# ------------------------------------------------------------
def aggregate_runs(files):
    if len(files) == 0:
        raise RuntimeError("No result files found")

    all_t0, all_t1 = [], []
    rays = None

    for f in files:
        d = load_one(f)
        if rays is None:
            rays = np.array(d["ray"])
        all_t0.append(d["task0_loss"])
        all_t1.append(d["task1_loss"])
        # # tabular case
        # all_t0.append(d["pred_loss"])
        # all_t1.append(d["fair_loss_ddp"])

    t0 = np.array(all_t0)
    t1 = np.array(all_t1)

    return {
        "rays": rays,
        "t0_mean": t0.mean(axis=0),
        "t0_std": t0.std(axis=0, ddof=1),
        "t1_mean": t1.mean(axis=0),
        "t1_std": t1.std(axis=0, ddof=1),
    }

# ------------------------------------------------------------
# Collect nested inference json (STRICT dataset match)
# ------------------------------------------------------------
def collect_runs(root, dataset, method, filename):
    root = Path(root)
    matches = []

    for d in root.iterdir():
        if not d.is_dir():
            continue

        name = d.name.lower()

        # ✅ strict dataset match (fix mnist/fmnist bug)
        ds_name = name.split("_")[0]
        if ds_name != dataset.lower():
            continue

        if method not in name:
            continue

        f = d / "inference" / filename

        if f.exists():
            print(f)
            matches.append(str(f))

    return sorted(matches)

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--file", default="test_50epochs_results_10rays.json")
    parser.add_argument("--out", default="pareto_final.png")
    args = parser.parse_args()

    hpn_files = collect_runs(args.root, args.dataset, "adam", args.file)
    mml_files = collect_runs(args.root, args.dataset, "velo", args.file)

    print("HPN runs:", len(hpn_files))
    print("MML runs:", len(mml_files))

    hpn = aggregate_runs(hpn_files)
    mml = aggregate_runs(mml_files)

    # -------------------------
    # sort by x (Pareto curve ordering)
    # -------------------------
    order = np.argsort(hpn["t0_mean"])
    for k in hpn:
        if k != "rays":
            hpn[k] = hpn[k][order]
    for k in mml:
        if k != "rays":
            mml[k] = mml[k][order]

    n = len(hpn["t0_mean"])

    # -------------------------
    # PHN-style ray colors
    # -------------------------
    colors = plt.cm.turbo(np.linspace(0.1, 0.9, n))

    fig, ax = plt.subplots()

    # -------------------------
    # STD bands
    # -------------------------
    ax.fill_between(
        hpn["t0_mean"],
        hpn["t1_mean"] - hpn["t1_std"],
        hpn["t1_mean"] + hpn["t1_std"],
        color="black",
        alpha=0.08,
        zorder=1,
    )

    ax.fill_between(
        mml["t0_mean"],
        mml["t1_mean"] - mml["t1_std"],
        mml["t1_mean"] + mml["t1_std"],
        color="black",
        alpha=0.08,
        zorder=1,
    )

    # -------------------------
    # main curves
    # -------------------------
    ax.plot(
        hpn["t0_mean"], hpn["t1_mean"],
        "--", color="black", linewidth=1.6, zorder=2
    )

    ax.plot(
        mml["t0_mean"], mml["t1_mean"],
        "-", color="black", linewidth=1.6, zorder=2
    )

    # -------------------------
    # ray-colored markers
    # -------------------------
    for i in range(n):

        ax.plot(
            hpn["t0_mean"][i], hpn["t1_mean"][i],
            marker="o",
            markersize=5,
            markerfacecolor=colors[i],
            markeredgecolor="black",
            markeredgewidth=0.5,
            linestyle="None",
            zorder=3,
        )

        ax.plot(
            mml["t0_mean"][i], mml["t1_mean"][i],
            marker="x",
            markersize=5,
            markeredgewidth=1.2,
            color=colors[i],
            linestyle="None",
            zorder=3,
        )

    # -------------------------
    # Legend (line + marker attached)
    # -------------------------
    legend_handles = [
        Line2D([0],[0],
               linestyle="-",
               color="black",
               marker="x",
               markersize=7,
               markeredgewidth=1.2,
               label="VeLO-PHN"),
        Line2D([0],[0],
               linestyle="--",
               color="black",
               marker="o",
               markersize=6,
               markerfacecolor="gray",
               markeredgecolor="black",
               label="PHN"),
    ]

    ax.legend(handles=legend_handles, frameon=False)

    # -------------------------
    # styling
    # -------------------------
    ax.set_xlabel(r"$\ell_1$")
    ax.set_ylabel(r"$\ell_2$")
    ax.set_title(f"{args.dataset} final Pareto front comparison", fontsize=13)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print("Saved:", args.out)


if __name__ == "__main__":
    main()