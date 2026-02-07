import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def get_last_epoch_key(data):
    keys = [k for k in data.keys() if k.startswith("epoch_")]
    return sorted(keys, key=lambda x: int(x.split("_")[1]))[-1]


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Final Pareto Front — HPN vs MML")

    parser.add_argument("--hpn", required=True)
    parser.add_argument("--mml", required=True)
    parser.add_argument("--datatype", required=True,
                        choices=["image", "tabular", "temporal"])
    parser.add_argument("--title", default="Final Pareto Front")
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    hpn_data = load_results(args.hpn)
    mml_data = load_results(args.mml)

    # ------------------------------------------------------------
    # pick last epoch automatically
    # ------------------------------------------------------------
    hpn_key = get_last_epoch_key(hpn_data)
    mml_key = get_last_epoch_key(mml_data)

    print("Using epochs:", hpn_key, mml_key)

    # ------------------------------------------------------------
    # Metric keys
    # ------------------------------------------------------------
    if args.datatype == "image":
        x_key, y_key = "task0_loss", "task1_loss"
        x_label, y_label = "Task 1 loss", "Task 2 loss"

    elif args.datatype == "tabular":
        x_key, y_key = "pred_loss", "fair_loss_ddp"
        x_label, y_label = "Prediction loss", "Fairness loss"

    else:
        x_key, y_key = "task0_loss", "task1_loss"
        x_label, y_label = "Drop anomaly loss", "Spike anomaly loss"

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    # =========================
    # HPN
    # =========================
    hpn_marker_color = "#d62728"   # red

    x = np.array(hpn_data[hpn_key][x_key]["mean"])
    y = np.array(hpn_data[hpn_key][y_key]["mean"])
    y_std = np.array(hpn_data[hpn_key][y_key]["std"])

    order = np.argsort(x)
    x, y, y_std = x[order], y[order], y_std[order]

    # std band (colored)
    ax.fill_between(
        x,
        y - y_std,
        y + y_std,
        color=hpn_marker_color,
        alpha=0.15,
        zorder=1,
    )

    # black dashed line + colored markers
    ax.plot(
        x, y,
        linestyle="--",
        linewidth=1.6,
        color="black",
        marker="o",
        markersize=4,
        markerfacecolor=hpn_marker_color,
        markeredgecolor="black",
        markeredgewidth=0.5,
        label="HPN",
        zorder=3,
    )

    # =========================
    # MML
    # =========================
    mml_marker_color = "#1f77b4"   # blue

    x = np.array(mml_data[mml_key][x_key]["mean"])
    y = np.array(mml_data[mml_key][y_key]["mean"])
    y_std = np.array(mml_data[mml_key][y_key]["std"])

    order = np.argsort(x)
    x, y, y_std = x[order], y[order], y_std[order]

    ax.fill_between(
        x,
        y - y_std,
        y + y_std,
        color=mml_marker_color,
        alpha=0.15,
        zorder=1,
    )

    ax.plot(
        x, y,
        linestyle="-",
        linewidth=1.6,
        color="black",
        marker="x",
        markersize=4.5,
        markeredgewidth=1.0,
        markeredgecolor=mml_marker_color,
        label="MML",
        zorder=3,
    )

    # ------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(args.title, fontsize=13)

    ax.legend(frameon=False, fontsize=11)
    ax.grid(alpha=0.22)

    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300, bbox_inches="tight")
        print("Saved:", args.out)
    else:
        plt.show()


# ------------------------------------------------------------
if __name__ == "__main__":
    main()