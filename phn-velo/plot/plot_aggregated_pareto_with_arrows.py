import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def annotate_epoch(ax, x, y, epoch, dx, dy):
    """Arrow + label pointing to the middle of a Pareto curve."""
    mid = len(x) // 2
    ax.annotate(
        f"epoch {epoch}",
        xy=(x[mid], y[mid]),
        xytext=(x[mid] + dx, y[mid] + dy),
        arrowprops=dict(arrowstyle="->", linewidth=0.9, alpha=0.85),
        fontsize=9,
        alpha=0.9,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Aggregated Pareto evolution (Adam vs VeLO, mean ± std)"
    )
    parser.add_argument("--adam-results", type=str, required=True)
    parser.add_argument("--velo-results", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--val-step", type=int, default=10)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    adam_data = load_results(args.adam_results)
    velo_data = load_results(args.velo_results)

    epochs = list(range(args.val_step, args.epochs + 1, args.val_step))
    marked_epochs = {10, args.epochs}

    fig, ax = plt.subplots(figsize=(7.2, 5.8))

    ax.set_xlabel("Task 0 loss", fontsize=12)
    ax.set_ylabel("Task 1 loss", fontsize=12)
    ax.set_title("Aggregated Pareto Front Evolution (Adam vs VeLO)", fontsize=13)

    # Shared colormap for epochs
    cmap = plt.cm.Reds
    norm = plt.Normalize(min(epochs), max(epochs))

    # ================= ADAM =================
    for e in epochs:
        key = f"epoch_{e}"
        if key not in adam_data:
            continue

        color = cmap(norm(e))

        x = np.array(adam_data[key]["task0_loss"]["mean"])
        y = np.array(adam_data[key]["task1_loss"]["mean"])
        x_std = np.array(adam_data[key]["task0_loss"]["std"])
        y_std = np.array(adam_data[key]["task1_loss"]["std"])

        ax.plot(
            x, y,
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.85,
        )

        # Std band (light)
        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

        if e in marked_epochs:
            annotate_epoch(ax, x, y, e, dx=0.03, dy=0.03)

    # ================= VELO =================
    for e in epochs:
        key = f"epoch_{e}"
        if key not in velo_data:
            continue

        color = cmap(norm(e))

        x = np.array(velo_data[key]["task0_loss"]["mean"])
        y = np.array(velo_data[key]["task1_loss"]["mean"])
        x_std = np.array(velo_data[key]["task0_loss"]["std"])
        y_std = np.array(velo_data[key]["task1_loss"]["std"])

        ax.plot(
            x, y,
            linestyle="-",
            linewidth=1.6,
            color=color,
            alpha=0.9,
        )

        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

        if e in marked_epochs:
            annotate_epoch(ax, x, y, e, dx=-0.04, dy=-0.04)

    # ================= LEGENDS =================
    optimizer_legend = [
        Line2D([0], [0], linestyle="--", color="black", label="Adam"),
        Line2D([0], [0], linestyle="-", color="black", label="VeLO"),
    ]

    ax.legend(
        handles=optimizer_legend,
        loc="upper left",
        frameon=False,
        fontsize=10,
        title="Optimizer",
        title_fontsize=10,
    )

    # Epoch colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label("Epoch", fontsize=10)

    ax.grid(alpha=0.3)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {args.out}")
    else:
        plt.show()
