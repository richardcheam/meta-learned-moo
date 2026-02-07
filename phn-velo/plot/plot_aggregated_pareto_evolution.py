import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

# # Evolution
# # ------------------------------------------------------------
# # Helpers
# # ------------------------------------------------------------
# def truncate_colormap(cmap, minval=0.35, maxval=1.0, n=256):
#     return LinearSegmentedColormap.from_list(
#         "trunc", cmap(np.linspace(minval, maxval, n))
#     )


# def load_results(path):
#     with open(path, "r") as f:
#         return json.load(f)


# # ------------------------------------------------------------
# # Main
# # ------------------------------------------------------------
# def main():
#     parser = argparse.ArgumentParser(
#         "Pareto Front Evolution — HPN vs MML"
#     )
#     parser.add_argument("--hpn", type=str, required=True,
#                         help="Aggregated JSON for HPN (Adam)")
#     parser.add_argument("--mml", type=str, required=True,
#                         help="Aggregated JSON for MML (VeLO)")
#     parser.add_argument("--datatype", type=str, required=True,
#                         choices=["image", "tabular", "temporal"])
#     parser.add_argument("--epochs", type=int, default=50)
#     parser.add_argument("--val-step", type=int, default=10)
#     parser.add_argument("--out", type=str, default=None)
#     parser.add_argument("--title", type=str, default="Pareto Front Evolution")
#     args = parser.parse_args()

#     hpn_data = load_results(args.hpn)
#     mml_data = load_results(args.mml)

#     epochs = list(range(args.val_step, args.epochs + 1, args.val_step))

#     # ------------------------------------------------------------
#     # Metric keys by datatype
#     # ------------------------------------------------------------
#     if args.datatype == "image":
#         x_key = "task0_loss"
#         y_key = "task1_loss"
#         x_label = "Task 1 loss"
#         y_label = "Task 2 loss"

#     elif args.datatype == "tabular":
#         x_key = "pred_loss"
#         y_key = "fair_loss_ddp"
#         x_label = "Prediction loss"
#         y_label = "Fairness loss (DDP)"

#     elif args.datatype == "temporal":
#         x_key = "task0_loss"
#         y_key = "task1_loss"
#         x_label = "Drop anomaly loss"
#         y_label = "Spike anomaly loss"

#     else:
#         raise ValueError

#     # ------------------------------------------------------------
#     # Figure
#     # ------------------------------------------------------------
#     fig, ax = plt.subplots(figsize=(7.2, 5.6))

#     ax.set_xlabel(x_label, fontsize=12)
#     ax.set_ylabel(y_label, fontsize=12)
#     ax.set_title(args.title, fontsize=13)

#     # distinct colormaps
#     cmap_hpn = truncate_colormap(plt.cm.Reds, 0.35, 1.0)
#     cmap_mml = truncate_colormap(plt.cm.Blues, 0.35, 1.0)
#     norm = plt.Normalize(min(epochs), max(epochs))

#     # ------------------------------------------------------------
#     # HPN curves
#     # ------------------------------------------------------------
#     for e in epochs:
#         key = f"epoch_{e}"
#         if key not in hpn_data:
#             continue

#         color = cmap_hpn(norm(e))

#         x = np.array(hpn_data[key][x_key]["mean"])
#         y = np.array(hpn_data[key][y_key]["mean"])
#         y_std = np.array(hpn_data[key][y_key]["std"])

#         order = np.argsort(x)
#         x, y, y_std = x[order], y[order], y_std[order]

#         ax.plot(
#             x, y,
#             linestyle="--",
#             linewidth=1.4,
#             color=color,
#             marker="x",
#             markersize=3,
#             alpha=0.95,
#         )

#         ax.fill_between(
#             x,
#             y - y_std,
#             y + y_std,
#             color=color,
#             alpha=0.18,
#             linewidth=0,
#         )

#     # ------------------------------------------------------------
#     # MML curves
#     # ------------------------------------------------------------
#     for e in epochs:
#         key = f"epoch_{e}"
#         if key not in mml_data:
#             continue

#         color = cmap_mml(norm(e))

#         x = np.array(mml_data[key][x_key]["mean"])
#         y = np.array(mml_data[key][y_key]["mean"])
#         y_std = np.array(mml_data[key][y_key]["std"])

#         order = np.argsort(x)
#         x, y, y_std = x[order], y[order], y_std[order]

#         ax.plot(
#             x, y,
#             linestyle="-",
#             linewidth=1.9,
#             color=color,
#             marker="o",
#             markersize=2.5,
#             alpha=0.98,
#         )

#         ax.fill_between(
#             x,
#             y - y_std,
#             y + y_std,
#             color=color,
#             alpha=0.22,
#             linewidth=0,
#         )

#     # ------------------------------------------------------------
#     # Legend (paper naming)
#     # ------------------------------------------------------------
#     legend_handles = [
#         Line2D([0], [0], linestyle="--", color=cmap_hpn(0.9),
#                label="HPN"),
#         Line2D([0], [0], linestyle="-", color=cmap_mml(0.9),
#                label="MML"),
#     ]

#     ax.legend(
#         handles=legend_handles,
#         loc="upper left",
#         frameon=False,
#         fontsize=10,
#     )

#     # ------------------------------------------------------------
#     # Epoch colorbars (annotation without clutter)
#     # ------------------------------------------------------------
#     from matplotlib.cm import ScalarMappable

#     sm_hpn = ScalarMappable(cmap=cmap_hpn, norm=norm)
#     sm_hpn.set_array([])

#     sm_mml = ScalarMappable(cmap=cmap_mml, norm=norm)
#     sm_mml.set_array([])

#     cbar_hpn = plt.colorbar(sm_hpn, ax=ax, pad=0.01, fraction=0.04)
#     cbar_hpn.set_label("HPN Epoch", fontsize=9)
#     cbar_hpn.set_ticks(epochs)
#     cbar_hpn.ax.tick_params(labelsize=8)

#     cbar_mml = plt.colorbar(sm_mml, ax=ax, pad=0.08, fraction=0.04)
#     cbar_mml.set_label("MML Epoch", fontsize=9)
#     cbar_mml.set_ticks(epochs)
#     cbar_mml.ax.tick_params(labelsize=8)

#     cbar_hpn.set_ticks(epochs[::2])
#     cbar_mml.set_ticks(epochs[::2])

#     # ax.grid(alpha=0.3)
#     plt.tight_layout()

#     if args.out:
#         plt.savefig(args.out, dpi=300, bbox_inches="tight")
#         print("Saved:", args.out)
#     else:
#         plt.show()


# if __name__ == "__main__":
#     main()


import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser("Pareto Evolution — Clean Paper Style")

    parser.add_argument("--hpn", required=True)
    parser.add_argument("--mml", required=True)
    parser.add_argument("--datatype", required=True,
                        choices=["image", "tabular", "temporal"])
    parser.add_argument("--epochs", nargs="+", type=int,
                        default=[10, 50, 100])
    parser.add_argument("--title", default="Pareto Front Evolution")
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    hpn_data = load_results(args.hpn)
    mml_data = load_results(args.mml)

    # -----------------------------
    # metric keys
    # -----------------------------
    if args.datatype == "image":
        x_key, y_key = "task0_loss", "task1_loss"
        x_label, y_label = "Task 1 loss", "Task 2 loss"
    elif args.datatype == "tabular":
        x_key, y_key = "pred_loss", "fair_loss_ddp"
        x_label, y_label = "Prediction loss", "Fairness loss"
    else:
        x_key, y_key = "task0_loss", "task1_loss"
        x_label, y_label = "Drop anomaly loss", "Spike anomaly loss"

    fig, ax = plt.subplots(figsize=(7, 5.2))

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(args.title, fontsize=13)

    reds = plt.cm.Reds
    blues = plt.cm.Blues

    # visual hierarchy
    widths = {args.epochs[0]:1.5, args.epochs[1]:2.2, args.epochs[-1]:2.8}
    alphas = {args.epochs[0]:0.6, args.epochs[1]:0.85, args.epochs[-1]:1.0}
    band_alpha = {args.epochs[0]:0.08, args.epochs[1]:0.12, args.epochs[-1]:0.16}

    # marker size decreases with epoch
    marker_sizes = {
        args.epochs[0]: 15,   # early → visible
        args.epochs[1]: 10,
        args.epochs[-1]: 5,   # late → small
    }

    # -----------------------------
    def draw(data, cmap, line_style, marker_style):

        for i, e in enumerate(args.epochs):

            key = f"epoch_{e}"
            if key not in data:
                continue

            color = cmap(0.55 + 0.35*i/len(args.epochs))

            x = np.array(data[key][x_key]["mean"])
            y = np.array(data[key][y_key]["mean"])
            y_std = np.array(data[key][y_key]["std"])

            order = np.argsort(x)
            x, y, y_std = x[order], y[order], y_std[order]

            lw = widths[e]
            alpha = alphas[e]

            # std band
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=band_alpha[e],
                linewidth=0,
                zorder=1,
            )

            # curve
            ax.plot(
                x, y,
                line_style,
                color=color,
                linewidth=0.65,
                alpha=alpha,
                zorder=2,
            )

            # markers — scaled by epoch
            ax.scatter(
                x, y,
                marker=marker_style,
                color="black",
                s=marker_sizes[e],
                linewidths=0.2,
                alpha=0.6,
                zorder=3,
            )

    # HPN = circles
    draw(hpn_data, reds, "--", "o")

    # MML = crosses
    draw(mml_data, blues, "-", "x")

    # -----------------------------
    legend = [
        Line2D([0],[0], linestyle="--", color=reds(0.9),
               linewidth=1, label="HPN (x)"),
        Line2D([0],[0], linestyle="-", color=blues(0.9),
               linewidth=1, label="MML (o)"),
    ]

    ax.legend(handles=legend, frameon=False, fontsize=10)

    ax.grid(alpha=0.22)
    plt.tight_layout()

    if args.out:
        plt.savefig(args.out, dpi=300, bbox_inches="tight")
        print("Saved:", args.out)
    else:
        plt.show()


if __name__ == "__main__":
    main()


import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def load_results(path):
    with open(path, "r") as f:
        return json.load(f)


def build_epoch_style_maps(epochs):
    """Create width/alpha/marker-size maps with visual hierarchy."""
    n = len(epochs)
    widths = {}
    alphas = {}
    band_alphas = {}
    marker_sizes = {}

    for i, e in enumerate(epochs):
        t = i / max(n - 1, 1)
        widths[e] = 0.9 + 1.4 * t        # thin → thicker
        alphas[e] = 0.55 + 0.45 * t      # light → solid
        band_alphas[e] = 0.07 + 0.09 * t # light → stronger
        marker_sizes[e] = 16 - 8 * t     # bigger → smaller

    return widths, alphas, band_alphas, marker_sizes


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser("Pareto Evolution — Clean Paper Style")

    parser.add_argument("--hpn", required=True)
    parser.add_argument("--mml", required=True)
    parser.add_argument("--datatype", required=True,
                        choices=["image", "tabular", "temporal"])
    parser.add_argument("--epochs", nargs="+", type=int,
                        default=[10, 50, 100])
    parser.add_argument("--title", default="Pareto Front Evolution")
    parser.add_argument("--out", default=None)

    args = parser.parse_args()

    hpn_data = load_results(args.hpn)
    mml_data = load_results(args.mml)

    # -----------------------------
    # metric keys
    # -----------------------------
    if args.datatype == "image":
        x_key, y_key = "task0_loss", "task1_loss"
        x_label, y_label = "Task 1 loss", "Task 2 loss"
    elif args.datatype == "tabular":
        x_key, y_key = "pred_loss", "fair_loss_ddp"
        x_label, y_label = "Prediction loss", "Fairness loss"
    else:
        x_key, y_key = "task0_loss", "task1_loss"
        x_label, y_label = "Drop anomaly loss", "Spike anomaly loss"

    fig, ax = plt.subplots(figsize=(7, 5.2))

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(args.title, fontsize=13)

    epochs = sorted(args.epochs)

    widths, alphas, band_alphas, marker_sizes = build_epoch_style_maps(epochs)

    # use truncated ranges for nicer tones
    reds = plt.cm.Reds
    blues = plt.cm.Blues

    # ------------------------------------------------------------
    def draw(data, cmap, line_style, marker_style):
        n = len(epochs)

        for i, e in enumerate(epochs):
            key = f"epoch_{e}"
            if key not in data:
                continue

            # color progression inside cmap
            color = cmap(0.55 + 0.35 * (i / max(n - 1, 1)))

            x = np.array(data[key][x_key]["mean"])
            y = np.array(data[key][y_key]["mean"])
            y_std = np.array(data[key][y_key]["std"])

            order = np.argsort(x)
            x, y, y_std = x[order], y[order], y_std[order]

            # ---------- std band ----------
            ax.fill_between(
                x,
                y - y_std,
                y + y_std,
                color=color,
                alpha=band_alphas[e],
                linewidth=0,
                zorder=1,
            )

            # ---------- curve ----------
            ax.plot(
                x, y,
                line_style,
                color=color,
                linewidth=widths[e],
                alpha=alphas[e],
                zorder=2,
            )

            # ---------- markers ----------
            ax.scatter(
                x, y,
                marker=marker_style,
                color="black",
                s=marker_sizes[e],
                linewidths=0.25,
                alpha=0.75,
                zorder=3,
            )

    # HPN = circles
    draw(hpn_data, reds, "--", "o")

    # MML = crosses
    draw(mml_data, blues, "-", "x")

    # ------------------------------------------------------------
    # Legend (corrected)
    # ------------------------------------------------------------
    legend = [
        Line2D([0], [0],
               linestyle="--",
               color=reds(0.9),
               linewidth=1.8,
               marker="o",
               markerfacecolor="black",
               markeredgecolor="black",
               markersize=4,
               label="HPN"),

        Line2D([0], [0],
               linestyle="-",
               color=blues(0.9),
               linewidth=1.8,
               marker="x",
               markeredgecolor="black",
               markersize=5,
               label="MML"),
    ]

    ax.legend(handles=legend, frameon=False, fontsize=10)

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