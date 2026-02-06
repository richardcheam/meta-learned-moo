#!/usr/bin/env python3
import argparse
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import numpy as np
from pathlib import Path

def _pad_limits(lo, hi, pad_frac):
    if lo == hi:
        # avoid zero-width; make a tiny window around the point
        span = max(abs(lo), 1e-9) * 0.05
        lo, hi = lo - span, hi + span
    span = hi - lo
    return lo - pad_frac * span, hi + pad_frac * span

def main():
    parser = argparse.ArgumentParser('plot with zoom inset')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--val-step', type=int, default=10, help='interval between validations')
    parser.add_argument('--resultspath', type=str, required=True, help='path to the results JSON file')
    parser.add_argument('--save', type=str, default='', help='optional path to save the figure (e.g., out.png)')
    parser.add_argument('--inset-pos', type=str, default='lower left',
                        help='inset location for inset_axes (e.g., upper right, lower left)')
    parser.add_argument('--inset-size', type=str, default='40%',
                        help='inset size as percentage string (e.g., 40%%) or float in inches')
    parser.add_argument('--zoom-pad', type=float, default=0.10,
                        help='fractional padding around auto zoom limits (default: 0.10)')
    args = parser.parse_args()

    with open(args.resultspath, 'r') as f:
        data = json.load(f)

    fig, ax = plt.subplots()
    ax.set_xlabel("Task 1 loss")
    ax.set_ylabel("Task 2 loss")

    # Plot epoch curves
    for i in range(args.val_step, args.epochs, args.val_step):
        x = data[f'epoch_{i}']['task0_loss']
        y = data[f'epoch_{i}']['task1_loss']
        ax.plot(x, y, 'x-', color=(1.0 * (i / args.epochs), 0.0, 0.0), label=f'epoch {i}', linewidth=1.0)

    # Last epoch (usually the Pareto front to zoom)
    x_last = np.asarray(data[f'epoch_{args.epochs}']['task0_loss'])
    y_last = np.asarray(data[f'epoch_{args.epochs}']['task1_loss'])
    ax.plot(x_last, y_last, 'x-', color=(1.0, 0.0, 0.0), label=f'epoch {args.epochs}', linewidth=1.0)

    # Create inset and re-plot everything inside for context
    axins = inset_axes(ax, width=args.inset_size, height=args.inset_size, loc=args.inset_pos)

    for i in range(args.val_step, args.epochs, args.val_step):
        xi = data[f'epoch_{i}']['task0_loss']
        yi = data[f'epoch_{i}']['task1_loss']
        axins.plot(xi, yi, 'x-', color=(1.0 * (i / args.epochs), 0.0, 0.0), linewidth=1.0)
    axins.plot(x_last, y_last, 'x-', color=(1.0, 0.0, 0.0), linewidth=1.0)

    # Auto-zoom to the extent of the last-epoch points, with padding
    xlo, xhi = float(np.min(x_last)), float(np.max(x_last))
    ylo, yhi = float(np.min(y_last)), float(np.max(y_last))
    xlo, xhi = _pad_limits(xlo, xhi, args.zoom_pad)
    ylo, yhi = _pad_limits(ylo, yhi, args.zoom_pad)
    axins.set_xlim(xlo, xhi)
    axins.set_ylim(ylo, yhi)
    
    #axins.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    axins.set_title(f"zoom on epoch {args.epochs}", fontsize=8)

    # Optional: outline the zoomed region on the main axes
    try:
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")
    except Exception:
        # mark_inset can fail if axes are identical; safe to ignore
        pass

    ax.legend(loc='best')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.tight_layout()

    if args.save:
        outpath = Path(args.save)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=200, bbox_inches='tight')
    else:
        plt.show()

if __name__ == '__main__':
    main()
