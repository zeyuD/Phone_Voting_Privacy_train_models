import os
import sys
import argparse

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config

config = load_machine_config()

data_dir = config["data_dir"] + "Phone_Privacy/data/webgazer_C/"

# Gaze was recorded on a horizontal 1920x1080 screen, plotted on a vertical 1080x1920 canvas
SRC_W, SRC_H = 1920, 1080
DST_W, DST_H = 1080, 1920


def load_gaze_points(csv_path, normalized=None, flip=True, x_stretch=2.0):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # Skip the first data row
    df = df.iloc[1:]
    df = df[["FPOGX", "FPOGY"]].apply(pd.to_numeric, errors="coerce").dropna()
    x, y = df["FPOGX"].to_numpy(), df["FPOGY"].to_numpy()

    # Gazepoint FPOG is normally a 0-1 fraction of the screen; otherwise assume source pixels
    if normalized is None:
        normalized = x.max() <= 1.5 and y.max() <= 1.5
    if normalized:
        x, y = x * DST_W, y * DST_H
    else:
        x, y = x * DST_W / SRC_W, y * DST_H / SRC_H

    cx = DST_W / 2
    if flip:
        x = DST_W - x
    # Stretch horizontally away from the vertical center line
    x = cx + (x - cx) * x_stretch
    return x, y


def plot_gaze_points(x, y, out_path, radius=20, color="red", alpha=1.0, linewidth=2, show_index=False, font_size=10):
    dpi = 100
    fig = plt.figure(figsize=(DST_W / dpi, DST_H / dpi), dpi=dpi)
    fig.patch.set_alpha(0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, DST_W)
    ax.set_ylim(DST_H, 0)  # screen coordinates: origin at top-left
    ax.set_aspect("equal")
    ax.axis("off")
    ax.patch.set_alpha(0)

    for i, (px, py) in enumerate(zip(x, y), start=1):
        ax.add_patch(Circle((px, py), radius, fill=False, edgecolor=color, linewidth=linewidth, alpha=alpha))
        if show_index:
            ax.text(px, py, str(i), ha="center", va="center", fontsize=font_size, color=color, alpha=alpha,
                    fontweight="bold")

    fig.savefig(out_path, dpi=dpi, transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot FPOGX/FPOGY gaze points on a transparent 1080x1920 canvas")
    parser.add_argument("csv", help="Gaze CSV path (absolute, or relative to data_dir)")
    parser.add_argument("-o", "--output", help="Output PNG path (default: <csv name>_gaze.png next to the CSV)")
    parser.add_argument("--index", action="store_true", help="Draw the point index inside each circle")
    parser.add_argument("--radius", type=float, default=20, help="Circle radius in pixels")
    parser.add_argument("--color", default="red", help="Circle outline (and index) color")
    parser.add_argument("--alpha", type=float, default=1.0, help="Circle opacity")
    parser.add_argument("--linewidth", type=float, default=2, help="Circle outline width")
    parser.add_argument("--no-flip", dest="flip", action="store_false", help="Don't mirror points horizontally")
    parser.add_argument("--x-stretch", type=float, default=2.0,
                        help="Scale x distance from the horizontal center by this factor")
    parser.add_argument("--font-size", type=float, default=10, help="Index label font size")
    coords = parser.add_mutually_exclusive_group()
    coords.add_argument("--normalized", dest="normalized", action="store_true", default=None,
                        help="Force FPOG values to be treated as 0-1 fractions")
    coords.add_argument("--pixels", dest="normalized", action="store_false",
                        help="Force FPOG values to be treated as 1920x1080 pixels")
    args = parser.parse_args()

    csv_path = args.csv if os.path.exists(args.csv) else os.path.join(data_dir, args.csv)
    out_path = args.output or os.path.splitext(csv_path)[0] + "_gaze.png"

    x, y = load_gaze_points(csv_path, normalized=args.normalized, flip=args.flip, x_stretch=args.x_stretch)
    plot_gaze_points(x, y, out_path, radius=args.radius, color=args.color, alpha=args.alpha,
                     linewidth=args.linewidth, show_index=args.index, font_size=args.font_size)
    print(f"Plotted {len(x)} points -> {out_path}")


if __name__ == "__main__":
    main()
