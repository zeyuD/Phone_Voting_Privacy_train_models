"""
Visualize which test samples each of two features classifies correctly.

Every shared test sample falls in one of four outcomes:
    both correct / only A correct / only B correct / both wrong

Outputs (in results/ by default):
    compare_<A>_vs_<B>.png          counts in total, per class, per user, and a map of every sample
    compare_<A>_vs_<B>_samples.csv  one row per sample with its outcome and both predictions

Inputs are the same as compare_features.py. With no arguments, the two features
in split_ref_features (variables_fuseFeat.py) are compared:
    python run_fusion/plot_compare_features.py

In the figure the features are called "eye feature" / "optical flow" (override with --names)
and users are shown as User 1, User 2, ... in the order of user_list. The csv keeps both names.

By default each sample uses one prediction ensembled over all tries; --try N uses a single try.
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.stats import binomtest

from compare_features import KEY, REPO_DIR, resolve_inputs, load_predictions, ensemble_over_tries

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"

# fill color, ink color for a label drawn inside that fill
OUTCOME_STYLE = [
    ("#dcdbd6", TEXT),       # both correct
    ("#2a78d6", "#ffffff"),  # only A correct
    ("#eb6834", TEXT),       # only B correct
    ("#52514e", "#ffffff"),  # both wrong
]


def display_name(file_path, fallback):
    """Readable feature name for the figure, guessed from the prediction file name."""
    base = os.path.basename(file_path)

    if '+' in base:
        return "fused"
    if 'processpos' in base:
        return "eye feature"
    if 'opticalflow' in base.lower():
        return "optical flow"

    return fallback


def anonymize_users(users):
    """Map real user names to 'User 1', 'User 2', ... following the order of user_list."""
    try:
        from variables_fuseFeat import user_list
    except ImportError:
        user_list = []

    ordered = [u for u in user_list if u in users] + sorted(set(users) - set(user_list), key=str.lower)

    return {user: f"User {i + 1}" for i, user in enumerate(ordered)}


def natural_key(name):
    """Sort Minjie_A_2.csv before Minjie_A_10.csv."""
    return [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', name)]


def build_paired(df_a, df_b, try_idx=None):
    """One row per shared test sample with both predictions and the outcome index (0-3)."""
    shared = df_a[KEY].drop_duplicates().merge(df_b[KEY].drop_duplicates(), on=KEY)

    if len(shared) == 0:
        raise SystemExit("No shared test samples, nothing to compare.")

    preds = []
    for df in (df_a, df_b):
        df = df.merge(shared, on=KEY)

        if try_idx is None:
            df = ensemble_over_tries(df)
        else:
            df = df[df['try'] == try_idx]
            if len(df) == 0:
                raise SystemExit(f"Try {try_idx} not found in the prediction files.")

        preds.append(df[KEY + ['target_labels', 'pred_labels']])

    paired = preds[0].merge(preds[1], on=KEY + ['target_labels'], suffixes=('_a', '_b'))

    correct_a = paired['pred_labels_a'] == paired['target_labels']
    correct_b = paired['pred_labels_b'] == paired['target_labels']

    paired['outcome'] = np.select(
        [correct_a & correct_b, correct_a & ~correct_b, ~correct_a & correct_b],
        [0, 1, 2],
        default=3,
    )

    return paired


def count_table(paired, column, order):
    """Rows: values of `column`, columns: outcome 0-3, values: number of samples."""
    table = pd.crosstab(paired[column], paired['outcome'])
    return table.reindex(index=order, columns=range(4), fill_value=0)


def draw_stacked(ax, table, title):
    """Horizontal stacked bars, one row per index entry, one segment per outcome."""
    y = np.arange(len(table))
    left = np.zeros(len(table))
    x_max = table.sum(axis=1).max()

    for outcome, (fill, ink) in enumerate(OUTCOME_STYLE):
        counts = table[outcome].to_numpy()
        ax.barh(y, counts, left=left, height=0.72, color=fill, edgecolor=SURFACE, linewidth=1.5)

        for yi, count, start in zip(y, counts, left):
            if count >= 0.05 * x_max:
                ax.text(start + count / 2, yi, str(count), ha='center', va='center', color=ink, fontsize=8)

        left += counts

    ax.set_yticks(y)
    ax.set_yticklabels(table.index)
    ax.set_ylim(max(len(table), 5) - 0.5, -0.5)  # keeps bars thin when there are few rows
    ax.set_xlim(0, x_max * 1.02)
    ax.set_xlabel("test samples")
    ax.set_title(title, loc='left', fontsize=11, fontweight='bold')


def draw_totals(ax, paired, labels):
    counts = paired['outcome'].value_counts().reindex(range(4), fill_value=0).to_numpy()
    y = np.arange(4)

    ax.barh(y, counts, height=0.62, color=[fill for fill, _ in OUTCOME_STYLE])

    for yi, count in zip(y, counts):
        ax.text(count, yi, f"  {count}  ({count / len(paired):.1%})", ha='left', va='center', color=TEXT, fontsize=10)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, counts.max() * 1.35)
    ax.set_xlabel("test samples")
    ax.set_title("All shared test samples", loc='left', fontsize=11, fontweight='bold')


def draw_sample_map(ax, paired, users, votes):
    """One tile per test sample: rows are users, columns are samples grouped by class."""
    block_width = paired.groupby(['vote', 'user_label']).size().groupby('vote').max().reindex(votes, fill_value=0)

    gap = 1
    starts = np.concatenate([[0], np.cumsum(block_width.to_numpy() + gap)[:-1]])
    n_cols = int(starts[-1] + block_width.iloc[-1])

    grid = np.full((len(users), n_cols), np.nan)

    for (user, vote), group in paired.groupby(['user_label', 'vote']):
        group = group.loc[sorted(group.index, key=lambda i: natural_key(group.at[i, 'file']))]
        row = users.index(user)
        col = int(starts[votes.index(vote)])
        grid[row, col:col + len(group)] = group['outcome'].to_numpy()

    cmap = ListedColormap([fill for fill, _ in OUTCOME_STYLE])
    cmap.set_bad(SURFACE)

    ax.pcolormesh(
        np.ma.masked_invalid(grid), cmap=cmap, vmin=-0.5, vmax=3.5,
        edgecolors=SURFACE, linewidth=1.0,
    )

    ax.set_yticks(np.arange(len(users)) + 0.5)
    ax.set_yticklabels(users)
    ax.invert_yaxis()
    ax.set_xticks(starts + block_width.to_numpy() / 2)
    ax.set_xticklabels([f"class {v}" for v in votes])
    ax.tick_params(axis='both', length=0)
    ax.grid(False)
    ax.set_title("Every test sample (one tile each, in file order within a class)",
                 loc='left', fontsize=11, fontweight='bold')


def print_sample_lists(paired, labels):
    """Which samples fall in each outcome where at least one feature is wrong."""
    for outcome in (1, 2, 3):
        subset = paired[paired['outcome'] == outcome]
        print(f"\n{labels[outcome]}: {len(subset)} samples")

        for user, group in subset.groupby('user'):
            names = sorted(group['file'], key=natural_key)
            names = [os.path.splitext(n)[0].replace(user + "_", "", 1) for n in names]
            print(f"  {user} ({len(names)}): {', '.join(names)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help="Two *_alltries.csv files; default: the split_ref_features pair")
    parser.add_argument('--names', nargs=2, default=None, help="Names for the two features, default: guessed from the file names")
    parser.add_argument('--try', dest='try_idx', type=int, default=None,
                        help="Use this single try instead of the ensemble over all tries")
    parser.add_argument('--out_dir', default=os.path.join(REPO_DIR, "results"), help="Where to save the png and csv")
    args = parser.parse_args()

    files, name_a, name_b = resolve_inputs(parser, args)

    if args.names is None:
        name_a, name_b = display_name(files[0], name_a), display_name(files[1], name_b)

    paired = build_paired(load_predictions(files[0]), load_predictions(files[1]), args.try_idx)

    # Real names stay in the terminal listing and the csv; the figure shows User 1, User 2, ...
    user_labels = anonymize_users(paired['user'].unique())
    paired['user_label'] = paired['user'].map(user_labels)

    labels = ["both correct", f"only {name_a} correct", f"only {name_b} correct", "both wrong"]
    users = list(user_labels.values())
    votes = sorted(paired['vote'].unique())

    counts = paired['outcome'].value_counts().reindex(range(4), fill_value=0)
    acc_a = (counts[0] + counts[1]) / len(paired)
    acc_b = (counts[0] + counts[2]) / len(paired)

    print(f"Shared test samples: {len(paired)}")
    for outcome in range(4):
        print(f"  {labels[outcome]}: {counts[outcome]} ({counts[outcome] / len(paired):.1%})")

    print_sample_lists(paired, labels)

    # ---------------------------------------------------------
    # Per-sample csv, disagreements first
    # ---------------------------------------------------------
    stem = f"compare_{name_a}_vs_{name_b}".replace(" ", "_") + ("" if args.try_idx is None else f"_try{args.try_idx}")
    os.makedirs(args.out_dir, exist_ok=True)

    label_to_vote = dict(zip(paired['target_labels'], paired['vote']))
    samples = pd.DataFrame({
        'outcome': [labels[o] for o in paired['outcome']],
        'user': paired['user'],
        'user_label': paired['user_label'],
        'vote': paired['vote'],
        'file': paired['file'],
        'pred_' + name_a: paired['pred_labels_a'].map(lambda x: label_to_vote.get(x, x)),
        'pred_' + name_b: paired['pred_labels_b'].map(lambda x: label_to_vote.get(x, x)),
    })

    order = paired['outcome'].map({1: 0, 2: 1, 3: 2, 0: 3})
    samples = samples.iloc[np.lexsort((paired['user_label'].map(users.index), order))]

    csv_path = os.path.join(args.out_dir, stem + "_samples.csv")
    samples.to_csv(csv_path, index=False)

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------
    plt.rcParams.update({
        'font.size': 9.5,
        'text.color': TEXT,
        'axes.labelcolor': TEXT_SECONDARY,
        'xtick.color': TEXT_SECONDARY,
        'ytick.color': TEXT,
        'axes.edgecolor': GRID,
        'axes.facecolor': SURFACE,
        'figure.facecolor': SURFACE,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.grid': True,
        'axes.grid.axis': 'x',
        'grid.color': GRID,
        'grid.linewidth': 0.8,
        'axes.axisbelow': True,
    })

    map_height = max(0.32 * len(users) + 0.8, 2.2)
    fig_height = 6.2 + map_height
    fig = plt.figure(figsize=(15, fig_height))
    grid_spec = fig.add_gridspec(
        2, 3, height_ratios=[4.2, map_height], width_ratios=[1.15, 1, 1.15],
        left=0.135, right=0.98, top=1 - 1.85 / fig_height, bottom=0.05, hspace=0.95 / ((4.2 + map_height) / 2), wspace=0.42,
    )

    draw_totals(fig.add_subplot(grid_spec[0, 0]), paired, labels)
    draw_stacked(fig.add_subplot(grid_spec[0, 1]), count_table(paired, 'vote', votes), "Per class")
    draw_stacked(fig.add_subplot(grid_spec[0, 2]), count_table(paired, 'user_label', users), "Per user")
    draw_sample_map(fig.add_subplot(grid_spec[1, :]), paired, users, votes)

    for ax in fig.axes:
        ax.tick_params(axis='y', length=0)

    source = "ensemble over tries" if args.try_idx is None else f"try {args.try_idx}"
    if counts[1] + counts[2] > 0:
        p_text = f"McNemar exact p = {binomtest(int(counts[1]), int(counts[1] + counts[2]), 0.5).pvalue:.3g}"
    else:
        p_text = "the two features never disagree"

    fig.suptitle(f"{name_a} vs {name_b}: who classifies each test sample correctly",
                 x=0.02, y=0.985, ha='left', fontsize=15, fontweight='bold')
    fig.text(0.02, 1 - 0.62 / fig_height,
             f"{len(paired)} shared test samples, {source}.  Accuracy: {name_a} {acc_a:.1%}, {name_b} {acc_b:.1%}.  {p_text}",
             ha='left', va='top', fontsize=10.5, color=TEXT_SECONDARY)

    fig.legend(
        handles=[Patch(facecolor=fill, label=label) for (fill, _), label in zip(OUTCOME_STYLE, labels)],
        loc='upper left', bbox_to_anchor=(0.012, 1 - 0.92 / fig_height),
        ncol=4 if len(name_a) + len(name_b) <= 30 else 2, frameon=False, fontsize=10, handlelength=1.2, columnspacing=2.0,
    )

    png_path = os.path.join(args.out_dir, stem + ".png")
    fig.savefig(png_path, dpi=150)

    print("\nSaved figure to", png_path)
    print("Saved per-sample table to", csv_path)


if __name__ == '__main__':
    main()
