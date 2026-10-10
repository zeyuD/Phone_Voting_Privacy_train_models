"""
Compare the classification results of two features on the same test samples.

Inputs are the per-sample prediction files written by main_5q_fuseFeat.py:
    results/pred_labels_<feature>_5q_fuseFeat_alltries.csv

Samples are matched on (user, vote, file); only samples present in both files are scored.

With no arguments, the two features in split_ref_features (variables_fuseFeat.py) are compared:
    python run_fusion/compare_features.py

Or give the two files explicitly:
    python run_fusion/compare_features.py \
        results/pred_labels_all_processpos_norm_downsample_480p_s22_5q_fuseFeat_alltries.csv \
        results/pred_labels_opticalflowRAFT_22_downsample_480p_s22_5q_fuseFeat_alltries.csv \
        --names eye flow
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

KEY = ['user', 'vote', 'file']

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def default_files():
    """Prediction files of the two features listed in split_ref_features."""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from variables_fuseFeat import split_ref_features, vote_list

    if len(split_ref_features) != 2:
        raise SystemExit(
            f"split_ref_features lists {len(split_ref_features)} features, need exactly 2 "
            "to pick the files automatically. Pass the two csv files as arguments instead."
        )

    files = [
        os.path.join(REPO_DIR, "results", f"pred_labels_{feature}_{len(vote_list)}q_fuseFeat_alltries.csv")
        for feature in split_ref_features
    ]

    return files, list(split_ref_features)


def load_predictions(path):
    df = pd.read_csv(path)

    missing = [c for c in KEY + ['try', 'pred_labels', 'target_labels'] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} has no column(s) {missing}. "
            "Re-run main_5q_fuseFeat.py and use the *_alltries.csv file."
        )

    if df.duplicated(KEY + ['try']).any():
        raise ValueError(f"{path} has repeated (user, vote, file) rows within one try.")

    return df


def balanced_accuracy(df):
    """Mean of per-class recall, same as 'Average accuracy' in main_5q_fuseFeat.py."""
    return (df['pred_labels'] == df['target_labels']).groupby(df['target_labels']).mean().mean()


def per_try_scores(df):
    rows = []
    for t, df_t in df.groupby('try'):
        rows.append({
            'try': t,
            'accuracy': (df_t['pred_labels'] == df_t['target_labels']).mean(),
            'balanced_accuracy': balanced_accuracy(df_t),
        })
    return pd.DataFrame(rows).set_index('try')


def ensemble_over_tries(df):
    """
    One prediction per sample: softmax the scores of each try, average over tries, take argmax.
    """
    score_cols = [c for c in df.columns if c.startswith('score_')]

    scores = df[score_cols].to_numpy(dtype=np.float64)
    scores = np.exp(scores - scores.max(axis=1, keepdims=True))
    probs = pd.DataFrame(scores / scores.sum(axis=1, keepdims=True), columns=score_cols)

    for k in KEY + ['target_labels']:
        probs[k] = df[k].values

    mean_probs = probs.groupby(KEY + ['target_labels'], sort=True)[score_cols].mean()

    out = mean_probs.index.to_frame(index=False)
    out['pred_labels'] = mean_probs.to_numpy().argmax(axis=1)
    out['correct'] = out['pred_labels'] == out['target_labels']

    return out


def mean_std(values):
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.4f} ± {std:.4f}"


def resolve_inputs(parser, args):
    """Return the two prediction files and the two feature names to use."""
    if len(args.files) == 0:
        files, names = default_files()
    elif len(args.files) == 2:
        files, names = args.files, ['A', 'B']
    else:
        parser.error("give either no files or exactly two files")

    for path in files:
        if not os.path.isfile(path):
            raise SystemExit(f"Not found: {path}\nRun main_5q_fuseFeat.py with this feature first.")

    name_a, name_b = args.names if args.names is not None else names
    print(f"{name_a}: {files[0]}\n{name_b}: {files[1]}\n")

    return files, name_a, name_b


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help="Two *_alltries.csv files; default: the split_ref_features pair")
    parser.add_argument('--names', nargs=2, default=None, help="Short names for the two features")
    parser.add_argument('--out', default=None, help="Optional csv path for the per-sample comparison")
    args = parser.parse_args()

    files, name_a, name_b = resolve_inputs(parser, args)

    df_a = load_predictions(files[0])
    df_b = load_predictions(files[1])

    # ---------------------------------------------------------
    # Keep only samples tested in both runs
    # ---------------------------------------------------------
    keys_a = df_a[KEY].drop_duplicates()
    keys_b = df_b[KEY].drop_duplicates()
    shared = keys_a.merge(keys_b, on=KEY)

    print("Test samples")
    print(f"  {name_a}: {len(keys_a)}   {name_b}: {len(keys_b)}   shared: {len(shared)}")

    if len(shared) < len(keys_a) or len(shared) < len(keys_b):
        print(f"  WARNING: {len(keys_a) - len(shared)} only in {name_a}, "
              f"{len(keys_b) - len(shared)} only in {name_b}. They are ignored below.")
        print("  Check that split_ref_features was the same in both runs.")

    if len(shared) == 0:
        raise SystemExit("No shared test samples, nothing to compare.")

    df_a = df_a.merge(shared, on=KEY)
    df_b = df_b.merge(shared, on=KEY)

    # ---------------------------------------------------------
    # Scores per try
    # ---------------------------------------------------------
    tries_a = per_try_scores(df_a)
    tries_b = per_try_scores(df_b)

    print("\nPer-try scores on shared samples (mean ± std over tries)")
    for name, tries in [(name_a, tries_a), (name_b, tries_b)]:
        print(f"  {name}: accuracy {mean_std(tries['accuracy'])}   "
              f"balanced accuracy {mean_std(tries['balanced_accuracy'])}   "
              f"({len(tries)} tries)")

    # ---------------------------------------------------------
    # Paired comparison, one ensembled prediction per sample
    # ---------------------------------------------------------
    ens_a = ensemble_over_tries(df_a)
    ens_b = ensemble_over_tries(df_b)

    paired = ens_a.merge(ens_b, on=KEY + ['target_labels'], suffixes=('_' + name_a, '_' + name_b))
    correct_a = paired['correct_' + name_a]
    correct_b = paired['correct_' + name_b]

    print("\nEnsemble over tries (mean softmax score)")
    print(f"  {name_a}: accuracy {correct_a.mean():.4f}")
    print(f"  {name_b}: accuracy {correct_b.mean():.4f}")

    both = int((correct_a & correct_b).sum())
    only_a = int((correct_a & ~correct_b).sum())
    only_b = int((~correct_a & correct_b).sum())
    neither = int((~correct_a & ~correct_b).sum())

    print("\nPaired outcome per sample")
    print(f"  both correct: {both}   only {name_a}: {only_a}   only {name_b}: {only_b}   both wrong: {neither}")

    if only_a + only_b > 0:
        p_value = binomtest(only_a, only_a + only_b, 0.5).pvalue
        print(f"  McNemar exact test: p = {p_value:.4g}")
    else:
        print("  McNemar exact test: not defined, the two features never disagree.")

    # ---------------------------------------------------------
    # Breakdown per class and per user
    # ---------------------------------------------------------
    for column, title in [('vote', 'class'), ('user', 'user')]:
        table = paired.groupby(column).agg(
            n=('file', 'size'),
            **{name_a: ('correct_' + name_a, 'mean'), name_b: ('correct_' + name_b, 'mean')},
        )
        table['diff'] = table[name_a] - table[name_b]

        print(f"\nAccuracy per {title} (ensemble)")
        print(table.to_string(float_format=lambda x: f"{x:.3f}"))

    if args.out is not None:
        paired.to_csv(args.out, index=False)
        print("\nSaved per-sample comparison to", args.out)


if __name__ == '__main__':
    main()
