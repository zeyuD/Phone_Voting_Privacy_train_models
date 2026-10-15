"""
Top-k accuracy of saved test predictions.

A sample counts as correct at k when its true class is among the k highest scores.
Inputs are the per-sample prediction files written by main_5q_fuseFeat.py:
    results/pred_labels_<features>_<N>q_fuseFeat_alltries.csv

Examples:
    # every *_alltries.csv in results/
    python run_fusion/topk_accuracy.py

    # chosen files, only top-1 and top-2, with per-class top-k and a csv of the table
    python run_fusion/topk_accuracy.py results/pred_labels_all_processpos_norm_downsample_480p_s22_5q_fuseFeat_alltries.csv \
        --k 1 2 --per_class --out results/topk.csv
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_scores(path):
    """
    Returns:
        scores: [num_rows, num_classes] model outputs
        targets: [num_rows] true class index
        tries: [num_rows] try number (0 for files without a 'try' column)
        class_names: list of class letters, in score order
    """
    df = pd.read_csv(path)

    # Only the classes of this run have scores; columns of other classes are empty
    score_cols = [c for c in df.columns if c.startswith('score_') and df[c].notna().all()]
    if not score_cols or 'target_labels' not in df.columns:
        raise ValueError(f"{path} has no score_* or target_labels columns.")

    scores = df[score_cols].to_numpy(dtype=np.float64)
    targets = df['target_labels'].to_numpy(dtype=int)
    tries = df['try'].to_numpy(dtype=int) if 'try' in df.columns else np.zeros(len(df), dtype=int)
    class_names = [c[len('score_'):] for c in score_cols]

    return scores, targets, tries, class_names


def true_class_rank(scores, targets):
    """
    0-based rank of the true class among the scores of each sample.
    Ties count in favour of the true class.
    """
    true_scores = scores[np.arange(len(targets)), targets]
    return (scores > true_scores[:, None]).sum(axis=1)


def topk_per_try(rank, tries, ks):
    """Rows: tries, columns: k, values: top-k accuracy."""
    return pd.DataFrame(
        {k: [np.mean(rank[tries == t] < k) for t in np.unique(tries)] for k in ks},
        index=np.unique(tries),
    )


def short_name(path):
    name = os.path.basename(path)
    for prefix in ('pred_labels_',):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ('_fuseFeat_alltries.csv', '_alltries.csv', '.csv'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return (name.replace('_downsample_480p', '')
                .replace('all_processpos_norm', 'eye')
                .replace('opticalflowRAFT', 'flow'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*', help="Prediction csv files; default: results/*_alltries.csv")
    parser.add_argument('--k', type=int, nargs='+', default=None,
                        help="Values of k; default: 1 up to the number of classes minus 1")
    parser.add_argument('--per_class', action='store_true', help="Also print top-k per true class (all tries pooled)")
    parser.add_argument('--out', default=None, help="Optional csv path for the summary table")
    args = parser.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(REPO_DIR, 'results', 'pred_labels_*_alltries.csv')),
                                 key=os.path.getmtime)
    if not files:
        raise SystemExit("No prediction files found.")

    summary = []

    for path in files:
        try:
            scores, targets, tries, class_names = load_scores(path)
        except ValueError as e:
            print(f"Skipping: {e}")
            continue

        num_classes = scores.shape[1]
        ks = [k for k in (args.k or range(1, num_classes)) if 1 <= k <= num_classes]
        rank = true_class_rank(scores, targets)
        per_try = topk_per_try(rank, tries, ks)

        print(f"\n{short_name(path)}  ({''.join(class_names)}, {len(per_try)} tries, "
              f"{int(np.sum(tries == tries[0]))} test samples per try)")
        print("  " + "  ".join(f"top-{k} {per_try[k].mean():.3f} ± {per_try[k].std(ddof=1) if len(per_try) > 1 else 0:.3f}"
                               for k in ks))
        print("  chance: " + "  ".join(f"top-{k} {k / num_classes:.3f}" for k in ks))

        if args.per_class:
            for c, name in enumerate(class_names):
                m = targets == c
                print(f"    class {name}: " + "  ".join(f"top-{k} {np.mean(rank[m] < k):.3f}" for k in ks))

        row = {'run': short_name(path), 'classes': ''.join(class_names), 'tries': len(per_try),
               'samples_per_try': int(np.sum(tries == tries[0]))}
        for k in ks:
            row[f'top{k}_mean'] = per_try[k].mean()
            row[f'top{k}_std'] = per_try[k].std(ddof=1) if len(per_try) > 1 else 0.0
        summary.append(row)

    if args.out is not None and summary:
        pd.DataFrame(summary).to_csv(args.out, index=False)
        print(f"\nSaved summary to {args.out}")


if __name__ == '__main__':
    main()
