import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === Load Data ===
tcn_df = pd.read_csv('tcn_predictions.csv')
transformer_df = pd.read_csv('pred_labels.csv', header=None)
transformer_pred_labels = transformer_df.iloc[0].astype(int).values
transformer_target_labels = transformer_df.iloc[1].astype(int).values

# === Process TCN ===
tcn_preds = tcn_df['pred_labels'].astype(int).values
tcn_targets = tcn_df['target_labels'].astype(int).values
tcn_matches = (tcn_preds == tcn_targets).astype(int)

# === Process Transformer ===
transformer_matches = (transformer_pred_labels == transformer_target_labels).astype(int)

# === Background Coloring Function ===
def add_background_color(ax, labels, color_map, alpha=0.1):
    prev_label = labels[0]
    start = 0
    for i in range(1, len(labels)):
        if labels[i] != prev_label:
            ax.axvspan(start, i, color=color_map[prev_label], alpha=alpha)
            start = i
            prev_label = labels[i]
    # Last span
    ax.axvspan(start, len(labels), color=color_map[prev_label], alpha=alpha)

# === Plotting ===
fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
color_map = {0: 'lightblue', 1: 'lightgreen', 2: 'lightcoral'}

# Plot TCN
axes[0].scatter(np.arange(len(tcn_matches)), tcn_matches, c='black', s=10, label='Match (1) / Mismatch (0)')
add_background_color(axes[0], tcn_targets, color_map)
axes[0].set_title('TCN Prediction Matches with Background = Target Class')
axes[0].set_ylabel('Correct')
axes[0].set_ylim(-0.1, 1.1)

# Plot Transformer
axes[1].scatter(np.arange(len(transformer_matches)), transformer_matches, c='black', s=10)
add_background_color(axes[1], transformer_target_labels, color_map)
axes[1].set_title('Transformer Prediction Matches with Background = Target Class')
axes[1].set_ylabel('Correct')
axes[1].set_xlabel('Instance Index')
axes[1].set_ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()
