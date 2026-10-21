"""
Attacker model: for every user in user_list, train on all recordings of the other users and
test on the left-out user (main_LOO.py). The left-out user is tested on its odd-position
recordings, the same test files as cust_run_perUser.py and cust_run_fewshot.py.

With two_stage, every left-out user first gets an eye-only model (the first of feature_names), then the
fused model of all feature_names with that user's eye branch loaded and frozen.

Results go to results/LOO/, with a per-user summary table per stage at the end.
"""

import subprocess
import os
import sys

import pandas as pd

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config
from run_fusion.variables_fuseFeat import user_list, vote_list, feature_names

config = load_machine_config()

num_vote = len(vote_list)
result_dir = os.path.join("results", "LOO")

# True: train the eye feature alone, then the fused model with the eye branch frozen.
# False: train only the fused model, all branches together.
two_stage = True
eye_feature = feature_names[0]

train_args = [
    "--output_dir", "experiments",
    "--comment", "classification from Scratch",
    "--name", "VRS_fromScratch",
    "--records_file", "VRS_records.xls",
    "--data_dir", config["data_dir"] + "Phone_Privacy/input_feature/",
    "--device", config["compdev"],
    "--no_crop_features", "all_processpos_norm_downsample_480p_s22", "all_processpos_norm_downsample_480p_op2",
          "all_processpos_norm_downsample_480p_downs22", "all_processpos_norm_downsample_480p_srs22",
          "all_processpos_norm_downsample_480p_180s22", "all_processpos_norm_downsample_480p_240s22",
          "opticalflowRAFT_edge_22_downsample_480p_srs22",
    "--no_crop_all", # uncomment to skip crop for every feature, regardless of --no_crop_features
    "--fusion_type", "same_time", # 'same_time', 'late', 'all_time', "feature", "both", "gated"
    "--use_loss_fusion", "True",
    "--pass_band", "1", "10", # 1-10 for optiflowRAFT edge
    "--lambda_aux", "0.1",
    "--wandb_project", "phone_vote", # uncomment to log training to Weights & Biases
    "--data_class", "vrs",
    "--val_ratio", "0.1",
    "--epochs", "100",
    "--lr", "0.0005",
    "--dropout", "0.2",
    "--batch_size", "8",
    "--d_model", "16",
    "--num_layers", "3",
    "--num_heads", "4",
    "--optimizer", "RAdam",
    "--pos_encoding", "fixed", # "learnable", "fixed"
    "--activation", "gelu", # "relu", "gelu"
    "--task", "classification",
    "--normalization", "none", # "none", "standardization", "minmax", "per_sample_std", "per_sample_minmax"
    "--key_metric", "accuracy",
]


def run(user, features, freeze_eye):
    """
    Train and test one model of these features for this user.
    freeze_eye: load the eye branch of this user's eye-only models (one per try) and keep it fixed.
    Returns the path of its test predictions, or None if the run failed.
    """
    args = ["python", "run_fusion/main_LOO.py", "--target_user", user, "--feature_names", *features] + train_args

    if freeze_eye:
        args += ["--pretrained_model", os.path.join(result_dir, "model_" + eye_feature + "_LOO_" + user + "_" + str(num_vote) + "q_fuseFeat_try{try}.pth"),
                 "--freeze_features", eye_feature]

    result = subprocess.run(args)

    pred_path = os.path.join(result_dir, "pred_labels_" + "+".join(features) + "_LOO_" + user + "_" + str(num_vote) + "q_fuseFeat_alltries.csv")
    if result.returncode != 0 or not os.path.exists(pred_path):
        print("Run of", "+".join(features), "for", user, "failed")
        return None

    return pred_path


def user_accuracy(pred_path):
    """Accuracy mean and std over tries, and accuracy per vote with all tries pooled."""
    df = pd.read_csv(pred_path)
    correct = df['pred_labels'] == df['target_labels']
    per_try = correct.groupby(df['try']).mean()

    row = {'acc': per_try.mean(), 'std': per_try.std(ddof=1) if len(per_try) > 1 else 0.0}
    for vote in vote_list:
        row[vote] = correct[df['vote'] == vote].mean()

    return row


stages = [[eye_feature], list(feature_names)] if two_stage and len(feature_names) > 1 else [list(feature_names)]
rows = {"+".join(features): [] for features in stages}

for user in user_list:
    for i, features in enumerate(stages):
        pred_path = run(user, features, freeze_eye=(i == 1))
        if pred_path is None:
            break  # the fused stage needs this user's eye-only models

        rows["+".join(features)].append({'User': user, **user_accuracy(pred_path)})

for tag, stage_rows in rows.items():
    if not stage_rows:
        continue

    summary = pd.DataFrame(stage_rows)
    summary.loc[len(summary)] = {'User': 'Average', **summary.drop(columns='User').mean().to_dict()}

    summary_path = os.path.join(result_dir, "summary_" + tag + "_" + str(num_vote) + "q_LOO.csv")
    summary.to_csv(summary_path, index=False)
    print("\n" + tag)
    print(summary.round(3).to_string(index=False))
    print("Saved summary to", summary_path)
