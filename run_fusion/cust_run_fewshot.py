"""
Few-shot attacker: the attacker model of cust_run_LOO.py, plus num_fewshot recordings per vote
of the target user in training (main_fewshot.py). Shots come from the target user's
even-position recordings and are drawn again for every try; the target user is tested on its
odd-position recordings, the same test files as cust_run_perUser.py and cust_run_LOO.py.
0 shots is the attacker model of cust_run_LOO.py.

With two_stage, every user and number of shots first gets an eye-only model (the first of
feature_names), then the fused model of all feature_names with that eye branch loaded and
frozen. Each try of the fused model loads the eye model of the same try, trained on the same shots.

Results go to results/fewshot/, with a user x shots summary table per stage at the end.
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
result_dir = os.path.join("results", "fewshot")

# Target-user files per vote added to training; -1 is full shot (all even-position files)
shot_list = [3]

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


def shot_name(num_fewshot):
    return "full" if num_fewshot < 0 else str(num_fewshot)


def run(user, num_fewshot, features, freeze_eye):
    """
    Train and test one model of these features for this user and number of shots.
    freeze_eye: load the eye branch of the eye-only models of the same user and shots (one per try)
    and keep it fixed.
    Returns the path of its test predictions, or None if the run failed.
    """
    split = shot_name(num_fewshot) + "shot_" + user
    args = ["python", "run_fusion/main_fewshot.py", "--target_user", user, "--num_fewshot", str(num_fewshot),
            "--feature_names", *features] + train_args

    if freeze_eye:
        args += ["--pretrained_model", os.path.join(result_dir, "model_" + eye_feature + "_" + split + "_" + str(num_vote) + "q_fuseFeat_try{try}.pth"),
                 "--freeze_features", eye_feature]

    result = subprocess.run(args)

    pred_path = os.path.join(result_dir, "pred_labels_" + "+".join(features) + "_" + split + "_" + str(num_vote) + "q_fuseFeat_alltries.csv")
    if result.returncode != 0 or not os.path.exists(pred_path):
        print("Run of", "+".join(features), "for", user, "with", shot_name(num_fewshot), "shots failed")
        return None

    return pred_path


def accuracy(pred_path):
    """Accuracy averaged over tries."""
    df = pd.read_csv(pred_path)
    correct = df['pred_labels'] == df['target_labels']
    return correct.groupby(df['try']).mean().mean()


stages = [[eye_feature], list(feature_names)] if two_stage and len(feature_names) > 1 else [list(feature_names)]
summaries = {
    "+".join(features): pd.DataFrame(index=user_list, columns=[shot_name(k) + "shot" for k in shot_list], dtype=float)
    for features in stages
}

for num_fewshot in shot_list:
    for user in user_list:
        for i, features in enumerate(stages):
            pred_path = run(user, num_fewshot, features, freeze_eye=(i == 1))
            if pred_path is None:
                break  # the fused stage needs the eye-only models of this user and shots

            summaries["+".join(features)].loc[user, shot_name(num_fewshot) + "shot"] = accuracy(pred_path)

for tag, summary in summaries.items():
    summary.loc['Average'] = summary.mean()
    summary.index.name = 'User'

    summary_path = os.path.join(result_dir, "summary_" + tag + "_" + str(num_vote) + "q_fewshot.csv")
    summary.to_csv(summary_path)
    print("\n" + tag)
    print(summary.round(3).to_string())
    print("Saved summary to", summary_path)
