import subprocess
import pandas as pd
import os
from variables import feature_names, user_list, vote_list, num_try
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions')))
from load_machine_config import load_machine_config

config = load_machine_config()
num_instance = 0

# 2 choices horizontal
# 720p: 87.3%
# 480p: 93.7%
# 360p: 92.6%
# 240p: 91.9%
# 2 choices vertical
# 720p: 64.4%
# 480p: 70.0%
# 360p: 69.8%
# 240p: 58.3%

for feature_name in feature_names:
    for v in range(len(vote_list)):
        vote = vote_list[v]
        train_path = os.path.join(config["data_dir"], "Vote_Privacy", "input_feature", feature_name, vote)
        for user in user_list:
            files = [file for file in os.listdir(train_path) if user in file and not file.startswith(".")]
            num_instance += int(len(files) / 2)

    # Create placeholder CSVs
    # df3 = pd.DataFrame(columns=range(num_instance))
    # df3.to_csv(f"TCN_pred_labels_{feature_name}.csv", index=False)

    # df4 = pd.DataFrame(columns=vote_list)
    # df4.to_csv(f"TCN_acc_avg_{feature_name}.csv", index=False)

    for t in range(num_try):
        print(f"[Trial {t}] Running feature: {feature_name}")

        subprocess.run([
            "python", "src/main_TCN.py",
            "--output_dir", "experiments",
            "--data_dir", config["data_dir"] + "Vote_Privacy/input_feature/" + feature_name,
            "--data_class", "vrs",
            "--val_ratio", "0.1",
            "--test_ratio", "0.0",
            "--epochs", "200",
            "--lr", "0.001",
            "--dropout", "0.2",
            "--batch_size", "8",
            "--normalization", "none"
        ])
