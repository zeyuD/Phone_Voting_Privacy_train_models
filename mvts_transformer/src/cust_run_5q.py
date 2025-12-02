import subprocess
import pandas as pd
import os
from variables import feature_names, user_list, vote_list, num_try
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

# determine number of instances
num_instance = 0

for feature_name in feature_names:
    # Create an empty dataframe, columns are user names
    # df = pd.DataFrame(columns=vote_list)
    # df.to_csv("F1_avg.csv", index=False)
    # df2 = pd.DataFrame(columns=vote_list)
    # df2.to_csv("FNR_avg.csv", index=False)
    df2 = pd.DataFrame(columns=['pred_labels', 'target_labels', 'score_A', 'score_B', 'score_C', 'score_D', 'score_E'])
    df2.to_csv("train_pred_labels.csv", index=False)
    df3 = pd.DataFrame(columns=['pred_labels', 'target_labels', 'score_A', 'score_B', 'score_C', 'score_D', 'score_E'])
    df3.to_csv("pred_labels.csv", index=False)
    df4 = pd.DataFrame(columns=vote_list)
    df4.to_csv("acc_avg.csv", index=False)
    for t in range(num_try):
        # print("User: ", user)
        subprocess.run(["python", "src/main_5q.py",
                        "--output_dir", "experiments",
                        "--comment", "classification from Scratch",
                        "--name", "VRS_fromScratch",
                        "--records_file", "VRS_records.xls",
                        "--data_dir", config["data_dir"] + "Phone_Privacy/input_feature/" + feature_name,
                        "--data_class", "vrs",
                        "--val_ratio", "0.1",
                        # "--test_ratio", "0.4",
                        "--epochs", "40", # Too long all B, default 30; <40 for selection frames, 55 for full frames
                        "--lr", "0.001", # Too high all B
                        "--dropout", "0.2",
                        "--batch_size", "8",
                        "--num_layers", "3",
                        "--num_heads", "4",
                        "--optimizer", "RAdam", # Adam all B
                        "--pos_encoding", "learnable",
                        "--task", "classification",
                        "--normalization", "none", # "none", "standardization", "minmax", "per_sample_std", "per_sample_minmax"
                        "--key_metric", "accuracy"])
    # Remove NaN values and move next row up
    # df = pd.read_csv("F1_avg.csv")
    # df1 = df.apply(lambda x: pd.Series(x.dropna().values))
    # df1.to_csv("F1_avg_"+feature_name+".csv", index=False)
    # df_ = pd.read_csv("FNR_avg.csv")
    # df2 = df_.apply(lambda x: pd.Series(x.dropna().values))
    # df2.to_csv("FNR_avg_"+feature_name+".csv", index=False)
    df = pd.read_csv("acc_avg.csv")
    df5 = df.apply(lambda x: pd.Series(x.dropna().values))
    df5.to_csv("acc_avg_"+feature_name+"_5q.csv", index=False)

    print(df5)
    acc_avg = df5.mean().mean()
    print("Accuracy average:", acc_avg)

    df6 = pd.read_csv("pred_labels.csv")
    df6.to_csv("pred_labels_"+feature_name+"_5q.csv", index=False)

    df7 = pd.read_csv("train_pred_labels.csv")
    df7.to_csv("train_pred_labels_"+feature_name+"_5q.csv", index=False)