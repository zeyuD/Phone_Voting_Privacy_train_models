import subprocess
import pandas as pd
import os
from variables import feature_names, user_list, vote_list, num_try

# determine number of instances
num_instance = 0

for feature_name in feature_names:
    for v in range(len(vote_list)):
        vote = vote_list[v]
        train_path = "/mnt/e/Vote_Privacy/data/" + feature_name + "/" + vote
        for user in user_list:
            files = [file for file in os.listdir(train_path) if user in file]
            num_instance += int(len(files) / 2)

    # Create an empty dataframe, columns are user names
    # df = pd.DataFrame(columns=vote_list)
    # df.to_csv("F1_avg.csv", index=False)
    # df2 = pd.DataFrame(columns=vote_list)
    # df2.to_csv("FNR_avg.csv", index=False)
    df3 = pd.DataFrame(columns=range(num_instance))
    df3.to_csv("pred_labels.csv", index=False)
    df4 = pd.DataFrame(columns=vote_list)
    df4.to_csv("acc_avg.csv", index=False)
    for t in range(num_try):
        # print("User: ", user)
        subprocess.run(["python", "src/main.py",
                        "--output_dir", "experiments",
                        "--comment", "classification from Scratch",
                        "--name", "VRS_fromScratch",
                        "--records_file", "VRS_records.xls",
                        "--data_dir", "/mnt/e/Vote_Privacy/data/"+feature_name,
                        # "--data_dir", "/mnt/e/Vote_Privacy/data_old/"+feature_name,
                        "--data_class", "vrs",
                        "--val_ratio", "0.1",
                        # "--test_ratio", "0.4",
                        "--epochs", "24", # Too long all B, <40 for selection frames, 55 for full frames
                        "--lr", "0.001", # Too high all B
                        "--dropout", "0.2",
                        "--batch_size", "64",
                        "--num_layers", "2",
                        "--num_heads", "8",
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
    df3 = df.apply(lambda x: pd.Series(x.dropna().values))
    df3.to_csv("acc_avg_"+feature_name+".csv", index=False)

    print(df3)
    acc_avg = df3.mean().mean()
    print("Accuracy average:", acc_avg)

    # df4 = pd.read_csv("pred_labels.csv")