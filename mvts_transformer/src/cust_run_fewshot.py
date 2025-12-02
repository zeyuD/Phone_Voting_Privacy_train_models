import subprocess
import pandas as pd
import os
from variables import feature_names, user_list, vote_list, num_try
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

# for num_fewshot in [0, 1, 3, 5, 7, 9, 11, 13, 15]:
for num_fewshot in [11, 9, 7, 5, 3, 1, 0]:

    df1 = pd.DataFrame(columns=vote_list)
    # Add a 1st column for user names, leave empty
    df1.insert(0, 'User', '')
    # Add a last column for average, leave empty
    df1.insert(len(df1.columns), 'Avg', '')
    df1.to_csv("acc_avg.csv", index=False)

    for user in user_list:

        # determine number of instances
        num_instance = 0

        for feature_name in feature_names:
            for v in range(len(vote_list)):
                vote = vote_list[v]
                train_path = config["data_dir"] + "Vote_Privacy/input_feature/" + feature_name + "/" + vote

                # files = [file for file in os.listdir(train_path) if user in file and not file.startswith(".")]
                files = [
                    file for file in os.listdir(train_path)
                    if user in file and not file.startswith('.') and
                    int(file.split('_')[-1].split('.')[0]) % 2 == 1
                ]
                num_instance += len(files)

            # Create an empty dataframe, columns are user names
            # df = pd.DataFrame(columns=vote_list)
            # df.to_csv("F1_avg.csv", index=False)
            # df2 = pd.DataFrame(columns=vote_list)
            # df2.to_csv("FNR_avg.csv", index=False)
            df3 = pd.DataFrame(columns=range(num_instance))
            df3.to_csv("pred_labels.csv", index=False)
            for t in range(num_try):
                # print("User: ", user)
                subprocess.run(["python", "src/main_fewshot.py",
                                "--output_dir", "experiments",
                                "--num_fewshot", str(num_fewshot),
                                "--target_user", user,
                                "--comment", "classification from Scratch",
                                "--name", "VRS_fromScratch",
                                "--records_file", "VRS_records.xls",
                                "--data_dir", config["data_dir"] + "Vote_Privacy/input_feature/" + feature_name,
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

            print(df5)
            acc_avg = df5['Avg'].mean()
            print("Accuracy average:", acc_avg)

            avg_row = pd.Series(["Average"] + df5.mean(numeric_only=True).tolist(), index=df5.columns)
            df5 = df5.append(avg_row, ignore_index=True)
            df5.to_csv("acc_avg_"+feature_name+"_"+str(num_fewshot)+"shot.csv", index=False)

            # df4 = pd.read_csv("pred_labels.csv")


    df_final = pd.read_csv("acc_avg.csv")
    # Add a new row, user name is "Average", each value average
    final_row = pd.Series(["Average"] + df_final.mean(numeric_only=True).tolist(), index=df_final.columns)
    df_final = df_final.append(final_row, ignore_index=True)
    print("Accuracy average:", df_final['Avg'].mean())
    df_final.to_csv("acc_avg_"+feature_name+"_"+str(num_fewshot)+"shot.csv", index=False)