# Remove data files exist in one folder but not in others
import os
import pandas as pd
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

work_directory = config["data_dir"] + "Phone_Privacy/input_feature/"

votes = ["A", "B", "C", "D", "E"]
suffix = "_downsample_480p_s22"
feature_1 = "all_processpos_norm"
feature_2 = "opticalflow_22"
exclude_keywords = ["Zoom"]
# feature_3 = "opticalflow_44"
# feature_4 = "opticalflow_84"
# feature_5 = "opticalflowRAFT_22"
# feature_6 = "opticalflowRAFT_44"
# feature_7 = "opticalflowRAFT_84"
# features = [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]
features = [feature_1, feature_2]

for vote in range(len(votes)):
    vote_name = votes[vote]

    feature_1_files = os.listdir(work_directory + feature_1 + suffix +"/" + vote_name + "/")
    feature_1_files_exclude = [f for f in feature_1_files if not any(keyword in f for keyword in exclude_keywords)]
    feature_2_files = os.listdir(work_directory + feature_2 + suffix +"/" + vote_name + "/")
    feature_2_files_exclude = [f for f in feature_2_files if not any(keyword in f for keyword in exclude_keywords)]
    # feature_3_files = os.listdir(work_directory + feature_3 + suffix + "/" + vote_name + "/")
    # feature_4_files = os.listdir(work_directory + feature_4 + suffix + "/" + vote_name + "/")
    # feature_5_files = os.listdir(work_directory + feature_5 + suffix + "/" + vote_name + "/")
    # feature_6_files = os.listdir(work_directory + feature_6 + suffix + "/" + vote_name + "/")
    # feature_7_files = os.listdir(work_directory + feature_7 + suffix + "/" + vote_name + "/")
    # feature_files = [feature_1_files, feature_2_files, feature_3_files, feature_4_files, feature_5_files, feature_6_files, feature_7_files]
    feature_files = [feature_1_files_exclude, feature_2_files_exclude]

    # Find common files in all features
    # common_files = set(feature_1_files) & set(feature_2_files) & set(feature_3_files) & set(feature_4_files) & set(feature_5_files) & set(feature_6_files) & set(feature_7_files)
    common_files = set(feature_1_files_exclude) & set(feature_2_files_exclude)

    # Remove files that are not in the common files list
    for feat in range(len(features)):
        feature_file = feature_files[feat]
        for f in feature_file:
            if f not in common_files:
                print("Removing file:", work_directory + features[feat] + suffix + "/" + vote_name + "/" + f)
                # os.remove(work_directory + features[feat] + suffix + "/" + vote_name + "/" + f)