import subprocess
import os
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config

config = load_machine_config()
device = config["compdev"]


# print("User: ", user)
subprocess.run(["python", "run/main_5q_combineFeat.py",
                "--output_dir", "experiments",
                "--comment", "classification from Scratch",
                "--name", "VRS_fromScratch",
                "--records_file", "VRS_records.xls",
                "--data_dir", config["data_dir"] + "Phone_Privacy/input_feature/",
                "--device", config["compdev"],
                "--data_class", "vrs",
                "--val_ratio", "0.1",
                # "--test_ratio", "0.4",
                "--epochs", "40", # Too long all B, default 40; <40 for selection frames, 55 for full frames
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
