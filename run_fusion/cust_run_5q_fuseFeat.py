import subprocess
import os
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config

config = load_machine_config()
device = config["compdev"]

# print("User: ", user)
subprocess.run(["python", "run_fusion/main_5q_fuseFeat.py",
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
                  #     "opticalflowRAFT_edge_22_downsample_480p_s22",
                "--no_crop_all", # uncomment to skip crop for every feature, regardless of --no_crop_features
                "--fusion_type", "same_time", # 'same_time', 'late', 'all_time', "feature", "both", "gated"
                # Stage 2 of "train eye first, then freeze": load the eye-only model of the same try and keep its branch fixed
                "--pretrained_model", "results/model_all_processpos_norm_downsample_480p_op2_3q_fuseFeat_try1.pth",
                "--freeze_features", "all_processpos_norm_downsample_480p_op2",
                "--use_loss_fusion", "True",
                "--pass_band", "1", "10", # 1-10 for optiflowRAFT edge
                "--lambda_aux", "0.1",
                "--wandb_project", "phone_vote", # uncomment to log training to Weights & Biases
                "--data_class", "vrs",
                "--val_ratio", "0.1",
                # "--test_ratio", "0.4",
                "--epochs", "100", # Too long all B, default 40; <40 for selection frames, 55 for full frames
                "--lr", "0.0005", # 0.0005
                # "--lr_scheduler", "cosine", "--warmup_epochs", "5", # "step" (default), "cosine", "plateau"
                "--dropout", "0.2",
                "--batch_size", "8",
                "--d_model", "16",
                "--num_layers", "3",
                "--num_heads", "4",
                "--optimizer", "RAdam", # Adam all B
                "--pos_encoding", "fixed", # "learnable", "fixed"
                "--activation", "gelu", # "relu", "gelu"
                "--task", "classification",
                "--normalization", "none", # "none", "standardization", "minmax", "per_sample_std", "per_sample_minmax"
                "--key_metric", "accuracy"])
