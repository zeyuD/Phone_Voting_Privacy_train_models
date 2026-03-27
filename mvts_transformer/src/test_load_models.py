import torch
import os
import time
import sys
import pandas as pd
import numpy as np
from copy import deepcopy
from datasets.data import data_factory, Normalizer
from models.ts_transformer import TSTransformerEncoderClassiregressor

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()
device = config["compdev"]

start_time = time.time()

data_dir = config["data_dir"] + "Phone_Privacy/input_feature/"
gaze_feat = "all_processpos_norm_downsample_480p_s22"
opticalflow_feat = "opticalflow_22_downsample_480p_s22"

# Load model
data_class = data_factory['vrs']
normalizer = Normalizer('standardization')

gaze_data = data_class('test', data_dir+gaze_feat, pattern=None, n_proc=1, limit_size=None, config=None)
test_indices = gaze_data.all_IDs
print(test_indices)
gaze_data.feature_df.loc[test_indices] = normalizer.normalize(gaze_data.feature_df.loc[test_indices])
gaze_data_feat_dim = gaze_data.feature_df.shape[1]
gaze_data_max_seq_len = gaze_data.max_seq_len

optiflow_data = data_class('test', data_dir+opticalflow_feat, pattern=None, n_proc=1, limit_size=None, config=None)
test_indices = optiflow_data.all_IDs
print(test_indices)
optiflow_data.feature_df.loc[test_indices] = normalizer.normalize(optiflow_data.feature_df.loc[test_indices])
optiflow_data_feat_dim = optiflow_data.feature_df.shape[1]
optiflow_data_max_seq_len = optiflow_data.max_seq_len

gaze_feat_check_point = torch.load(data_dir + gaze_feat + "/" + "model_best.pth")
gaze_feat_state_dict = deepcopy(gaze_feat_check_point['state_dict'])
gaze_feat_model = TSTransformerEncoderClassiregressor(gaze_data_feat_dim, gaze_data_max_seq_len, 64, 
                                                      8, 
                                                      3, 256, 
                                                      num_classes=5,
                                                      dropout=0.1, pos_encoding='learnable')
gaze_feat_model.load_state_dict(gaze_feat_state_dict, strict=False)

opticalflow_check_point = torch.load(data_dir + opticalflow_feat + "/" + "model_best.pth")
opticalflow_state_dict = deepcopy(opticalflow_check_point['state_dict'])
opticalflow_model = TSTransformerEncoderClassiregressor(optiflow_data_feat_dim, optiflow_data_max_seq_len, 64, 
                                                        8,
                                                        3, 256, 
                                                        num_classes=5,
                                                        dropout=0.1, pos_encoding='learnable')
opticalflow_model.load_state_dict(opticalflow_state_dict, strict=False)