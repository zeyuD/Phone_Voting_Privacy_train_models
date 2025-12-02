import subprocess
import pandas as pd
import torch
import math
import os
import numpy as np
from torch import nn
from copy import deepcopy

from datasets.data import data_factory, Normalizer
from models.ts_transformer import TSTransformerEncoderClassiregressor, get_pos_encoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Our own dataset
user_list = ["Jingwei", "Long", "Ruxin", "Zeyu"]
user_list = user_list
data_names = ["ArmHand"]
# data_names = ["rawleftpos", "rawrightpos", "rawheadpos", "rawleftrot", "rawrightrot", "rawheadrot",
#               "interleft", "interright", "left_hr", "right_hr", 
#               "feature_all"]

# Create an empty dataframe, columns are user names
df = pd.DataFrame(columns=user_list)
df.to_csv("F1_test.csv", index=False)
df2 = pd.DataFrame(columns=user_list)
df2.to_csv("FNR_test.csv", index=False)

normalizer = Normalizer('standardization')
for data_name in data_names:
    # Read model
    for user in user_list:
        config = {
                    "output_dir": "experiments",
                    "comment": "classification from Scratch",
                    "name": "VRS_fromScratch",
                    "records_file": "VRS_records.xls",
                    "data_dir": "/mnt/e/Avatar_Attack/data/"+data_name+"/"+user,
                    "data_class": "vrs",
                    "val_ratio": "0.1",
                    # "test_ratio": "0.4",
                    "epochs": "10",
                    "lr": "0.001",
                    "optimizer": "RAdam",
                    "pos_encoding": "learnable",
                    "task": "classification",
                    "key_metric": "accuracy"
                    }
        # check_point = torch.load("/mnt/e/VR_Skeleton/data/processed_"+data_name+"/"+user+"/model_last.pth")
        # state_dict = deepcopy(check_point['state_dict'])
        # print(check_point['epoch'])

        # Create a directory to store the input
        if not os.path.exists(config['data_dir']+"/inptest"):
            os.makedirs(config['data_dir']+"/inptest")

        # Load model
        data_class = data_factory[config['data_class']]
        my_data = data_class('test', config['data_dir'], pattern=None, n_proc=1, limit_size=None, config=config)
        # print(my_data)
        test_indices = my_data.all_IDs
        my_data.feature_df.loc[test_indices] = normalizer.normalize(my_data.feature_df.loc[test_indices])
        feat_dim = my_data.feature_df.shape[1]
        max_seq_len = my_data.max_seq_len
        d_model = 64
        project_inp = nn.Linear(feat_dim, d_model)
        pos_enc = get_pos_encoder('learnable')(d_model, dropout=0.1*(1.0 - False), max_len=max_seq_len)
        
        print("Feature dimension:", feat_dim)
        print("Max sequence length:", max_seq_len)
        # model = TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, 64, 
        #                                             8, 
        #                                             3, 256, 
        #                                             num_classes=2,
        #                                             dropout=0.1, pos_encoding='learnable')

        per_PIN = {'targets': [], 'predictions': []}
        # len_seg = 8
    
        # First, some target user's data
        acm = 0
        for ins in range(0, 10):
            num_frag = my_data.num_frag_list[ins]
            for idx in range(acm+1, acm+num_frag):
                # print(idx)
                input_i = my_data.feature_df.loc[idx].values
                input_i = torch.tensor(input_i, dtype=torch.float32)
                # make input a batch of size 1
                input_i = input_i.unsqueeze(0)
                # print(idx, input_i.shape)
                inp = input_i.permute(1, 0, 2)
                inp = project_inp(inp) * math.sqrt(d_model)  # [seq_length, batch_size, d_model] project input vectors to d_model dimensional space
                # print(inp.shape)
                inp = pos_enc(inp)  # add positional encoding
                inp_1 = inp[:, 0, :]
                # Save the input as a csv file
                pd.DataFrame(inp_1.detach().numpy()).to_csv(config['data_dir']+"/inptest/"+str(idx)+".csv", index=False, header=False)
            acm += num_frag