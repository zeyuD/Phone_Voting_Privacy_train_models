import subprocess
import pandas as pd
import torch
import numpy as np
from copy import deepcopy

from datasets.data_test import data_factory, Normalizer
from models.ts_transformer import TSTransformerEncoderClassiregressor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Our own dataset
user_list = ["Aditya", "Bingzhe", "Boyang", "Bruce", "Cheng", "Dai", "Feifei", "Hanfei", "Huan", "Jianshu",
             "Jingwei", "Koosha", "Lalita", "Long", "Mike", "Paige", "Peiman", "Rui", "Ruxin", "Steven", "Taibiao",
             "Xiaobing", "Xiyan", "Xuhang", "Yehong", "Yifeng", "Yuting", "Yunpeng", "Zeyu", "Zhuoqun"]
user8_list = ["Aditya8", "Bingzhe8", "Boyang8", "Bruce8", "Cheng8", "Dai8", "Feifei8", "Hanfei8", "Huan8", "Jianshu8",
             "Jingwei8", "Koosha8", "Lalita8", "Long8", "Mike8", "Paige8", "Peiman8", "Rui8", "Ruxin8", "Steven8", "Taibiao8",
             "Xiaobing8", "Xiyan8", "Xuhang8", "Yehong8", "Yifeng8", "Yuting8", "Yunpeng8", "Zeyu8", "Zhuoqun8"]
user_test_list = ["Jingwei8", "Zeyu8"]
user_list = user_test_list
data_names = ["feature_all"]
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
                    "data_dir": "/mnt/e/VR_Skeleton/data/processed_"+data_name+"/"+user,
                    "data_class": "vrs",
                    "data_window_len": "None",
                    "val_ratio": "0.25",
                    "test_ratio": "0.4",
                    "epochs": "20",
                    "lr": "0.001",
                    "optimizer": "RAdam",
                    "pos_encoding": "learnable",
                    "task": "classification",
                    "key_metric": "accuracy"
                    }
        check_point = torch.load("/mnt/e/VR_Skeleton/data/processed_"+data_name+"/"+user+"/model_last.pth")
        state_dict = deepcopy(check_point['state_dict'])
        print(check_point['epoch'])

        # Load model
        data_class = data_factory[config['data_class']]
        my_data = data_class('test', config['data_dir'], pattern=None, n_proc=1, limit_size=None, config=None)
        # print(my_data)
        test_indices = my_data.all_IDs
        my_data.feature_df.loc[test_indices] = normalizer.normalize(my_data.feature_df.loc[test_indices])
        feat_dim = my_data.feature_df.shape[1]
        max_seq_len = my_data.max_seq_len
        print("Feature dimension:", feat_dim)
        print("Max sequence length:", max_seq_len)
        model = TSTransformerEncoderClassiregressor(feat_dim, max_seq_len, 64, 
                                                    8, 
                                                    3, 256, 
                                                    num_classes=2,
                                                    dropout=0.1, pos_encoding='learnable')
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model = model.eval()

        per_PIN = {'targets': [], 'predictions': []}
        len_seg = 8
    
        # First, some target user's data
        for pin in range(0, 18):
            # Read 8 data and make a batch of size 8 (len_seg)
            input = my_data.feature_df.loc[pin*len_seg].values
            label = my_data.labels_df.iloc[pin*len_seg].values
            if label == 0:
                target = [1, 0]
            else:
                target = [0, 1]
            input = torch.tensor(input, dtype=torch.float32)
            # make input a batch of size 1
            input = input.unsqueeze(0)
            for idx in range(pin*len_seg+1, pin*len_seg+len_seg):
                # print(idx)
                input_i = my_data.feature_df.loc[idx].values
                input_i = torch.tensor(input_i, dtype=torch.float32)
                # make input a batch of size 1
                input_i = input_i.unsqueeze(0)
                # print(idx, input_i.shape)
                input = torch.cat((input, input_i), dim=0)
            # print(input)
            padding_masks = torch.ones((len_seg, max_seq_len), dtype=torch.bool)
            # print(padding_masks)
            # print(padding_masks.shape)
            predictions = model(input.to(device), padding_masks.to(device))
            # print(predictions)
            # Make decision based on sum of predictions
            predictions = torch.sum(predictions, dim=0)
            targets = torch.tensor(target, dtype=torch.float32)
            # append predictions and targets
            per_PIN['predictions'].append(predictions.detach().cpu().numpy())
            per_PIN['targets'].append(targets.detach().cpu().numpy())
        
        # Stack each PIN's predictions and targets
        predictions = torch.from_numpy(np.stack(per_PIN['predictions']))
        targets = torch.from_numpy(np.stack(per_PIN['targets']))
        probs_p = torch.nn.functional.softmax(predictions, dim=1)
        probs_t = torch.nn.functional.softmax(targets, dim=1)
        predictions = torch.argmax(probs_p, dim=1).cpu().numpy()
        targets = torch.argmax(probs_t, dim=1).cpu().numpy()
        print(predictions)
        print(targets)
        TP, FP, TN, FN = 0, 0, 0, 0
        for true, pred in zip(targets, predictions):
            if true == 0 and pred == 0:
                TP += 1
            elif true == 1 and pred == 0:
                FP += 1
            elif true == 0 and pred == 1:
                FN += 1
            elif true == 1 and pred == 1:
                TN += 1
        # Calculating FPR
        if FP + TN == 0:  # handle edge case to avoid division by zero
            fpr = 0
        else:
            fpr = FP / (FP + TN)
        fnr = FN / (FN + TP)
        f1 = 2 * TP / (2 * TP + FP + FN)

        # Save test results
        df = pd.read_csv('F1_test.csv')
        target_user = config['data_dir'].rsplit('/', 1)[1]
        # print(df[target_user])
        df.loc[df.shape[0], target_user] = f1
        df.to_csv('F1_test.csv', index=False)
        
        df = pd.read_csv('FNR_test.csv')
        target_user = config['data_dir'].rsplit('/', 1)[1]
        # print(df[target_user])
        df.loc[df.shape[0], target_user] = fnr
        df.to_csv('FNR_test.csv', index=False)


    # Remove NaN values and move next row up
    df = pd.read_csv("F1_test.csv")
    df1 = df.apply(lambda x: pd.Series(x.dropna().values))
    df1.to_csv("F1_test_"+data_name+".csv", index=False)
    df_ = pd.read_csv("FNR_test.csv")
    df2 = df_.apply(lambda x: pd.Series(x.dropna().values))
    df2.to_csv("FNR_test_"+data_name+".csv", index=False)