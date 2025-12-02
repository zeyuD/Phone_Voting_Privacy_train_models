import pandas as pd
import numpy as np

data_names = ["raw"]

for data_idx in range(len(data_names)):
    data_name = data_names[data_idx]
    # data_dir = "/mnt/e/VR_Skeleton/data/processed_" + data_name + "/"
    # target_user = "Aditya"
    # seg_data_1 = pd.read_csv(data_dir + target_user + '/' + '1.csv', header=None)
    print("Feature:", data_name)
    # print("Feature length:", len(seg_data_1.columns))

    # f1_data_file = "F1_avg_" + data_name + ".csv"
    # f1_data = pd.read_csv(f1_data_file, header=None)
    # f1_values = f1_data.iloc[1:]
    # for row in range(f1_values.shape[0]):
    #     f1_values.iloc[row] = pd.to_numeric(f1_data.iloc[row+1])
    # f1_avg = f1_values.mean().mean()
    # f1_max = f1_values.max().mean()

    # fnr_data_file = "FNR_avg_" + data_name + ".csv"
    # fnr_data = pd.read_csv(fnr_data_file, header=None)
    # fnr_values = fnr_data.iloc[1:]
    # for row in range(fnr_values.shape[0]):
    #     fnr_values.iloc[row] = pd.to_numeric(fnr_data.iloc[row+1])
    # fnr_avg = fnr_values.mean().mean()
    # fnr_min = fnr_values.min().mean()

    # print("F1 average:", f1_avg)
    # print("F1 max average:", f1_max)
    # print("F1 for each user:")
    # print(f1_values.mean(axis=0))
    # print("FNR average:", fnr_avg)
    # print("FNR min average:", fnr_min)

    acc_data = pd.read_csv("acc_avg_" + data_name + ".csv")
    print(acc_data)
    acc_avg = acc_data.mean().mean()
    print("Accuracy average:", acc_avg)