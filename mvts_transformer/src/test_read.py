import pandas as pd
import numpy as np
import random

data_dir = "../data/processed/"
num_ins = 4

user_list = ["Aditya", "Bingzhe", "Boyang", "Bruce", "Cheng", "Dai", "Feifei", "Hanfei", "Huan", "Jianshu",
             "Jingwei", "Koosha", "Lalita", "Long", "Mike", "Paige", "Peiman", "Rui", "Ruxin", "Steven", "Taibiao",
             "Xiaobing", "Xiyan", "Xuhang", "Yehong", "Yifeng", "Yuting", "Yunpeng", "Zeyu", "Zhuoqun"]
target_user = "Aditya"

# Initialize data and labels
seg_data_1 = pd.read_csv(data_dir + target_user + '/' + '1.csv', header=None)
num_dimensions = len(seg_data_1.columns)
header_list = []
for dim in range(0, num_dimensions):
    header_list.append('dim_' + str(dim))
data = pd.DataFrame(dtype=np.float32, columns=header_list)
labels = pd.DataFrame([0 for i in range(2*num_ins)],dtype=np.int32)

# Add target user data
print("Target user: ", target_user)
for idx in range(num_ins):
    seg_data_i = pd.read_csv(data_dir + target_user + '/' + str(idx+1) + '.csv', header=None)
    data_i = pd.DataFrame(dtype=np.float32, columns=header_list)
    for dim in range(0, num_dimensions):
        data_i['dim_' + str(dim)] = seg_data_i[dim]
    data_i.index = [idx] * len(data_i)
    data = pd.concat([data, data_i])
    labels.iloc[idx] = 0

# Create a list of other users
oU = list(user_list)
oU.remove(target_user)
rdminx = random.sample(range(len(oU)), int(num_ins/2))
otherUsers = [oU[i] for i in rdminx]
# Add other users data
print("Other users: ", otherUsers)
other_i = num_ins
for other_user in otherUsers:
    other_idx = random.sample(range(num_ins), 2)
    for idx in other_idx:
        seg_data_i = pd.read_csv(data_dir + other_user + '/' + str(idx+1) + '.csv', header=None)
        data_i = pd.DataFrame(dtype=np.float32, columns=header_list)
        for dim in range(0, num_dimensions):
            data_i['dim_' + str(dim)] = seg_data_i[dim]
        data_i.index = [other_i] * len(data_i)
        data = pd.concat([data, data_i])
        labels.iloc[other_i] = 1
        other_i += 1

print(data)
print(labels)


classes = labels[0].astype("category")
print(classes)
class_names = classes.cat.categories
print(class_names)