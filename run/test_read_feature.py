import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config
from functions.detect_two_taps import detect_two_taps
from variables_combineFeat import user_list, vote_list, feature_names

config = load_machine_config()


data_dir = config["data_dir"] + "Phone_Privacy/input_feature/"
feature_name = feature_names[0]

seg_data_f_1 = pd.read_csv(data_dir + feature_name + '/' + vote_list[0] + '/' + user_list[1] + '_' + vote_list[0] + '_2.csv', header=None)

plot_data, centers = detect_two_taps(seg_data_f_1)

# Plot every dimension of the feature
plt.figure(figsize=(12, 8))
for i in range(plot_data.shape[1]):
    plt.plot(plot_data.iloc[:, i], label=f'Feature {i}')
# Add lines for the detected centers
for center in centers:
    plt.axvline(x=center, color='red', linestyle='--')
# plt.title(f'{feature_name} for {user_list[0]}_{vote_list[0]}_1')
plt.xlabel('Time Steps')
plt.ylabel('Feature Values')
# plt.legend()
# plt.show()
plt.savefig('test_read_feature.png')