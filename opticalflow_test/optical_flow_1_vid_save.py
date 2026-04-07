import torch
import tempfile
import os
import time
import threading
import cv2
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))
from load_machine_config import load_machine_config
from normalization import normalization
from interpolate_multiD import interpolate_multiD

config = load_machine_config()

# If you can, run this example on a GPU, it will be a lot faster.
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", device)

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()


votes = ["A", "B", "C", "D", "E"]
M = 2
N = 2

idx = 10

# Create a dictionary of setups, each will have different setup and users
setups = {
    "phone_s22": {
        # "users": ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", "Junwei", "Minjie", "minglei", "Mingxuan", "Rosie", "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", "Ziyue", "Ziyue1"],
        "users": ["Zeyu"],
        # "users": ["ZeyuZoom"],
        "save_dir": "phone_s22/"
          }
}


def preprocess(batch):
    transforms = T.Compose(
        [
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=0.5, std=0.5),  # map [0, 1] into [-1, 1]
            T.Resize(size=(848, 480)),
        ]
    )
    batch = transforms(batch)
    return batch


def flow_to_arrow(flow, target_grid=[2, 2], threshold=1.0):
    # print(f"Flow shape: {flow.shape}") # [1, 2, 848, 480]
    # Calculate average flow in each step x step block
    flow = flow.cpu().detach()
    flow_u = flow[0, 0].numpy()  # horizontal flow
    flow_v = flow[0, 1].numpy()  # vertical flow
    h, w = flow_u.shape
    x, y, u, v = [], [], [], []
    # Average the flow in each target_grid size
    # E.g., [2, 2] means average flow into 2 row and 2 column blocks
    step_h = h // target_grid[0]
    step_w = w // target_grid[1]
    for i in range(0, h, step_h):
        for j in range(0, w, step_w):
            block_u = flow_u[i:i+step_h, j:j+step_w]
            block_v = flow_v[i:i+step_h, j:j+step_w]
            avg_u = np.mean(block_u)
            avg_v = np.mean(block_v)
            # if np.sqrt(avg_u**2 + avg_v**2) > threshold:
            x.append(j + step_w // 2)
            y.append(i + step_h // 2)
            u.append(avg_u)
            v.append(avg_v)
    return x, y, u, v



proj_path = os.path.join(config["data_dir"], "Phone_Privacy")
video_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + setups["phone_s22"]["users"][0] + "_" + votes[0] + "_" + str(idx) + "_downsample_480p.mp4"
# save_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + setups["phone_s22"]["users"][0] + "_" + votes[0] +"_1_optical_flow_arrow_gaussian_480p.mp4"
save_csv = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + setups["phone_s22"]["users"][0] + "_" + votes[0] + "_" + str(idx) + "_optical_flow_arrow_480p.csv"

cap = cv2.VideoCapture(video_file)
frame_count = 0
last_frame = None
current_frame = None

# Create a dataframe to save the flow data
# Each column is a flow vector # of flow u and v
# Such as M*N*2 column for each frame, where M and N are the target grid size in flow_to_arrow function
# No header, save as csv without index
num_row = M * N * 2
df_save = pd.DataFrame(columns=[f"flow_{i}" for i in range(num_row)])

# Get optical flow for all frames in the video
while True:
    ret, frame = cap.read()
    if ret:
        # Convert the frame from BGR to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = F.to_tensor(frame_rgb)
        # Add batch dimension
        if frame_count == 0:
            last_frame = frame_tensor
            current_frame = frame_tensor
            frame_count += 1
            continue
        else:
            # last_frame = current_frame
            current_frame = frame_tensor
            frame_count += 1
        
        last_batch = torch.stack([last_frame])
        current_batch = torch.stack([current_frame])
        # print(f"Processing frame {frame_count} with shape {current_batch.shape}")
        # plot(last_batch)
        # plt.show()

        last_batch = preprocess(last_batch).to(device)
        current_batch = preprocess(current_batch).to(device)
        predicted_flow = model(last_batch, current_batch)
        # Get the flow from the last iteration of the model
        final_predicted_flow = predicted_flow[-1]
        # Convert the flow to arrow on image
        x, y, u, v = flow_to_arrow(final_predicted_flow, [M, N], threshold=1.0)
        print(f"Processing frame {frame_count} with {len(x)} arrows")
        # Save the flow data into dataframe
        flow_data = []
        for i in range(len(x)):
            flow_data.append(u[i])
        for i in range(len(x)):
            flow_data.append(v[i])

        df_save = pd.concat([df_save, pd.DataFrame([flow_data], columns=[f"flow_{i}" for i in range(num_row)])])

    else:
        break

flow_data_interp = interpolate_multiD(np.array(df_save, dtype=np.float64), target_length=80)
flow_data_norm = normalization(flow_data_interp, method='zscore')
# Save the dataframe into csv
pd.DataFrame(flow_data_norm).to_csv(save_csv, index=False, header=False)