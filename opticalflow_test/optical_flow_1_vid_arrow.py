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

config = load_machine_config()

# If you can, run this example on a GPU, it will be a lot faster.
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model = raft_large(pretrained=True, progress=False).to(device)
model = model.eval()


votes = ["A", "B", "C", "D", "E"]

# Create a dictionary of setups, each will have different setup and users
setups = {
    "phone_s22": {
        # "users": ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", "Junwei", "Minjie", "minglei", "Mingxuan", "Rosie", "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", "Ziyue", "Ziyue1"],
        "users": ["ZeyuZoom"],
        "save_dir": "phone_s22/"
          }
}

plt.rcParams["savefig.bbox"] = "tight"
# sphinx_gallery_thumbnail_number = 2


def plot(imgs, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


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


def arrow_on_image(image, x, y, u, v, color=(255, 0, 0), thickness=2):
    image = image.copy()
    scale = 5  # scale factor for better visualization
    for (x_i, y_i, u_i, v_i) in zip(x, y, u, v):
        end_point = (int(x_i + u_i * scale), int(y_i + v_i * scale))
        cv2.arrowedLine(image, (x_i, y_i), end_point, color, thickness)
    return image 


proj_path = os.path.join(config["data_dir"], "Phone_Privacy")
video_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + setups["phone_s22"]["users"][0] + "_" + votes[0] +"_1_downsample_480p.mp4"
save_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + setups["phone_s22"]["users"][0] + "_" + votes[0] +"_1_optical_flow_arrow_480p.mp4"
# save_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + setups["phone_s22"]["users"][0] + "_" + votes[0] +"_1_optical_flow_arrow_gaussian_480p.mp4"

# save optical flow video
vid = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (480, 848))

cap = cv2.VideoCapture(video_file)
frame_count = 0
last_frame = None
current_frame = None

# all_frames, _, _ = read_video(str(video_file))
# all_frames = all_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

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
            last_frame = current_frame
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
        # Smooth the flow for better visualization
        smoothed_flow = T.GaussianBlur(kernel_size=(5, 5), sigma=(1.0, 1.0))(final_predicted_flow)
 
        # Convert the flow to arrow on image
        x, y, u, v = flow_to_arrow(final_predicted_flow, [2, 2], threshold=1.0)
        arrow_image = arrow_on_image(cv2.resize(frame, (480, 848)), x, y, u, v, color=(255, 0, 0), thickness=2)
        save_frame = arrow_image
        vid.write(save_frame)
    else:
        break

vid.release()
# plt.show()