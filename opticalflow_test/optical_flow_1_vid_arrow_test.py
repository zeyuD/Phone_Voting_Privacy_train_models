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
            T.Resize(size=(360, 640)),
        ]
    )
    batch = transforms(batch)
    return batch


def flow_to_arrow(flow, step=16, threshold=1.0):
    # print(f"Flow shape: {flow.shape}") # [1, 2, 360, 640]
    # Calculate average flow in each step x step block
    flow = flow.cpu().detach()
    flow_u = flow[0, 0].numpy()  # horizontal flow
    flow_v = flow[0, 1].numpy()  # vertical flow
    h, w = flow_u.shape
    x, y, u, v = [], [], [], []
    for i in range(0, h, step):
        for j in range(0, w, step):
            block_u = flow_u[i:i+step, j:j+step]
            block_v = flow_v[i:i+step, j:j+step]
            avg_u = np.mean(block_u)
            avg_v = np.mean(block_v)
            # if np.sqrt(avg_u**2 + avg_v**2) > threshold:
            x.append(j + step // 2)
            y.append(i + step // 2)
            u.append(avg_u)
            v.append(avg_v)
    return x, y, u, v


def arrow_on_image(image, x, y, u, v, color=(255, 0, 0), thickness=2):
    image = image.copy()
    scale = 10  # scale factor for better visualization
    for (x_i, y_i, u_i, v_i) in zip(x, y, u, v):
        end_point = (int(x_i + u_i * scale), int(y_i + v_i * scale))
        cv2.arrowedLine(image, (x_i, y_i), end_point, color, thickness)
    return image 


proj_path = os.path.join(config["data_dir"], "Phone_Privacy")
video_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + "AAA_test.mp4"
save_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + "AAA_test_optical_flow.mp4"

# save optical flow video
vid = cv2.VideoWriter(save_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 360))

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
        print(f"Frame {frame_count}: Final predicted flow shape: {final_predicted_flow.shape}, dtype: {final_predicted_flow.dtype}")
        
        # Convert the flow to arrow on image
        x, y, u, v = flow_to_arrow(final_predicted_flow, step=16, threshold=1.0)
        # print("X shape:", len(x))
        # print(f"Frame {frame_count}: Detected {len(x)} arrows")
        # print(f"Sample arrows (x, y, u, v): {list(zip(x[:5], y[:5], u[:5], v[:5]))}")
        arrow_image = arrow_on_image(cv2.resize(frame, (640, 360)), x, y, u, v, color=(255, 0, 0), thickness=2)
        # print(f"Frame {frame_count}: Arrow image shape: {arrow_image.shape}, dtype: {arrow_image.dtype}")

        # plot(F.to_tensor(arrow_image))
        # plt.show()

        save_frame = arrow_image
        vid.write(save_frame)
    else:
        break

vid.release()
# plt.show()