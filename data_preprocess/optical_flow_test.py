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


proj_path = os.path.join(config["data_dir"], "Phone_Privacy")
video_file = proj_path + "/data/" + setups["phone_s22"]["save_dir"] + votes[0] + "/downsample_480p/" + setups["phone_s22"]["users"][0] + "_" + votes[0] +"_1_downsample_480p.mp4"

cap = cv2.VideoCapture(video_file)
frame_count = 0
last_frame = None
current_frame = None

all_frames, _, _ = read_video(str(video_file))
all_frames = all_frames.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

print(f"Total shape of the video: {all_frames.shape}")

img1_batch = torch.stack([all_frames[20], all_frames[50]])
img2_batch = torch.stack([all_frames[21], all_frames[51]])

# plot(img1_batch)

# # Check if the video was opened successfully
# if not cap.isOpened():
#     print("Error opening video file")
# else:
#     # Read the first frame of the video
#     ret, frame = cap.read()
#     if ret:
#         # Convert the frame from BGR to RGB format
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_tensor = F.to_tensor(frame_rgb)

#         if frame_count == 0:
#             last_frame = frame_tensor
#             current_frame = frame_tensor
#         else:
#             last_frame = current_frame
#             current_frame = frame_tensor
#             img1_batch = torch.stack([last_frame, current_frame])
#             # img2_batch = torch.stack([frame_tensor[101], frame_tensor[151]])

#             plot(img1_batch)

#     else:
#         print("Error reading the first frame of the video")


img1_batch = preprocess(img1_batch).to(device)
img2_batch = preprocess(img2_batch).to(device)

print(f"shape = {img1_batch.shape}, dtype = {img1_batch.dtype}")

list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
print(f"type = {type(list_of_flows)}")
print(f"length = {len(list_of_flows)} = number of iterations of the model")
print(f"shape of each flow = {list_of_flows[0].shape} = (N, 2, H, W) where N is the batch size, H and W are the height and width of the input images")

predicted_flows = list_of_flows[-1]
print(f"dtype = {predicted_flows.dtype}")
print(f"shape = {predicted_flows.shape} = (N, 2, H, W)")
print(f"min = {predicted_flows.min()}, max = {predicted_flows.max()}")

flow_imgs = flow_to_image(predicted_flows)
# The images have been mapped into [-1, 1] but for plotting we want them in [0, 1]
img1_batch = [(img1 + 1) / 2 for img1 in img1_batch]

grid = [[img1, flow_img] for (img1, flow_img) in zip(img1_batch, flow_imgs)]
plot(grid)


plt.show()