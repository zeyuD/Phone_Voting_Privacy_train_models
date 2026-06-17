import torch
import os
import cv2
import sys
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))
from load_machine_config import load_machine_config
from normalization import normalization
from interpolate_multiD import interpolate_multiD

config = load_machine_config()

data_directory = config["data_dir"] + "Phone_Privacy/data/"
save_directory = config["data_dir"] + "Phone_Privacy/input_feature/"

# If you can, run this example on a GPU, it will be a lot faster.
compdev = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
print("Using Device:", compdev)

model = raft_large(pretrained=True, progress=False).to(compdev)
model = model.eval()


votes = ["A", "B", "C", "D", "E"]
M = 2
N = 2

idx = 10

# Create a dictionary of setups, each will have different setup and users
setups = {
    "phone_s22": {
        # "users": ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", "Junwei", "Minjie", "minglei", "Mingxuan", "Rosie", "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", "Ziyue", "Ziyue1"],
        "users": ["JingweiObj", "ZeyuObj"],
        "device": "phone_s22/"
          }
}
suffix = "_downsample_480p_s22"
prefix = "objYOLO"

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
    h, w = flow.shape[2], flow.shape[3]
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


def frame_to_pos_YOLO(frame):
    # Use a pre-trained YOLO model to detect objects in the frame, then average flow on those objects
    model = YOLO("yolo26m.pt")
    results = model(frame, verbose=False)
    
    # Get the x and y coordinate of the center of the detected objects if is cup
    found_cup = False
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "cup":
                x_center = int((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
                y_center = int((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
                x_obj = x_center
                y_obj = y_center
                found_cup = True
                break
        else:
            continue
        break
    if not found_cup:
        print("No cup detected in the frame.")
        x_obj = np.NaN
        y_obj = np.NaN
    return x_obj, y_obj


# Find all file directories
video_files = []
for vote in votes:
    for setup_name, setup_info in setups.items():
        users = setup_info["users"]
        device = setup_info["device"]
        file_path_dir = data_directory + device + vote + '/downsample_480p/'
        for user in users:
            # All files that match the pattern
            file_list_i = [f for f in os.listdir(file_path_dir) if f.startswith(user + '_' + vote) and f.endswith('_downsample_480p.mp4')]
            # print(f"Found {len(file_list_i)} files for user {user} in vote {vote} for setup {setup_name}.")
            video_files.extend([os.path.join(file_path_dir, f) for f in file_list_i])

# Usage
save_path_folder = save_directory + prefix + suffix
if not os.path.exists(save_path_folder):
    os.makedirs(save_path_folder)
for vote in votes:
    if not os.path.exists(save_path_folder + '/' + vote):
        os.makedirs(save_path_folder + '/' + vote)
for file in video_files:
    # Print progress by n/N
    n = video_files.index(file) + 1
    num_vid = len(video_files)
    print(f"Processing file {n}/{num_vid}: {file}")

    vote = file.split('/')[-3]  # Extract vote from the file path
    # print(f"Processing file: {file}")

    # save as user + '_' + vote + '_' + idx + '.csv'
    # that is the name remove file_path_dir and replace "_downsample_480p.mp4" with ".csv"
    file_name = os.path.basename(file).replace('file_path_dir', '')
    save_file_name = os.path.basename(file_name).replace('_downsample_480p.mp4', '.csv')
    # print(f"Saving file: {save_path_folder + '/' + vote + '/' + save_file_name}")

    # try:
    cap = cv2.VideoCapture(file)
    frame_count = 0
    last_frame = None
    current_frame = None

    # Create a dataframe to save the data
    num_row = 2
    df_save = pd.DataFrame(columns=[f"pos_{i}" for i in range(num_row)])

    # Get optical flow for all frames in the video
    while True:
        ret, frame = cap.read()
        if ret:
            # Convert the frame from BGR to RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results_pos = frame_to_pos_YOLO(frame_rgb)
            # Skip if NaN (no cup detected)
            if np.isnan(results_pos).any():
                print("Skipping frame due to no cup detected.")
                continue
            df_save = pd.concat([df_save, pd.DataFrame([results_pos], columns=[f"pos_{i}" for i in range(num_row)])])

        else:
            break        

    # except Exception as e:
    #     print(f"Error processing file {file}: {e}")
    #     continue

    # print("Output raw data:", df_save)

    flow_data_interp = interpolate_multiD(np.array(df_save, dtype=np.float64), target_length=80)
    flow_data_norm = normalization(flow_data_interp, method='zscore')
    # Save the dataframe into csv
    pd.DataFrame(flow_data_norm).to_csv(save_path_folder + '/' + vote + '/' + save_file_name, index=False, header=False)

print(" ")
print("All files processed and saved.")