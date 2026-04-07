import torch
import os
import cv2
import sys
import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.models.optical_flow import raft_large

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
        "users": ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", "Junwei", "Minjie", "minglei", "Mingxuan", "Rosie", "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", "Ziyue", "Ziyue1"],
        # "users": ["ZeyuZoom"],
        "device": "phone_s22/"
          }
}
suffix = "_downsample_480p_s22"
prefix = "opticalflowRAFT_border_22"


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


def flow_to_arrow_border(flow):
    # only use the frame line pixels to calculate the average flow
    flow = flow.cpu().detach()
    flow_u = flow[0, 0].numpy()  # horizontal flow
    flow_v = flow[0, 1].numpy()  # vertical flow
    h, w = flow.shape[2], flow.shape[3]

    outline_u = []
    outline_v = []

    # top row
    outline_u.append(flow_u[0, :])
    outline_v.append(flow_v[0, :])

    # bottom row
    outline_u.append(flow_u[-1, :])
    outline_v.append(flow_v[-1, :])

    # left and right columns, excluding corners to avoid double counting
    if h > 2:
        outline_u.append(flow_u[1:-1, 0])
        outline_v.append(flow_v[1:-1, 0])

        outline_u.append(flow_u[1:-1, -1])
        outline_v.append(flow_v[1:-1, -1])

    outline_u = np.concatenate([arr.ravel() for arr in outline_u])
    outline_v = np.concatenate([arr.ravel() for arr in outline_v])

    avg_u = np.mean(outline_u)
    avg_v = np.mean(outline_v)

    return avg_u, avg_v


def flow_to_arrow_edge(frame, flow):
    # Use Canny edge detection to find edges in the flow magnitude, then average flow on those edges
    flow = flow.cpu().detach()
    flow_u = flow[0, 0].numpy()  # horizontal flow
    flow_v = flow[0, 1].numpy()  # vertical flow
    edges = cv2.Canny(frame, 100, 200)
    edge_u = flow_u[edges > 0]
    edge_v = flow_v[edges > 0]
    avg_u = np.mean(edge_u)
    avg_v = np.mean(edge_v)
    return avg_u, avg_v


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

    # Create a dataframe to save the flow data
    # Each column is a flow vector # of flow u and v
    # Such as M*N*2 column for each frame, where M and N are the target grid size in flow_to_arrow function
    # No header, save as csv without index
    # num_row = M * N * 2
    num_row = 2
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

            last_batch = preprocess(last_batch).to(compdev)
            current_batch = preprocess(current_batch).to(compdev)
            predicted_flow = model(last_batch, current_batch)
            # Get the flow from the last iteration of the model
            final_predicted_flow = predicted_flow[-1]
            # Convert the flow to arrow on image
            # x, y, u, v = flow_to_arrow(final_predicted_flow, [M, N], threshold=1.0)
            u, v = flow_to_arrow_border(final_predicted_flow)
            # print(f"Processing frame {frame_count} with {len(x)} arrows")
            # Save the flow data into dataframe
            flow_data = [u, v]
            # flow_data = []
            # for i in range(len(x)):
            #     flow_data.append(u[i])
            # for i in range(len(x)):
            #     flow_data.append(v[i])

            df_save = pd.concat([df_save, pd.DataFrame([flow_data], columns=[f"flow_{i}" for i in range(num_row)])])

        else:
            break        

    # except Exception as e:
    #     print(f"Error processing file {file}: {e}")
    #     continue

    flow_data_interp = interpolate_multiD(np.array(df_save, dtype=np.float64), target_length=80)
    flow_data_norm = normalization(flow_data_interp, method='zscore')
    # Save the dataframe into csv
    pd.DataFrame(flow_data_norm).to_csv(save_path_folder + '/' + vote + '/' + save_file_name, index=False, header=False)

print(" ")
print("All files processed and saved.")