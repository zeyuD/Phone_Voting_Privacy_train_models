import os
import cv2
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))
from load_machine_config import load_machine_config
from get_ego_motion import track_camera_motion
from interpolate_multiD import interpolate_multiD
from normalization import normalization

config = load_machine_config()


data_directory = config["data_dir"] + "Phone_Privacy/data/"
save_directory = config["data_dir"] + "Phone_Privacy/input_feature/"

votes = ["A", "B", "C", "D", "E"]

# Create a dictionary of setups, each will have different setup and users
setups = {
    # "phone_s22": {
    #     # "users": ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", "Junwei", "Minjie", "minglei", "Mingxuan", "Rosie", "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", "Ziyue", "Ziyue1"],
    #     "users": ["JingweiObj", "ZeyuObj"],
    #     "device": "phone_s22/"
    #       },
    "pad_op2": {
        "users": ["JingweiPad", "ZeyuPad"],
        "device": "pad_op2/"
          }
}
suffix = "_downsample_480p_s22"
prefix = "egomotion_rot"

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
    N = len(video_files)
    print(f"Processing file {n}/{N}: {file}")

    vote = file.split('/')[-3]  # Extract vote from the file path
    # print(f"Processing file: {file}")

    # save as user + '_' + vote + '_' + idx + '.csv'
    # that is the name remove file_path_dir and replace "_downsample_480p.mp4" with ".csv"
    file_name = os.path.basename(file).replace('file_path_dir', '')
    save_file_name = os.path.basename(file_name).replace('_downsample_480p.mp4', '.csv')
    # print(f"Saving file: {save_path_folder + '/' + vote + '/' + save_file_name}")

    try:
        motion_data = track_camera_motion(file)
    except Exception as e:
        print(f"Error processing file {file}: {e}")
        continue

    # Rotation data
    rotations = np.array([data['rotation'] for data in motion_data])
    euler_angles = np.array([cv2.Rodrigues(rot)[0].flatten() for rot in rotations])
    # print("Rotation data shape:", euler_angles.shape)

    euler_angles_interp = interpolate_multiD(euler_angles, target_length=80)
    # print("Interpolated rotation data shape:", euler_angles_interp.shape)
    euler_angles_norm = normalization(euler_angles_interp, method='zscore')

    # Save to CSV no header
    pd.DataFrame(euler_angles_norm).to_csv(save_path_folder + '/' + vote + '/' + save_file_name, index=False, header=False)

print(" ")
print("All files processed and saved.")