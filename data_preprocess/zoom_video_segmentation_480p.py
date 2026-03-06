import cv2
import yaml
import os
import subprocess
import sys
import pandas as pd

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

work_directory = config["data_dir"] + "Phone_Privacy/"

data_folder = work_directory + "data/phone_s22/"

# user = "JingweiZoom"
# offset = 7.7 # for JingweiZoom
# end_offset = 0.2 # for JingweiZoom

# user = "mingleiZoom"
# offset = 5.0 # for mingleiZoom
# end_offset = 0.2 # for mingleiZoom

# user = "MinjieZoom"
# offset = 41.0 # for MinjieZoom
# end_offset = 0.4 # for MinjieZoom

# user = "WeiZoom"
# offset = 7.0 # for WeiZoom
# end_offset = 0.4 # for WeiZoom

user = "WenZoom"
offset = 7.4 # for WenZoom
end_offset = 0.4 # for WenZoom

# user = "ZeyuZoom"
# offset = 3.9 # for ZeyuZoom
# end_offset = 0.4 # for ZeyuZoom

label_file = data_folder + user + "/" + user + "_correct_options.csv"
zoom_video_file = data_folder + user + "/" + user + ".mov"

cap = cv2.VideoCapture(zoom_video_file)
# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get original video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Original video properties: width={frame_width}, height={frame_height}, fps={fps}")

# Set the new resolution
new_width, new_height = 480, 848
phone_width, phone_height = 1080, 1920

# Define the codec and create VideoWriter object
ext = 'mp4'  # or 'mp4'
if ext == 'mp4':
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
elif ext == 'avi':
    fourcc = cv2.VideoWriter_fourcc(*'PIM1')

label_data = pd.read_csv(label_file)
timestamps = label_data["timestamp"]
labels = label_data["label"].tolist()  # skip the first row which is the starting frame with label 3
timestamps = timestamps - timestamps[0]  # make the first time step as 0
votes = label_data["vote"].tolist()


# If label is 2, add end_offset to the timestamp to get the ending frame index
for i in range(len(labels)):
    if labels[i] == 2:
        timestamps[i] += end_offset

frame_indices = [int((time_step + offset) * fps) for time_step in timestamps]
print("Frame indices: ", frame_indices)

frame_idx = 0
A_segment_idx = 1
B_segment_idx = 1
C_segment_idx = 1
D_segment_idx = 1
E_segment_idx = 1
save_frame = False
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi

# Create folders if not exist
for vote in ["A", "B", "C", "D", "E"]:
    save_vote_directory = data_folder + vote + "/"
    if not os.path.exists(save_vote_directory):
        os.makedirs(save_vote_directory)
    downsample_directory = save_vote_directory + "/downsample_480p/"
    if not os.path.exists(downsample_directory):
        os.makedirs(downsample_directory)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Exit loop if no frames are left
    
    # Flip the frame horizontally
    # frame = cv2.flip(frame, 1)
    # Crop the frame to the center vertical region (scaled to frame_height)
    crop_height = frame_height
    crop_width = phone_width * crop_height // phone_height
    # print(f"Crop dimensions: width={crop_width}, height={crop_height}")
    # Calculate crop coordinates to center the crop
    x_start = (frame_width - crop_width) // 2
    y_start = (frame_height - crop_height) // 2
    cropped_frame = frame[y_start:y_start+crop_height, x_start:x_start+crop_width]
    # Resize the frame to the new resolution
    resized_frame = cv2.resize(cropped_frame, (new_width, new_height))

    frame_to_save = resized_frame.copy()

    # Sync with labels
    # Headers are timestamp, label and vote
    # every 2 rows represents a segment
    # first row is the starting frame, second row is the ending frame, label is 1 or 2
    # vote is at the second row
    # base on the video fps and timestamp, we can calculate the frame index for each segment
    # save each segment as a separate video file
    
    if frame_idx in frame_indices:
        label = labels[frame_indices.index(frame_idx)]
        # save the current segment
        if label == 1:
            vote = votes[frame_indices.index(frame_idx)+1]
            print("Start saving a segment")
            save_frame = True
            
            if vote == "A":
                output_vid_path = data_folder + vote + "/downsample_480p/" + user + "_" + vote + "_" + str(A_segment_idx) + "_downsample_480p." + ext
                A_segment_idx += 1
            elif vote == "B":
                output_vid_path = data_folder + vote + "/downsample_480p/" + user + "_" + vote + "_" + str(B_segment_idx) + "_downsample_480p." + ext
                B_segment_idx += 1
            elif vote == "C":
                output_vid_path = data_folder + vote + "/downsample_480p/" + user + "_" + vote + "_" + str(C_segment_idx) + "_downsample_480p." + ext
                C_segment_idx += 1
            elif vote == "D":
                output_vid_path = data_folder + vote + "/downsample_480p/" + user + "_" + vote + "_" + str(D_segment_idx) + "_downsample_480p." + ext
                D_segment_idx += 1
            elif vote == "E":
                output_vid_path = data_folder + vote + "/downsample_480p/" + user + "_" + vote + "_" + str(E_segment_idx) + "_downsample_480p." + ext
                E_segment_idx += 1

            out = cv2.VideoWriter(output_vid_path, fourcc, fps, (new_width, new_height))
        elif label == 2:
            vote = votes[frame_indices.index(frame_idx)]
            print("Stop saving a segment")
            save_frame = False
            out.release()
            

        print(f"Frame index: {frame_idx}, Label: {label}, Vote: {vote}")
        
    if save_frame:
        out.write(frame_to_save)

    frame_idx += 1

# Release video objects
cap.release()

print(f"Video segmentation completed. Segments saved in {data_folder} with resolution {new_width}x{new_height}.")