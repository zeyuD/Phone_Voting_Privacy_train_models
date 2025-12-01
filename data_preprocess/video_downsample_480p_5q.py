import cv2
import yaml
import os
import subprocess
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

work_directory = config["data_dir"] + "Phone_Privacy/"

# Sample lists for setups, votes, users, and instances
votes = ["A", "B", "C", "D", "E"]
# votes = ["E"]

# Create a dictionary of setups, each will have different setup and users
setups = {
    "phone_s22": ["Zeyu"],
          }

# Loop through the variables and construct the video path for each combination
for setup in setups:
    users = setups[setup]
    print("Running setup: ", setup)
    print("Users: ", users)
    for vote in votes:
        for user in users:
            data_directory = work_directory + "data/" + setup + "/" + vote
            output_directory = data_directory + "/downsample_480p"
            # create the output directory if it does not exist
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            # find all video files in the data path end with ".mp4"
            video_files = os.listdir(data_directory)
            video_files = [f for f in video_files if f.endswith(".mp4")]
            for ins, video in enumerate(video_files):
                # Construct the video path dynamically based on the current loop variables
                video_path = os.path.join(data_directory, video)
                # Replace the "record" to "angle" in the video name
                output_vid_name = video.replace("record", "downsample_480p")
                output_vid_path = os.path.join(output_directory, output_vid_name)

                cap = cv2.VideoCapture(video_path)
                # Check if video opened successfully
                if not cap.isOpened():
                    print("Error: Could not open video.")
                    exit()

                # Get original video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Set the new resolution
                new_width, new_height = 480, 848

                # Define the codec and create VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
                out = cv2.VideoWriter(output_vid_path, fourcc, fps, (new_width, new_height))

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break  # Exit loop if no frames are left

                    # Resize the frame to the new resolution
                    resized_frame = cv2.resize(frame, (new_width, new_height))

                    # Write the resized frame to the output video
                    out.write(resized_frame)

                # Release video objects
                cap.release()
                out.release()

                print(f"Video saved as {output_vid_path} with resolution {new_width}x{new_height}.")