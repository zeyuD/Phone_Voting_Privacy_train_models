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

# Create a dictionary of setups, each will have different setup and users
setups = {
    "phone180sr_s22": ["Chuan", "Jimmy", "minglei", "Zeyu", "Ziyue"],
    "phone240sr_s22": ["Chuan", "Jimmy", "minglei", "Zeyu", "Ziyue"],
    }
# Set to False to actually move the files, True only prints what would be done
dry_run = False

# Reorganize "[setup]/[user]/[vote]/[3digit]_ESRGAN_1080p.mp4" into "[setup]/[vote]/[user]_[vote]_[idx]_record.mp4"
for setup in setups:
    users = setups[setup]
    setup_directory = work_directory + "data/" + setup
    print("Running setup: ", setup)
    print("Users: ", users)
    moved, skipped = 0, 0
    for user in users:
        for vote in votes:
            source_directory = os.path.join(setup_directory, user, vote)
            if not os.path.isdir(source_directory):
                print(f"Warning: {source_directory} does not exist, skipping.")
                continue
            output_directory = os.path.join(setup_directory, vote)
            if not dry_run:
                os.makedirs(output_directory, exist_ok=True)
            # find all video files in the source path end with ".mp4"
            video_files = sorted(f for f in os.listdir(source_directory) if f.endswith(".mp4"))
            for video in video_files:
                # "001_ESRGAN_1080p.mp4" -> 1, to match the "[user]_[vote]_[idx]_record.mp4" naming of other setups
                idx = int(video.split("_")[0])
                source_path = os.path.join(source_directory, video)
                output_path = os.path.join(output_directory, f"{user}_{vote}_{idx}_record.mp4")
                if os.path.exists(output_path):
                    print(f"Warning: {output_path} already exists, skipping {source_path}.")
                    skipped += 1
                    continue
                if dry_run:
                    print(f"[dry run] {source_path} -> {output_path}")
                else:
                    os.rename(source_path, output_path)
                moved += 1
            # remove the now empty "[user]/[vote]" directory
            if not dry_run and not os.listdir(source_directory):
                os.rmdir(source_directory)
        # remove the now empty "[user]" directory
        user_directory = os.path.join(setup_directory, user)
        if not dry_run and os.path.isdir(user_directory) and not os.listdir(user_directory):
            os.rmdir(user_directory)
    print(f"{setup}: {'would move' if dry_run else 'moved'} {moved} files, skipped {skipped}.")
