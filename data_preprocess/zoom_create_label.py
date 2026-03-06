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

votes = ["A", "B", "C", "D", "E"]
users = ["JingweiZoom", "mingleiZoom", "MinjieZoom", "WeiZoom", "WenZoom", "ZeyuZoom"]

for user in users:
    session_file = data_folder + user + "/" + user + "_active_session.csv"

    # Read the session file
    session_data = pd.read_csv(session_file)

    # Keep only the rows where "label" column is 1 or 2, and drop the rest
    # Keep the "timestamp" and "label" columns
    session_data = session_data[session_data["label"].isin([1, 2])]
    session_data = session_data[["timestamp", "label"]]
    # Timestamp is in milliseconds, convert it to seconds by dividing by 1000
    session_data["timestamp"] = session_data["timestamp"] / 1000

    # In the same directory, there are vote marker filesnamed [vote]/[timestamp].vote
    # Get vote sequence from the vote marker files, by timestamp, and add the vote sequence to the session data
    # Vote at each label '2', where label '1' keep empty as ''
    vote_sequence = []
    for vote in votes:
        vote_dir = data_folder + user + "/" + vote + "/"
        if os.path.exists(vote_dir):
            vote_files = os.listdir(vote_dir)
            for vote_file in vote_files:
                if vote_file.endswith(".vote"):
                    timestamp = float(vote_file.split(".")[0])
                    vote_sequence.append((timestamp, vote))
    vote_sequence.sort(key=lambda x: x[0])  # sort by timestamp
    # print("Vote sequence: ", vote_sequence)
    vote_idx = 0
    for idx, row in session_data.iterrows():
        if row["label"] == 2:
            # If the label is 2, assign the vote based on the index
            if vote_idx < len(vote_sequence):
                session_data.at[idx, "vote"] = vote_sequence[vote_idx][1]
                vote_idx += 1
            else:
                session_data.at[idx, "vote"] = ''  # If there are no more votes, assign empty string
        else:
            session_data.at[idx, "vote"] = ''  # If the label is not 2, assign empty string

    # Save the processed session data to a new CSV file
    processed_session_file = data_folder + user + "/" + user + "_correct_options.csv"
    session_data.to_csv(processed_session_file, index=False)