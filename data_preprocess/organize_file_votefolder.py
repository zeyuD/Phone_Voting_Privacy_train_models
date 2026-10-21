import os
import shutil
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
    # "phone_s22": ["Haochen", "Linwei", "Rui", "Yipei", "Yu"],
    "phonehori_s22": ["Jingwei", "Shengyan", "Wen"],
    "padhori_op2": ["JingweiPad", "ShengyanPad", "WenPad"],
    }
# Set to False to actually copy the files, True only prints what would be done
dry_run = False

# Copy "[setup]/[user]/[vote]/*" into "[setup]/[vote]/*", keeping the user folder as it is
for setup in setups:
    users = setups[setup]
    setup_directory = work_directory + "data/" + setup
    print("Running setup: ", setup)
    print("Users: ", users)
    copied, skipped = 0, 0
    for user in users:
        for vote in votes:
            source_directory = os.path.join(setup_directory, user, vote)
            if not os.path.isdir(source_directory):
                print(f"Warning: {source_directory} does not exist, skipping.")
                continue
            output_directory = os.path.join(setup_directory, vote)
            if not dry_run:
                os.makedirs(output_directory, exist_ok=True)
            data_files = sorted(f for f in os.listdir(source_directory)
                                if os.path.isfile(os.path.join(source_directory, f)) and not f.startswith("."))
            for data_file in data_files:
                source_path = os.path.join(source_directory, data_file)
                output_path = os.path.join(output_directory, data_file)
                if os.path.exists(output_path):
                    print(f"Warning: {output_path} already exists, skipping {source_path}.")
                    skipped += 1
                    continue
                if dry_run:
                    print(f"[dry run] {source_path} -> {output_path}")
                else:
                    shutil.copy2(source_path, output_path)
                copied += 1
    print(f"{setup}: {'would copy' if dry_run else 'copied'} {copied} files, skipped {skipped}.")
