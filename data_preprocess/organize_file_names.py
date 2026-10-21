import os
import sys

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

work_directory = config["data_dir"] + "Phone_Privacy/"

# Create a dictionary of setups, each will have different setup and users
setups = {
    # "phone_s22": ["haochen Li", "linweili", "RuiT", "YipeiMa_2", "YU_2"],
    "phonehori_s22": ["jingwei_h", "shengyan_h", "wen_h"],
    "padhori_op2": ["jingweipad_h", "shengyan_pad_h", "wen_pad_h"],
    }
# New user names, in the same order as setups
setups_rename = {
    # "phone_s22": ["Haochen", "Linwei", "Rui", "Yipei", "Yu"],
    "phonehori_s22": ["Jingwei", "Shengyan", "Wen"],
    "padhori_op2": ["JingweiPad", "ShengyanPad", "WenPad"],
    }
# Set to False to actually rename the files, True only prints what would be done
dry_run = False

# Rename every folder/file named "[user]" or "[user]_..." under "data/[setup]" to use the new user name,
# e.g. "phoneds_s22/jingwei_d/A/jingwei_d_A_1_record.mp4" -> "phoneds_s22/Jingwei/A/Jingwei_A_1_record.mp4"
for setup in setups:
    users = setups[setup]
    users_rename = setups_rename[setup]
    assert len(users) == len(users_rename), f"{setup}: setups and setups_rename have different lengths"
    rename_map = dict(zip(users, users_rename))
    setup_directory = work_directory + "data/" + setup
    print("Running setup: ", setup)
    print("Rename: ", rename_map)
    renamed, skipped = 0, 0
    # walk bottom-up so files are renamed before the folders that contain them
    for root, dirs, files in os.walk(setup_directory, topdown=False):
        for name in files + dirs:
            for user, user_rename in rename_map.items():
                if name == user or name.startswith(user + "_"):
                    new_name = user_rename + name[len(user):]
                    break
            else:
                continue
            source_path = os.path.join(root, name)
            output_path = os.path.join(root, new_name)
            if os.path.exists(output_path):
                print(f"Warning: {output_path} already exists, skipping {source_path}.")
                skipped += 1
                continue
            if dry_run:
                print(f"[dry run] {source_path} -> {output_path}")
            else:
                os.rename(source_path, output_path)
            renamed += 1
    print(f"{setup}: {'would rename' if dry_run else 'renamed'} {renamed} files/folders, skipped {skipped}.")
