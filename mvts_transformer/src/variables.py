import sys
import os

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

compdev = config["compdev"]

# Our own dataset
# vote_list = ["A", "C"]
# vote_list = ["A", "B", "C"]
vote_list = ["A", "B", "C", "D", "E"]

# user_list = ["jingwei", "Wen", "Zeyu"]
user_list = ["jingwei"]

feature_names = ["all_norm_downsample_480p_s22"]
num_try = 10
num_round = 100