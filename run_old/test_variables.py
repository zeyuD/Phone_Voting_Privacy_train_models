import sys
import os

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config

config = load_machine_config()

compdev = config["compdev"]

# Our own dataset
# vote_list = ["A", "C"]
# vote_list = ["A", "B", "C"]
vote_list = ["A", "B", "C", "D", "E"]

num_try = 1
num_round = 100

user_list = ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", 
             "Junwei", "minglei", "Minjie", "Mingxuan","Rosie", 
             "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", 
             "Ziyue1"
             ]
# user_list = ["Chuan", "Gujing", "Haofan", "Junwei", "Yirui"]
# user_list = ["Zeyu"]
# user_list = ["ZeyuZoom"]

# user_list = ["Jingwei", "minglei", "Minjie", "Wen", "Zeyu"]
# user_othersetting_list = ["Jingwei", "minglei", "Minjie", "Wen", "Zeyu"]
# user_othersetting_list = ["ZeyuZoom"]

# user_list = ["JingweiZoom", "mingleiZoom", "MinjieZoom", "WeiZoom", "WenZoom", "ZeyuZoom"]
# user_othersetting_list = ["JingweiZoom", "mingleiZoom", "MinjieZoom", "WeiZoom", "WenZoom", "ZeyuZoom"]

grid_dim = "22"

# feature_names = ["combine_opticalflow_" + grid_dim + "_all_feat_downsample_480p_s22"]
# feature_names = ["opticalflow_" + grid_dim + "_downsample_480p_s22"]
# feature_names = ["opticalflowRAFT_" + grid_dim + "_downsample_480p_s22"]
# feature_names = ["opticalflowRAFT_border_" + grid_dim + "_downsample_480p_s22"]
# feature_names = ["opticalflowRAFT_edge_" + grid_dim + "_downsample_480p_s22"]
# feature_names = ["opticalflowRAFT_obj_" + grid_dim + "_downsample_480p_s22"]
feature_names = ["all_processpos_norm_downsample_480p_s22"]
# feature_names = ["egomotion_rot_downsample_480p_s22"]

# feature_names = ["all_norm_downsample_480p_s22"]
# feature_names = ["all_processpos_nonorm_downsample_480p_s22"]
# feature_names = ["all_noprocesspos_norm_downsample_480p_s22"]
# feature_names = ["all_noprocesspos_nonorm_downsample_480p_s22"]
# feature_names = ["all_nonorm_downsample_480p_s22"]