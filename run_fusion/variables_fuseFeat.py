import sys
import os

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from functions.load_machine_config import load_machine_config

config = load_machine_config()

compdev = config["compdev"]

# Our own dataset
# vote_list = ["A", "C"]
# vote_list = ["A", "B"]
# vote_list = ["A", "C", "E"]
# vote_list = ["A", "B", "C"]
# vote_list = ["A", "B", "C", "D"]
vote_list = ["A", "B", "C", "D", "E"]

num_try = 3
num_round = 100
crop_overlap = 5
interp_len = 50

# user_list = ["Chuan", "Gujing", "Haofan", "Jimmy", "Jingwei", 
#              "Junwei", "minglei", "Minjie", "Mingxuan","Rosie", 
#              "Sihang", "Wen", "Yirui", "Zeyu", "Zidan", 
#              "Ziyue1"
#              ]
# user_list = ["Chuan", "Gujing", "Haofan", "Junwei", "Yirui"]
# user_list = ["Chuan"]
# user_list = ["Zeyu"]
# user_list = ["JingweiObj", "ZeyuObj"]
user_list = ["Jingwei", "Zeyu"]

# user_list = ["Jingwei", "minglei", "Minjie", "Wen", "Zeyu"]
# user_othersetting_list = ["Jingwei", "minglei", "Minjie", "Wen", "Zeyu"]
# user_othersetting_list = ["ZeyuZoom"]

# user_list = ["JingweiZoom", "mingleiZoom", "MinjieZoom", "WeiZoom", "WenZoom", "ZeyuZoom"]
# user_othersetting_list = ["JingweiZoom", "mingleiZoom", "MinjieZoom", "WeiZoom", "WenZoom", "ZeyuZoom"]

grid_dim = "22"

# feature_names = ["combine_opticalflow_" + grid_dim + "_all_feat_downsample_480p_s22"]
# feature_names = ["opticalflow_" + grid_dim + "_downsample_480p_s22"]
# feature_names = ["opticalflowRAFT_" + grid_dim + "_downsample_480p_s22"]
# feature_names = ["all_processpos_norm_downsample_480p_s22"]
# feature_names = ["egomotion_rot_downsample_480p_s22"]
feature_names = [
    "all_processpos_norm_downsample_480p_s22",
    # "opticalflowRAFT_varyRef_" + grid_dim + "_downsample_480p_s22",
    # "opticalflowRAFT_" + grid_dim + "_downsample_480p_s22",
    # "opticalflowRAFT_border_" + grid_dim + "_downsample_480p_s22",
    # "egomotion_rot_downsample_480p_s22",
    # "opticalflowRAFT_border_varyRef_" + grid_dim + "_downsample_480p_s22",
    "opticalflowRAFT_edge_" + grid_dim + "_downsample_480p_s22", # Best after fuse with eyeFeat?
    # "objYOLO_downsample_480p_s22",
    # "opticalflowRAFT_edge_varyRef_" + grid_dim + "_downsample_480p_s22",
    # "opticalflowRAFT_obj_" + grid_dim + "_downsample_480p_s22"
    # "opticalflowRAFT_obj_varyRef_" + grid_dim + "_downsample_480p_s22"
    ]
