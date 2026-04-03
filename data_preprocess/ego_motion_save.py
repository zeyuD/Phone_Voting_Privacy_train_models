import torch
import tempfile
import os
import time
import threading
import cv2
import random
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torchvision.transforms as T
from torchvision.io import read_video
from torchvision.models.optical_flow import raft_large
from torchvision.utils import flow_to_image
from torchvision.io import write_jpeg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'functions')))
from load_machine_config import load_machine_config
from get_ego_motion import track_camera_motion

config = load_machine_config()

# Usage
motion_data = track_camera_motion('Zeyu_A_1_downsample_480p.mp4')

# Rotation data
rotations = np.array([data['rotation'] for data in motion_data])
print("Rotation data shape:", rotations.shape)