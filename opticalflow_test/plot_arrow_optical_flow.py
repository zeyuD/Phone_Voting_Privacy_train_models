import os
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory of this file to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'functions')))

from load_machine_config import load_machine_config

config = load_machine_config()

work_directory = config["data_dir"] + "Phone_Privacy/"

data_dir = os.path.join(config["data_dir"], "Phone_Privacy") + os.sep
print(f"Using data from: {data_dir}")

vote = "A"
user = "Zeyu"
idx = 1

vid_file_path = os.path.join(data_dir, "data", "phone_s22", vote, "downsample_480p")
vid_name = f"{user}_{vote}_{idx}_downsample_480p.mp4"
vid_file_name = os.path.join(vid_file_path, vid_name)

# ----------------------------
# Read video metadata
# ----------------------------
cap = cv2.VideoCapture(vid_file_name)
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {vid_file_name}")

vid_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
vid_fps = cap.get(cv2.CAP_PROP_FPS)
vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ----------------------------
# Read reference frame (MATLAB read(vid, 5))
# Note: MATLAB uses 1-based indexing; OpenCV uses 0-based.
# MATLAB frame 5 -> OpenCV index 4
# ----------------------------
ref_index = 4
cap.set(cv2.CAP_PROP_POS_FRAMES, ref_index)
ok, ref_bgr = cap.read()
if not ok:
    raise RuntimeError(f"Failed to read reference frame at index {ref_index}")

ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

# ----------------------------
# Restart video and compute flow over time
# ----------------------------
cap.release()
cap = cv2.VideoCapture(vid_file_name)

frame_flow_x_time = np.zeros((vid_length, vid_height, vid_width), dtype=np.float32)
frame_flow_y_time = np.zeros((vid_length, vid_height, vid_width), dtype=np.float32)

# Farnebäck parameters (tweak if needed)
fb_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

prev_gray = ref_gray.copy()  # matches your "initialize with ref" idea

count = 0
while True:
    ok, frame_bgr = cap.read()
    if not ok or count >= vid_length:
        break

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # MATLAB does:
    # frame_flow = estimateFlow(opticFlow, ref_gray_frame); % initialize
    # frame_flow = estimateFlow(opticFlow, grayFrame);
    #
    # In OpenCV, calcOpticalFlowFarneback needs prev and next frames.
    # We'll use ref_gray as "prev" for the first calc, then you can keep
    # prev_gray fixed or update it. Here we keep it fixed to ref_gray to
    # stay closer to your intent (flow relative to reference).
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, **fb_params)

    frame_flow_x_time[count] = flow[..., 0]  # Vx
    frame_flow_y_time[count] = flow[..., 1]  # Vy

    count += 1

cap.release()

# ----------------------------
# Visualize a specific frame (MATLAB num = 10)
# MATLAB frame 10 -> OpenCV index 9
# ----------------------------
num = 10
frame_index = num - 1

cap = cv2.VideoCapture(vid_file_name)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
ok, frame_bgr = cap.read()
cap.release()
if not ok:
    raise RuntimeError(f"Failed to read frame {num} (index {frame_index})")

frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

Vx = frame_flow_x_time[frame_index]
Vy = frame_flow_y_time[frame_index]

scale = 50  # same as MATLAB

# MATLAB does imresize(Vx, 1/scale). We can sample every 'scale' pixels.
Vx_ds = Vx[::scale, ::scale]
Vy_ds = Vy[::scale, ::scale]

# Meshgrid in original image coordinates
h_ds, w_ds = Vx_ds.shape
x = np.arange(0, w_ds * scale, scale)
y = np.arange(0, h_ds * scale, scale)
X, Y = np.meshgrid(x, y)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(frame_gray, cmap="gray")
plt.quiver(X, Y, Vx_ds, Vy_ds, color="g", angles="xy", scale_units="xy", scale=0.03)
plt.axis("off")
plt.tight_layout()
plt.show()
