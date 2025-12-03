import cv2
import numpy as np
import json

# --- Load RGB image ---
rgb_bgr = cv2.imread('./sample_scene_ucn/input/from_rgbd-color.png')      # (480, 640, 3), BGR
rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)   # -> RGB uint8, like 0.npy['rgb']

# --- Load depth and convert to meters ---
depth_raw = cv2.imread('./sample_scene_ucn/input/from_rgbd-depth.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
# depth_raw is uint16 mm → meters
depth_m = depth_raw / 1000.0                     # (480, 640) float32

# --- Load segmentation (either im_label.npy or segmentation.json) ---
seg = np.load('./sample_scene_ucn/output/segmentation_from_rgbd/im_label.npy').astype(np.float32)  # (480, 640) float32; matches json["instance_ids"]

# --- Build K (you must fill in fx, fy, cx, cy for your camera) ---
fx, fy = 615.0, 615.0   # example values – replace with your camera’s
cx, cy = 320.0, 240.0
K = np.array([[fx, 0.0, cx],
              [0.0, fy, cy],
              [0.0, 0.0, 1.0]], dtype=np.float64)

# --- Pack into dict matching 0.npy’s structure ---
scene_dict = {
    'rgb':   rgb,       # HxWx3 uint8
    'depth': depth_m,   # HxW float32, meters
    'K':     K,         # 3x3 float64
    'seg':   seg        # HxW float32 labels
}

np.save('./sample_scene_ucn/scene_from_ucn.npy', scene_dict)
