import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------- paths --------
base = Path(".")  # adjust if needed
# pc_path = base / "scene_from_ucn.npy"
# json_path = "../results/predictions_scene_from_ucn.json"
pc_path = base / "scene_live.npy"
json_path = "../results/predictions_scene_live.json"

print("Loading:")
print("  point cloud:", pc_path)
print("  predictions:", json_path)




# -------- load point cloud --------
pc = np.load(pc_path, allow_pickle=True)
pc_arr = np.asarray(pc)

# Handle common formats:
# - Nx3 or Nx>=3 (XYZ or XYZRGB...)
# - 0-D np.object_ with dict (e.g., {"xyz": ..., ...})
if pc_arr.ndim == 0 and hasattr(pc_arr, "item"):
    d = pc_arr.item()
    if "xyz" in d:
        pc_arr = np.asarray(d["xyz"]).reshape(-1, 3)
    elif "pc" in d:
        arr = np.asarray(d["pc"])
        pc_arr = arr[:, :3]
    else:
        raise ValueError(f"Unsupported {pc_path} format (no 'xyz' or 'pc' key).")
elif pc_arr.ndim == 2 and pc_arr.shape[1] > 3:
    pc_arr = pc_arr[:, :3]

if not (pc_arr.ndim == 2 and pc_arr.shape[1] == 3 and pc_arr.shape[0] > 0):
    raise ValueError(f"Point cloud not Nx3: shape={pc_arr.shape}")

print("Point cloud shape (XYZ):", pc_arr.shape)

# -------- load predictions JSON --------
with open(json_path, "r") as f:
    preds = json.load(f)

print("JSON keys:", list(preds.keys()))

pred_grasps_cam = preds.get("pred_grasps_cam", {})
scores = preds.get("scores", {})

# -------- flatten grasps across all object ids --------
poses_list = []   # list of np.array (pose)
scores_list = []  # float or None
obj_ids_list = [] # int

for obj_id_str, grasps_list in pred_grasps_cam.items():
    try:
        obj_id = int(obj_id_str)
    except Exception:
        obj_id = -1
    obj_scores = scores.get(obj_id_str, [])
    for i, g in enumerate(grasps_list):
        arr = np.asarray(g, dtype=float)
        poses_list.append(arr)
        sc = obj_scores[i] if i < len(obj_scores) else None
        scores_list.append(sc)
        obj_ids_list.append(obj_id)

n_grasps = len(poses_list)
print("Total grasps:", n_grasps)

if n_grasps == 0:
    raise RuntimeError("No grasps found in predictions_scene_live.json")

# -------- extract positions from poses --------
def pose_to_xyz(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape == (4, 4):
        # homogeneous transform
        return pose[0, 3], pose[1, 3], pose[2, 3]
    if pose.ndim == 1 and pose.size >= 3:
        # [x,y,z,...]
        return pose[0], pose[1], pose[2]
    raise ValueError(f"Unsupported pose shape: {pose.shape}")

def quat_to_R(qx,qy,qz,qw):
    # standard quaternion→rotation (right-handed)
    x,y,z,w = qx,qy,qz,qw
    R = np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),   1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),   2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])
    return R

xyz_list = []
for pose in poses_list:
    try:
        xyz_list.append(pose_to_xyz(pose))
    except Exception:
        xyz_list.append((np.nan, np.nan, np.nan))

xyz = np.asarray(xyz_list, dtype=float)
finite_mask = np.isfinite(xyz).all(axis=1)
xyz_f = xyz[finite_mask]

print("Finite grasp positions:", xyz_f.shape[0], "/", xyz.shape[0])

if xyz_f.shape[0] == 0:
    raise RuntimeError("All grasp poses had NaN/Inf positions.")

# -------- 3D visualization --------
# Downsample point cloud if huge
n_pc = pc_arr.shape[0]
max_pc = 40000
if n_pc > max_pc:
    idx_pc = np.random.choice(n_pc, max_pc, replace=False)
    pc_vis = pc_arr[idx_pc]
else:
    pc_vis = pc_arr

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot point cloud
ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2], s=1, alpha=0.25)

# Plot grasp positions
ax.scatter(xyz_f[:, 0], xyz_f[:, 1], xyz_f[:, 2], s=20, marker="o")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Contact-GraspNet grasps over scene_live point cloud")

plt.tight_layout()
plt.axis('equal')
out_img = base / "scene_live_grasps_3d.png"
plt.savefig(out_img, dpi=200)
plt.show()

print("Saved 3D plot to:", out_img.resolve())
