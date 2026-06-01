import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- paths ----------
# base = Path(".")  # adjust if needed
base = Path("/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/test_data/") #samples_flexbe/Cam45_ViewFar5_wholeScene_filterZ0.28_Succ2/")
# result_path = Path("/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/")
pc_path = base / "scene_live.npy"
json_path = base / "predictions_scene_live.json"

print("Loading:")
print("  point cloud:", pc_path)
print("  predictions:", json_path)

# ---------- load point cloud ----------
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
        raise ValueError("Unsupported scene_live.npy format (no 'xyz' or 'pc' key).")
elif pc_arr.ndim == 2 and pc_arr.shape[1] > 3:
    pc_arr = pc_arr[:, :3]

if not (pc_arr.ndim == 2 and pc_arr.shape[1] == 3 and pc_arr.shape[0] > 0):
    raise ValueError(f"Point cloud not Nx3: shape={pc_arr.shape}")

print("Point cloud shape (XYZ):", pc_arr.shape)

# ---------- load predictions JSON ----------
with open(json_path, "r") as f:
    preds = json.load(f)

print("JSON top-level keys:", list(preds.keys()))

pred_grasps_cam = preds.get("pred_grasps_cam", {})
scores_dict = preds.get("scores", {})

# ---------- flatten grasps across all object ids ----------
poses_list = []    # list of 4x4 np.array
scores_list = []   # list of floats or None
obj_ids_list = []  # list of ints

# print(f"pred_grasps_cam.items() = {pred_grasps_cam.items()}")
for obj_id_str, grasps_list in pred_grasps_cam.items():
    try:
        obj_id = int(obj_id_str)
    except Exception:
        obj_id = -1

    obj_scores = scores_dict.get(obj_id_str, [])

    for i, g in enumerate(grasps_list):
        arr = np.asarray(g, dtype=float)
        if arr.shape != (4, 4):
            raise ValueError(f"Expected 4x4 pose, got shape {arr.shape} for object {obj_id_str}")
        poses_list.append(arr)
        sc = obj_scores[i] if i < len(obj_scores) else None
        scores_list.append(sc)
        obj_ids_list.append(obj_id)

n_grasps = len(poses_list)
print("Total grasps:", n_grasps)

if n_grasps == 0:
    raise RuntimeError("No grasps found in predictions_scene_live.json")

# ---------- extract positions and rotations ----------
origins = np.zeros((n_grasps, 3), dtype=float)
rot_mats = np.zeros((n_grasps, 3, 3), dtype=float)

for i, T in enumerate(poses_list):
    # T is 4x4: [R t; 0 0 0 1]
    R = T[:3, :3]
    t = T[:3, 3]
    origins[i, :] = t
    rot_mats[i, :, :] = R

finite_mask = np.isfinite(origins).all(axis=1)
origins_f = origins[finite_mask]
rot_mats_f = rot_mats[finite_mask]

print("Finite grasp poses:", origins_f.shape[0], "/", origins.shape[0])
if origins_f.shape[0] == 0:
    raise RuntimeError("All grasp origins had NaN/Inf.")

# ---------- 3D visualization ----------
# Downsample point cloud if huge
n_pc = pc_arr.shape[0]
max_pc = 40000
if n_pc > max_pc:
    idx_pc = np.random.choice(n_pc, max_pc, replace=False)
    pc_vis = pc_arr[idx_pc]
else:
    pc_vis = pc_arr

# Optionally downsample grasp frames for readability
max_grasps_for_frames = 10  # tweak as needed
n_g = origins_f.shape[0]
if n_g > max_grasps_for_frames:
    idx_g = np.random.choice(n_g, max_grasps_for_frames, replace=False)
    origins_plot = origins_f[idx_g]
    rot_plot = rot_mats_f[idx_g]
else:
    origins_plot = origins_f
    rot_plot = rot_mats_f

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot point cloud (small markers)
ax.scatter(pc_vis[:, 0], pc_vis[:, 1], pc_vis[:, 2], s=1, alpha=0.25)

# Plot grasp origins
ax.scatter(origins_f[:, 0], origins_f[:, 1], origins_f[:, 2], s=20, marker="o")

# Draw local frames at a subset of grasps
axis_length = 0.03  # meters; adjust if needed

for t, R in zip(origins_plot, rot_plot):
    # Columns of R are the directions of the local X,Y,Z axes in world coords
    ex = R[:, 0] * axis_length
    ey = R[:, 1] * axis_length
    ez = R[:, 2] * axis_length

    # X axis
    ax.plot(
        [t[0], t[0] + ex[0]],
        [t[1], t[1] + ex[1]],
        [t[2], t[2] + ex[2]],
        color="red",
    )
    # Y axis
    ax.plot(
        [t[0], t[0] + ey[0]],
        [t[1], t[1] + ey[1]],
        [t[2], t[2] + ey[2]],
        color="green",
    )
    # Z axis
    ax.plot(
        [t[0], t[0] + ez[0]],
        [t[1], t[1] + ez[1]],
        [t[2], t[2] + ez[2]],
        color="blue",
    )

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("Contact-GraspNet grasps with orientation frames over scene_live")

plt.tight_layout()
plt.axis('equal')
out_img = base / "scene_live_grasps_frames_3d.png"
plt.savefig(out_img, dpi=200)
plt.show()

print("Saved 3D plot to:", out_img.resolve())
