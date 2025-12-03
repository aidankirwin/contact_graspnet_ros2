# inspect_graspnet_outputs.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# -------- paths --------
base = Path(".")        # change to /mnt/data if you want
K_path = base / "K.npy"
pc_path = base / "scene_live.npy"

# -------- load K --------
K = np.load(K_path)
print("K.npy:")
print("  shape:", K.shape, "dtype:", K.dtype)
print(K)

# -------- load point cloud --------
pc = np.load(pc_path, allow_pickle=True)
pc_arr = np.asarray(pc)
print("\nscene_live.npy:")
print("  shape:", pc_arr.shape, "dtype:", pc_arr.dtype)

# Handle common cases:
# - Nx3 (XYZ)
# - Nx6 (XYZRGB)
# - 0-dim object array (dict saved via np.save)
if pc_arr.ndim == 0 and hasattr(pc_arr, "item"):
    d = pc_arr.item()
    print("  Detected object npy (dict) keys:", list(d.keys()))
    if "xyz" in d:
        pc_arr = np.asarray(d["xyz"]).reshape(-1, 3)
    elif "pc" in d:
        arr = np.asarray(d["pc"])
        pc_arr = arr[:, :3] if arr.ndim == 2 and arr.shape[1] >= 3 else arr
    else:
        raise ValueError("Unsupported object npy format — need key 'xyz' or 'pc'")

# basic sanity
if not (pc_arr.ndim == 2 and pc_arr.shape[1] >= 3 and pc_arr.shape[0] > 0):
    raise ValueError("Point array not in expected Nx3 shape or it is empty.")

finite_mask = np.isfinite(pc_arr[:, :3]).all(axis=1)
nan_rows = (~finite_mask).sum()
valid = pc_arr[finite_mask, :3]

print(f"  n_points (raw): {pc_arr.shape[0]}")
print(f"  n_points (finite XYZ): {valid.shape[0]}")
print(f"  NaN/Inf rows: {nan_rows}")

print("  bbox min XYZ:", valid.min(axis=0))
print("  bbox max XYZ:", valid.max(axis=0))
print("  mean XYZ:", valid.mean(axis=0))

# save a small CSV preview (first 1000 rows, up to 6 columns if present)
preview_cols = min(pc_arr.shape[1], 6)
limit = min(pc_arr.shape[0], 1000)
df_preview = pd.DataFrame(pc_arr[:limit, :preview_cols], columns=[f"col{i}" for i in range(preview_cols)])
csv_path = base / "scene_live_preview.csv"
df_preview.to_csv(csv_path, index=False)
print(f"\nSaved CSV preview → {csv_path.resolve()}")

# plots: keep each chart in its own figure; no explicit colors
sample_n = min(5000, valid.shape[0])
idx = np.random.choice(valid.shape[0], sample_n, replace=False) if valid.shape[0] > sample_n else np.arange(valid.shape[0])
samp = valid[idx]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d') # '111' means 1x1 grid, first subplot
ax.scatter(samp[:, 0], samp[:, 1], samp[:, 2])
plt.title("scene_live: XYZ scatter (sample)")
plt.axis('equal') 
plt.xlabel("X"); plt.ylabel("Y"); 
plt.savefig("scene_live.png")
plt.show()

# plt.figure()
# plt.scatter(samp[:, 0], samp[:, 2], s=1)
# plt.title("scene_live: XZ scatter (sample)")
# plt.xlabel("X"); plt.ylabel("Z")
# plt.show()
