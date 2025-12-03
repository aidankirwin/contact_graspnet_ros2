# inspect_predictions_npz.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# npz_path = Path("predictions_scene_live.npz")  # adjust if needed
npz_path = Path("./sample_scene_ucn/predictions_scene_from_ucn.npz")  # adjust if needed
npz = np.load(npz_path, allow_pickle=True)
keys = list(npz.files)

print("=== NPZ Summary ===")
print(f"File: {npz_path}")
print(f"Keys ({len(keys)}): {keys}")

# Print basic info for every key
rows = []
for k in keys:
    arr = npz[k]
    rows.append((k, getattr(arr, "shape", None), getattr(arr, "dtype", None)))
    print(f" - {k:24s} shape={getattr(arr,'shape',None)} dtype={getattr(arr,'dtype',None)}")

def pick_key(cands):
    for c in cands:
        for k in keys:
            if c.lower() in k.lower():
                return k
    return None

k_scores  = pick_key(["score","scores","grasp_score","grasp_scores"])
k_poses   = pick_key(["pose","poses","grasp_pose","grasp_poses","T_world_grasp","T_obj_grasp","T_world_target"])
k_samples = pick_key(["sample","samples","contact","contacts","approach"])
k_ids     = pick_key(["object_ids","obj_id","obj_ids","ids"])

print("\n=== Heuristic picks ===")
print("scores :", k_scores)
print("poses  :", k_poses)
print("samples:", k_samples)
print("ids    :", k_ids)

scores  = npz[k_scores]  if k_scores  else None
poses   = npz[k_poses]   if k_poses   else None
samples = npz[k_samples] if k_samples else None
obj_ids = npz[k_ids]     if k_ids     else None

# Score summary + histogram
if isinstance(scores, np.ndarray) and scores.ndim == 1 and scores.size > 0:
    print("\nScores summary:")
    print(" n =", scores.size,
          " min =", float(scores.min()),
          " max =", float(scores.max()),
          " mean =", float(scores.mean()))
    plt.figure()
    plt.hist(scores, bins=30)
    plt.title("Grasp scores distribution")
    plt.xlabel("score")
    plt.ylabel("count")
    plt.show()

# Interpret poses into a dataframe (best-effort)
pose_df = None
pose_note = ""
if isinstance(poses, np.ndarray) and poses.size > 0:
    if poses.ndim == 2 and poses.shape[1] == 7:
        # [x,y,z,qx,qy,qz,qw]
        pose_note = "Assuming poses columns [x, y, z, qx, qy, qz, qw]."
        cols = ["x","y","z","qx","qy","qz","qw"]
        pose_df = pd.DataFrame(poses, columns=cols)
    elif poses.ndim == 2 and poses.shape[1] == 6:
        # [x,y,z,rx,ry,rz] (Euler or axis-angle)
        pose_note = "Assuming poses columns [x, y, z, r1, r2, r3] (rotation params)."
        cols = ["x","y","z","r1","r2","r3"]
        pose_df = pd.DataFrame(poses, columns=cols)
    elif poses.ndim == 3 and poses.shape[1:] == (4,4):
        # Convert homogeneous to [x,y,z,qx,qy,qz,qw]
        def mat_to_quat(T):
            M = T[:3,:3]; t = T[:3,3]
            tr = np.trace(M)
            if tr > 0:
                S = np.sqrt(tr + 1.0) * 2.0
                qw = 0.25 * S
                qx = (M[2,1] - M[1,2]) / S
                qy = (M[0,2] - M[2,0]) / S
                qz = (M[1,0] - M[0,1]) / S
            elif (M[0,0] > M[1,1]) and (M[0,0] > M[2,2]):
                S = np.sqrt(1.0 + M[0,0] - M[1,1] - M[2,2]) * 2.0
                qw = (M[2,1] - M[1,2]) / S
                qx = 0.25 * S
                qy = (M[0,1] + M[1,0]) / S
                qz = (M[0,2] + M[2,0]) / S
            elif M[1,1] > M[2,2]:
                S = np.sqrt(1.0 + M[1,1] - M[0,0] - M[2,2]) * 2.0
                qw = (M[0,2] - M[2,0]) / S
                qx = (M[0,1] + M[1,0]) / S
                qy = 0.25 * S
                qz = (M[1,2] + M[2,1]) / S
            else:
                S = np.sqrt(1.0 + M[2,2] - M[0,0] - M[1,1]) * 2.0
                qw = (M[1,0] - M[0,1]) / S
                qx = (M[0,2] + M[2,0]) / S
                qy = (M[1,2] + M[2,1]) / S
                qz = 0.25 * S
            return np.array([t[0], t[1], t[2], qx, qy, qz, qw], dtype=float)
        rows = [mat_to_quat(T) for T in poses]
        cols = ["x","y","z","qx","qy","qz","qw"]
        pose_df = pd.DataFrame(np.vstack(rows), columns=cols)
        pose_note = "Detected homogeneous transforms (N,4,4) → converted to [x,y,z,qx,qy,qz,qw]."
    else:
        # generic preview
        print(f"Unrecognized pose shape: {poses.shape}; showing a small raw preview.")
        if poses.ndim == 2 and poses.shape[0] > 0 and poses.shape[1] > 0:
            pose_df = pd.DataFrame(poses[:min(10, poses.shape[0]), :min(10, poses.shape[1])])

# Join scores and show top-20
if pose_df is not None and not pose_df.empty:
    if isinstance(scores, np.ndarray) and scores.ndim == 1 and len(scores) == len(pose_df):
        pose_df = pose_df.copy()
        pose_df["score"] = scores
        pose_df = pose_df.sort_values("score", ascending=False).reset_index(drop=True)
    print("\nPreview (top 20):")
    print(pose_df.head(20).to_string(index=False))

    # Save CSV
    out_csv = npz_path.with_name("predictions_scene_live_top.csv")
    pose_df.head(1000).to_csv(out_csv, index=False)
    print(f"\nSaved CSV: {out_csv.resolve()}")

    # 2D scatter if x,y,z present
    if all(c in pose_df.columns for c in ["x","y","z"]):
        P = pose_df[["x","y","z"]].to_numpy()
        ns = min(5000, P.shape[0])
        idx = np.random.choice(P.shape[0], ns, replace=False) if P.shape[0] > ns else np.arange(P.shape[0])
        Ps = P[idx]
        plt.figure()
        plt.scatter(Ps[:,0], Ps[:,1], s=2)
        plt.title("Grasp pose XY (sample)")
        plt.xlabel("x"); plt.ylabel("y")
        plt.show()

        plt.figure()
        plt.scatter(Ps[:,0], Ps[:,2], s=2)
        plt.title("Grasp pose XZ (sample)")
        plt.xlabel("x"); plt.ylabel("z")
        plt.show()

# Save a simple text report
lines = [f"Keys: {keys}"]
for k,sh,dt in rows:
    lines.append(f"{k}: shape={sh}, dtype={dt}")
if k_scores:  lines.append(f"picked scores: {k_scores}")
if k_poses:   lines.append(f"picked poses: {k_poses}")
if k_samples: lines.append(f"picked samples: {k_samples}")
if k_ids:     lines.append(f"picked ids: {k_ids}")
if pose_note: lines.append(pose_note)

txt_out = npz_path.with_name("predictions_scene_live_report.txt")
with open(txt_out, "w") as f:
    f.write("\n".join(lines))
print("\nReport saved:", txt_out.resolve())
