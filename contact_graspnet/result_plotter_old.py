import numpy as np
import open3d as o3d

def depth_to_point_cloud(depth, K):
    """Convert a depth map to 3D point cloud using intrinsics."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth.shape
    xmap, ymap = np.meshgrid(np.arange(W), np.arange(H))
    mask = depth > 0
    x, y, z = xmap[mask], ymap[mask], depth[mask]
    X = (x - cx) * z / fx
    Y = (y - cy) * z / fy
    Z = z
    return np.stack((X, Y, Z), axis=-1)

# Load predictions
npz_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/predictions_scene_from_ucn.npz"
data = np.load(npz_path, allow_pickle=True)
grasps = data["pred_grasps_cam"].item()
scores = data["scores"].item()
contacts = data["contact_pts"].item()

print("Loaded grasp predictions for object IDs:", list(grasps.keys()))

# Load original depth data and intrinsics
pc_data = np.load("/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/test_data/scene_from_ucn.npy", allow_pickle=True).item()
depth = pc_data["depth"]
K = pc_data["K"]
pc_points = depth_to_point_cloud(depth, K)
print(f"Point cloud shape: {pc_points.shape}")

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.paint_uniform_color([0.5, 0.5, 0.5])  # grey base
# geometries = [pcd]

pc_points = depth_to_point_cloud(depth, K)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_points)

# ---------------------------------------------------------
# Zoom in: crop point cloud around its center
# ---------------------------------------------------------
pts_np = np.asarray(pcd.points)
center = np.array([0,0,0]) #pts_np.mean(axis=0)

# Size of the zoom box (meters) – tweak as needed
dx, dy, dz = 2.30, 2.30, 2.30   # +/- 30 cm in x,y,z around the center

min_bound = center - np.array([dx, dy, dz])
max_bound = center + np.array([dx, dy, dz])

print("[INFO] Cropping around center:", center)
print("[INFO] min_bound:", min_bound, "max_bound:", max_bound)

bbox = o3d.geometry.AxisAlignedBoundingBox(
    min_bound=min_bound,
    max_bound=max_bound
)

pcd_cropped = pcd.crop(bbox)
print("[INFO] Original points:", pts_np.shape[0],
      " -> Cropped points:", np.asarray(pcd_cropped.points).shape[0])

# Use cropped cloud in the viewer
geometries = [pcd_cropped]




# Predefined colors for visualization
colors = [
    [1, 0, 0],    # red
    [0, 1, 0],    # green
    [0, 0, 1],    # blue
    [1, 1, 0],    # yellow
    [1, 0, 1],    # magenta
    [0, 1, 1],    # cyan
    [1, 0.5, 0],  # orange
    [0.6, 0.4, 1] # purple
]


# Visualize top grasps per object
for idx, obj_id in enumerate(grasps.keys()):
    g_mats = grasps[obj_id]
    sc = scores[obj_id]
    top_idxs = np.argsort(-sc) #[:5]  # top 5 grasps
    # color = colors[idx % len(colors)]

    for i in top_idxs:
        g = g_mats[i]
        T = np.eye(4)
        T[:4, :4] = g
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(T)
        # frame.paint_uniform_color(color)
        geometries.append(frame)
    

# Render
o3d.visualization.draw_geometries(geometries)
