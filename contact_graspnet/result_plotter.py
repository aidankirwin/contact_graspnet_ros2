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


# ------------------------------------------------------------------
# Paths (adjust as you like)
# ------------------------------------------------------------------
npz_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/predictions_scene_from_ucn.npz"
scene_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/test_data/scene_from_ucn.npy"
save_image_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/scene_from_ucn_plot.png"

# npz_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/predictions_0.npz"
# scene_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/test_data/0.npy"
# save_image_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/scene_0.png"


# ------------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------------
data = np.load(npz_path, allow_pickle=True)
grasps = data["pred_grasps_cam"].item()
scores = data["scores"].item()
contacts = data["contact_pts"].item()

print("Loaded grasp predictions for object IDs:", list(grasps.keys()))

# ------------------------------------------------------------------
# Load original depth data and intrinsics
# ------------------------------------------------------------------
pc_data = np.load(scene_path, allow_pickle=True).item()
depth = pc_data["depth"]
K = pc_data["K"]
pc_points = depth_to_point_cloud(depth, K)
print(f"Point cloud shape: {pc_points.shape}")

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_points)
pcd.paint_uniform_color([0.5, 0.5, 0.5])  # grey base


# ------------------------------------------------------------------
# Optional zoom: crop point cloud around its center
# ------------------------------------------------------------------
pts_np = np.asarray(pcd.points)
center = [0.0, 0, 0.3]  # center around the actual points

# Size of the zoom box (meters) – tweak as needed
dx, dy, dz = 2.30, 2.30, 2.30

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

geometries = [pcd_cropped]


# ------------------------------------------------------------------
# Add coordinate frames for predicted grasps
# (X=red, Y=green, Z=blue by default in Open3D)
# ------------------------------------------------------------------
for obj_id in grasps.keys():
    g_mats = grasps[obj_id]
    sc = scores[obj_id]

    # Sort by score, highest first
    top_idxs = np.argsort(-sc)  # you can slice [:N] if you want fewer
    top_idxs = top_idxs[:15]

    for i in top_idxs:
        g = g_mats[i]
        T = np.eye(4)
        T[:4, :4] = g
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(T)
        # IMPORTANT: do NOT paint_uniform_color here
        # so that X=red, Y=green, Z=blue remain intact.
        geometries.append(frame)

# obj_id = 5.0
# g_mats = grasps[obj_id]
# sc = scores[obj_id]

# # Sort by score, highest first
# top_idxs = np.argsort(-sc)  # you can slice [:N] if you want fewer
# top_idxs = top_idxs[:15]

# for i in top_idxs:
#     g = g_mats[i]
#     T = np.eye(4)
#     T[:4, :4] = g
#     frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
#     frame.transform(T)
#     # IMPORTANT: do NOT paint_uniform_color here
#     # so that X=red, Y=green, Z=blue remain intact.
#     geometries.append(frame)


# ------------------------------------------------------------------
# Custom draw: set view & save screenshot
# ------------------------------------------------------------------
def custom_draw(vis: o3d.visualization.Visualizer):
    """
    This callback runs once when the window is created.

    It:
      - sets a nicer default view (avoid upside-down scene),
      - saves a screenshot to `save_image_path`,
      - and then returns False to close the window.
    """
    ctr = vis.get_view_control()

    # Set a reasonable view:
    #   - look at the point cloud center
    #   - front points roughly from +Y towards -Z (tweak if needed)
    #   - up is +Z
    # ctr.set_lookat([0, 0, 0.3])
    # ctr.set_front([0.1, -0.8, 1.0])  # direction camera looks towards
    # ctr.set_up([0, 0, -1])       # "up" direction
    # ctr.set_zoom(1)                 # zoom factor (tweak as needed)

    vis.update_renderer()
    vis.capture_screen_image(save_image_path, do_render=True)
    

    # Return False to stop after one frame,
    # or True if you want the window to stay interactive.
    return True



o3d.visualization.draw_geometries(geometries)

# Render with our custom callback
# o3d.visualization.draw_geometries_with_animation_callback(
#     geometries,
#     custom_draw
# )

print(f"[INFO] Saved screenshot to: {save_image_path}")