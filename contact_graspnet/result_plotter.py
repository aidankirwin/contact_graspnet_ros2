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


def make_faded_axis_lines(T, size=0.025):
    """
    Make a small, faded-looking coordinate-frame line set.

    Open3D's legacy draw_geometries() does not reliably support alpha
    transparency for these simple geometries, so the non-top grasp frames are
    drawn smaller with lighter RGB colors to visually de-emphasize them.
    """
    T = np.asarray(T, dtype=np.float64)
    origin = T[:3, 3]
    x_end = origin + size * T[:3, 0]
    y_end = origin + size * T[:3, 1]
    z_end = origin + size * T[:3, 2]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(
        np.vstack([origin, x_end, y_end, z_end])
    )
    line_set.lines = o3d.utility.Vector2iVector(
        np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
    )

    # Faded red/green/blue axis colors. This approximates "half transparent"
    # while staying compatible with Open3D's default legacy visualizer.
    line_set.colors = o3d.utility.Vector3dVector(
        np.array([
            [0.70, 0.35, 0.35],  # faded X/red
            [0.35, 0.70, 0.35],  # faded Y/green
            [0.35, 0.35, 0.70],  # faded Z/blue
        ], dtype=np.float64)
    )
    return line_set


# ------------------------------------------------------------------
# Paths (adjust as you like)
# ------------------------------------------------------------------
npz_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/predictions_scene_from_ucn.npz"
scene_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/scene_from_ucn.npy"
save_image_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/results/scene_from_ucn_plot.png"

# ------------------------------------------------------------------
# Visualization options
# ------------------------------------------------------------------
MAX_GRASPS_PER_OBJECT = 15
TOP_FRAME_SIZE = 0.060
OTHER_FRAME_SIZE = 0.025
SHOW_OTHER_GRASPS = True


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
# Add coordinate frames for predicted grasps.
#
# Top grasp:
#   - full Open3D coordinate frame, normal size/color
# Other high-score grasps:
#   - smaller, faded-looking line frames
#
# The list is sorted by score descending for visualization, so index 0 in
# ordered_idxs is the highest-score grasp for that object.
# ------------------------------------------------------------------
for obj_id in grasps.keys():
    g_mats = grasps[obj_id]
    sc = np.asarray(scores[obj_id], dtype=np.float64)

    if len(g_mats) == 0 or len(sc) == 0:
        print(f"[WARN] object {obj_id} has zero predicted grasps; skipping visualization for this object.")
        continue

    ordered_idxs = np.argsort(-sc)
    ordered_idxs = ordered_idxs[:min(MAX_GRASPS_PER_OBJECT, len(ordered_idxs))]

    if len(ordered_idxs) == 0:
        continue

    top_idx = int(ordered_idxs[0])
    top_T = np.eye(4)
    top_T[:4, :4] = np.asarray(g_mats[top_idx], dtype=np.float64)

    top_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=TOP_FRAME_SIZE)
    top_frame.transform(top_T)
    geometries.append(top_frame)

    print(
        f"[INFO] Object {obj_id}: top grasp index={top_idx}, "
        f"score={float(sc[top_idx]):.6f}; showing {len(ordered_idxs)} grasps total."
    )

    if SHOW_OTHER_GRASPS:
        for idx in ordered_idxs[1:]:
            idx = int(idx)
            T = np.eye(4)
            T[:4, :4] = np.asarray(g_mats[idx], dtype=np.float64)
            geometries.append(make_faded_axis_lines(T, size=OTHER_FRAME_SIZE))


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
    # ctr.set_zoom(1)              # zoom factor (tweak as needed)

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
