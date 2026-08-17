#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert UCN segmentation outputs + RGBD images into a Contact-GraspNet-style
"scene" file, and provide helpers to build XYZ images / masked point clouds.

Inputs (default names, can be overridden via CLI):
  - from_rgbd-color.png  : RGB image (BGR PNG from Gazebo / segmentation_rgbd)
  - from_rgbd-depth.png  : depth PNG (uint16 mm)
  - im_label.npy         : segmentation labels (H,W) int32
  - segmentation.json    : optional; contains "instance_ids" (same as im_label)
  - sample.npz           : optional; UCN debug sample (not required here)

Output:
  - scene_from_ucn.npy   : dict with keys ['rgb','depth','K','seg']
      'rgb'   : (H,W,3) uint8, RGB
      'depth' : (H,W) float32, meters
      'K'     : (3,3) float64 intrinsics
      'seg'   : (H,W) float32 labels (instance ids)

Helpers:
  - depth_to_xyz(depth_m, K) -> xyz_img (H,W,3)
  - scene_to_masked_points(scene_dict, target_id=None, background_id=0)
      -> xyz_img (H,W,3), pts_xyz (N,3), used_id
"""

import argparse
import json
import os

from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def load_rgb_depth(rgb_path: str, depth_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load RGB and depth PNGs and convert to:
      - rgb: (H,W,3) uint8, RGB
      - depth_m: (H,W) float32, meters  (assumes uint16 mm in the PNG)
    """
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB file not found: {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth file not found: {depth_path}")

    # RGB comes from OpenCV as BGR
    rgb_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Failed to load RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"Failed to load depth image: {depth_path}")
    depth_raw = depth_raw.astype(np.float32)

    # Assume uint16 in millimeters → meters
    depth_m = depth_raw / 1000.0

    return rgb, depth_m


def load_segmentation(
    im_label_path: str,
    seg_json_path: Optional[str] = None
) -> np.ndarray:
    """
    Load segmentation labels as (H,W) float32.

    Preference:
      1) im_label.npy if present
      2) segmentation.json["instance_ids"]
    """
    if os.path.exists(im_label_path):
        seg = np.load(im_label_path)
        seg = seg.astype(np.float32)
        return seg

    if seg_json_path is None:
        raise FileNotFoundError("No im_label.npy or segmentation.json provided")

    with open(seg_json_path, "r") as f:
        data = json.load(f)
    if "instance_ids" not in data:
        raise KeyError("segmentation.json missing 'instance_ids' key")
    seg = np.array(data["instance_ids"], dtype=np.float32)
    return seg


def build_K(
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> np.ndarray:
    """
    Construct a 3x3 intrinsics matrix from fx, fy, cx, cy.
    """
    K = np.array(
        [[fx, 0.0, cx],
         [0.0, fy, cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return K


def depth_to_xyz(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project a depth image (meters) into camera-centric XYZ coordinates.

    Args:
      depth_m: (H,W) float32, meters
      K:      (3,3) intrinsics, with fx = K[0,0], fy = K[1,1],
              cx = K[0,2], cy = K[1,2]

    Returns:
      xyz_img: (H,W,3) float32, where xyz_img[v,u,:] = (X,Y,Z) in meters.
               Pixels with depth <= 0 have xyz = (0,0,0).
    """
    H, W = depth_m.shape
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    v_coords, u_coords = np.indices((H, W), dtype=np.float32)  # v = row (y), u = col (x)

    Z = depth_m
    X = (u_coords - cx) * Z / fx
    Y = (v_coords - cy) * Z / fy

    xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)

    # Optionally: zero out invalid depths
    invalid = Z <= 0
    xyz[invalid] = 0.0

    return xyz


def scene_to_masked_points(
    scene: dict,
    target_id: Optional[int] = None,
    background_id: int = 0
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Given a scene dict {'rgb','depth','K','seg'}, compute:
      - xyz_img: (H,W,3) organized XYZ
      - pts_xyz: (N,3) masked points for Contact-GraspNet (N≥0)
      - used_id: instance id actually used for the mask

    If target_id is None or <= background_id, automatically choose the
    non-background instance label with the largest number of pixels.
    """
    rgb = scene["rgb"]
    depth = scene["depth"]
    K = scene["K"]
    seg = scene["seg"]

    if seg.shape != depth.shape:
        raise ValueError(f"seg shape {seg.shape} != depth shape {depth.shape}")

    seg_int = seg.astype(np.int32)
    xyz_img = depth_to_xyz(depth, K)

    # Choose instance id
    unique_ids = [int(v) for v in np.unique(seg_int) if v != background_id]
    if not unique_ids:
        # no foreground; return empty
        return xyz_img, np.zeros((0, 3), dtype=np.float32), background_id

    if target_id is not None and target_id > background_id and target_id in unique_ids:
        used_id = target_id
    else:
        # pick largest area
        best_id = None
        best_area = -1
        for iid in unique_ids:
            area = int(np.count_nonzero(seg_int == iid))
            if area > best_area:
                best_area = area
                best_id = iid
        used_id = best_id

    mask_inst = (seg_int == used_id)
    valid_depth = depth > 0
    mask = mask_inst & valid_depth

    pts_xyz = xyz_img[mask].reshape(-1, 3)

    return xyz_img, pts_xyz, used_id


# ---------------------------------------------------------------------------
# Main conversion: UCN outputs → scene_from_ucn.npy
# ---------------------------------------------------------------------------

def build_scene_from_ucn(
    rgb_path: str,
    depth_path: str,
    im_label_path: str,
    seg_json_path: Optional[str],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    out_path: str
) -> dict:
    """
    High-level helper:
      1) Load RGB + depth PNGs
      2) Load segmentation labels
      3) Build K
      4) Pack dict {'rgb','depth','K','seg'}
      5) Save to out_path (scene_from_ucn.npy)

    Returns:
      scene dict (same object that is saved)
    """
    rgb, depth_m = load_rgb_depth(rgb_path, depth_path)
    seg = load_segmentation(im_label_path, seg_json_path)

    if seg.shape != depth_m.shape:
        raise ValueError(
            f"Segmentation shape {seg.shape} does not match depth shape {depth_m.shape}"
        )
    if rgb.shape[:2] != depth_m.shape:
        raise ValueError(
            f"RGB shape {rgb.shape[:2]} does not match depth shape {depth_m.shape}"
        )

    K = build_K(fx, fy, cx, cy)

    scene = {
        "rgb": rgb.astype(np.uint8),
        "depth": depth_m.astype(np.float32),
        "K": K.astype(np.float64),
        "seg": seg.astype(np.float32),
    }

    np.save(out_path, scene)
    print(f"[INFO] Saved scene dict to {out_path}")
    return scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert UCN outputs + RGBD into Contact-GraspNet-style scene_from_ucn.npy"
    )

    unseen_obj_clst_seg_path = os.path.expanduser("~/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd") # "/home/csrobot/graspnet_ws/src/unseen_obj_clst_ros2/compare_UnseenObjectClustering/results/segmentation_rgbd"
    cgn_data_path = os.path.expanduser("~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet") # "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/test_data"

    parser.add_argument("--rgb", type=str, default=f"{unseen_obj_clst_seg_path}/input/from_rgbd-color.png",
                        help="Path to RGB color PNG")
    parser.add_argument("--depth", type=str, default=f"{unseen_obj_clst_seg_path}/input/from_rgbd-depth.png",
                        help="Path to depth PNG (uint16 mm)")
    parser.add_argument("--im_label", type=str, default=f"{unseen_obj_clst_seg_path}/output/im_label.npy",
                        help="Path to segmentation labels (H,W) int32")
    parser.add_argument("--seg_json", type=str, default=f"{unseen_obj_clst_seg_path}/output/segmentation.json",
                        help="Path to segmentation JSON (optional)")

    parser.add_argument("--fx", type=float, default= 615.0, #required=True,
                        help="Camera focal length fx")
    parser.add_argument("--fy", type=float, default= 615.0, #required=True,
                        help="Camera focal length fy")
    parser.add_argument("--cx", type=float, default= 320.0, #required=True,
                        help="Principal point cx")
    parser.add_argument("--cy", type=float, default= 240.0, #required=True,
                        help="Principal point cy")

    parser.add_argument("--out_scene", type=str, default=f"{cgn_data_path}/results/scene_from_ucn.npy",
                        help="Output .npy path for scene dict")

    parser.add_argument("--dump_cloud", action="store_true",
                        help="If set, also dump xyz_img.npy and pts_xyz.npy for inspection")
    parser.add_argument("--target_id", type=int, default=-1,
                        help="Target instance id for masking (default: auto-select largest)")

    return parser.parse_args()


def main():
    args = parse_args()

    seg_json_path = args.seg_json if os.path.exists(args.seg_json) else None

    # 1) Build & save scene dict
    scene = build_scene_from_ucn(
        rgb_path=args.rgb,
        depth_path=args.depth,
        im_label_path=args.im_label,
        seg_json_path=seg_json_path,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        out_path=args.out_scene,
    )




    # # --- Load RGB image ---
    # rgb_bgr = cv2.imread('./sample_scene_ucn/sample_3/input/from_rgbd-color.png')      # (480, 640, 3), BGR
    # rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)   # -> RGB uint8, like 0.npy['rgb']

    # # --- Load depth and convert to meters ---
    # depth_raw = cv2.imread('./sample_scene_ucn/sample_3/input/from_rgbd-depth.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
    # # depth_raw is uint16 mm → meters
    # depth_m = depth_raw / 1000.0                     # (480, 640) float32

    # # --- Load segmentation (either im_label.npy or segmentation.json) ---
    # seg = np.load('./sample_scene_ucn/sample_3/output/segmentation_from_rgbd/im_label.npy').astype(np.float32)  # (480, 640) float32; matches json["instance_ids"]

    # # --- Build K (you must fill in fx, fy, cx, cy for your camera) ---
    # fx, fy = 615.0, 615.0   # example values – replace with your camera’s
    # cx, cy = 320.0, 240.0
    # K = np.array([[fx, 0.0, cx],
    #               [0.0, fy, cy],
    #               [0.0, 0.0, 1.0]], dtype=np.float64)

    # # --- Pack into dict matching 0.npy’s structure ---
    # scene_dict = {
    #     'rgb':   rgb,       # HxWx3 uint8
    #     'depth': depth_m,   # HxW float32, meters
    #     'K':     K,         # 3x3 float64
    #     'seg':   seg        # HxW float32 labels
    # }

    # np.save('./scene_from_ucn.npy', scene_dict)


    # 2) Optionally: compute XYZ image + masked Nx3 and dump them for CGN debugging
    if args.dump_cloud:
        xyz_img, pts_xyz, used_id = scene_to_masked_points(
            scene,
            target_id=args.target_id if args.target_id >= 0 else None,
            background_id=0,
        )
        np.save("xyz_img.npy", xyz_img)
        np.save("pts_xyz.npy", pts_xyz)
        print(f"[INFO] Dumped xyz_img.npy (H,W,3) and pts_xyz.npy (N,3), used instance_id = {used_id}")
        print(f"[INFO] pts_xyz shape: {pts_xyz.shape}")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
