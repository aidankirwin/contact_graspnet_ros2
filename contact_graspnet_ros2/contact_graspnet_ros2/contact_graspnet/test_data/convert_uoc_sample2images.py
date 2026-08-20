#!/usr/bin/env python3
"""
Convert a .npy file (storing a Python dict) into:
  - RGB image (PNG)
  - depth image (16-bit PNG; depth stored in millimeters; 0 can represent invalid)
  - JSON metadata (intrinsics K if present, filenames, depth scale)

Typical .npy content (dict keys): rgb, depth, K, seg, ...
"""

import argparse
import json
import os
import numpy as np
from PIL import Image


def load_npy_dict(npy_path: str) -> dict:
    data = np.load(npy_path, allow_pickle=True)
    if isinstance(data, np.ndarray) and data.shape == () and data.dtype == object:
        data = data.item()
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict stored in {npy_path}, got {type(data)}")
    return data


def save_rgb(rgb: np.ndarray, out_path: str) -> None:
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb).save(out_path)


def save_depth_as_16bit_png(depth: np.ndarray, out_path: str, depth_unit: str = "m") -> dict:
    """
    Saves a float depth array as uint16 PNG in millimeters.

    Returns a dict describing the encoding:
      - depth_scale_to_meters: meters per integer unit in PNG (default 0.001)
      - depth_units_in_png: descriptive string
    """
    if depth_unit.lower() in ["m", "meter", "meters"]:
        depth_mm = np.rint(depth * 1000.0)
    elif depth_unit.lower() in ["mm", "millimeter", "millimeters"]:
        depth_mm = np.rint(depth)
    else:
        raise ValueError("depth_unit must be 'm' or 'mm'")

    depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)

    img = Image.fromarray(depth_mm)  # 16-bit PNG
    img.save(out_path)

    return {
        "depth_encoding": "16-bit PNG",
        "depth_units_in_png": "mm (uint16 units)",
        "depth_scale_to_meters": 0.001,  # depth_meters = png_value * 0.001
    }


def convert(npy_path: str, out_dir: str, stem: str | None, depth_unit: str) -> tuple[str, str, str]:
    data = load_npy_dict(npy_path)

    if "rgb" not in data or "depth" not in data:
        raise ValueError(f"Expected keys 'rgb' and 'depth'. Found keys: {list(data.keys())}")

    rgb = data["rgb"]
    depth = data["depth"]
    K = data.get("K", None)

    if stem is None:
        stem = os.path.splitext(os.path.basename(npy_path))[0]

    os.makedirs(out_dir, exist_ok=True)

    rgb_path = os.path.join(out_dir, f"{stem}-color.png")
    depth_path = os.path.join(out_dir, f"{stem}-depth.png")
    json_path = os.path.join(out_dir, f"{stem}-meta.json")

    save_rgb(rgb, rgb_path)
    depth_info = save_depth_as_16bit_png(depth, depth_path, depth_unit=depth_unit)

    h, w = depth.shape[:2]
    meta = {
        "source_npy": os.path.abspath(npy_path),
        "width": int(w),
        "height": int(h),
        "rgb_file": os.path.basename(rgb_path),
        "depth_file": os.path.basename(depth_path),
        "depth_min_m": float(np.min(depth)),
        "depth_max_m": float(np.max(depth)),
        **depth_info,
    }
    if K is not None:
        meta["K"] = np.asarray(K).tolist()

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return rgb_path, depth_path, json_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy_path", default='0.npy',help="Name of the .npy file")
    ap.add_argument("--out_dir", default='./image_form', help="Output directory (default: same folder as npy)")
    ap.add_argument("--stem", default=None, help="Output filename stem (default: npy basename)")
    ap.add_argument("--depth_unit", default="m", choices=["m", "mm"], help="Unit of depth array stored in the npy")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.npy_path))
    rgb_path, depth_path, json_path = convert(args.npy_path, out_dir, args.stem, args.depth_unit)

    print("Wrote:")
    print("  RGB  :", rgb_path)
    print("  Depth:", depth_path)
    print("  Meta :", json_path)


if __name__ == "__main__":
    main()