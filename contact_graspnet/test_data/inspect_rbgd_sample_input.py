import numpy as np, json, os

zero = np.load('./0.npy', allow_pickle=True)
print("0.npy repr type:", type(zero))
print("0.npy scalar:", bool(zero.shape == ()))
if zero.shape == ():
    obj = zero.item()
    print("inner type:", type(obj))
    if isinstance(obj, dict):
        print("dict keys:", list(obj.keys()))
        for k,v in obj.items():
            if isinstance(v, np.ndarray):
                print(" ", k, "shape", v.shape, "dtype", v.dtype)
            else:
                print(" ", k, "type", type(v))
else:
    print("0.npy shape:", zero.shape, "dtype:", zero.dtype)

sample = np.load('./sample_scene_ucn/output/segmentation_from_rgbd/sample.npz')
print("sample.npz keys:", sample.files)
for k in sample.files:
    arr = sample[k]
    print("  ", k, arr.shape, arr.dtype)

im_label = np.load('./sample_scene_ucn/output/segmentation_from_rgbd/im_label.npy')
print("im_label shape:", im_label.shape, "dtype:", im_label.dtype)

with open('./sample_scene_ucn/output/segmentation_from_rgbd/segmentation.json') as f:
    seg_json = json.load(f)
print("segmentation.json keys:", seg_json.keys())
print("  num_objects:", seg_json.get("num_objects"))

obj = np.load('./0.npy', allow_pickle=True).item()
print("0.npy depth range:", float(obj['depth'].min()), float(obj['depth'].max()))
print("0.npy seg unique:", np.unique(obj['seg'])[:10])

im_color_depth = np.load('./sample_scene_ucn/output/segmentation_from_rgbd/sample.npz')
depth_ours = im_color_depth['depth'][0,0]  # guess first channel is depth?
print("sample depth range:", float(depth_ours.min()), float(depth_ours.max()))
