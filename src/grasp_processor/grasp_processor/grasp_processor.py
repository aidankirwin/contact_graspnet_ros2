#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

# extra math stuff
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import random
from torch_geometric.nn import fps

# ROS2 stuff
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray
from grasp_interface.msg import Grasps
from sensor_msgs.msg import CameraInfo

# Approximate time synchronizer libraries
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge
import cv2

# To time processing
import threading
import time

# PyTorch Contact-GraspNet API
import cgn_pytorch

# UOIS (object segmentation)
import grasp_processor.uois.src.data_augmentation as data_augmentation
import grasp_processor.uois.src.segmentation as segmentation
import grasp_processor.uois.src.evaluation as evaluation
import grasp_processor.uois.src.util.utilities as util_
import grasp_processor.uois.src.util.flowlib as flowlib

class GraspProcessor(Node):
    def __init__(self):
        super().__init__('grasp_processor')
        self.bridge = CvBridge()

        self.save_plots = True
        self.plt_ctr = 0

        # the topic names are slightly different in sim so grab gazebo parameter
        self.declare_parameter('is_gazebo', 'true')
        self.is_gazebo = self.get_parameter('is_gazebo').get_parameter_value().string_value
        self.get_logger().info(f'Received argument: {self.is_gazebo}')

        # TODO: need to check these topic names with real camera
        if self.is_gazebo == 'true':
            rgb_topic = '/depth_camera/image'
            depth_topic = 'depth_camera/depth_image'
        else:
            rgb_topic = '/camera/camera/color/image_raw'
            depth_topic = '/camera/depth/color/depth_raw'
        
        # Output publishers configurations
        self.grasp_pub = self.create_publisher(Grasps, '/predicted_grasps', 10)
        self.get_logger().info('Grasp Processor active.')

        self.seg_pub = self.create_publisher(Image, '/segmentation', 10) # we publish the segmentation map for viz

        #### Camera data subscribers
        self.camera_info_sub = self.create_subscription(CameraInfo, '/depth_camera/camera_info', self.camera_info_callback, 10)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Define localized message filters for real-time tracking streams
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.rgb_sub = Subscriber(self, Image, rgb_topic)

        # Synchronize depth channels and mask frames within a 0.1-second window
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_scene_callback)

        #### CGN SETUP
        torch.cuda.empty_cache()
        # NOTE: currently I don't know what the optimizer or config_dict are needed for
        # also from_pretrained handles the torch.device('cuda' if torch.cuda.is_available() else 'cpu') line
        # from_pretrained doesn't print anything so below we check if cuda is available, but self.device isn't used anywhere
        # Initialize PyTorch device and model directly in memory
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Loading cgn_pytorch onto device: {self.device}")
        self.model, optimizer, config_dict  = cgn_pytorch.from_pretrained()

        #### UOIS SETUP
        dsn_config = {
            # Feature Dimensions: controls the dimensionality of the learned feature representation used by the network
            #    Potentially better separation of visually/geometrically similar objects
            #    More model computation and memory
            #    Potentially useful if seeing objects being merged
            #    Doesn't directly control the size or number of clusters
            'feature_dim' : 64, # 32 would be normal

            # Mean Shift parameters (for 3D voting)
            'max_GMS_iters' : 10, 

            # Epsilon: controls the spatial tolerance used when determining whether things belong to the same 
            # connected component/cluster after the voting stage
            #    smaller epsilon --> stricter connectivity --> more likely to split things apart
            #    larger epsilon --> looser connectivity --> more likely to merge things together
            'epsilon' : 0.07, # Connected Components parameter

            # Sigma: the Gaussian bandwidth used during mean shift, essentially determines how much influence neighboring votes have
            #    smaller sigma --> better separation of nearby objects, more fragmented objects, more sensitivity to noisy predictions
            #    larger sigma --> smoother clustering, more robust to noise, more likely to merge nearby objects
            'sigma' : 0.05, # Gaussian bandwidth parameter

            'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
            'subsample_factor' : 5,
            
            # Minimum Pixel Threshold: controls min number of pixels for a cluster to be considered a valid object
            'min_pixels_thresh' : 500,

            'tau' : 15.,
        }
        rrn_config = {
            # Sizes
            'feature_dim' : 64, # 32 would be normal
            'img_H' : 224,
            'img_W' : 224,
            
            # architecture parameters
            'use_coordconv' : False,
        }
        uois3d_config = {
            # Padding for RGB Refinement Network
            'padding_percentage' : 0.25,
            
            # Open/Close Morphology for IMP (Initial Mask Processing) module
            'use_open_close_morphology' : True,
            'open_close_morphology_ksize' : 9,
            
            # Largest Connected Component for IMP module
            'use_largest_connected_component' : True,  
        }
        checkpoint_dir = get_package_share_directory('grasp_processor') + '/uois_model/'
        dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
        rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
        uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
        self.uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                            dsn_filename,
                                            dsn_config,
                                            rrn_filename,
                                            rrn_config
                                            )
        self.get_logger().info(f"UOIS configured")

        ### WORKER THREAD SETUP
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.processing = False

        self.inference_thread = threading.Thread(
            target=self._inference_worker,
            daemon=True
        )
        self.inference_thread.start()
        self.get_logger().info(f"Inference thread started")


    def synchronized_scene_callback(self, rgb_msg: Image, depth_msg: Image):
        with self.frame_lock:
            self.latest_frame = (
                rgb_msg,
                depth_msg
            )
        # self.get_logger().info('TESTING: RECEIVED DATA.')

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

        self.get_logger().info(
            f"Camera intrinsics: "
            f"fx={self.fx:.2f}, fy={self.fy:.2f}, "
            f"cx={self.cx:.2f}, cy={self.cy:.2f}"
        )

        # No longer need CameraInfo
        self.destroy_subscription(self.camera_info_sub)
        self.camera_info_sub = None

    def _inference_worker(self):
        while rclpy.ok():
            with self.frame_lock:
                frame = self.latest_frame
                self.latest_frame = None
            if frame is None:
                time.sleep(0.001)
                continue

            rgb_msg, depth_msg = frame

            try:
                self._process_frame(rgb_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(
                    f'Inference failed: {e}'
                )

    def _process_frame(self, rgb_msg, depth_msg):
        self.get_logger().info('Processing data.')

        if (self.fx is None) or (self.fy is None) or (self.cx is None) or (self.cy is None):
            self.get_logger().info('Camera intrinsics not yet set.')
            return

        self.model.eval()

        #### PARSE DATA
        try:
            # Convert incoming RGB and depth data to np arrays
            rgb_data = np.frombuffer(rgb_msg.data, dtype=np.uint8 ).reshape(rgb_msg.height, rgb_msg.width, 3)
            # 32FC1 encoding
            dep_data = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width)
            dep_data = dep_data.astype(np.float32)

            # Replace NaN / +/-inf with 0
            dep_data[~np.isfinite(dep_data)] = 0.0

            self.get_logger().info(
                f"Depth: shape={dep_data.shape}, "
                f"dtype={dep_data.dtype}, "
                f"min={np.nanmin(dep_data)}, "
                f"max={np.nanmax(dep_data)}, "
                f"finite={np.isfinite(dep_data).all()}, "
                f"valid={(np.isfinite(dep_data) & (dep_data > 0)).sum()}"
            )

        except Exception as e:
            self.get_logger().error(f'Failed parsing input messages: {str(e)}')
            return

        #### GET SEGMENTATION MASK
        try:
            # first we will convert dep_np to an organized pt cloud
            organized_pcd = self._depth_to_organized_pc(dep_data, self.fx, self.fy, self.cx, self.cy)

            # NOTE: for now let's just assume the camera data is already 480x640
            # return a warning if it isn't
            if rgb_data.shape[0] != 480 or rgb_data.shape[1] != 640:
                self.get_logger().warn(
                    f"Input RGB image is not 480x640: "
                    f"height={rgb_data.shape[0]}, width={rgb_data.shape[1]}"
                )
            # resize everything to 480x640
            # target_h, target_w = 480, 640
            # rgb_data = cv2.resize(rgb_data, (target_w, target_h), interpolation=cv2.INTER_LINEAR,).astype(np.float32)
            # organized_pcd = cv2.resize(organized_pcd, (target_w, target_h), interpolation=cv2.INTER_NEAREST,).astype(np.float32)
            rgb_data = rgb_data.astype(np.float32)
            organized_pcd = organized_pcd.astype(np.float32)
            # remove nans from organized_pcd
            organized_pcd = np.nan_to_num(organized_pcd, nan=0.0)
            pcd = organized_pcd.reshape(-1, 3)
            pcd = pcd[np.isfinite(pcd).all(axis=1)]

            # then pass to UOIS
            seg_mask = self._get_segmentation_mask(rgb_data, organized_pcd)
            mask = np.asarray(seg_mask)

            self.get_logger().info(
                f"UOIS mask: shape={mask.shape}, "
                f"dtype={mask.dtype}, "
                f"min={mask.min()}, "
                f"max={mask.max()}, "
                f"unique={np.unique(mask)}"
            )

            # then reshape mask so it can be used by cgn
            # (H, W, 1) -> (H, W)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]
            # (H, W) -> (H*W,)
            mask = mask.reshape(-1)

        except Exception as e:
            self.get_logger().error(f'Failed segmentation: {str(e)}')
            return   

        #### GENERATE GRASPS
        try:
            grasps_matrices, scores, object_ids, _ = self._cgn_infer(pcd, mask)

            # Log grasps info
            self.get_logger().info(
                f"CGN generated {len(grasps_matrices)} grasps, "
                f"Highest score: {scores.max():.2f}, "
                f"Grasp matrices shape: {grasps_matrices.shape}, "
                f"Object IDs: {np.unique(object_ids).tolist()}"
            )

            # sort by confidence
            sorted_indices = np.argsort(scores)[::-1]

            grasps_matrices = grasps_matrices[sorted_indices]
            scores = scores[sorted_indices]
            object_ids = object_ids[sorted_indices]

            # construct grasps msg
            grasp_msg = Grasps()

            for T, score, object_id in zip(grasps_matrices, scores, object_ids):
                pose = Pose()

                # Position
                pose.position.x = float(T[0, 3])
                pose.position.y = float(T[1, 3])
                pose.position.z = float(T[2, 3])

                # Orientation
                q = R.from_matrix(
                    T[:3, :3]
                ).as_quat()

                pose.orientation.x = float(q[0])
                pose.orientation.y = float(q[1])
                pose.orientation.z = float(q[2])
                pose.orientation.w = float(q[3])

                # Add aligned data
                grasp_msg.poses.append(pose)
                grasp_msg.scores.append(float(score))
                grasp_msg.object_ids.append(int(object_id))

            self.grasp_pub.publish(grasp_msg)
            self.get_logger().info(
                f"Published {len(grasp_msg.poses)} grasps. "
                f"Highest score: {scores[0]:.2f}, "
                f"Lowest score: {scores[-1]:.2f}, "
                f"Object IDs: {np.unique(object_ids).tolist()}"
            )

            if self.save_plots:
                self._save_grasp_visualization(
                    rgb=rgb_data,
                    grasps_matrices=grasps_matrices,
                    scores=scores,
                    object_ids=object_ids,
                    output_path=f"grasps_{self.plt_ctr}.png"
                )
                self.plt_ctr += 1

        except Exception as e:
            self.get_logger().error(f"CGN inference crash: {str(e)}")


    def _get_segmentation_mask(self, rgb: np.array, xyz: np.array):
        """Generate segmentation mask using UOIS.

        Args:
            rgb: np.array (HxWx3) containing the RGB data from the current frame
            xyz: np.array (HxWx3) containing the depth information
                (organized pt cloud) from the current frame

        Returns:
            seg_mask: np.array containing the segmentation data in the
                original image shape.
        """

        # remove nans from xyz
        xyz = np.nan_to_num(xyz, nan=0.0)

        self.get_logger().info(
            f"RGB: shape={rgb.shape}, dtype={rgb.dtype}, "
            f"min={rgb.min()}, max={rgb.max()}"
        )

        self.get_logger().info(
            f"XYZ: shape={xyz.shape}, dtype={xyz.dtype}, "
            f"min={np.nanmin(xyz)}, max={np.nanmax(xyz)}, "
            f"finite={np.isfinite(xyz).all()}"
        )

        self.get_logger().info(
            f"XYZ NaNs: {np.isnan(xyz).sum()}, "
            f"XYZ infs: {np.isinf(xyz).sum()}"
        )

        N = 1
        rgb_imgs = np.zeros((N, rgb.shape[0], rgb.shape[1], 3), dtype=np.float32)
        xyz_imgs = np.zeros((N, xyz.shape[0], xyz.shape[1], 3), dtype=np.float32)

        rgb_imgs[0] = data_augmentation.standardize_image(rgb.astype(np.float32))
        xyz_imgs[0] = xyz.astype(np.float32)

        batch = {
            'rgb': data_augmentation.array_to_tensor(rgb_imgs),
            'xyz': data_augmentation.array_to_tensor(xyz_imgs),
        }

        if self.save_plots:
            # save the batch to a temporary file for debugging
            np.savez(f'batch_debug_{self.plt_ctr}.npz', rgb=rgb_imgs, xyz=xyz_imgs)

            # also save the rgb and xyz images to image files for debugging
            cv2.imwrite(f'rgb_debug_{self.plt_ctr}.png', rgb)
            # save xyz as a 3-channel image for debugging
            xyz_debug = (xyz - np.nanmin(xyz)) / (np.nanmax(xyz) - np.nanmin(xyz)) * 255
            xyz_debug = np.nan_to_num(xyz_debug, nan=0.0)
            xyz_debug = xyz_debug.astype(np.uint8)
            cv2.imwrite(f'xyz_debug_{self.plt_ctr}.png', xyz_debug)

        fg_masks, center_offsets, initial_masks, seg_masks = (self.uois_net_3d.run_on_batch(batch))
        seg_masks = seg_masks.cpu().numpy()

        self.get_logger().info(
            f"UOIS raw seg_masks: "
            f"shape={seg_masks.shape}, "
            f"dtype={seg_masks.dtype}, "
            f"min={seg_masks.min()}, "
            f"max={seg_masks.max()}, "
            f"unique={np.unique(seg_masks)[:20]}"
        )

        # extract the single mask and resize it back to the original resolution
        seg_mask = seg_masks[0]
        # seg_mask = cv2.resize(seg_mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

        if self.save_plots:
            # also save the segmentation mask to an image file for debugging
            seg_mask_debug = (seg_mask / seg_mask.max() * 255).astype(np.uint8)
            # overlay object labels
            seg_img = cv2.cvtColor(seg_mask_debug, cv2.COLOR_RGB2BGR)
            for label in np.unique(seg_masks):
                if label == 0:
                    continue

                # Get pixels belonging to this object
                ys, xs = np.where(seg_mask == label)

                if len(xs) == 0:
                    continue

                # Compute centroid
                cx = int(xs.mean())
                cy = int(ys.mean())

                # Draw label
                cv2.putText(
                    seg_img,
                    f"Object {label}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                
            cv2.imwrite(f'seg_mask_debug_{self.plt_ctr}.png', seg_img)

        return seg_mask


    def _cgn_infer(self, pcd, obj_mask=None, threshold=0.5):
        # adapted from https://github.com/sebjperalta/cgn_pytorch/blob/main/eval.py 
        cgn = self.model
        cgn.eval()

        # The model should work on any pointcloud of shape (Nx3). 
        # For most consistent results, please make sure to put the pointcloud in the world frame and center it by subtracting the mean.
        # Do not normalize the pointcloud to a unit sphere or unit box, as "graspability" naturally changes depending on the size of the objects 
        # (so we don't want to lose that information about the scene by scaling it).
        # pcd = pcd - np.mean(pcd, axis=0, keepdims=True)

        if pcd.shape[0] > 20000:
            downsample = np.array(
                random.sample(range(pcd.shape[0]), 20000)
            )
        else:
            downsample = np.arange(pcd.shape[0])

        pcd = pcd[downsample, :]
        pcd = torch.as_tensor(
            pcd,
            dtype=torch.float32,
            device=cgn.device
        )
        batch = torch.zeros(
            pcd.shape[0],
            dtype=torch.int64,
            device=cgn.device
        )
        idx = fps(
            pcd,
            batch,
            2048 / pcd.shape[0]
        )

        if obj_mask is not None:
            # obj_mask should be shape (original_num_points,)
            object_ids = torch.as_tensor(
                obj_mask[downsample],
                dtype=torch.int64,
                device=cgn.device
            )
            # Keep only the object ID corresponding to each FPS point
            object_ids = object_ids[idx]
        else:
            object_ids = torch.ones(
                idx.shape[0],
                dtype=torch.int64,
                device=cgn.device
            )
        
        # RUN CGN
        gripper_depth = 0.1034
        gripper_width = 0.08
        points, pred_grasps, confidence, pred_widths, _, _ = cgn(
            pcd[:, 3:],
            pcd_poses=pcd[:, :3],
            batch=batch,
            idxs=idx,
            gripper_depth=gripper_depth,
            gripper_width=gripper_width,
        )

        confidence = torch.sigmoid(confidence)
        # Expected shape: confidence = (num_points, num_grasps_per_point)
        #
        # Flatten it to match flattened pred_grasps.
        confidence = confidence.reshape(-1)

        pred_grasps = torch.flatten(
            pred_grasps,
            start_dim=0,
            end_dim=1
        )

        num_grasps = pred_grasps.shape[0]
        num_points = object_ids.shape[0]

        if num_grasps % num_points != 0:
            raise RuntimeError(
                f"Cannot associate object IDs with grasps: "
                f"{num_grasps} grasps for {num_points} points."
            )

        grasps_per_point = num_grasps // num_points
        object_ids = torch.repeat_interleave(
            object_ids,
            grasps_per_point
        )

        # only allow grasps belonging to segmented objects
        valid_object = object_ids > 0
        confidence[~valid_object] = 0.0

        # convert to numpy
        pred_grasps = pred_grasps.detach().cpu().numpy()
        confidence = confidence.detach().cpu().numpy()
        object_ids = object_ids.detach().cpu().numpy()

        # confidence threshold
        success_mask = confidence > threshold

        if not np.any(success_mask):
            self.get_logger().warn(
                "CGN failed to find successful grasps."
            )
            raise Exception("No successful grasps found")

        pred_grasps = pred_grasps[success_mask]
        confidence = confidence[success_mask]
        object_ids = object_ids[success_mask]

        return (
            pred_grasps,
            confidence,
            object_ids,
            downsample
        )

    def _depth_to_organized_pc(self, depth_map, fx, fy, cx, cy):
        """
        Converts a depth map into an organized point cloud of shape (H, W, 3).
        
        Parameters:
        depth_map (np.ndarray): HxW or HxWx1 float array (depth in meters).
        fx, fy (float): Camera focal lengths from camera_info.
        cx, cy (float): Camera principal point (optical center) from camera_info.
        """
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)

        depth_map = depth_map.astype(np.float32)

        h, w = depth_map.shape

        u, v = np.meshgrid(
            np.arange(w, dtype=np.float32),
            np.arange(h, dtype=np.float32),
            indexing='xy'
        )

        valid = np.isfinite(depth_map) & (depth_map > 0.0)

        x = (u - cx) * depth_map / fx
        y = (v - cy) * depth_map / fy
        z = depth_map

        # Mark invalid depth as invalid XYZ
        x[~valid] = np.nan
        y[~valid] = np.nan
        z[~valid] = np.nan

        return np.stack((x, y, z), axis=-1)

    def _save_grasp_visualization(
        self,
        rgb: np.ndarray,
        grasps_matrices: np.ndarray,
        scores: np.ndarray,
        object_ids: np.ndarray,
        output_path: str,
        axis_length: float = 0.05,
    ):
        """Visualize 3D grasp poses projected onto an RGB image and save as PNG.

        Args:
            rgb: RGB image, shape (H, W, 3).
            grasps_matrices: Grasp transforms, shape (N, 4, 4).
                Assumed to be expressed in the camera optical frame.
            scores: Grasp scores, shape (N,).
            object_ids: Object ID associated with each grasp, shape (N,).
            output_path: Path to save the PNG.
            axis_length: Length of grasp-frame axes in meters.

        Grasp frame visualization:
            X axis -> red
            Y axis -> green
            Z axis -> blue
        
        NOTE: this is mostly AI generated code and may not be correct. 
        the logic appears to check out though: project grasp poses into the image using the camera intrinsics, then draw them onto the image
        """

        ## PREPARE IMAGE
        # OpenCV expects BGR for drawing/saving.
        if rgb.dtype != np.uint8:
            # Handle float images in either [0, 1] or [0, 255].
            if rgb.max() <= 1.0:
                rgb_vis = (rgb * 255.0).clip(0, 255).astype(np.uint8)
            else:
                rgb_vis = rgb.clip(0, 255).astype(np.uint8)
        else:
            rgb_vis = rgb.copy()

        vis = cv2.cvtColor(rgb_vis, cv2.COLOR_RGB2BGR)
        image_h, image_w = vis.shape[:2]

        ## PROJECT POINTS INTO IMAGE
        def project_point(point):
            x, y, z = point

            # Point is behind camera or on camera plane.
            if z <= 0:
                return None

            u = self.fx * x / z + self.cx
            v = self.fy * y / z + self.cy

            return int(round(u)), int(round(v))

        ## DRAW GRASPS
        for grasp_idx, (T, score, object_id) in enumerate(zip(grasps_matrices, scores, object_ids)):
            # Grasp position in camera frame
            position = T[:3, 3]

            p0 = project_point(position)

            if p0 is None:
                self.get_logger().warn(
                    f"Grasp {grasp_idx} projected behind camera: "
                    f"position={position}"
                )
                continue

            u, v = p0

            # dont draw if grasp position is outside image
            if not (0 <= u < image_w and 0 <= v < image_h):
                self.get_logger().warn(
                    f"Grasp {grasp_idx} projected outside image: "
                    f"u={u}, v={v}, image_w={image_w}, image_h={image_h}"
                )
                continue

            # draw grasp center
            cv2.circle(
                vis,
                p0,
                radius=6,
                color=(0, 255, 255),  # Yellow
                thickness=-1,
            )

            cv2.circle(
                vis,
                p0,
                radius=8,
                color=(0, 0, 0),
                thickness=2,
            )

            # draw coordinate frame
            rotation = T[:3, :3]
            axes = [
                (rotation[:, 0], (0, 0, 255)),   # X -> red
                (rotation[:, 1], (0, 255, 0)),   # Y -> green
                (rotation[:, 2], (255, 0, 0)),   # Z -> blue
            ]

            for axis, color in axes:
                endpoint_3d = position + axis * axis_length
                p1 = project_point(endpoint_3d)

                if p1 is None:
                    continue

                cv2.arrowedLine(
                    vis,
                    p0,
                    p1,
                    color,
                    thickness=2,
                    tipLength=0.2,
                )

            # draw label
            label = f"obj={int(object_id)} score={float(score):.3f}"

            text_x = u + 10
            text_y = v - 10

            # Keep text inside image reasonably
            text_x = max(0, min(text_x, image_w - 200))
            text_y = max(20, text_y)

            # Black background for readability
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1,
            )

            cv2.rectangle(
                vis,
                (text_x - 2, text_y - text_h - baseline - 2),
                (text_x + text_w + 2, text_y + 2),
                (0, 0, 0),
                thickness=-1,
            )

            cv2.putText(
                vis,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # save png
        success = cv2.imwrite(output_path, vis)

        if not success:
            raise RuntimeError(f"Failed to save grasp visualization to: {output_path}")

        self.get_logger().info(
            f"Saved grasp visualization with "
            f"{len(grasps_matrices)} grasps to {output_path}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GraspProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
