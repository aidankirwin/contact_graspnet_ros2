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
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray
from sensor_msgs_py import point_cloud2
from grasp_interface.msg import Grasps
from sensor_msgs.msg import CameraInfo

# Approximate time synchronizer libraries
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge

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

        # the topic names are slightly different in sim so grab gazebo parameter
        self.declare_parameter('is_gazebo', 'true')
        self.is_gazebo = self.get_parameter('is_gazebo').get_parameter_value().string_value
        self.get_logger().info(f'Received argument: {self.is_gazebo}')

        # TODO: need to check these topic names with real camera
        if self.is_gazebo == 'true':
            rgb_topic = '/depth_camera/image'
            depth_topic = 'depth_camera/depth_image'
            pts_topic = '/depth_camera/points'
        else:
            rgb_topic = '/camera/camera/color/image_raw'
            depth_topic = '/camera/depth/color/depth_raw'
            pts_topic = '/camera/depth/color/points'
        
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
        self.cloud_sub = Subscriber(self, PointCloud2, pts_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.rgb_sub = Subscriber(self, Image, rgb_topic)

        # Synchronize depth channels and mask frames within a 0.1-second window
        self.sync = ApproximateTimeSynchronizer(
            [self.cloud_sub, self.rgb_sub, self.depth_sub], 
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
            # Sizes
            'feature_dim' : 64, # 32 would be normal

            # Mean Shift parameters (for 3D voting)
            'max_GMS_iters' : 10, 
            'epsilon' : 0.05, # Connected Components parameter
            'sigma' : 0.02, # Gaussian bandwidth parameter
            'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
            'subsample_factor' : 5,
            
            # Misc
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

        ### FOR TESTING
        self.organized_pcd_pub = self.create_publisher(
            PointCloud2,
            '/grasp_processor/organized_pcd',
            10
        )

        self.unorganized_pcd_pub = self.create_publisher(
            PointCloud2,
            '/grasp_processor/unorganized_pcd',
            10
        )


    def synchronized_scene_callback(self, cloud_msg: PointCloud2, rgb_msg: Image, depth_msg: Image):
        with self.frame_lock:
            self.latest_frame = (
                cloud_msg,
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

            cloud_msg, rgb_msg, depth_msg = frame

            try:
                self._process_frame(cloud_msg, rgb_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(
                    f'Inference failed: {e}'
                )

    def _publish_xyz_cloud(self, xyz, header, publisher, organized=False):
        """
        Publish an XYZ numpy array as PointCloud2.

        xyz:
            Organized:   (H, W, 3)
            Unorganized: (N, 3)

        Invalid XYZ values should be NaN/inf and will be preserved.
        """

        xyz = np.asarray(xyz, dtype=np.float32)

        if organized:
            if xyz.ndim != 3 or xyz.shape[-1] != 3:
                raise ValueError(
                    f"Expected organized XYZ shape (H,W,3), got {xyz.shape}"
                )

            points = xyz.reshape(-1, 3)

        else:
            if xyz.ndim != 2 or xyz.shape[-1] != 3:
                raise ValueError(
                    f"Expected unorganized XYZ shape (N,3), got {xyz.shape}"
                )

            points = xyz

        # create_cloud_xyz32 accepts NaNs, which is what we want for
        # invalid pixels in the organized cloud.
        msg = point_cloud2.create_cloud_xyz32(
            header,
            points.tolist()
        )

        publisher.publish(msg)


    def _process_frame(self, cloud_msg, rgb_msg, depth_msg):
        self.get_logger().info('Processing data.')

        if (self.fx is None) or (self.fy is None) or (self.cx is None) or (self.cy is None):
            self.get_logger().info('Camera intrinsics not yet set.')
            return

        self.model.eval()

        #### PARSE DATA
        try:
            # Parse ROS PointCloud2 to an Nx3 numpy array (ignoring RGB/Intensity data fields)
            # NOTE: this is an unorganized pc, it's faster to convert depth img --> organized pc, which is why we subscribe to both pcd and depth data
            # we use unorganized pcd for CGN and organized pcd for UOIS
            cloud = point_cloud2.read_points(
                cloud_msg,
                field_names=("x", "y", "z"),
                skip_nans=True
            )
            pcd = np.column_stack((
                cloud["x"],
                cloud["y"],
                cloud["z"]
            )).astype(np.float32)

            self._publish_xyz_cloud(
                pcd,
                cloud_msg.header,
                self.unorganized_pcd_pub,
                organized=False
            )
            
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

        if pcd.shape[0] == 0:
            self.get_logger().warn("Empty point cloud received. Skipping frame.")
            return

        #### GET SEGMENTATION MASK
        try:
            # first we will convert dep_np to an organized pt cloud
            organized_pcd = self._depth_to_organized_pc(dep_data, self.fx, self.fy, self.cx, self.cy)
            self._publish_xyz_cloud(
                organized_pcd,
                depth_msg.header,
                self.organized_pcd_pub,
                organized=True
            )

            valid_xyz = np.isfinite(organized_pcd).all(axis=-1) & (organized_pcd[..., 2] > 0)

            self.get_logger().info(
                f"Valid XYZ pixels: {valid_xyz.sum()} / {valid_xyz.size}"
            )

            if valid_xyz.any():
                xyz_valid = organized_pcd[valid_xyz]

                self.get_logger().info(
                    f"Valid XYZ range: "
                    f"X=[{xyz_valid[:,0].min():.3f}, {xyz_valid[:,0].max():.3f}], "
                    f"Y=[{xyz_valid[:,1].min():.3f}, {xyz_valid[:,1].max():.3f}], "
                    f"Z=[{xyz_valid[:,2].min():.3f}, {xyz_valid[:,2].max():.3f}]"
                )

            # then pass to UOIS
            seg_mask = self._get_segmentation_mask(rgb_data, organized_pcd)
            mask = np.asarray(seg_mask)
            
            # then reshape mask so it can be used by cgn
            # (H, W, 1) -> (H, W)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask[..., 0]

            self.get_logger().info(
                f"UOIS mask: shape={mask.shape}, "
                f"dtype={mask.dtype}, "
                f"min={mask.min()}, "
                f"max={mask.max()}, "
                f"unique={np.unique(mask)}"
            )

            # Anything belonging to an object becomes white
            mask_viz = (mask > 0).astype(np.uint8) * 255
            mask_msg = self.bridge.cv2_to_imgmsg(
                mask_viz,
                encoding='mono8'
            )
            mask_msg.header = cloud_msg.header
            self.seg_pub.publish(mask_msg)

            # (H, W) -> (H*W,)
            mask = mask.reshape(-1)

        except Exception as e:
            self.get_logger().error(f'Failed segmentation: {str(e)}')
            return   

        #### GENERATE GRASPS
        try:
            grasps_matrices, scores, object_ids, _ = self._cgn_infer(pcd, mask)

            # sort by confience
            sorted_indices = np.argsort(scores)[::-1]

            grasps_matrices = grasps_matrices[sorted_indices]
            scores = scores[sorted_indices]
            object_ids = object_ids[sorted_indices]

            # construct grasps msg
            grasp_msg = Grasps()
            grasp_msg.header = cloud_msg.header

            for T, score, object_id in zip(grasps_matrices, scores, object_ids):
                pose = Pose()

                # Position
                pose.position.x = float(T[0, 3])
                pose.position.y = float(T[1, 3])
                pose.position.z = float(T[2, 3])

                # Orientation
                q = R.from_matrix(
                    T[:3, :3]
                ).as_quat(scalar_first=False)

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

        except Exception as e:
            self.get_logger().error(f"CGN inference crash: {str(e)}")


    def _get_segmentation_mask(self, rgb: np.array, xyz: np.array):
        """Generate segmentation mask using UOIS
        Args:
            rgb: np.array (HxWx3) containing the RGB data from the current frame
            xyz: np.array (HxWx3) containing the depth information (organized pt cloud) from the current frame

        Returns:
            seg_mask: np.array containing the segmentation data
        """

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

        N = 1   # NOTE: we could modify this later to generate segmentation masks in bulk from a buffer, this would probably be faster
        rgb_imgs = np.zeros((N, rgb.shape[0], rgb.shape[1], 3))
        xyz_imgs = np.zeros((N, rgb.shape[0], rgb.shape[1], 3))
        rgb_imgs[0] = data_augmentation.standardize_image(rgb)
        xyz_imgs[0] = xyz

        batch = {
            'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
            'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
        }
        fg_masks, center_offsets, initial_masks, seg_masks = self.uois_net_3d.run_on_batch(batch)
        seg_masks = seg_masks.cpu().numpy()

        self.get_logger().info(
            f"UOIS raw seg_masks: "
            f"shape={seg_masks.shape}, "
            f"dtype={seg_masks.dtype}, "
            f"min={seg_masks.min()}, "
            f"max={seg_masks.max()}, "
            f"unique={np.unique(seg_masks)[:20]}"
        )

        return seg_masks[0]

    def _cgn_infer(self, pcd, obj_mask=None, threshold=0.5):
        # adapted from https://github.com/sebjperalta/cgn_pytorch/blob/main/eval.py 
        cgn = self.model
        cgn.eval()

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
