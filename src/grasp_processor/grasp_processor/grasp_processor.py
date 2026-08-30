#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory

# extra math stuff
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# ROS2 stuff
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray
from sensor_msgs_py import point_cloud2
from grasp_interface.msg import Grasps

# Approximate time synchronizer libraries
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge

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
            rgb_topic = '/depth_camera/color/image'
            depth_topic = 'depth_camera/depth/image'
            pts_topic = '/depth_camera/points'
        else:
            rgb_topic = '/camera/camera/color/image_raw'
            depth_topic = '/camera/depth/color/points'
        
        # Output publishers configurations
        self.grasp_pub = self.create_publisher(Grasps, '/predicted_grasps', 10)
        self.get_logger().info('Grasp Processor active.')

        self.seg_pub = self.create_publisher(Image, '/segmentation', 10) # we publish the segmentation map for viz

        # TODO: get real intrinsics from the camera
        # these are assumed based on the expected values for a D435 @ 1280x720
        self.fx, self.fy = 1230.0, 1230.0
        self.cx, self.cy = 640.0, 360.0

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
        uois_net_3d = segmentation.UOISNet3D(uois3d_config, 
                                            dsn_filename,
                                            dsn_config,
                                            rrn_filename,
                                            rrn_config
                                            )

    def synchronized_scene_callback(self, cloud_msg: PointCloud2, rgb_msg: Image, depth_msg: Image):
        self.get_logger().info('Received synchronized point cloud, RGB, and depth data.')
        self.model.eval()

        #### PARSE DATA
        try:
            # Parse ROS PointCloud2 to an Nx3 numpy array (ignoring RGB/Intensity data fields)
            # NOTE: this is an unorganized pc, it's faster to convert depth img --> organized pc, which is why we subscribe to both pcd and depth data
            # we use unorganized pcd for CGN and organized pcd for UOIS
            cloud_gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
            pcd = np.array(list(cloud_gen), dtype=np.float32)
            
            # Convert incoming RGB and depth data to np arrays
            rgb_data = np.frombuffer(rgb_msg.data, dtype=np.uint8).reshape(rgb_msg.height, rgb_msg.width)
            dep_data = np.frombuffer(depth_msg.data, dtype=np.uint8).reshape(depth_msg.height, depth_msg.width)
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
            # then pass to UOIS
            seg_mask = self._get_segmentation_mask(rgb_data, organized_pcd)
            mask = np.asarray(seg_mask)
            
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
            self.get_logger().error(f"In-memory CGN PyTorch inference crash: {str(e)}")


    def _get_segmentation_mask(self, rgb: np.array, xyz: np.array):
        """Generate segmentation mask using UOIS
        Args:
            rgb: np.array (HxWx3) containing the RGB data from the current frame
            xyz: np.array (HxWx3) containing the depth information (organized pt cloud) from the current frame

        Returns:
            seg_mask: np.array containing the segmentation data
        """

        N = 1   # NOTE: we could modify this later to generate segmentation masks in bulk from a buffer, this would probably be faster
        rgb_imgs = np.zeros((N, rgb.shape[0], rgb.shape[1], 3))
        xyz_imgs = np.zeros((N, rgb.shape[0], rgb.shape[1], 3))
        rgb_imgs[0] = rgb
        xyz_imgs[0] = xyz

        batch = {
            'rgb' : data_augmentation.array_to_tensor(rgb_imgs),
            'xyz' : data_augmentation.array_to_tensor(xyz_imgs),
        }
        fg_masks, center_offsets, initial_masks, seg_masks = uois_net_3d.run_on_batch(batch)
        seg_masks = seg_masks.cpu().numpy()

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
        points, pred_grasps, confidence, pred_widths, _, pred_collide = cgn(
            pcd[:, 3:],
            pos=pcd[:, :3],
            batch=batch,
            idx=idx
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
        # Ensure depth map is 2D (H, W)
        if depth_map.ndim == 3:
            depth_map = depth_map.squeeze(-1)
            
        h, w = depth_map.shape

        # Create a 2D grid of pixel coordinates (u, v)
        # indexing='xy' ensures u corresponds to columns (width) and v to rows (height)
        u, v = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')

        # Compute X and Y matrices using the standard pinhole camera formula
        x_coords = (u - cx) * depth_map / fx
        y_coords = (v - cy) * depth_map / fy

        # Stack along the third axis to get an organized (H, W, 3) matrix
        organized_pc = np.stack((x_coords, y_coords, depth_map), axis=-1)

        return organized_pc

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
