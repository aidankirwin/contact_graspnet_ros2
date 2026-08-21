#!/usr/bin/env python3
import os
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from std_msgs.msg import Int32MultiArray
from sensor_msgs_py import point_cloud2

# Approximate time synchronizer libraries
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge

# Direct in-memory import of the PyTorch Contact-GraspNet API
from cgn_pytorch import ContactGraspNet

# NOTE: this is mostly AI garbage

class GraspProcessor(Node):
    def __init__(self):
        super().__init__('grasp_processor')
        self.bridge = CvBridge()
        
        # Output publishers configurations
        self.grasp_pub = self.create_publisher(PoseArray, '/predicted_grasps', 10)
        self.obj_id_pub = self.create_publisher(Int32MultiArray, '/predicted_grasp_object_ids', 10)
        self.get_logger().info('Grasp Processor active with Object ID Mapping. Synchronizing input streams...')

        # TODO: get real intrinsics from the camera
        # these are assumed based on the expected values for a D435 @ 1280x720
        self.fx, self.fy = 1230.0, 1230.0
        self.cx, self.cy = 640.0, 360.0

        # Define localized message filters for real-time tracking streams
        # RealSense topic is typically /camera/camera/depth/color/points or /camera/depth/color/points
        self.cloud_sub = Subscriber(self, PointCloud2, '/camera/depth/color/points')
        self.seg_sub = Subscriber(self, Image, '/camera/segmentation/mask')


        # Synchronize depth channels and mask frames within a 0.1-second window
        self.sync = ApproximateTimeSynchronizer(
            [self.cloud_sub, self.seg_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_scene_callback)

        # Initialize PyTorch device and model directly in memory
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Loading cgn_pytorch onto device: {self.device}")
        
        # Initialize network weights
        self.model = ContactGraspNet().to(self.device)
        self.model.eval()

    def depth_and_seg_to_point_cloud(self, depth: np.ndarray, mask: np.ndarray):
        """
        Converts depth map and segmentation mask into an Nx3 point cloud, 
        and extracts an Nx1 array representing the corresponding Object ID for each point.
        """
        # TODO: what's going on here

        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        
        # Match only parts of the scene containing valid depth info
        valid_indices = (mask > 0) & (depth > 0)
        
        if not np.any(valid_indices):
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.int32)

        z = depth[valid_indices]
        x = (u[valid_indices] - self.cx) * z / self.fx
        y = (v[valid_indices] - self.cy) * z / self.fy
        
        pts = np.vstack((x, y, z)).T
        point_object_ids = mask[valid_indices]
        
        return pts, point_object_ids

    def synchronized_scene_callback(self, cloud_msg: PointCloud2, seg_msg: Image):
        self.get_logger().info('Received synchronized point cloud and segmentation frame pairs.')

        try:
            # 1. Parse ROS PointCloud2 to an Nx3 numpy array (ignoring RGB/Intensity data fields)
            cloud_gen = point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
            pts = np.array(list(cloud_gen), dtype=np.float32)
            
            # 2. Convert incoming segmentation mask to a clean matrix
            # Using custom mapping if cv_bridge isn't used, or basic mapping for raw data
            # Assuming seg_msg data parsing via manual buffer conversion to avoid cv_bridge dependencies
            seg_np = np.frombuffer(seg_msg.data, dtype=np.uint8).reshape(seg_msg.height, seg_msg.width)

        except Exception as e:
            self.get_logger().error(f'Failed parsing input messages: {str(e)}')
            return

        if pts.shape[0] == 0:
            self.get_logger().warn("Empty point cloud received. Skipping frame.")
            return

        # 3. Project 3D points back to 2D image coordinates to map them against the segmentation mask
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # Prevent division by zero
        valid_z = z > 0.01
        pts = pts[valid_z]
        x, y, z = x[valid_z], y[valid_z], z[valid_z]

        u = np.round((x * self.fx / z) + self.cx).astype(int)
        v = np.round((y * self.fy / z) + self.cy).astype(int)

        # 4. Filter out points that project outside the image bounds
        valid_pixel = (u >= 0) & (u < seg_msg.width) & (v >= 0) & (v < seg_msg.height)
        pts = pts[valid_pixel]
        u = u[valid_pixel]
        v = v[valid_pixel]

        # 5. Extract Object IDs from the segmentation mask using the projected pixels
        point_object_ids = seg_np[v, u]

        # 6. Filter point cloud to ONLY contain segmented objects (Object ID > 0)
        mask_indices = point_object_ids > 0
        pts = pts[mask_indices]
        point_object_ids = point_object_ids[mask_indices]

        if pts.shape[0] == 0:
            self.get_logger().warn("No 3D points matched your target segmentation mask. Skipping.")
            return

        # 7. Downsample cloud to fit CGN expectations safely (~20k points max)
        if pts.shape[0] > 20000:
            indices = np.random.choice(pts.shape[0], 20000, replace=False)
            pts = pts[indices]
            point_object_ids = point_object_ids[indices]

        # Convert to PyTorch Tensor adding batch size [1, N, 3]
        cloud_tensor = torch.from_numpy(pts).unsqueeze(0).to(self.device)

        # Direct native model inference execution
        try:
            with torch.no_grad():
                predictions = self.model(cloud_tensor)
                
            # Extract raw predictions onto system CPU memory
            grasps_matrices = predictions['grasp_poses'].squeeze(0).cpu().numpy() 
            scores = predictions['grasp_scores'].squeeze(0).cpu().numpy()        

            # Filter out poor choices using a confidence threshold
            valid_mask = scores > 0.40
            grasps_matrices = grasps_matrices[valid_mask]
            scores = scores[valid_mask]
            object_ids = point_object_ids[valid_mask]

            if len(scores) == 0:
                self.get_logger().warn("No grasps met the confidence threshold.")
                return

            # Sort the remaining poses by score in descending order (highest score first)
            sorted_indices = np.argsort(scores)[::-1]
            grasps_matrices = grasps_matrices[sorted_indices]
            scores = scores[sorted_indices]
            object_ids = object_ids[sorted_indices]  

            # Construct and publish output PoseArray
            grasp_poses = PoseArray()
            grasp_poses.header = cloud_msg.header # Keeps the frame context (e.g. "camera_depth_optical_frame")
            
            for T in grasps_matrices:
                pose = Pose()
                pose.position.x = float(T[0, 3])
                pose.position.y = float(T[1, 3])
                pose.position.z = float(T[2, 3])
                
                q = R.from_matrix(T[:3, :3]).as_quat(scalar_first=False) 
                pose.orientation.x = q[0]
                pose.orientation.y = q[1]
                pose.orientation.z = q[2]
                pose.orientation.w = q[3]
                
                grasp_poses.poses.append(pose)
            
            # Construct and publish the mapping Int32MultiArray
            obj_id_msg = Int32MultiArray()
            obj_id_msg.data = [int(x) for x in object_ids]
            
            # Synchronous message publishing loop
            self.grasp_pub.publish(grasp_poses)
            self.obj_id_pub.publish(obj_id_msg)
            
            self.get_logger().info(
                f'Published {len(grasp_poses.poses)} poses and matching object IDs from native cloud. '
                f'Highest Score: {scores[0]:.2f}, Lowest Score: {scores[-1]:.2f}'
            )

        except Exception as e:
            self.get_logger().error(f'In-memory CGN PyTorch inference crash: {str(e)}')

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
