import os
import json
import subprocess
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose

# Approximate time synchronizer libraries
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge

class GraspProcessor(Node):
    def __init__(self):
        super().__init__('grasp_processor')
        self.bridge = CvBridge()
        
        # Output publisher topic setup configuration
        self.grasp_pub = self.create_publisher(PoseArray, '/predicted_grasps', 10)
        self.get_logger().info('Grasp Processor active. Synchronizing input streams...')

        # Hardcoded dummy Camera Intrinsics 
        # (Replace these with a dynamic CameraInfo subscriber lookup if preferred)
        self.fx, self.fy = 615.0, 615.0
        self.cx, self.cy = 320.0, 240.0

        # Define localized message filters for real-time tracking streams
        self.depth_sub = Subscriber(self, Image, '/camera/depth/image_raw')
        self.seg_sub = Subscriber(self, Image, '/camera/segmentation/mask')

        # Synchronize depth channels and mask frames within a 0.1-second window
        self.sync = ApproximateTimeSynchronizer(
            [self.depth_sub, self.seg_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.sync.registerCallback(self.synchronized_scene_callback)

    def synchronized_scene_callback(self, depth_msg: Image, seg_msg: Image):
        self.get_logger().info('Received synchronized depth and segmentation frame pairs.')

        try:
            # Convert incoming messages into clean numeric matrices
            cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            cv_seg = self.bridge.imgmsg_to_cv2(seg_msg, desired_encoding='passthrough')
            
            depth_np = np.array(cv_depth, dtype=np.float32)
            seg_np = np.array(cv_seg, dtype=np.int32)

        except Exception as e:
            self.get_logger().error(f'Failed parsing image arrays: {str(e)}')
            return

        # Build payload mapping packet strings
        payload = {
            "depth": depth_np.tolist(),
            "seg_mask": seg_np.tolist(),
            "fx": self.fx, "fy": self.fy,
            "cx": self.cx, "cy": self.cy,
            "local_regions": True,
            "filter_grasps": True
        }

        # Run streaming system processes over isolated input channels
        try:
            output_json = self.stream_to_docker_container(payload)
            results = json.loads(output_json)
            
            # Construct and publish output PoseArray
            grasp_poses = PoseArray()
            grasp_poses.header = depth_msg.header # Bind coordinate timestamp configurations
            
            # Assuming '0' matches the background extraction layer id loop
            if "0" in results:
                grasps_matrices = np.array(results["0"]["grasps"])
                scores = np.array(results["0"]["scores"])
                
                for i in range(len(grasps_matrices)):
                    T = grasps_matrices[i] # 4x4 homogenous matrix transformation 
                    
                    # (Optional) Map coordinate changes using your custom 
                    # frame conversion logic: T = self.cgn_optical_to_ros_cam(T)
                    
                    pose = Pose()
                    pose.position.x = float(T[0, 3])
                    pose.position.y = float(T[1, 3])
                    pose.position.z = float(T[2, 3])
                    
                    # Helper translation logic mapping matrix components to quaternions
                    q = self.matrix_to_quaternion(T[:3, :3])
                    pose.orientation.x = q[0]
                    pose.orientation.y = q[1]
                    pose.orientation.z = q[2]
                    pose.orientation.w = q[3]
                    
                    grasp_poses.poses.append(pose)
            
            self.grasp_pub.publish(grasp_poses)
            self.get_logger().info(f'Published {len(grasp_poses.poses)} evaluated grasps.')

        except Exception as e:
            self.get_logger().error(f'Inference subprocess pipe failed: {str(e)}')

    def stream_to_docker_container(self, payload: dict) -> str:
        json_input_bytes = json.dumps(payload)

        # Call the script inside the container using the interactive (-i) flag
        cmd = [
            "docker", "exec", "-i", "contact_graspnet_container",
            "python3", "/workspace/contact_graspnet/custom_inference.py"
        ]

        result = subprocess.run(
            cmd, 
            input=json_input_bytes, 
            capture_output=True, 
            text=True
        )

        if result.returncode != 0:
            raise RuntimeError(f"Container internal pipeline crash: {result.stderr}")

        start_marker = "<<<BEGIN_JSON>>>"
        end_marker = "<<<END_JSON>>>"

        start = result.stdout.find(start_marker)
        end = result.stdout.find(end_marker, start)
        
        if start != -1 and end != -1:
            return result.stdout[start + len(start_marker):end].strip()
            
        raise RuntimeError("No bounded json data packets found inside system output channels.")

    def matrix_to_quaternion(self, R):
        """Converts a standard 3x3 rotation matrix into a flat unit quaternion array [x,y,z,w]."""
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        
        tr = m00 + m11 + m22
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m21 - m12) / S
            qy = (m02 - m20) / S
            qz = (m10 - m01) / S
        else:
            if (m00 > m11) and (m00 > m22):
                S = np.sqrt(1.0 + m00 - m11 - m22) * 2
                qw = (m21 - m12) / S
                qx = 0.25 * S
                qy = (m01 + m10) / S
                qz = (m02 + m20) / S
            elif m11 > m22:
                S = np.sqrt(1.0 + m11 - m00 - m22) * 2
                qw = (m02 - m20) / S
                qx = (m01 + m10) / S
                qy = 0.25 * S
                qz = (m12 + m21) / S
            else:
                S = np.sqrt(1.0 + m22 - m00 - m11) * 2
                qw = (m10 - m01) / S
                qx = (m02 + m20) / S
                qy = (m12 + m21) / S
                qz = 0.25 * S
        return [qx, qy, qz, qw]

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
