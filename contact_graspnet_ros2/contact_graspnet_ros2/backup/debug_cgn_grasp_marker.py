#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import json
import os
import numpy as np

import tf2_ros
import tf_transformations as tft
import rclpy.duration

from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker


class DebugCgnGraspMarker(Node):
    def __init__(self):
        super().__init__('debug_cgn_grasp_marker')

        # Adjust if needed
        self.base_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet"
        self.scene_name = "scene_from_ucn"
        self.camera_frame = "rgbd_camera/camera_link/rgbd_camera"
        self.base_frame = "panda_link0"  # or "simple_pedestal"

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Marker publisher
        self.marker_pub = self.create_publisher(Marker, "debug_cgn_grasp", 1)

        # Run once after a short delay to give TF some time
        self.timer = self.create_timer(1.0, self.timer_cb)
        self.has_published = False

        self.get_logger().info("DebugCgnGraspMarker node ready.")

    # Same optical->camera conversion as in the server
    def cgn_optical_to_ros_cam(self, T_cgn: np.ndarray) -> np.ndarray:
        R = np.array([
            [0.0,  0.0, 1.0],   # z_opt -> X_cam
            [-1.0, 0.0, 0.0],   # x_opt -> -Y_cam
            [0.0, -1.0, 0.0],   # y_opt -> -Z_cam
        ], dtype=np.float64)

        T_ros = np.eye(4, dtype=np.float64)
        T_ros[:3, :3] = R @ T_cgn[:3, :3]
        T_ros[:3, 3] = R @ T_cgn[:3, 3]
        return T_ros

    def timer_cb(self):
        if self.has_published:
            return

        # --------------------------------------------------
        # Load predictions JSON
        # --------------------------------------------------
        json_path = os.path.join(
            self.base_path, "results", f"predictions_{self.scene_name}.json"
        )
        if not os.path.exists(json_path):
            self.get_logger().error(f"JSON file not found: {json_path}")
            return

        with open(json_path, "r") as f:
            results = json.load(f)

        pred_grasps_cam = {
            k: [np.array(g) for g in v]
            for k, v in results["pred_grasps_cam"].items()
        }

        # Pick the first object and first grasp
        if not pred_grasps_cam:
            self.get_logger().error("No grasps found in JSON.")
            return

        first_obj_id = list(pred_grasps_cam.keys())[0]
        if len(pred_grasps_cam[first_obj_id]) == 0:
            self.get_logger().error(f"No grasps for object {first_obj_id}.")
            return

        T_cgn = pred_grasps_cam[first_obj_id][0]
        self.get_logger().info(f"Using grasp 0 of object {first_obj_id} for debug marker.")

        # --------------------------------------------------
        # Optical -> camera_link
        # --------------------------------------------------
        T_cam = self.cgn_optical_to_ros_cam(T_cgn)

        # Pose in camera frame
        p_cam = Pose()
        p_cam.position.x = float(T_cam[0, 3])
        p_cam.position.y = float(T_cam[1, 3])
        p_cam.position.z = float(T_cam[2, 3])
        q_cam = tft.quaternion_from_matrix(T_cam)
        p_cam.orientation.x = float(q_cam[0])
        p_cam.orientation.y = float(q_cam[1])
        p_cam.orientation.z = float(q_cam[2])
        p_cam.orientation.w = float(q_cam[3])

        # --------------------------------------------------
        # Camera -> base via TF
        # --------------------------------------------------
        try:
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(
                f"TF lookup {self.base_frame} <- {self.camera_frame} failed: {e}"
            )
            return

        trans = t.transform.translation
        rot = t.transform.rotation
        T_bc = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        T_bc[0, 3] = trans.x
        T_bc[1, 3] = trans.y
        T_bc[2, 3] = trans.z

        # T_cp: camera -> pose
        T_cp = tft.quaternion_matrix(
            [p_cam.orientation.x, p_cam.orientation.y,
             p_cam.orientation.z, p_cam.orientation.w]
        )
        T_cp[0, 3] = p_cam.position.x
        T_cp[1, 3] = p_cam.position.y
        T_cp[2, 3] = p_cam.position.z

        T_bp = T_bc @ T_cp

        pos = T_bp[:3, 3]
        q_bp = tft.quaternion_from_matrix(T_bp)

        # --------------------------------------------------
        # Publish marker in base frame
        # --------------------------------------------------
        m = Marker()
        m.header.frame_id = self.base_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "debug_cgn"
        m.id = 0
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.x = float(q_bp[0])
        m.pose.orientation.y = float(q_bp[1])
        m.pose.orientation.z = float(q_bp[2])
        m.pose.orientation.w = float(q_bp[3])
        m.scale.x = 0.03
        m.scale.y = 0.03
        m.scale.z = 0.03
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        m.lifetime = rclpy.duration.Duration(seconds=0.0).to_msg()  # forever

        self.marker_pub.publish(m)
        self.has_published = True

        self.get_logger().info(
            f"Published debug grasp marker at "
            f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}] in frame {self.base_frame}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = DebugCgnGraspMarker()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
