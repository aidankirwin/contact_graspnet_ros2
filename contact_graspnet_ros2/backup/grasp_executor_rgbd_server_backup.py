#!/usr/bin/env python3


import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose, Point
from contact_graspnet_ros2.srv import GetGrasps
from contact_graspnet_ros2.msg import Grasps
# TF rotations
import tf_transformations as tfs
import os
import subprocess
import json

import tf2_ros
import tf_transformations as tft
from geometry_msgs.msg import Pose, PoseArray


class GraspServer_Backup(Node):
    def __init__(self):
        super().__init__('grasp_server')
        self.srv = self.create_service(GetGrasps, 'get_grasps_back', self.handle_grasp_request)
        self.get_logger().info('Grasp server ready (executing inference inside a docker container).')
        self.base_path = "/home/csrobot/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet"
        self.result_loading = "_use_json" #["_use_json", "_use_npz"]
        # self.result_loading = "_use_npz" 

        self.base_frame = 'panda_link0'
        self.camera_frame = 'rgbd_camera/camera_link/rgbd_camera'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


    def run_inference_in_docker(self,scene_name):
        container_name = "contact_graspnet_container"
        # container_name = "magical_lovelace"
        np_path = f"test_data/{scene_name}.npy"

        # cmd = [
        #     "docker", "exec", container_name,
        #     "bash", "-lc",
        #     f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
        #     f"conda run -n contact-graspnet bash compile_pointnet_tfops.sh && "
        #     f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
        #     f"conda run -n contact-graspnet python contact_graspnet/inference.py --np_path={np_path} --local_regions --filter_grasps"
        # ]

         # The shared object we expect if tf_ops are compiled
        compiled_lib = "/root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/pointnet2/tf_ops/sampling/tf_sampling_so.so"

        compile_cmd = (
            f"if [ ! -f {compiled_lib} ]; then "
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet bash compile_pointnet_tfops.sh; "
            f"fi"
        )

        inference_cmd = (
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet python contact_graspnet/inference.py "
            f"--np_path={np_path} --local_regions --filter_grasps" # --json_out
        )

        cmd = [
            "docker", "exec", container_name,
            "bash", "-lc", f"{compile_cmd} && {inference_cmd}"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Inference failed: {result.stderr}")
        # return result.stdout

        start_marker = "<<<BEGIN_JSON>>>"
        end_marker = "<<<END_JSON>>>"

        # Extract the JSON block from stdout
        json_text = None
        # First try: marker-based extraction
        start = result.stdout.find(start_marker)
        end = result.stdout.find(end_marker, start)
        if start != -1 and end != -1:
            json_text = result.stdout[start+len(start_marker):end].strip()
            self.get_logger().info("Extracted JSON using markers.")
        else:
            # Fallback: scan line by line for something that looks like JSON
            for line in result.stdout.splitlines():
                if line.strip().startswith("{") and line.strip().endswith("}"):
                    json_text = line.strip()
                    break

        if json_text is None:
            # Log some of the noisy output for debugging
            self.get_logger().error(f"No JSON found in inference output.\nFirst 500 chars:\n{result.stdout[:500]}")
            raise RuntimeError("Inference did not return valid JSON")

        return json_text


    def transform_pose_array(self, pose_array, from_frame, to_frame):
        """
        Transform a PoseArray from 'from_frame' to 'to_frame' using TF2.
        Returns a new PoseArray in the target frame; if TF fails, returns the input.
        """
        try:
            # Latest available transform
            t = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time(),  # 0 time = latest
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup {to_frame} <- {from_frame} failed: {e}")
            return pose_array

        trans = t.transform.translation
        rot = t.transform.rotation

        # 4x4 transform matrix base<-camera
        T_bc = tft.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        T_bc[0, 3] = trans.x
        T_bc[1, 3] = trans.y
        T_bc[2, 3] = trans.z

        out = PoseArray()
        out.header.frame_id = to_frame
        out.header.stamp = pose_array.header.stamp

        for p in pose_array.poses:
            # Pose in camera frame as 4x4
            q = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
            T_cp = tft.quaternion_matrix(q)
            T_cp[0, 3] = p.position.x - 0.5
            T_cp[1, 3] = p.position.y
            T_cp[2, 3] = p.position.z + 0.75

            T_bp = T_bc @ T_cp  # base <- camera <- pose

            pos = T_bp[:3, 3]
            q_bp = tft.quaternion_from_matrix(T_bp)

            p_out = Pose()
            p_out.position.x, p_out.position.y, p_out.position.z = pos
            p_out.orientation.x = q_bp[0]
            p_out.orientation.y = q_bp[1]
            p_out.orientation.z = q_bp[2]
            p_out.orientation.w = q_bp[3]

            out.poses.append(p_out)

        return out


    def handle_grasp_request(self, request, response):
        self.scene_name = request.scene_name
        self.get_logger().info(f"Running inference in Docker for scene {self.scene_name}...")

        output = self.run_inference_in_docker(self.scene_name)
        self.get_logger().info("Inference finished")

        # ---------------------------------------------------
        # Load inference results (JSON or NPZ)
        # ---------------------------------------------------
        if self.result_loading == "_use_json":
            # Save stdout for debugging
            with open(f"{self.base_path}/results/inference_output_{self.scene_name}.txt", "w") as f:
                f.write(output)
            self.get_logger().info(
                f"Saved raw inference output to "
                f"{self.base_path}/results/inference_output_{self.scene_name}.txt"
            )

            results = json.loads(output)
            pred_grasps_cam = {k: [np.array(g) for g in v]
                               for k, v in results["pred_grasps_cam"].items()}
            scores = {k: np.array(v) for k, v in results["scores"].items()}
            contact_pts = {k: np.array(v) for k, v in results["contact_pts"].items()}

            self.get_logger().info(
                f"Received grasp results from docker for scene {self.scene_name}:"
            )

        elif self.result_loading == "_use_npz":
            result_path = os.path.join(
                self.base_path, "results", f"predictions_{self.scene_name}.npz"
            )
            data = np.load(result_path, allow_pickle=True)
            pred_grasps_cam = data["pred_grasps_cam"].item()
            scores = data["scores"].item()
            contact_pts = data["contact_pts"].item()

            self.get_logger().info(
                f"Loaded grasp results from docker for scene {self.scene_name}:"
            )

        # ---------------------------------------------------
        # Build PoseArray in camera frame
        # ---------------------------------------------------
        grasps_cam_pa = PoseArray()
        grasps_cam_pa.header.frame_id = self.camera_frame
        grasps_cam_pa.header.stamp = self.get_clock().now().to_msg()

        score_list = []
        sample_list = []
        object_list = []

        for obj_id in pred_grasps_cam.keys():
            for pose, score, sample in zip(
                pred_grasps_cam[obj_id], scores[obj_id], contact_pts[obj_id]
            ):
                ros_pose = Pose()
                ros_pose.position.x = float(pose[0, 3])
                ros_pose.position.y = float(pose[1, 3])
                ros_pose.position.z = float(pose[2, 3])

                quat = tfs.quaternion_from_matrix(pose)
                ros_pose.orientation.x = float(quat[0])
                ros_pose.orientation.y = float(quat[1])
                ros_pose.orientation.z = float(quat[2])
                ros_pose.orientation.w = float(quat[3])

                grasps_cam_pa.poses.append(ros_pose)
                score_list.append(float(score))
                sample_list.append(
                    Point(
                        x=float(sample[0]),
                        y=float(sample[1]),
                        z=float(sample[2]),
                    )
                )
                object_list.append(int(float(obj_id)))

            self.get_logger().info(
                f"Obtained {len(scores[obj_id])} grasps for object {obj_id}"
            )

        # ---------------------------------------------------
        # Transform PoseArray cam -> base using TF2
        # ---------------------------------------------------
        grasps_in_base_pa = self.transform_pose_array(
            grasps_cam_pa,
            from_frame=self.camera_frame,
            to_frame=self.base_frame,
        )

        # ---------------------------------------------------
        # Fill Grasps msg (poses now in base frame)
        # ---------------------------------------------------
        grasps_msg = Grasps()
        grasps_msg.poses = list(grasps_in_base_pa.poses)
        grasps_msg.scores = score_list
        grasps_msg.samples = sample_list
        grasps_msg.object_ids = object_list

        response.grasps = grasps_msg

        self.get_logger().info(
            f"Responded with {len(grasps_msg.poses)} grasps in frame "
            f"'{self.base_frame}' for scene {self.scene_name}"
        )

        return response



def main(args=None):
    rclpy.init(args=args)
    server = GraspServer()
    rclpy.spin(server)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
