#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import Pose, Point, PoseArray
from contact_graspnet_ros2.srv import GetGrasps
from contact_graspnet_ros2.msg import Grasps

import tf_transformations as tfs
import os
import subprocess
import json
import shlex

import tf2_ros
import tf_transformations as tft
import rclpy.duration


class GraspServer(Node):
    def __init__(self):
        super().__init__('grasp_server')
        self.srv = self.create_service(GetGrasps, '/get_grasps_rgbd', self.handle_grasp_request)
        self.get_logger().info('Grasp server ready (executing inference inside a docker container).')

        # Base path inside the docker container and host
        self.base_path = os.path.expanduser('~/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet')

        # Whether to parse JSON from stdout or load the .npz file directly
        self.result_loading = "_use_json"  # ["_use_json", "_use_npz"]

        # Contact-GraspNet scene input path. For the real UOC->CGN pipeline,
        # scene_from_ucn.npy is now written under contact_graspnet/results/
        # instead of contact_graspnet/test_data/.
        self.declare_parameter('scene_input_dir', 'results')
        self.declare_parameter('cgn_k_path', 'results/K_kinova_640x360.npy')
        self.declare_parameter('retry_without_filter_if_empty', True)
        self.scene_input_dir = str(self.get_parameter('scene_input_dir').value).strip() or 'results'
        self.cgn_k_path = str(self.get_parameter('cgn_k_path').value).strip()
        self.retry_without_filter_if_empty = bool(self.get_parameter('retry_without_filter_if_empty').value)

        # Frames
        # Original simulation frames:
        # self.base_frame = 'simple_pedestal' # 'panda_link0'  #'simple_workstation'  # 
        # self.camera_frame = 'rgbd_camera/camera_link/rgbd_camera'

        # GEN3/real-robot additions:
        # Make the same server usable with the real Kinova camera/TF tree.
        # Recommended starting values for your current Gen3 setup:
        #   base_frame:=base_link
        #   camera_frame:=camera_color_frame or camera_depth_frame, depending on TF and UCN output
        # If camera_frame is an optical frame, set convert_cgn_optical_to_ros_camera_link:=False.
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_color_frame')
        self.declare_parameter('convert_cgn_optical_to_ros_camera_link', False)
        self.declare_parameter('apply_gripper_frame_offset', False)
        self.declare_parameter('gripper_offset_rx_deg', 0.0)
        self.declare_parameter('gripper_offset_ry_deg', 0.0)
        self.declare_parameter('gripper_offset_rz_deg', 0.0)
        self.declare_parameter('apply_scene_replica_xy_swap', False)

        # GEN3/real-robot safety offset:
        # The CGN pose is treated as an end_effector_link target in base_frame.
        # On the real Robotiq gripper, the fingers can extend below/forward from
        # end_effector_link and collide with the table. Add a small positive
        # base-frame Z offset BEFORE returning poses to FlexBE/MoveToPose, so the
        # arm never receives the too-low grasp pose. Tune from launch if needed.
        # Original behavior: no Z offset.
        # self.grasp_base_z_safety_offset = 0.0
        self.declare_parameter('grasp_base_z_safety_offset', 0.0)
        self.declare_parameter('base_x_offset', 0)
        self.declare_parameter('base_y_offset', 0.018)
        self.declare_parameter('strict_scene_mask_filter', True)
        self.declare_parameter('mask_projection_radius_px', 8)
        self.declare_parameter('max_contact_to_mask_dist_m', 0.035)

        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.convert_cgn_optical_to_ros_camera_link = bool(
            self.get_parameter('convert_cgn_optical_to_ros_camera_link').value
        )
        self.apply_gripper_frame_offset = bool(
            self.get_parameter('apply_gripper_frame_offset').value
        )
        self.apply_scene_replica_xy_swap = bool(
            self.get_parameter('apply_scene_replica_xy_swap').value
        )
        self.gripper_offset_rx_deg = float(self.get_parameter('gripper_offset_rx_deg').value)
        self.gripper_offset_ry_deg = float(self.get_parameter('gripper_offset_ry_deg').value)
        self.gripper_offset_rz_deg = float(self.get_parameter('gripper_offset_rz_deg').value)
        self.grasp_base_z_safety_offset = float(
            self.get_parameter('grasp_base_z_safety_offset').value
        )
        self.base_x_offset = float(self.get_parameter('base_x_offset').value)
        self.base_y_offset = float(self.get_parameter('base_y_offset').value)
        self.strict_scene_mask_filter = bool(self.get_parameter('strict_scene_mask_filter').value)
        self.mask_projection_radius_px = int(self.get_parameter('mask_projection_radius_px').value)
        self.max_contact_to_mask_dist_m = float(self.get_parameter('max_contact_to_mask_dist_m').value)

        self.get_logger().info(
            f'Using base_frame={self.base_frame}, camera_frame={self.camera_frame}, '
            f'convert_cgn_optical_to_ros_camera_link={self.convert_cgn_optical_to_ros_camera_link}, '
            f'apply_gripper_frame_offset={self.apply_gripper_frame_offset}, '
            f'grasp_base_z_safety_offset={self.grasp_base_z_safety_offset:.3f} m, '
            f'base_x_offset={self.base_x_offset:.3f} m, base_y_offset={self.base_y_offset:.3f} m'
        )
        self.get_logger().info(
            f"CGN input settings: scene_input_dir='{self.scene_input_dir}', "
            f"cgn_k_path='{self.cgn_k_path}', "
            f"retry_without_filter_if_empty={self.retry_without_filter_if_empty}, "
            f"strict_scene_mask_filter={self.strict_scene_mask_filter}"
        )

        # TF2
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


    # ------------------------------------------------------------------
    # Docker inference
    # ------------------------------------------------------------------
    def _extract_json_from_stdout(self, stdout: str) -> str:
        start_marker = "<<<BEGIN_JSON>>>"
        end_marker = "<<<END_JSON>>>"

        start = stdout.find(start_marker)
        end = stdout.find(end_marker, start)
        if start != -1 and end != -1:
            self.get_logger().info("Extracted JSON using markers.")
            return stdout[start + len(start_marker):end].strip()

        for line in stdout.splitlines():
            if line.strip().startswith("{") and line.strip().endswith("}"):
                return line.strip()

        self.get_logger().error(
            f"No JSON found in inference output.\nFirst 500 chars:\n{stdout[:500]}"
        )
        raise RuntimeError("Inference did not return valid JSON")

    @staticmethod
    def _count_json_grasps(json_text: str) -> int:
        try:
            data = json.loads(json_text)
            pg = data.get("pred_grasps_cam", {})
            return int(sum(len(v) for v in pg.values()))
        except Exception:
            return -1

    def run_inference_in_docker(self, scene_name) -> str:
        container_name = "contact_graspnet_container"

        # Relative path inside contact_graspnet/. The UOC converter now writes
        # scene_from_ucn.npy to results/ by default.
        scene_input_dir = self.scene_input_dir.strip("/ ")
        np_path = f"{scene_input_dir}/{scene_name}.npy"

        # Host-side sanity checks. The Docker container sees the same workspace
        # through the bind mount.
        host_scene_path = os.path.join(self.base_path, np_path)
        if not os.path.exists(host_scene_path):
            self.get_logger().warn(
                f"CGN scene file not found on host before Docker inference: {host_scene_path}"
            )

        if self.cgn_k_path:
            host_k_path = os.path.join(self.base_path, self.cgn_k_path)
            if not os.path.exists(host_k_path):
                self.get_logger().warn(
                    f"CGN K_path not found on host before Docker inference: {host_k_path}. "
                    "inference.py will warn and may fall back to K stored in the scene file."
                )

        compiled_lib = (
            "/root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet/"
            "pointnet2/tf_ops/sampling/tf_sampling_so.so"
        )

        compile_cmd = (
            f"if [ ! -f {compiled_lib} ]; then "
            f"cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
            f"conda run -n contact-graspnet bash compile_pointnet_tfops.sh; "
            f"fi"
        )

        k_arg = ""
        if self.cgn_k_path:
            k_arg = f" --K_path={shlex.quote(self.cgn_k_path)}"

        def _run_once(filter_grasps: bool, tag: str) -> str:
            filter_arg = " --filter_grasps" if filter_grasps else ""
            inference_cmd = (
                "cd /root/graspnet_ws/src/contact_graspnet_ros2/contact_graspnet && "
                f"conda run -n contact-graspnet python contact_graspnet/inference.py "
                f"--np_path={shlex.quote(np_path)} --local_regions{filter_arg}{k_arg}"
            )

            cmd = [
                "docker", "exec", container_name,
                "bash", "-lc", f"{compile_cmd} && {inference_cmd}"
            ]

            self.get_logger().info(
                f"Running Contact-GraspNet inference ({tag}): "
                f"np_path={np_path}, filter_grasps={filter_grasps}, K_path={self.cgn_k_path}"
            )

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Inference failed during {tag} run.\n"
                    f"STDOUT:\n{result.stdout[-3000:]}\n"
                    f"STDERR:\n{result.stderr[-3000:]}"
                )

            json_text = self._extract_json_from_stdout(result.stdout)
            n_grasps = self._count_json_grasps(json_text)
            self.get_logger().info(f"Contact-GraspNet {tag} run returned {n_grasps} grasps.")

            # Keep useful debug logs visible if CGN reports zero grasps.
            if n_grasps == 0:
                self.get_logger().warn(
                    f"Contact-GraspNet {tag} run returned zero grasps. "
                    f"STDERR tail:\n{result.stderr[-2000:]}"
                )

            return json_text

        # First try the stricter setting used before.
        json_text = _run_once(filter_grasps=True, tag="strict local_regions+filter_grasps")
        if self._count_json_grasps(json_text) != 0:
            return json_text

        # For a scene already filtered to a single selected object, --filter_grasps can
        # be overly strict and reject all grasps. Retry without it before failing.
        if self.retry_without_filter_if_empty:
            self.get_logger().warn(
                "Strict CGN inference returned zero grasps; retrying with "
                "--local_regions but without --filter_grasps."
            )
            json_text_retry = _run_once(filter_grasps=False, tag="fallback local_regions_only")
            return json_text_retry

        return json_text


    # ------------------------------------------------------------------
    # Scene-mask filtering helpers
    # ------------------------------------------------------------------
    def _load_filtered_scene_info(self, scene_name: str):
        """Load the filtered CGN scene and prepare target-mask information."""
        if not self.strict_scene_mask_filter:
            return None

        scene_input_dir = self.scene_input_dir.strip("/ ")
        scene_path = os.path.join(self.base_path, scene_input_dir, f"{scene_name}.npy")
        if not os.path.exists(scene_path):
            self.get_logger().warn(
                f"Strict scene-mask filter enabled, but scene file was not found: {scene_path}"
            )
            return None

        try:
            data = np.load(scene_path, allow_pickle=True)
            scene = data.item() if isinstance(data, np.ndarray) and data.shape == () else data
            seg = np.asarray(scene.get('seg', scene.get('segmap', scene.get('mask', None))))
            depth = np.asarray(scene.get('depth', None))
            K = np.asarray(scene.get('K', None), dtype=np.float64)
            if seg is None or depth is None or K is None or K.shape != (3, 3):
                self.get_logger().warn(
                    f"Scene file {scene_path} does not contain usable seg/depth/K keys; "
                    f"keys={list(scene.keys()) if isinstance(scene, dict) else type(scene)}"
                )
                return None

            if seg.shape != depth.shape:
                self.get_logger().warn(
                    f"Scene seg/depth shape mismatch in {scene_path}: seg={seg.shape}, depth={depth.shape}"
                )
                return None

            seg_i = seg.astype(np.int32)
            unique_nonzero = sorted(int(v) for v in np.unique(seg_i) if int(v) != 0)
            if not unique_nonzero:
                self.get_logger().warn(
                    f"Strict scene-mask filter enabled, but scene has no nonzero target id. "
                    f"unique ids={np.unique(seg_i).tolist()}"
                )
                return None

            # The SelectInstance state filters the scene to [0, target_id], so the
            # only nonzero id is the selected target.  If more than one id remains,
            # choose the largest and warn.
            areas = {iid: int(np.count_nonzero(seg_i == iid)) for iid in unique_nonzero}
            selected_id = max(areas, key=areas.get)
            if len(unique_nonzero) != 1:
                self.get_logger().warn(
                    f"Expected one selected nonzero id after scene filtering, got {areas}. "
                    f"Using largest id {selected_id}."
                )

            valid_target = (seg_i == selected_id) & (depth > 0)
            target_pts = self._depth_to_xyz(depth.astype(np.float32), K)[valid_target]
            self.get_logger().info(
                f"Loaded strict scene filter from {scene_path}: selected_id={selected_id}, "
                f"target_pixels={int(np.count_nonzero(seg_i == selected_id))}, "
                f"valid_target_depth={target_pts.shape[0]}, "
                f"unique_nonzero={unique_nonzero}"
            )
            if target_pts.shape[0] == 0:
                self.get_logger().warn("Selected object has no valid depth points; strict scene filter will reject candidates.")

            return {
                'scene_path': scene_path,
                'selected_id': int(selected_id),
                'seg': seg_i,
                'depth': depth.astype(np.float32),
                'K': K,
                'target_pts': target_pts.astype(np.float32),
            }
        except Exception as e:
            self.get_logger().warn(f"Failed to load strict scene-mask filter info: {e}")
            return None

    @staticmethod
    def _depth_to_xyz(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
        H, W = depth_m.shape
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        v, u = np.indices((H, W), dtype=np.float32)
        Z = depth_m
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        xyz = np.stack([X, Y, Z], axis=-1).astype(np.float32)
        xyz[Z <= 0] = 0.0
        return xyz

    def _point_matches_selected_mask(self, point_cam, scene_info, label='point') -> bool:
        if scene_info is None:
            return True
        p = np.asarray(point_cam, dtype=np.float64).reshape(-1)
        if p.shape[0] < 3 or not np.all(np.isfinite(p[:3])) or p[2] <= 1e-6:
            return False

        K = scene_info['K']
        seg = scene_info['seg']
        target_pts = scene_info['target_pts']
        target_id = scene_info['selected_id']
        H, W = seg.shape

        u = int(round(float(K[0, 0]) * float(p[0]) / float(p[2]) + float(K[0, 2])))
        v = int(round(float(K[1, 1]) * float(p[1]) / float(p[2]) + float(K[1, 2])))

        proj_ok = False
        if 0 <= u < W and 0 <= v < H:
            r = max(0, int(self.mask_projection_radius_px))
            y0, y1 = max(0, v - r), min(H, v + r + 1)
            x0, x1 = max(0, u - r), min(W, u + r + 1)
            proj_ok = bool(np.any(seg[y0:y1, x0:x1] == target_id))

        dist_ok = False
        min_dist = float('inf')
        if target_pts.shape[0] > 0:
            pts = target_pts
            # Downsample for speed if the mask is large.
            if pts.shape[0] > 5000:
                step = max(1, pts.shape[0] // 5000)
                pts = pts[::step]
            d2 = np.sum((pts - p[:3].astype(np.float32)) ** 2, axis=1)
            min_dist = float(np.sqrt(np.min(d2)))
            dist_ok = min_dist <= float(self.max_contact_to_mask_dist_m)

        ok = bool(proj_ok or dist_ok)
        if not ok:
            self.get_logger().debug(
                f"Rejected {label}: p={p[:3].tolist()}, uv=({u},{v}), "
                f"proj_ok={proj_ok}, min_dist_to_target={min_dist:.4f} m, target_id={target_id}"
            )
        return ok

    def _candidate_matches_selected_scene(self, obj_id, T_cgn: np.ndarray, sample, scene_info) -> bool:
        if scene_info is None:
            return True

        target_id = int(scene_info['selected_id'])
        try:
            obj_id_int = int(float(obj_id))
        except Exception:
            obj_id_int = None

        if obj_id_int is not None and obj_id_int != target_id:
            self.get_logger().debug(
                f"Rejected CGN candidate because obj_id={obj_id_int} != selected target_id={target_id}."
            )
            return False

        # Contact-GraspNet contact_pts are the most meaningful target-object check.
        # The pose origin can be a gripper-frame point offset from the surface, so
        # it is only used as a fallback if contact_pts is not usable.
        sample_np = np.asarray(sample, dtype=np.float64).reshape(-1)
        if sample_np.shape[0] >= 3 and np.all(np.isfinite(sample_np[:3])):
            return self._point_matches_selected_mask(sample_np[:3], scene_info, label='contact_pt')

        return self._point_matches_selected_mask(T_cgn[:3, 3], scene_info, label='grasp_origin')

    # ------------------------------------------------------------------
    # Coordinate-frame helpers
    # ------------------------------------------------------------------
    def cgn_optical_to_ros_cam(self, T_cgn: np.ndarray) -> np.ndarray:
        """
        Contact-GraspNet grasps are expressed in the camera *optical* frame:
          x_right, y_down, z_forward.

        The URDF/TF frame `rgbd_camera/camera_link/rgbd_camera` is a standard
        ROS camera_link frame:
          X_forward, Y_left, Z_up.

        This applies the fixed rotation R_opt->cam_link so that the resulting
        4x4 matrix is in the ROS camera_link convention, rooted at the camera.
        """
        R = np.array([
            [0.0,  0.0, 1.0],   # z_opt -> X_cam
            [-1.0, 0.0, 0.0],   # x_opt -> -Y_cam
            [0.0, -1.0, 0.0],   # y_opt -> -Z_cam
        ], dtype=np.float64)

        # R = np.array([
        #     [1.0, 0.0, 0.0],   
        #     [0.0, 1.0, 0.0],   
        #     [0.0, 0.0, 1.0],  
        # ], dtype=np.float64)

        T_ros = np.eye(4, dtype=np.float64)
        T_ros[:3, :3] = R @ T_cgn[:3, :3]
        T_ros[:3, 3] = R @ T_cgn[:3, 3]
        return T_ros

    def transform_pose_array(self, pose_array: PoseArray,
                             from_frame: str,
                             to_frame: str) -> PoseArray:
        """
        Transform a PoseArray from `from_frame` to `to_frame` using TF2.
        Returns a new PoseArray in the target frame;
        if TF fails, returns the input pose_array.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time(),  # latest
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup {to_frame} <- {from_frame} failed: {e}")
            return pose_array

        trans = t.transform.translation
        rot = t.transform.rotation

        # 4x4 transform matrix base <- camera
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
            T_cp[0, 3] = p.position.x
            T_cp[1, 3] = p.position.y
            T_cp[2, 3] = p.position.z

            # base <- camera <- pose
            T_bp = T_bc @ T_cp

            pos = T_bp[:3, 3]
            q_bp = tft.quaternion_from_matrix(T_bp)

            p_out = Pose()
            p_out.position.x = float(pos[0]) + self.base_x_offset # manual displacement in base frame to account for physical robot/camera TF mismatch
            p_out.position.y = float(pos[1]) + self.base_y_offset
            p_out.position.z = float(pos[2])
            p_out.orientation.x = float(q_bp[0])
            p_out.orientation.y = float(q_bp[1])
            p_out.orientation.z = float(q_bp[2])
            p_out.orientation.w = float(q_bp[3])

            out.poses.append(p_out)

        return out

    # ------------------------------------------------------------------
    # Service callback
    # ------------------------------------------------------------------
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
            out_path = f"{self.base_path}/results/inference_output_{self.scene_name}.txt"
            with open(out_path, "w") as f:
                f.write(output)
            self.get_logger().info(f"Saved raw inference output to {out_path}")

            results = json.loads(output)
            pred_grasps_cam = {
                k: [np.array(g) for g in v]
                for k, v in results["pred_grasps_cam"].items()
            }
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

        scene_info = self._load_filtered_scene_info(self.scene_name)

        # ---------------------------------------------------
        # Build PoseArray in ROS camera_link frame
        # (first convert CGN optical -> camera_link)
        # ---------------------------------------------------
        grasps_cam_pa = PoseArray()
        grasps_cam_pa.header.frame_id = self.camera_frame
        grasps_cam_pa.header.stamp = self.get_clock().now().to_msg()

        score_list = []
        sample_list = []
        object_list = []
        rejected_by_scene_filter = 0

        for obj_id, T_list in pred_grasps_cam.items():
            obj_scores = scores[obj_id]
            obj_samples = contact_pts[obj_id]

            for T_cgn, score, sample in zip(T_list, obj_scores, obj_samples):
                if not self._candidate_matches_selected_scene(obj_id, T_cgn, sample, scene_info):
                    rejected_by_scene_filter += 1
                    continue

                # 1) CGN optical frame -> ROS camera_link
                # Original line:
                # T_cam = self.cgn_optical_to_ros_cam(T_cgn)
                # GEN3/real robot: make this configurable.
                # Keep True when camera_frame is a ROS camera_link-like frame.
                # Set False when camera_frame is already an optical frame.
                if self.convert_cgn_optical_to_ros_camera_link:
                    T_cam = self.cgn_optical_to_ros_cam(T_cgn)
                else:
                    T_cam = T_cgn


                # ----- NEW: constant “gripper frame” rotation offset -----
                #
                # This encodes the fixed transform between the Contact-GraspNet
                # grasp frame and your robot’s *gripper frame* (e.g. panda_hand).
                #
                # Start with identity; you can tune rx, ry, rz as needed.
                # (values are in *degrees* here for convenience)
                # Original hard-coded offset values:
                # rx_deg, ry_deg, rz_deg = 0.0, 0.0, 0.0
                # GEN3/real robot: expose them as ROS parameters for tuning.
                rx_deg = self.gripper_offset_rx_deg
                ry_deg = self.gripper_offset_ry_deg
                rz_deg = self.gripper_offset_rz_deg
                rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])

                # 4x4 homogeneous transform in the *grasp frame*:
                # T_grasp->gripper  (rotation only, no translation)
                T_gripper_offset = tft.euler_matrix(rx, ry, rz, 'sxyz')
                T_gripper_offset[0:3, 3] = [0.0, 0.0, 0.0]

                # If you decide you *also* want the SceneReplica-style X/Y swap,
                # you can fold it into the same offset like this:
                
                swap_xy = np.array([
                    [0.,  1., 0., 0.],
                    [-1., 0., 0., 0.],
                    [0.,  0., 1., 0.],
                    [0.,  0., 0., 1.],
                ])
                # Original line:
                # T_gripper_offset = T_gripper_offset @ swap_xy
                # GEN3/real robot: make the SceneReplica-style X/Y swap optional.
                if self.apply_scene_replica_xy_swap:
                    T_gripper_offset = T_gripper_offset @ swap_xy

                 # ----- NEW: apply constant gripper-frame rotation -----
                # T_cam is (camera_link -> CGN_graspFrame).
                # We want (camera_link -> robot_gripperFrame):
                # Original line:
                # T_cam = T_cam @ T_gripper_offset
                # GEN3/real robot: allow disabling this for debugging raw CGN poses.
                if self.apply_gripper_frame_offset:
                    T_cam = T_cam @ T_gripper_offset
                # ------------------------------------------------------


                 # 2) Convert to Pose in camera frame
                ros_pose = Pose()
                ros_pose.position.x = float(T_cam[0, 3])
                ros_pose.position.y = float(T_cam[1, 3])
                ros_pose.position.z = float(T_cam[2, 3])

                quat = tfs.quaternion_from_matrix(T_cam)
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
                f"Obtained {len(obj_scores)} grasps for object {obj_id}"
            )

        if rejected_by_scene_filter > 0:
            self.get_logger().warn(
                f"Strict selected-mask filtering rejected {rejected_by_scene_filter} CGN grasp candidates."
            )

        # Optional: quick Z debug
        if grasps_cam_pa.poses:
            zs_cam = [p.position.z for p in grasps_cam_pa.poses]
            self.get_logger().info(
                f"Camera-frame grasp Z range: [{min(zs_cam):.3f}, {max(zs_cam):.3f}]"
            )

        # ---------------------------------------------------
        # Transform PoseArray cam -> base using TF2
        # ---------------------------------------------------
        grasps_in_base_pa = self.transform_pose_array(
            grasps_cam_pa,
            from_frame=self.camera_frame,
            to_frame=self.base_frame,
        )

        if grasps_in_base_pa.poses:
            zs_base = [p.position.z for p in grasps_in_base_pa.poses]
            self.get_logger().info(
                f"Base-frame grasp Z range before safety offset: "
                f"[{min(zs_base):.3f}, {max(zs_base):.3f}]"
            )

        # ---------------------------------------------------
        # GEN3/real-robot safety offset in base-frame +Z
        # ---------------------------------------------------
        # Proper location for the first safety correction:
        #   after camera->base TF, before filling response.grasps.
        # This means FlexBE and MoveToPose receive the lifted pose directly.
        # It avoids sending a low raw CGN pose to /move_to_pose first.
        if grasps_in_base_pa.poses and abs(self.grasp_base_z_safety_offset) > 1e-9:
            for p in grasps_in_base_pa.poses:
                # Original behavior:
                # p.position.z = p.position.z
                p.position.z += self.grasp_base_z_safety_offset

                if p.position.z < 0.15:
                    p.position.z = 0.15


            zs_base_offset = [p.position.z for p in grasps_in_base_pa.poses]
            self.get_logger().info(
                f"Applied base-frame +Z safety offset: "
                f"{self.grasp_base_z_safety_offset:.3f} m; "
                f"Z range after offset: [{min(zs_base_offset):.3f}, {max(zs_base_offset):.3f}]"
            )

        # ---------------------------------------------------
        # Fill Grasps msg (poses now in base frame, with optional safety offset)
        # ---------------------------------------------------
        grasps_msg = Grasps()
        grasps_msg.poses = list(grasps_in_base_pa.poses)
        grasps_msg.scores = score_list
        grasps_msg.samples = sample_list
        grasps_msg.object_ids = list(object_list)

        response.grasps = grasps_msg

        self.get_logger().info(
            f"Responded with {len(grasps_msg.poses)} grasps in frame "
            f"'{self.base_frame}' for scene {self.scene_name}, grasps_msg.object_ids = {np.unique(grasps_msg.object_ids)}"
            # f"'{self.base_frame}' for scene {self.scene_name}"
            # f" object_list = {object_list}"
            # f" grasps_msg.object_ids = {grasps_msg.object_ids}"
            # f" list(grasps_msg.object_ids) = {list(grasps_msg.object_ids)}"
        )

        return response


def main(args=None):
    rclpy.init(args=args)
    node = GraspServer()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()