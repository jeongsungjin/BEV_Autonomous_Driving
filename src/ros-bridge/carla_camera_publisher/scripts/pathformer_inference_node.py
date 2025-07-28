#!/usr/bin/env python3
"""Real-time Pathformer inference node.
Subscribes to YOLOP BEV OccupancyGrid topics and odometry, buffers the latest seq_len grids,
feeds Pathformer model, rescales output, publishes nav_msgs/Path for the planned trajectory.
"""
import os
import sys
import math
from pathlib import Path
from collections import deque
from typing import Deque, Tuple

import numpy as np
import rospy
import torch
from nav_msgs.msg import OccupancyGrid, Odometry, Path as RosPath
from geometry_msgs.msg import PoseStamped

# numpy의 polyfit을 사용한 다항식 피팅

# === Import Pathformer modules ===
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[2]  # capstone_2025/src/ros-bridge
# Fallback search: walk up to find 'pathformer' directory
candidate = THIS_DIR
for _ in range(8):
    pf_dir = candidate / 'pathformer'
    if pf_dir.exists():
        sys.path.insert(0, str(candidate))
        sys.path.insert(0, str(pf_dir))
        break
    candidate = candidate.parent
from pathformer.models.PathFormer import Model as PathFormerModel  # type: ignore

class PathformerInferenceNode:
    def __init__(self):
        rospy.init_node("pathformer_inference_node", anonymous=True)

        # --- parameters ---
        self.seq_len = int(rospy.get_param("~seq_len", 8))
        self.pred_len = int(rospy.get_param("~pred_len", 12))
        self.height = int(rospy.get_param("~bev_h", 48))
        self.width = int(rospy.get_param("~bev_w", 80))
        self.pos_scale = float(rospy.get_param("~pos_scale", 20.0))
        # Channel weights for (det, da, ll) to emphasize specific cues
        weights_str = rospy.get_param("~channel_weights", "1.0,1.0,3.0")
        try:
            self.w_det, self.w_da, self.w_ll = [float(x) for x in weights_str.split(',')]
        except ValueError:
            rospy.logwarn("~channel_weights parameter malformed, using 1,1,3")
            self.w_det, self.w_da, self.w_ll = 1.0, 1.0, 3.0
        # If predictions are in vehicle body frame, set this param true to rotate by yaw.
        self.body_frame = rospy.get_param("~body_frame", True)
        # Debug flag: enable detailed logging when ~debug:=true
        self.debug = rospy.get_param("~debug", False)
        ckpt_path = Path(rospy.get_param("~checkpoint_path"))
        if not ckpt_path.is_file():
            rospy.logerr(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        # --- model config ---
        # gather dynamic hyperparameters first to avoid NameError in class-level list comps
        layer_nums_cfg = int(rospy.get_param("~layer_nums", 2))
        ps_param = rospy.get_param("~patch_size_list", "4,2")
        ps_flat = [int(x) for x in str(ps_param).split(',') if x]
        if len(ps_flat) >= layer_nums_cfg:
            per_layer = len(ps_flat) // layer_nums_cfg or 1
            patch_size_cfg = [ps_flat[i*per_layer:(i+1)*per_layer] for i in range(layer_nums_cfg)]
        else:
            patch_size_cfg = [[4,2] for _ in range(layer_nums_cfg)]

        ne_param = rospy.get_param("~num_experts_list", "2")
        if isinstance(ne_param, int):
            num_experts_cfg = [ne_param]
        elif isinstance(ne_param, list):
            num_experts_cfg = [int(x) for x in ne_param]
        else:
            num_experts_cfg = [int(x) for x in str(ne_param).split(',') if x]

        class Args:
            layer_nums = layer_nums_cfg
            num_nodes = 3 * self.height * self.width
            pred_len = self.pred_len
            seq_len = self.seq_len
            k = int(rospy.get_param("~k", 2))
            num_experts_list = num_experts_cfg
            patch_size_list = patch_size_cfg
            d_model = int(rospy.get_param("~d_model", 16))
            d_ff = int(rospy.get_param("~d_ff", 64))
            residual_connection = 0
            revin = 0
            gpu = 0
            batch_norm = 1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = PathFormerModel(Args()).to(self.device)
        state = torch.load(ckpt_path, map_location=self.device)
        try:
            self.model.load_state_dict(state)
        except RuntimeError as e:
            rospy.logwarn(f"[Pathformer] state_dict mismatch: {e}. Loading with strict=False")
            self.model.load_state_dict(state, strict=False)
        self.model.eval()
        if self.debug:
            rospy.loginfo("[Pathformer] Model state dict loaded and set to eval mode.")
        rospy.loginfo("Pathformer model loaded.")

        # --- buffers ---
        self.grid_buf: Deque[np.ndarray] = deque(maxlen=self.seq_len)
        self.latest_pose = None  # type: Odometry

        # --- publishers ---
        self.path_pub = rospy.Publisher("/planned_path", RosPath, queue_size=10)
        self.smooth_path_pub = rospy.Publisher("/planned_smooth_path", RosPath, queue_size=10)

        # --- subscribers ---
        rospy.Subscriber("/carla/yolop/det_grid", OccupancyGrid, self._grid_callback)
        rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, self._odom_callback)

        rospy.loginfo("Pathformer inference node started.")
        rospy.spin()

    def _odom_callback(self, msg: Odometry):
        self.latest_pose = msg

    def _grid_callback(self, msg: OccupancyGrid):
        stamp = msg.header.stamp
        det = self._grid_to_numpy(msg)
        try:
            da = self._grid_to_numpy(rospy.wait_for_message("/carla/yolop/da_grid", OccupancyGrid, timeout=0.1))
            ll = self._grid_to_numpy(rospy.wait_for_message("/carla/yolop/ll_grid", OccupancyGrid, timeout=0.1))
        except rospy.ROSException:
            return
        # Apply channel weights, clip to [0,1] then convert to uint8
        raster_f = np.stack([
            det * self.w_det,
            da  * self.w_da,
            ll  * self.w_ll,
        ], axis=0)
        raster_f = np.clip(raster_f, 0, 1)
        raster = raster_f.astype(np.uint8)
        if self.debug:
            rospy.loginfo(f"[weights] det={self.w_det} da={self.w_da} ll={self.w_ll}")
        self.grid_buf.append(raster)
        if self.debug:
            # Compute simple metrics on raster and temporal change
            if len(self.grid_buf) >= 2:
                diff = np.abs(self.grid_buf[-1].astype(int) - self.grid_buf[-2].astype(int))
                diff_ratio = diff.mean()
            else:
                diff_ratio = 0.0
            ch_means = raster.mean(axis=(1, 2))  # per-channel mean
            rospy.loginfo(
                f"[grid_cb] t={stamp.to_sec():.3f} buf_len={len(self.grid_buf)} pose={'yes' if self.latest_pose else 'no'} diff_mean={diff_ratio:.3f} ch_mean={ch_means}" 
            )
        if len(self.grid_buf) < self.seq_len or self.latest_pose is None:
            return
        self._run_inference(stamp)

    def _grid_to_numpy(self, grid: OccupancyGrid) -> np.ndarray:
        data = np.array(grid.data, dtype=np.int8).reshape((grid.info.height, grid.info.width))
        binary = (data > 50).astype(np.uint8)  # occupied threshold
        # resize to desired H,W if necessary
        if data.shape != (self.height, self.width):
            import cv2
            binary = cv2.resize(binary, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        return binary

    def _run_inference(self, stamp):
        # Prepare input tensor
        seq = np.stack(self.grid_buf, axis=0)  # (seq_len, 3, H, W)
        if self.debug:
            rospy.loginfo(
                "[run_inf] seq stats preNorm min={:.3f} max={:.3f} mean={:.3f}".format(
                    seq.min(), seq.max(), seq.mean()
                )
            )
        seq = seq.astype(np.float32)
        # If raster values are 0/255 scale, normalize to 0-1. If already 0/1, keep as is.
        if seq.max() > 1.0:
            seq = seq / 255.0
        seq = seq.reshape(self.seq_len, -1)  # flatten
        tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)  # (1, seq_len, num_nodes)
        with torch.no_grad():
            preds, _ = self.model(tensor)  # (1, pred_len, 2)
        preds_raw = preds.squeeze(0).cpu().numpy()  # 원본 예측값 (스케일링 전)
        preds = preds_raw * self.pos_scale  # rescale meters

        # 예측된 경로가 차량 전방으로 향하도록 강제
        for i in range(len(preds)):
            dx, dy = preds[i]
            # 전방 방향으로 최소 거리 보장
            # Pathformer 모델의 좌표계: x축이 전방, y축이 좌우
            # 항상 양의 x 방향으로 전방 설정 (차량 전방)
            if dx < 3.0:  # 전방 거리가 3미터 미만이면 강제로 전방으로
                dx = 3.0 + i * 2.0  # 각 waypoint마다 2미터씩 증가
            # 항상 양수로 강제 (전방 방향)
            dx = abs(dx)
            preds[i] = [dx, dy]

        if self.debug:
            rospy.loginfo(f"[run_inf] preds_raw(first3) = {preds_raw[:3]}")
            rospy.loginfo(f"[run_inf] preds_scaled(first3) = {preds[:3]}")
            rospy.loginfo(f"[run_inf] preds shape = {preds.shape}")
            rospy.loginfo(f"[run_inf] pos_scale = {self.pos_scale}")
            rospy.loginfo(f"[run_inf] preds_raw range = [{preds_raw.min():.4f}, {preds_raw.max():.4f}]")

        # Build nav_msgs/Path relative to current pose
        path_msg = RosPath()
        path_msg.header.stamp = stamp
        path_msg.header.frame_id = "map"
        base_pose = self.latest_pose.pose.pose
        # Compute vehicle yaw once
        yaw = self._get_yaw(base_pose.orientation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        for i, (dx, dy) in enumerate(preds):
            if self.body_frame:
                # Pathformer 예측은 차량 전방 방향을 x축으로 하는 좌표계
                # dx: 전방 거리, dy: 좌우 거리
                # 차량의 yaw를 고려하여 월드 좌표계로 변환
                # 차량 헤딩 방향으로 전방 계산
                # x방향을 뒤집어서 차량 전방으로 waypoints가 나가도록 강제
                # 차량의 yaw 방향으로 전방 계산 (x 부호 반전)
                x_rel = -dx * cos_yaw - dy * sin_yaw
                y_rel = -dx * sin_yaw + dy * cos_yaw
            else:
                x_rel, y_rel = dx, dy

            # 최소 거리 보장 (너무 가까운 waypoint 제거)
            min_distance = 2.0  # 최소 2미터
            if i == 0 or (abs(x_rel) > min_distance or abs(y_rel) > min_distance):
                x_world = base_pose.position.x + x_rel
                y_world = base_pose.position.y + y_rel
                
                if self.debug and i < 3:  # 처음 3개 waypoint만 디버그 출력
                    rospy.loginfo(f"[coord] i={i} dx={dx:.3f} dy={dy:.3f} x_rel={x_rel:.3f} y_rel={y_rel:.3f} x_world={x_world:.3f} y_world={y_world:.3f}")
                    # 좌표계 변환 확인
                    rospy.loginfo(f"[coord] yaw={yaw:.3f} cos_yaw={cos_yaw:.3f} sin_yaw={sin_yaw:.3f}")
                    # 차량 헤딩 방향 확인
                    heading_deg = math.degrees(yaw)
                    rospy.loginfo(f"[coord] heading_deg={heading_deg:.1f}")
                
                pose = PoseStamped()
                pose.header = path_msg.header
                pose.pose.position.x = x_world
                pose.pose.position.y = y_world
                pose.pose.position.z = base_pose.position.z
                path_msg.poses.append(pose)
        self.path_pub.publish(path_msg)

        # 곡선 생성 및 발행
        if len(path_msg.poses) > 2:
            smooth_path = self._create_smooth_path(path_msg.poses)
            self.smooth_path_pub.publish(smooth_path)

        if self.debug and len(path_msg.poses):
            yaw = self._get_yaw(base_pose.orientation)
            first = path_msg.poses[0].pose.position
            last = path_msg.poses[-1].pose.position
            rospy.loginfo(
                f"[publish] base=({base_pose.position.x:.2f},{base_pose.position.y:.2f}) yaw={yaw:.2f} first_pred=({first.x:.2f},{first.y:.2f}) last_pred=({last.x:.2f},{last.y:.2f})"
            )
            # 예측된 경로의 방향성 확인
            if len(path_msg.poses) > 1:
                dx = first.x - base_pose.position.x
                dy = first.y - base_pose.position.y
                path_angle = math.atan2(dy, dx)
                angle_diff = abs(path_angle - yaw)
                rospy.loginfo(f"[publish] path_angle={path_angle:.2f} angle_diff={angle_diff:.2f}")

    def _create_smooth_path(self, poses) -> RosPath:
        """waypoints를 평균화하여 부드러운 경로 생성"""
        if len(poses) < 3:
            return RosPath()
        
        # waypoints 좌표 추출
        points = np.array([[pose.pose.position.x, pose.pose.position.y] for pose in poses])
        
        # 2-3차 다항식으로 단순한 path 피팅
        smooth_points = self._create_averaged_path(points)
        
        # RosPath 메시지 생성
        smooth_path = RosPath()
        smooth_path.header = poses[0].header
        smooth_path.header.frame_id = "map"
        
        for point in smooth_points:
            pose = PoseStamped()
            pose.header = smooth_path.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = poses[0].pose.position.z
            smooth_path.poses.append(pose)
        
        return smooth_path

    def _create_averaged_path(self, points):
        """추론된 점들의 평균점들을 만들고 차량 전방 방향으로 인도하는 완만한 곡선"""
        if len(points) < 3:
            return points
        
        # 점들을 전방 거리(y값) 기준으로 정렬 (차량에서 멀어지는 순서)
        sorted_indices = np.argsort(points[:, 1])  # y값 기준 정렬
        sorted_points = points[sorted_indices]
        
        # 점들을 3-4개 그룹으로 나누어 중심점 생성
        n_groups = min(4, len(points) // 2)
        if n_groups < 2:
            return points
        
        group_size = len(sorted_points) // n_groups
        center_points = []
        
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = start_idx + group_size if i < n_groups - 1 else len(sorted_points)
            group_points = sorted_points[start_idx:end_idx]
            
            # 각 그룹의 평균을 중심점으로 사용
            center_point = np.mean(group_points, axis=0)
            center_points.append(center_point)
        
        center_points = np.array(center_points)
        
        # x축 변화량과 y축 변화량 비교하여 피팅 방향 결정
        x_coords = center_points[:, 0]
        y_coords = center_points[:, 1]
        
        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()
        
        # x축 변화량이 적으면 x = f(y) 형태로 피팅
        if x_range < y_range * 0.5:
            # x = f(y) 형태로 피팅 (y를 독립변수로)
            if len(center_points) >= 4:
                degree = 3  # 3차 다항식
            else:
                degree = 2  # 2차 다항식
            
            try:
                coeffs = np.polyfit(y_coords, x_coords, degree)
                poly_func = np.poly1d(coeffs)
                
                # y 범위에서 더 많은 점 생성
                y_min, y_max = y_coords.min(), y_coords.max()
                y_range_extended = np.linspace(y_min, y_max + (y_max - y_min) * 0.3, len(points) * 3)
                x_fitted = poly_func(y_range_extended)
                
                smooth_points = np.column_stack([x_fitted, y_range_extended])
                
            except:
                smooth_points = points
        else:
            # y = f(x) 형태로 피팅 (기존 방식)
            if len(center_points) >= 4:
                degree = 3  # 3차 다항식
            else:
                degree = 2  # 2차 다항식
            
            try:
                coeffs = np.polyfit(x_coords, y_coords, degree)
                poly_func = np.poly1d(coeffs)
                
                # x 범위에서 더 많은 점 생성
                x_min, x_max = x_coords.min(), x_coords.max()
                x_range_extended = np.linspace(x_min, x_max + (x_max - x_min) * 0.3, len(points) * 3)
                y_fitted = poly_func(x_range_extended)
                
                smooth_points = np.column_stack([x_range_extended, y_fitted])
                
            except:
                smooth_points = points
        
        return smooth_points

    def _get_yaw(self, q) -> float:
        import math
        # assuming quaternion is geometry_msgs/Quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

if __name__ == "__main__":
    try:
        PathformerInferenceNode()
    except rospy.ROSInterruptException:
        pass 