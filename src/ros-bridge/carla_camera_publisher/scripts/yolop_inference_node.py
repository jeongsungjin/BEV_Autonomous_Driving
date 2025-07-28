#!/usr/bin/env python3

# (기존 import 생략 없이 그대로 유지)
import os
import sys
import time
import threading
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import rospy
import torch
import torchvision.transforms as transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Path as RosPath
from sensor_msgs.msg import CameraInfo
import tf2_ros
from geometry_msgs.msg import PointStamped
from tf2_geometry_msgs import do_transform_point  # type: ignore

# OccupancyGrid 메시지 작성을 위해 추가
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, Point, Quaternion
from nav_msgs.msg import Odometry

# === YOLOP 경로 설정 ===
THIS_DIR = Path(__file__).resolve().parent
POSSIBLE_DIRS = [p / "YOLOP" for p in [THIS_DIR, *list(THIS_DIR.parents)]]
YOLOP_DIR = next((d for d in POSSIBLE_DIRS if d.exists()), None)

if YOLOP_DIR is None:
    raise RuntimeError("YOLOP 디렉터리를 찾을 수 없습니다.")
if str(YOLOP_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOP_DIR))

# === YOLOP 모듈 import ===
from lib.config import cfg, update_config
from lib.models import get_net
from lib.utils.utils import select_device
from lib.utils.augmentations import letterbox_for_img
from lib.core.general import non_max_suppression, scale_coords


class YOLOPInferenceNode:
    def __init__(self):
        rospy.init_node("yolop_inference_node", anonymous=True)

        # 파라미터
        # 기본 가중치 경로를 새로 학습한 모델 경로로 변경
        self.weight_path = rospy.get_param(
            "~weight_path",
            "/home/carla/capstone_2025/src/ros-bridge/YOLOP/weights/epoch-240.pth",
        )
        self.conf_thres = float(rospy.get_param("~conf_thres", 0.5))
        self.iou_thres = float(rospy.get_param("~iou_thres", 0.45))
        self.input_size = tuple(cfg.MODEL.IMAGE_SIZE)

        # --- additional params for BEV mask tuning ---
        self.da_thresh = float(rospy.get_param("~da_thresh", 0.53))  # drivable area prob threshold
        self.ll_thresh = float(rospy.get_param("~ll_thresh", 0.5))  # lane line prob threshold

        # global debug flag
        self.debug = rospy.get_param("~debug", False)

        self.device = select_device(device="0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.to(self.device).eval()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.bridge = CvBridge()

        # === Waypoint overlay 준비 ===
        self.latest_path = None  # type: RosPath
        self.latest_smooth_path = None  # type: RosPath
        self.camera_info = None  # type: CameraInfo
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        cam_info_topic = rospy.get_param("~camera_info_topic", "/carla/yolop/camera_info")
        rospy.Subscriber(cam_info_topic, CameraInfo, self._camerainfo_callback, queue_size=1)
        rospy.Subscriber("/planned_path", RosPath, self._path_callback, queue_size=1)
        rospy.Subscriber("/planned_smooth_path", RosPath, self._smooth_path_callback, queue_size=1)

        # 최신 프레임 저장용
        self.latest_msg = None
        self.lock = threading.Lock()

        # 결과 발행용 퍼블리셔
        self.pub_vis = rospy.Publisher("/carla/yolop/inference", Image, queue_size=10)
        self.pub_det = rospy.Publisher("/carla/yolop/det_grid", OccupancyGrid, queue_size=10)
        self.pub_da = rospy.Publisher("/carla/yolop/da_grid", OccupancyGrid, queue_size=10)
        self.pub_ll = rospy.Publisher("/carla/yolop/ll_grid", OccupancyGrid, queue_size=10)

        # --- costmap 모드용 퍼블리셔(통합 OccupancyGrid) ---
        self.enable_costmap = rospy.get_param("~enable_costmap", True)
        if self.enable_costmap:
            self.pub_costmap = rospy.Publisher(
                "/carla/yolop/costmap", OccupancyGrid, queue_size=10
            )

        # === BEV Raster 저장 디렉터리 (옵션) ===
        if rospy.get_param("~save_bev", False):
            bev_dir_param = rospy.get_param("~bev_save_dir", str(Path.home() / ".ros" / "bev_rasters"))
            self.bev_save_dir = Path(bev_dir_param)
            self.bev_save_dir.mkdir(parents=True, exist_ok=True)

        self.sub = rospy.Subscriber(
            "/carla/yolop/image_raw", Image, self._image_callback,
            queue_size=5, buff_size=1 << 24,
        )

        # 추론 스레드 시작
        self.inference_thread = threading.Thread(target=self._inference_loop)
        self.inference_thread.daemon = True
        self.inference_thread.start()

        rospy.loginfo("YOLOP 실시간 추론 노드가 시작되었습니다.")
        rospy.spin()

    def _load_model(self):
        class SimpleArgs:
            modelDir = ""
            logDir = ""
            conf_thres = None
            iou_thres = None
            weights = None
        args = SimpleArgs()
        update_config(cfg, args)

        model = get_net(cfg)
        if not os.path.isfile(self.weight_path):
            rospy.logerr(f"가중치 파일이 없습니다: {self.weight_path}")
            rospy.signal_shutdown("weights not found")
        ckpt = torch.load(self.weight_path, map_location=self.device)
        state_dict = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state_dict)
        rospy.loginfo("YOLOP 모델 가중치 로드 완료")
        return model

    def _image_callback(self, msg: Image):
        with self.lock:
            self.latest_msg = msg  # 가장 최신 메시지만 저장

    def _path_callback(self, msg: RosPath):
        self.latest_path = msg

    def _smooth_path_callback(self, msg: RosPath):
        self.latest_smooth_path = msg

    def _camerainfo_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def _inference_loop(self):
        rate = rospy.Rate(30)  # 최대 30Hz 추론 루프
        while not rospy.is_shutdown():
            msg = None
            with self.lock:
                if self.latest_msg is not None:
                    msg = self.latest_msg
                    self.latest_msg = None  # 소비 후 초기화

            if msg is not None:
                frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                # vis 이미지와 occupancy grid 메시지 반환
                result_img, det_grid_msg, da_grid_msg, ll_grid_msg = self._run_inference(frame_bgr, msg.header)

                # 시각화용 이미지 발행
                ros_image = self.bridge.cv2_to_imgmsg(result_img, encoding="bgr8")
                # 입력 이미지와 동일한 헤더(프레임) 사용 → TF 변환 일관성 확보
                ros_image.header = msg.header
                self.pub_vis.publish(ros_image)

                # Grid 발행
                if det_grid_msg is not None:
                    self.pub_det.publish(det_grid_msg)
                if da_grid_msg is not None:
                    self.pub_da.publish(da_grid_msg)
                if ll_grid_msg is not None:
                    self.pub_ll.publish(ll_grid_msg)

            rate.sleep()

    def _build_occupancy_grid(self, mask: np.ndarray, header_stamp: rospy.Time, frame_id: str,
                              occupied_val: int = 100, free_val: int = 0, unknown_val: int = -1) -> OccupancyGrid:
        """Binary/graded mask를 OccupancyGrid 메시지로 변환"""
        h, w = mask.shape

        # 값 매핑
        data = np.full(mask.shape, unknown_val, dtype=np.int8)
        data[mask == 0] = free_val
        data[mask == 1] = occupied_val

        # BEV 그리드 해상도를 80x48로 고정 (Pathformer 모델에 맞춤)
        target_width = 80
        target_height = 48
        
        # 마스크를 목표 해상도로 리사이즈
        import cv2
        data_resized = cv2.resize(data, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        meta = MapMetaData()
        meta.map_load_time = rospy.Time.now()
        meta.resolution = 1.0  # 1 pixel == 1 grid cell
        meta.width = target_width
        meta.height = target_height
        meta.origin = Pose(position=Point(0.0, 0.0, 0.0),
                           orientation=Quaternion(0.0, 0.0, 0.0, 1.0))

        grid = OccupancyGrid()
        grid.header.stamp = header_stamp
        grid.header.frame_id = frame_id
        grid.info = meta
        grid.data = data_resized.flatten().tolist()
        return grid

    def _run_inference(self, frame_bgr: np.ndarray, header) -> Tuple[np.ndarray, OccupancyGrid, OccupancyGrid, OccupancyGrid]:
        orig_shape = frame_bgr.shape[:2]
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized, ratio, pad = letterbox_for_img(
            img_rgb, new_shape=self.input_size, auto=False, scaleFill=False, scaleup=False
        )
        img_tensor = transforms.ToTensor()(img_resized)
        img_tensor = self.normalize(img_tensor).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)

        det_out, da_seg_out, ll_seg_out = outputs
        det_pred = det_out[0]
        det_nms = non_max_suppression(det_pred, self.conf_thres, self.iou_thres)[0]

        vis_img = frame_bgr.copy()

        # === Detection Occupancy Grid 생성 ===
        det_mask = np.zeros(orig_shape, dtype=np.uint8)  # (h, w)

        if det_nms is not None and len(det_nms):
            det_nms[:, :4] = scale_coords(
                img_tensor.shape[2:], det_nms[:, :4], orig_shape
            )
            for *xyxy, conf, cls in det_nms:
                x1, y1, x2, y2 = map(int, xyxy)
                # 파란색(BGR) 박스로 시각화
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_img, f"obj {conf:.2f}", (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)

                # det mask 표시 (1: occupied)
                det_mask[y1:y2, x1:x2] = 1

        # === 마스크 확장(팽창)으로 가시성 향상 ===
        kernel = np.ones((5, 5), np.uint8)
        det_mask = cv2.dilate(det_mask, kernel, iterations=1)

        da_mask = da_seg_out.squeeze().cpu().numpy()
        ll_mask = ll_seg_out.squeeze().cpu().numpy()
        if da_mask.ndim == 3:
            da_mask = da_mask[1]
        if ll_mask.ndim == 3:
            ll_mask = ll_mask[1]

        # 1. input_size 크기로 upsample
        da_mask = cv2.resize(da_mask, self.input_size[::-1])
        ll_mask = cv2.resize(ll_mask, self.input_size[::-1])

        # 2. letterbox 제거 (pad 제거 + scaling 역변환)
        h, w = orig_shape
        scale = min(self.input_size[0] / h, self.input_size[1] / w)
        pad_x = int((self.input_size[1] - w * scale) / 2)
        pad_y = int((self.input_size[0] - h * scale) / 2)

        x1, x2 = pad_x, self.input_size[1] - pad_x
        y1, y2 = pad_y, self.input_size[0] - pad_y

        da_mask_cropped = da_mask[y1:y2, x1:x2]
        ll_mask_cropped = ll_mask[y1:y2, x1:x2]

        # 3. 원래 이미지 크기로 리사이즈
        da_mask_resized = cv2.resize(da_mask_cropped, (w, h))
        ll_mask_resized = cv2.resize(ll_mask_cropped, (w, h))

        # Thresholding
        _, da_bin = cv2.threshold(da_mask_resized, self.da_thresh, 1, cv2.THRESH_BINARY)
        _, ll_bin = cv2.threshold(ll_mask_resized, self.ll_thresh, 1, cv2.THRESH_BINARY)
        da_bin = da_bin.astype(np.uint8)
        ll_bin = ll_bin.astype(np.uint8)

        # Lane 마스크를 조금 두껍게
        ll_bin = cv2.dilate(ll_bin, kernel, iterations=1)

        # === drivable area 종방향(세로) 팽창 (더 크게) ===
        vertical_kernel = np.ones((15, 3), np.uint8)  # 더 큰 세로 커널
        da_bin = cv2.dilate(da_bin, vertical_kernel, iterations=3)

        if self.debug:
            rospy.loginfo(
                "[YOLOP] mask ratios det={:.3f} da={:.3f} ll={:.3f}".format(
                    det_mask.mean(), da_bin.mean(), ll_bin.mean()
                )
            )

        # 색상 합성 (녹색: da, 파란색: ll)
        color_da = np.zeros_like(vis_img)
        color_da[:, :, 1] = da_bin * 255
        color_ll = np.zeros_like(vis_img)
        color_ll[:, :, 2] = ll_bin * 255

        vis_img = cv2.addWeighted(vis_img, 1.0, color_da, 0.4, 0)
        vis_img = cv2.addWeighted(vis_img, 1.0, color_ll, 0.4, 0)

        # ---- Ego vehicle purple box (center of BEV image) ----
        h, w = vis_img.shape[:2]
        box_w, box_h = 30, 50  # 가로, 세로 픽셀 크기 (원하면 파라미터화 가능)
        cx, cy = w // 2, h // 2
        top_left = (cx - box_w // 2, cy - box_h // 2)
        bottom_right = (cx + box_w // 2, cy + box_h // 2)
        cv2.rectangle(vis_img, top_left, bottom_right, (255, 0, 255), 2)  # 보라색

        # === Waypoint 경로 오버레이 ===
        vis_img = self._overlay_waypoints(vis_img, header)

        # OccupancyGrid 메시지 생성
        det_grid_msg = self._build_occupancy_grid(det_mask, header.stamp, "map", occupied_val=100, free_val=0)
        # DA: 주행 가능 영역은 free(0), 그 외는 occupied(100)
        da_occ_mask = np.where(da_bin == 1, 0, 1).astype(np.uint8)  # 1이면 occupied
        da_grid_msg = self._build_occupancy_grid(da_occ_mask, header.stamp, "map", occupied_val=100, free_val=0)
        # LL: 차선은 occupied(50)로 표현, free_val=0, occupied_val=50
        ll_grid_msg = self._build_occupancy_grid(ll_bin, header.stamp, "map", occupied_val=100, free_val=0)

        # === 통합 Costmap OccupancyGrid 생성 ===
        if self.enable_costmap:
            # DA 외부는 99로 더 높임
            combined_mask = np.full(det_mask.shape, 99, dtype=np.int8)

            # 1) Drivable Area (DA): 내부 0, 경계 90
            kernel3 = np.ones((3, 3), np.uint8)
            da_eroded = cv2.erode(da_bin, kernel3, iterations=2)
            da_border = cv2.bitwise_and(da_bin, cv2.bitwise_not(da_eroded))

            combined_mask[da_eroded == 1] = 0      # 자유 공간
            combined_mask[da_border == 1] = 90     # 경계부 cost ↑

            # 2) Lane Line (LL): 장애물과 동일하게 100
            ll_dilated = cv2.dilate(ll_bin, kernel3, iterations=1)
            combined_mask[ll_dilated == 1] = 100

            # 3) Detection (Det): 장애물 100
            det_dilated = cv2.dilate(det_mask, kernel3, iterations=2)
            combined_mask[det_dilated == 1] = 100

            # 4) Ego vehicle 중앙 박스 → DA(0)로 강제 설정 (임시)
            box_w, box_h = 30, 50
            h_cm, w_cm = combined_mask.shape
            cx_cm, cy_cm = w_cm // 2, h_cm // 2
            tl_x, tl_y = cx_cm - box_w // 2, cy_cm - box_h // 2
            br_x, br_y = cx_cm + box_w // 2, cy_cm + box_h // 2
            combined_mask[tl_y:br_y, tl_x:br_x] = 0

            meta = MapMetaData()
            meta.map_load_time = rospy.Time.now()
            meta.resolution = 1.0
            meta.width = combined_mask.shape[1]
            meta.height = combined_mask.shape[0]
            meta.origin = Pose(
                position=Point(0.0, 0.0, 0.0),
                orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
            )

            costmap_msg = OccupancyGrid()
            costmap_msg.header.stamp = header.stamp
            costmap_msg.header.frame_id = "map"
            costmap_msg.info = meta
            costmap_msg.data = combined_mask.flatten().tolist()
            self.pub_costmap.publish(costmap_msg)

        # === BEV Raster 저장 ===
        if hasattr(self, 'bev_save_dir'):
            try:
                raster = np.stack([det_mask, da_bin, ll_bin], axis=0).astype(np.uint8)
                fname = self.bev_save_dir / f"{header.stamp.to_sec():.6f}.npy"
                np.save(fname, raster)
            except Exception as e:
                rospy.logwarn_once(f"BEV raster 저장 실패: {e}")

        return vis_img, det_grid_msg, da_grid_msg, ll_grid_msg

    # ------------------- Waypoint overlay helper -------------------
    def _overlay_waypoints(self, img: np.ndarray, header) -> np.ndarray:
        """/planned_path 의 waypoints 를 이미지 평면에 투영하여 파란 점으로 표시하고, 곡선 path를 초록색 선으로 표시"""
        if self.latest_path is None or self.camera_info is None:
            return img

        # 카메라 내참수
        K = self.camera_info.K  # 3x3 row-major list
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]

        try:
            # map -> camera transform (at latest available)
            trans = self.tf_buffer.lookup_transform(
                header.frame_id,  # target: camera frame
                self.latest_path.header.frame_id,  # source: map
                rospy.Time(0),
                rospy.Duration(0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return img

        pts_drawn = 0
        pts_total = len(self.latest_path.poses)
        skipped_z = 0
        skipped_bounds = 0
        for pose_st in self.latest_path.poses:
            pt = PointStamped()
            pt.header = self.latest_path.header
            pt.point = pose_st.pose.position
            try:
                pt_cam = do_transform_point(pt, trans)
            except Exception as e:
                rospy.logdebug_once(f"Transform error for waypoint: {e}")
                continue
            X, Y, Z = pt_cam.point.x, pt_cam.point.y, pt_cam.point.z
            # 투영 시 Z 절대값 사용 (위/아래 모두 허용). 너무 가까우면 스킵.
            if abs(Z) < 0.1:
                skipped_z += 1
                continue
            u = int(fx * X / abs(Z) + cx)
            v = int(fy * Y / abs(Z) + cy)
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                # 파란색(BGR) 원으로 waypoint 표시
                cv2.circle(img, (u, v), 4, (255, 0, 0), -1)
                pts_drawn += 1
            else:
                skipped_bounds += 1
        rospy.loginfo(f"[overlay] total:{pts_total} drawn:{pts_drawn} skipZ:{skipped_z} skipOut:{skipped_bounds}")
        
        # 곡선 path 시각화
        if self.latest_smooth_path is not None:
            img = self._overlay_smooth_path(img, header)
        
        return img

    def _overlay_smooth_path(self, img: np.ndarray, header) -> np.ndarray:
        """곡선 path를 이미지 평면에 투영하여 초록색 선으로 표시"""
        if self.camera_info is None:
            return img

        # 카메라 내참수
        K = self.camera_info.K
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]

        try:
            # map -> camera transform
            trans = self.tf_buffer.lookup_transform(
                header.frame_id,
                self.latest_smooth_path.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return img

        # 곡선 path의 모든 점을 이미지 좌표로 변환
        path_points = []
        for pose_st in self.latest_smooth_path.poses:
            pt = PointStamped()
            pt.header = self.latest_smooth_path.header
            pt.point = pose_st.pose.position
            try:
                pt_cam = do_transform_point(pt, trans)
            except Exception:
                continue
            X, Y, Z = pt_cam.point.x, pt_cam.point.y, pt_cam.point.z
            if abs(Z) < 0.1:
                continue
            u = int(fx * X / abs(Z) + cx)
            v = int(fy * Y / abs(Z) + cy)
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                path_points.append((u, v))

        # 곡선 path를 보라색 선으로 그리기
        if len(path_points) > 1:
            for i in range(len(path_points) - 1):
                cv2.line(img, path_points[i], path_points[i+1], (255, 0, 255), 2)

        return img

    def _get_yaw(self, q) -> float:
        import math
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


if __name__ == "__main__":
    try:
        YOLOPInferenceNode()
    except rospy.ROSInterruptException:
        pass
