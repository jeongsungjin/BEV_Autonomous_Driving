#!/usr/bin/env python3

import os
import sys
import time
import threading
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import rospy
import torch
import torchvision.transforms as transforms
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from nav_msgs.msg import Path as RosPath, OccupancyGrid, MapMetaData
import message_filters
import tf2_ros
from geometry_msgs.msg import PointStamped, Pose, Point, Quaternion
from tf2_geometry_msgs import do_transform_point  # type: ignore

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


class YOLOPBatchInferenceNode:
    def __init__(self):
        rospy.init_node("yolop_batch_inference_node", anonymous=True)

        # 파라미터
        self.weight_path = rospy.get_param(
            "~weight_path",
            "/home/carla/capstone_2025/src/ros-bridge/YOLOP/weights/epoch-240.pth",
        )
        self.conf_thres = float(rospy.get_param("~conf_thres", 0.5))
        self.iou_thres = float(rospy.get_param("~iou_thres", 0.45))
        self.input_size = tuple(cfg.MODEL.IMAGE_SIZE)
        self.num_vehicles = 3

        # --- additional params for BEV mask tuning ---
        self.da_thresh = float(rospy.get_param("~da_thresh", 0.53))  # drivable area prob threshold
        self.ll_thresh = float(rospy.get_param("~ll_thresh", 0.5))  # lane line prob threshold

        # global debug flag
        self.debug = rospy.get_param("~debug", False)

        # 모델 초기화
        self.device = select_device(device="0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.model.to(self.device).eval()

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.bridge = CvBridge()

        # === Message filters for synchronization ===
        self.image_subs = []
        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            topic_name = f"/carla/vehicle{vehicle_id}/image_raw"
            sub = message_filters.Subscriber(topic_name, Image)
            self.image_subs.append(sub)
            rospy.loginfo(f"Subscribed to {topic_name}")

        # 시간 동기화 (더 관대한 허용 오차로 설정)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            self.image_subs, queue_size=10, slop=0.1, allow_headerless=True
        )
        self.sync.registerCallback(self.synchronized_callback)
        
        # 디버깅을 위한 개별 콜백도 등록 (필요시 활성화)
        # for i, sub in enumerate(self.image_subs):
        #     vehicle_id = i + 1
        #     sub.registerCallback(lambda msg, vid=vehicle_id: self._debug_image_callback(msg, vid))

        # === Waypoint overlay 준비 ===
        self.latest_paths = [None] * self.num_vehicles  # type: List[RosPath]
        self.latest_smooth_paths = [None] * self.num_vehicles  # type: List[RosPath]
        self.camera_infos = [None] * self.num_vehicles  # type: List[CameraInfo]
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # 각 차량별 카메라 정보 구독
        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            cam_info_topic = f"/carla/vehicle{vehicle_id}/camera_info"
            rospy.Subscriber(
                cam_info_topic, CameraInfo, 
                lambda msg, vid=vehicle_id: self._camerainfo_callback(msg, vid), 
                queue_size=1
            )

        # Path 구독 (전체 공유 - 필요시 각 차량별로 변경 가능)
        rospy.Subscriber("/planned_path", RosPath, self._path_callback, queue_size=1)
        rospy.Subscriber("/planned_smooth_path", RosPath, self._smooth_path_callback, queue_size=1)

        # === 결과 발행용 퍼블리셔 ===
        self.vis_pubs = []
        self.det_pubs = []
        self.da_pubs = []
        self.ll_pubs = []
        self.costmap_pubs = []

        # --- costmap 모드용 퍼블리셔(통합 OccupancyGrid) ---
        self.enable_costmap = rospy.get_param("~enable_costmap", True)

        for i in range(self.num_vehicles):
            vehicle_id = i + 1
            self.vis_pubs.append(
                rospy.Publisher(f"/carla/vehicle{vehicle_id}/yolop/inference", Image, queue_size=10)
            )
            self.det_pubs.append(
                rospy.Publisher(f"/carla/vehicle{vehicle_id}/yolop/det_grid", OccupancyGrid, queue_size=10)
            )
            self.da_pubs.append(
                rospy.Publisher(f"/carla/vehicle{vehicle_id}/yolop/da_grid", OccupancyGrid, queue_size=10)
            )
            self.ll_pubs.append(
                rospy.Publisher(f"/carla/vehicle{vehicle_id}/yolop/ll_grid", OccupancyGrid, queue_size=10)
            )
            
            # 통합 costmap
            if self.enable_costmap:
                self.costmap_pubs.append(
                    rospy.Publisher(f"/carla/vehicle{vehicle_id}/yolop/costmap", OccupancyGrid, queue_size=10)
                )

        # === BEV Raster 저장 디렉터리 (옵션) ===
        if rospy.get_param("~save_bev", False):
            bev_dir_param = rospy.get_param("~bev_save_dir", str(Path.home() / ".ros" / "bev_rasters_batch"))
            self.bev_save_dir = Path(bev_dir_param)
            self.bev_save_dir.mkdir(parents=True, exist_ok=True)

        # 추론 통계
        self.inference_count = 0
        self.total_inference_time = 0.0

        rospy.loginfo("YOLOP 배치 추론 노드가 시작되었습니다.")
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

    def _debug_image_callback(self, msg: Image, vehicle_id: int):
        """개별 이미지 메시지 디버깅용 콜백"""
        rospy.loginfo(f"[DEBUG] Vehicle {vehicle_id} 이미지 수신: stamp={msg.header.stamp.to_sec():.3f}")

    def _camerainfo_callback(self, msg: CameraInfo, vehicle_id: int):
        self.camera_infos[vehicle_id - 1] = msg
        # rospy.loginfo(f"[DEBUG] Vehicle {vehicle_id} 카메라 정보 수신")

    def _path_callback(self, msg: RosPath):
        # 모든 차량이 같은 경로를 공유한다고 가정
        for i in range(self.num_vehicles):
            self.latest_paths[i] = msg
        # rospy.loginfo(f"[DEBUG] 경로 메시지 수신: {len(msg.poses)} 개 waypoints")

    def _smooth_path_callback(self, msg: RosPath):
        # 모든 차량이 같은 smooth 경로를 공유한다고 가정
        for i in range(self.num_vehicles):
            self.latest_smooth_paths[i] = msg
        # rospy.loginfo(f"[DEBUG] Smooth 경로 메시지 수신: {len(msg.poses)} 개 waypoints")

    def synchronized_callback(self, *image_msgs):
        """시간 동기화된 이미지들을 받아서 배치 추론 수행"""
        start_time = time.time()
        
        # rospy.loginfo(f"[SYNC] 동기화된 이미지 {len(image_msgs)}개 수신")
        
        if len(image_msgs) != self.num_vehicles:
            rospy.logwarn(f"Expected {self.num_vehicles} images, got {len(image_msgs)}")
            return

        try:
            # 이미지 전처리 - 배치로 준비
            batch_frames = []
            batch_orig_shapes = []
            batch_headers = []

            for img_msg in image_msgs:
                frame_bgr = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
                batch_frames.append(frame_bgr)
                batch_orig_shapes.append(frame_bgr.shape[:2])
                batch_headers.append(img_msg.header)

            # 배치 추론 수행
            batch_results = self._run_batch_inference(batch_frames, batch_orig_shapes, batch_headers)

            # 결과 발행
            for vehicle_id, (vis_img, det_grid, da_grid, ll_grid) in enumerate(batch_results, 1):
                # 시각화 이미지
                ros_image = self.bridge.cv2_to_imgmsg(vis_img, encoding="bgr8")
                ros_image.header = batch_headers[vehicle_id - 1]
                self.vis_pubs[vehicle_id - 1].publish(ros_image)

                # Grid 메시지들
                if det_grid is not None:
                    self.det_pubs[vehicle_id - 1].publish(det_grid)
                if da_grid is not None:
                    self.da_pubs[vehicle_id - 1].publish(da_grid)
                if ll_grid is not None:
                    self.ll_pubs[vehicle_id - 1].publish(ll_grid)

            # 성능 통계
            inference_time = time.time() - start_time
            self.inference_count += 1
            self.total_inference_time += inference_time
            avg_time = self.total_inference_time / self.inference_count

            if self.inference_count % 30 == 0:  # 30번마다 로그
                rospy.loginfo(f"배치 추론 통계 - 평균 시간: {avg_time:.3f}s, FPS: {1.0/avg_time:.1f}")

        except Exception as e:
            rospy.logerr(f"배치 추론 중 오류 발생: {e}")

    def _run_batch_inference(self, batch_frames: List[np.ndarray], batch_orig_shapes: List[Tuple], 
                           batch_headers: List) -> List[Tuple]:
        """배치로 YOLOP 추론을 수행"""
        
        # 1. 이미지 전처리 - 배치로 처리
        batch_tensors = []
        batch_ratios = []
        batch_pads = []

        for frame_bgr in batch_frames:
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            img_resized, ratio, pad = letterbox_for_img(
                img_rgb, new_shape=self.input_size, auto=False, scaleFill=False, scaleup=False
            )
            img_tensor = transforms.ToTensor()(img_resized)
            img_tensor = self.normalize(img_tensor)
            batch_tensors.append(img_tensor)
            batch_ratios.append(ratio)
            batch_pads.append(pad)

        # 배치 텐서 생성
        batch_input = torch.stack(batch_tensors).to(self.device)  # [batch_size, 3, H, W]

        # 2. 배치 추론
        with torch.no_grad():
            batch_outputs = self.model(batch_input)

        # 모델 출력 구조 확인 및 처리
        if not isinstance(batch_outputs, (list, tuple)) or len(batch_outputs) != 3:
            rospy.logerr(f"Unexpected model output structure: {type(batch_outputs)}, length: {len(batch_outputs) if hasattr(batch_outputs, '__len__') else 'N/A'}")
            return []

        batch_det_out, batch_da_seg_out, batch_ll_seg_out = batch_outputs
        
        # 디버그 정보 출력 (첫 번째 배치에서만)
        if self.debug and self.inference_count == 1:
            rospy.loginfo(f"Batch output types: det={type(batch_det_out)}, da={type(batch_da_seg_out)}, ll={type(batch_ll_seg_out)}")
            if hasattr(batch_det_out, 'shape'):
                rospy.loginfo(f"Detection output shape: {batch_det_out.shape}")
            elif isinstance(batch_det_out, (list, tuple)):
                rospy.loginfo(f"Detection output is tuple/list with length: {len(batch_det_out)}")
                if len(batch_det_out) > 0 and hasattr(batch_det_out[0], 'shape'):
                    rospy.loginfo(f"Detection output[0] shape: {batch_det_out[0].shape}")
            
            if hasattr(batch_da_seg_out, 'shape'):
                rospy.loginfo(f"DA segmentation output shape: {batch_da_seg_out.shape}")
            if hasattr(batch_ll_seg_out, 'shape'):
                rospy.loginfo(f"LL segmentation output shape: {batch_ll_seg_out.shape}")
        
        # Detection 출력이 tuple인 경우 첫 번째 요소 사용
        if isinstance(batch_det_out, (list, tuple)):
            batch_det_pred = batch_det_out[0]
        else:
            batch_det_pred = batch_det_out
        
        # 3. 후처리 - 각 이미지별로 처리
        results = []
        for i in range(len(batch_frames)):
            frame_bgr = batch_frames[i]
            orig_shape = batch_orig_shapes[i]
            header = batch_headers[i]
            
            try:
                # Detection NMS - 개별 이미지의 prediction 추출
                single_det_pred = batch_det_pred[i:i+1]  # [1, num_anchors, 85]
                det_nms = non_max_suppression(single_det_pred, self.conf_thres, self.iou_thres)[0]
                
                # Segmentation 마스크 추출 - 안전한 방식으로
                try:
                    # DA mask 처리
                    if isinstance(batch_da_seg_out, (list, tuple)):
                        da_tensor = batch_da_seg_out[i] if len(batch_da_seg_out) > i else batch_da_seg_out[0][i]
                    else:
                        da_tensor = batch_da_seg_out[i]
                    
                    da_mask = da_tensor.squeeze().cpu().numpy()
                    if da_mask.ndim == 3:
                        da_mask = da_mask[1]  # 클래스 1 (drivable area)
                    
                    # LL mask 처리  
                    if isinstance(batch_ll_seg_out, (list, tuple)):
                        ll_tensor = batch_ll_seg_out[i] if len(batch_ll_seg_out) > i else batch_ll_seg_out[0][i]
                    else:
                        ll_tensor = batch_ll_seg_out[i]
                    
                    ll_mask = ll_tensor.squeeze().cpu().numpy()
                    if ll_mask.ndim == 3:
                        ll_mask = ll_mask[1]  # 클래스 1 (lane line)
                        
                except Exception as e:
                    rospy.logwarn(f"Error extracting segmentation masks for image {i}: {e}")
                    # 기본값으로 빈 마스크 사용
                    da_mask = np.zeros(orig_shape, dtype=np.float32)
                    ll_mask = np.zeros(orig_shape, dtype=np.float32)
                
            except Exception as e:
                rospy.logerr(f"Error processing image {i} in batch: {e}")
                # 기본값으로 설정
                det_nms = None
                da_mask = np.zeros(orig_shape, dtype=np.float32)
                ll_mask = np.zeros(orig_shape, dtype=np.float32)

            # 개별 후처리 및 시각화
            vis_img, det_grid, da_grid, ll_grid = self._postprocess_single(
                frame_bgr, orig_shape, header, det_nms, da_mask, ll_mask, i + 1
            )
            
            results.append((vis_img, det_grid, da_grid, ll_grid))

        return results

    def _postprocess_single(self, frame_bgr: np.ndarray, orig_shape: Tuple, header, 
                           det_nms, da_mask: np.ndarray, ll_mask: np.ndarray, vehicle_id: int) -> Tuple:
        """개별 이미지에 대한 후처리"""
        
        vis_img = frame_bgr.copy()
        
        # === Detection 처리 ===
        det_mask = np.zeros(orig_shape, dtype=np.uint8)
        
        if det_nms is not None and len(det_nms):
            det_nms[:, :4] = scale_coords(
                self.input_size, det_nms[:, :4], orig_shape
            )
            for *xyxy, conf, cls in det_nms:
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis_img, f"obj {conf:.2f}", (x1, max(y1 - 5, 0)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
                det_mask[y1:y2, x1:x2] = 1

        # 마스크 확장
        kernel = np.ones((5, 5), np.uint8)
        det_mask = cv2.dilate(det_mask, kernel, iterations=1)

        # === Segmentation 처리 ===
        # 1. input_size 크기로 upsample
        da_mask = cv2.resize(da_mask, self.input_size[::-1])
        ll_mask = cv2.resize(ll_mask, self.input_size[::-1])

        # 2. letterbox 제거
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

        # 마스크 후처리
        ll_bin = cv2.dilate(ll_bin, kernel, iterations=1)
        vertical_kernel = np.ones((15, 3), np.uint8)
        da_bin = cv2.dilate(da_bin, vertical_kernel, iterations=3)

        # if self.debug:
        #     rospy.loginfo(
        #         f"[YOLOP-V{vehicle_id}] mask ratios det={det_mask.mean():.3f} da={da_bin.mean():.3f} ll={ll_bin.mean():.3f}"
        #     )

        # === 시각화 ===
        # 색상 합성
        color_da = np.zeros_like(vis_img)
        color_da[:, :, 1] = da_bin * 255
        color_ll = np.zeros_like(vis_img)
        color_ll[:, :, 2] = ll_bin * 255

        vis_img = cv2.addWeighted(vis_img, 1.0, color_da, 0.4, 0)
        vis_img = cv2.addWeighted(vis_img, 1.0, color_ll, 0.4, 0)

        # Ego vehicle 박스
        h_img, w_img = vis_img.shape[:2]
        box_w, box_h = 30, 50
        cx, cy = w_img // 2, h_img // 2
        top_left = (cx - box_w // 2, cy - box_h // 2)
        bottom_right = (cx + box_w // 2, cy + box_h // 2)
        cv2.rectangle(vis_img, top_left, bottom_right, (255, 0, 255), 2)

        # Waypoint 오버레이
        vis_img = self._overlay_waypoints(vis_img, header, vehicle_id)

        # 좌표계 정합: 원본 그대로 사용해서 실제 환경 확인
        # 회전 없이 YOLOP 원본 BEV 그리드 사용
        det_mask_rotated = det_mask.copy()  # 회전 없음
        da_bin_rotated = da_bin.copy()
        ll_bin_rotated = ll_bin.copy()
        
        # OccupancyGrid 메시지 생성 (ego_vehicle 프레임으로 통일)
        det_grid_msg = self._build_occupancy_grid(det_mask_rotated, header.stamp, "ego_vehicle", occupied_val=100, free_val=0)
        # DA: 주행 가능 영역은 free(0), 그 외는 occupied(100)
        da_occ_mask = np.where(da_bin_rotated == 1, 0, 1).astype(np.uint8)
        da_grid_msg = self._build_occupancy_grid(da_occ_mask, header.stamp, "ego_vehicle", occupied_val=100, free_val=0)
        # LL: 차선은 occupied(100)로 표현
        ll_grid_msg = self._build_occupancy_grid(ll_bin_rotated, header.stamp, "ego_vehicle", occupied_val=100, free_val=0)

        # === 통합 Costmap OccupancyGrid 생성 ===
        if self.enable_costmap and vehicle_id <= len(self.costmap_pubs):
            # 회전된 마스크 사용
            combined_mask = np.full(det_mask_rotated.shape, 99, dtype=np.int8)

            # 1) Drivable Area (DA): 내부 0, 경계 90
            kernel3 = np.ones((3, 3), np.uint8)
            da_eroded = cv2.erode(da_bin_rotated, kernel3, iterations=2)
            da_border = cv2.bitwise_and(da_bin_rotated, cv2.bitwise_not(da_eroded))

            combined_mask[da_eroded == 1] = 0      # 자유 공간
            combined_mask[da_border == 1] = 90     # 경계부 cost ↑

            # 2) Lane Line (LL): 장애물과 동일하게 100
            ll_dilated = cv2.dilate(ll_bin_rotated, kernel3, iterations=1)
            combined_mask[ll_dilated == 1] = 100

            # 3) Detection (Det): 장애물 100
            det_dilated = cv2.dilate(det_mask_rotated, kernel3, iterations=2)
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
            costmap_msg.header.frame_id = "ego_vehicle"  # 다른 그리드들과 프레임 통일
            costmap_msg.info = meta
            costmap_msg.data = combined_mask.flatten().tolist()
            self.costmap_pubs[vehicle_id - 1].publish(costmap_msg)

        # === BEV Raster 저장 ===
        if hasattr(self, 'bev_save_dir'):
            try:
                # 좌표계 정합된 회전 마스크 저장
                raster = np.stack([det_mask_rotated, da_bin_rotated, ll_bin_rotated], axis=0).astype(np.uint8)
                fname = self.bev_save_dir / f"v{vehicle_id}_{header.stamp.to_sec():.6f}.npy"
                np.save(fname, raster)
            except Exception as e:
                rospy.logwarn_once(f"BEV raster 저장 실패 (Vehicle {vehicle_id}): {e}")

        return vis_img, det_grid_msg, da_grid_msg, ll_grid_msg

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

    def _build_occupancy_grid_direct(self, data: np.ndarray, header_stamp: rospy.Time, frame_id: str) -> OccupancyGrid:
        """직접 데이터를 OccupancyGrid로 변환"""
        meta = MapMetaData()
        meta.map_load_time = rospy.Time.now()
        meta.resolution = 1.0
        meta.width = data.shape[1]
        meta.height = data.shape[0]
        meta.origin = Pose(position=Point(0.0, 0.0, 0.0),
                          orientation=Quaternion(0.0, 0.0, 0.0, 1.0))

        grid = OccupancyGrid()
        grid.header.stamp = header_stamp
        grid.header.frame_id = frame_id
        grid.info = meta
        grid.data = data.flatten().tolist()
        return grid

    def _overlay_waypoints(self, img: np.ndarray, header, vehicle_id: int) -> np.ndarray:
        """/planned_path 의 waypoints 를 이미지 평면에 투영하여 파란 점으로 표시하고, 곡선 path를 초록색 선으로 표시"""
        if (self.latest_paths[vehicle_id - 1] is None or 
            self.camera_infos[vehicle_id - 1] is None):
            return img

        camera_info = self.camera_infos[vehicle_id - 1]
        latest_path = self.latest_paths[vehicle_id - 1]

        # 카메라 내참수
        K = camera_info.K  # 3x3 row-major list
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]

        try:
            # map -> camera transform (at latest available)
            trans = self.tf_buffer.lookup_transform(
                header.frame_id,  # target: camera frame
                latest_path.header.frame_id,  # source: map
                rospy.Time(0),
                rospy.Duration(0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return img

        pts_drawn = 0
        pts_total = len(latest_path.poses)
        skipped_z = 0
        skipped_bounds = 0
        for pose_st in latest_path.poses:
            pt = PointStamped()
            pt.header = latest_path.header
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
        # rospy.loginfo(f"[overlay-V{vehicle_id}] total:{pts_total} drawn:{pts_drawn} skipZ:{skipped_z} skipOut:{skipped_bounds}")
        
        # 곡선 path 시각화
        if self.latest_smooth_paths[vehicle_id - 1] is not None:
            img = self._overlay_smooth_path(img, header, vehicle_id)
        
        return img

    def _overlay_smooth_path(self, img: np.ndarray, header, vehicle_id: int) -> np.ndarray:
        """곡선 path를 이미지 평면에 투영하여 초록색 선으로 표시"""
        if self.camera_infos[vehicle_id - 1] is None:
            return img

        camera_info = self.camera_infos[vehicle_id - 1]
        smooth_path = self.latest_smooth_paths[vehicle_id - 1]

        # 카메라 내참수
        K = camera_info.K
        fx, fy = K[0], K[4]
        cx, cy = K[2], K[5]

        try:
            # map -> camera transform
            trans = self.tf_buffer.lookup_transform(
                header.frame_id,
                smooth_path.header.frame_id,
                rospy.Time(0),
                rospy.Duration(0.05)
            )
        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
            return img

        # 곡선 path의 모든 점을 이미지 좌표로 변환
        path_points = []
        for pose_st in smooth_path.poses:
            pt = PointStamped()
            pt.header = smooth_path.header
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
        YOLOPBatchInferenceNode()
    except rospy.ROSInterruptException:
        pass 