#!/usr/bin/env python3
"""
YOLOP Grid Fusion Node
======================

det_grid, da_grid, ll_grid 세 OccupancyGrid 토픽을 동기화하여 하나의 글로벌 OccupancyGrid 로 통합한다.

규칙
-----
1. 주행 가능 영역(da)
   - 값이 0 (free)  -> 기본적으로 free (0)
   - 값이 100       -> 장애물 (100)
2. 차선 영역(ll)
   - 값이 100       -> 차선 셀로 표시 (50)
3. 탐지 결과(det)
   - 값이 100       -> 최우선으로 장애물 (100)

최종 가중치
===========
- -1 : unknown (초기화 시)
- 0  : free
- 50 : 차선 (low cost)
- 100: 장애물 (high cost)

"""
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from message_filters import ApproximateTimeSynchronizer, Subscriber

class YOLOPGridFusionNode:
    def __init__(self):
        rospy.init_node("yolop_grid_fusion_node", anonymous=True)

        # 파라미터
        self.output_topic = rospy.get_param("~output_topic", "/carla/yolop/fused_grid")
        queue_size = rospy.get_param("~queue_size", 10)
        sync_slop = rospy.get_param("~sync_slop", 0.05)  # seconds

        # 구독자 & 동기화
        sub_det = Subscriber("/carla/yolop/det_grid", OccupancyGrid)
        sub_da = Subscriber("/carla/yolop/da_grid", OccupancyGrid)
        sub_ll = Subscriber("/carla/yolop/ll_grid", OccupancyGrid)

        self.ts = ApproximateTimeSynchronizer([sub_det, sub_da, sub_ll], queue_size, sync_slop, allow_headerless=False)
        self.ts.registerCallback(self._callback)

        self.pub = rospy.Publisher(self.output_topic, OccupancyGrid, queue_size=10)

        rospy.loginfo("YOLOP Grid Fusion Node started. Output: %s", self.output_topic)
        rospy.spin()

    def _callback(self, det: OccupancyGrid, da: OccupancyGrid, ll: OccupancyGrid):
        # 기본 메타데이터는 da grid 사용 (3개가 동일해야 함)
        h, w = da.info.height, da.info.width
        total_cells = h * w
        if any(grid.info.height != h or grid.info.width != w for grid in [det, ll]):
            rospy.logwarn("Grid size mismatch. Skipping fusion.")
            return

        # numpy 배열로 변환
        det_np = np.array(det.data, dtype=np.int8).reshape(h, w)
        da_np = np.array(da.data, dtype=np.int8).reshape(h, w)
        ll_np = np.array(ll.data, dtype=np.int8).reshape(h, w)

        # 초기 unknown
        fused_np = np.full((h, w), -1, dtype=np.int8)

        # 1) 주행 가능 영역 기준 free / obstacle
        fused_np[da_np == 0] = 0          # free
        fused_np[da_np == 100] = 100      # obstacle

        # 2) 차선 덮어쓰기 (lane = 50) – 단, obstacle 셀은 유지
        mask_lane = (ll_np == 100) & (fused_np != 100)
        fused_np[mask_lane] = 50

        # 3) 탐지 결과 최우선 덮어쓰기 (obstacle=100)
        fused_np[det_np == 100] = 100

        # OccupancyGrid 메시지 생성
        fused_msg = OccupancyGrid()
        fused_msg.header.stamp = rospy.Time.now()
        fused_msg.header.frame_id = da.header.frame_id if da.header.frame_id else "yolop_fused_grid"
        fused_msg.info = da.info  # meta 복사
        fused_msg.data = fused_np.flatten().tolist()

        self.pub.publish(fused_msg)

if __name__ == "__main__":
    try:
        YOLOPGridFusionNode()
    except rospy.ROSInterruptException:
        pass 