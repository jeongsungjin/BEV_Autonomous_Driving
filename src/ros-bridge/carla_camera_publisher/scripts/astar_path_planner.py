#!/usr/bin/env python3
import math
import heapq
from typing import List, Tuple

import numpy as np
import rospy
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Odometry

# 8방향 이동 허용
DIRS = [
    (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
    (-1,  0, 1.0),                     (1,  0, 1.0),
    (-1,  1, math.sqrt(2)), (0,  1, 1.0), (1,  1, math.sqrt(2)),
]

def _normalize_angle(rad: float) -> float:
    while rad > math.pi:
        rad -= 2 * math.pi
    while rad < -math.pi:
        rad += 2 * math.pi
    return rad

# Vectorised angle difference (|wrap_to_pi(a - ref)|) for NumPy arrays
def _angle_diff(angle_arr, ref: float):
    """Return absolute smallest difference between angles (radians) and a reference angle.

    Parameters
    ----------
    angle_arr : np.ndarray or float
        Angle(s) in radians.
    ref : float
        Reference yaw angle.
    """
    return np.abs((angle_arr - ref + np.pi) % (2 * np.pi) - np.pi)

class AStarPlanner:
    def __init__(self):
        rospy.init_node("astar_path_planner", anonymous=True)

        # Params
        self.goal_row_offset = int(rospy.get_param("~goal_row_offset", 100))
        self.costmap_topic = rospy.get_param("~costmap_topic", "/carla/yolop/costmap")
        self.path_topic = rospy.get_param("~path_topic", "/planned_path")

        self.num_goal_candidates = int(rospy.get_param("~num_goal_candidates", 5))
        self.goal_search_half_width = float(rospy.get_param("~goal_search_half_width", 0.25))
        self.distance_weight = float(rospy.get_param("~distance_weight", 0.1))
        self.heading_penalty_weight = float(rospy.get_param("~heading_penalty_weight", 1.0))
        self.lateral_penalty_weight = float(rospy.get_param("~lateral_penalty_weight", 0.05))
        self.smoothness_weight = float(rospy.get_param("~smoothness_weight", 10.0))
        self.avg_cost_weight = float(rospy.get_param("~avg_cost_weight", 5.0))
        self.cell_cost_weight = float(rospy.get_param("~cell_cost_weight", 1.0))
        self.smoothing_step = float(rospy.get_param("~smoothing_step", 0.5))
        # How many additional cells to extend the path forward beyond the A* goal
        self.path_extension_cells = int(rospy.get_param("~path_extension_cells", 20))

        # Publishers/Subscribers
        self.pub_path = rospy.Publisher(self.path_topic, Path, queue_size=1)
        self.pub_candidates = rospy.Publisher("/astar_candidate_paths", MarkerArray, queue_size=1)
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self._costmap_callback, queue_size=1)
        rospy.Subscriber("/carla/ego_vehicle/odometry", Odometry, self._odom_callback, queue_size=1)

        self.current_yaw = 0.0
        rospy.loginfo("Improved A* Path Planner initialized.")
        rospy.spin()

    def _odom_callback(self, msg: Odometry):
        q = msg.pose.pose.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def _costmap_callback(self, msg: OccupancyGrid):
        width, height = msg.info.width, msg.info.height
        if width == 0 or height == 0:
            return

        grid = np.array(msg.data, dtype=np.int8).reshape((height, width))
        grid[grid < 0] = 100  # unknown(-1)은 장애물로 간주

        sx, sy = width // 2, height // 2
        scored_cells = []
        half_width = int(width * self.goal_search_half_width)

        # Only consider goal candidates that are at least `goal_row_offset` cells ahead
        max_front_row = max(0, sy - self.goal_row_offset)

        y_inds = np.arange(0, max_front_row)
        x_inds = np.arange(max(0, sx - half_width), min(width, sx + half_width))

        if y_inds.size == 0 or x_inds.size == 0:
            return

        # Build mesh grid for vectorised computation
        X, Y = np.meshgrid(x_inds, y_inds)
        C = grid[Y, X]

        valid = (C >= 0) & (C < 100)
        if not np.any(valid):
            rospy.logwarn("No valid goal candidates found.")
            return

        Xv = X[valid]
        Yv = Y[valid]
        Cv = C[valid].astype(np.float32)

        dist = sy - Yv
        vec_x = Xv - sx
        heading_diff = _angle_diff(np.arctan2(dist, vec_x), self.current_yaw)
        lateral_dist = np.abs(vec_x)

        score = Cv + dist * self.distance_weight + heading_diff * self.heading_penalty_weight + lateral_dist * self.lateral_penalty_weight

        # Convert to list of tuples for downstream logic (score, x, y)
        scored_cells = list(zip(score, Xv, Yv))

        if not scored_cells:
            rospy.logwarn("No valid goal candidates found.")
            return

        scored_cells.sort(key=lambda t: t[0])
        candidates = [(x, y, s) for s, x, y in scored_cells[:self.num_goal_candidates]]

        best_path, best_metric = None, float("inf")
        candidate_markers = MarkerArray()
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1)]

        for idx, (gx, gy, _) in enumerate(candidates):
            path_cells = self._astar(grid.copy(), (sx, sy), (gx, gy))
            if not path_cells:
                continue

            length = len(path_cells)
            smoothness = sum(
                abs(_normalize_angle(
                    math.atan2(np.array(path_cells[i + 1])[1] - np.array(path_cells[i])[1],
                               np.array(path_cells[i + 1])[0] - np.array(path_cells[i])[0]) -
                    math.atan2(np.array(path_cells[i])[1] - np.array(path_cells[i - 1])[1],
                               np.array(path_cells[i])[0] - np.array(path_cells[i - 1])[0])
                ))
                for i in range(1, len(path_cells) - 1)
            )
            avg_cost = np.mean([grid[int(p[1]), int(p[0])] for p in path_cells])

            metric = length + (smoothness * self.smoothness_weight) + (avg_cost * self.avg_cost_weight)
            if metric < best_metric:
                best_metric = metric
                best_path = path_cells

            marker = Marker()
            marker.header = msg.header
            marker.ns = "astar_candidates"
            marker.id = idx
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.1
            r, g, b = colors[idx % len(colors)]
            marker.color.r = r
            marker.color.g = g
            marker.color.b = b
            marker.color.a = 1.0
            marker.points = [Point(p[0], p[1], 0.0) for p in path_cells]
            candidate_markers.markers.append(marker)

        self.pub_candidates.publish(candidate_markers)

        if best_path is None:
            rospy.logwarn("A* failed for all candidates.")
            return

        # Optionally lengthen the path by continuing in the last direction
        best_path = self._extend_path(best_path, grid)

        rospy.loginfo(f"Best path selected with metric {best_metric:.2f}")
        self._publish_path(best_path, msg.header, msg.info)

    def _astar(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        h, w = grid.shape
        sx, sy = start
        gx, gy = goal

        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, goal), 0, start))
        came_from = {}
        g_score = {start: 0}

        while open_set:
            _, g_current, current = heapq.heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            cx, cy = current
            for dx, dy, move_cost in DIRS:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < w and 0 <= ny < h):
                    continue
                if ny >= cy:
                    continue

                cell_cost = int(grid[ny, nx])
                if cell_cost >= 100:
                    continue

                tentative_g = g_current + move_cost + (cell_cost / 100.0) * self.cell_cost_weight
                neighbor = (nx, ny)
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
        return []

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # collinear simplification
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            prev, curr, nxt = simplified[-1], path[i], path[i + 1]
            if (prev[0] - curr[0]) * (nxt[1] - curr[1]) == (prev[1] - curr[1]) * (nxt[0] - curr[0]):
                continue
            simplified.append(curr)
        simplified.append(path[-1])

        # interpolation
        dense = []
        for p0, p1 in zip(simplified[:-1], simplified[1:]):
            dense.append(p0)
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            dist = math.hypot(dx, dy)
            n_div = max(1, int(dist / self.smoothing_step))
            for j in range(1, n_div):
                t = j / n_div
                ix = p0[0] + dx * t
                iy = p0[1] + dy * t
                dense.append((ix, iy))
        dense.append(simplified[-1])

        # Chaikin smoothing
        smoothed = []
        if len(dense) < 3:
            return dense
        smoothed.append(dense[0])
        for i in range(0, len(dense) - 2):
            p0 = np.array(dense[i])
            p1 = np.array(dense[i + 1])
            Q = 0.75 * p0 + 0.25 * p1
            R = 0.25 * p0 + 0.75 * p1
            smoothed.append(tuple(Q))
            smoothed.append(tuple(R))
        smoothed.append(dense[-1])
        return smoothed

    def _extend_path(self, path: List[Tuple[int, int]], grid: np.ndarray) -> List[Tuple[int, int]]:
        """Extend the final path forward in the same heading to make it longer."""
        if len(path) < 2 or self.path_extension_cells <= 0:
            return path

        # direction of last segment
        x_prev, y_prev = path[-2]
        x_last, y_last = path[-1]
        dx, dy = x_last - x_prev, y_last - y_prev
        norm = math.hypot(dx, dy)
        if norm == 0:
            return path

        step_x, step_y = dx / norm, dy / norm
        x, y = x_last, y_last
        h, w = grid.shape

        for _ in range(self.path_extension_cells):
            x += step_x
            y += step_y
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < w and 0 <= iy < h):
                break
            if grid[iy, ix] >= 100:
                break
            path.append((x, y))

        return path

    def _publish_path(self, path_cells: List[Tuple[int, int]], header, info):
        path_msg = Path()
        path_msg.header = header
        origin = info.origin.position
        res = info.resolution
        poses = []
        for (x, y) in path_cells:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = origin.x + (x + 0.5) * res
            pose.pose.position.y = origin.y + (y + 0.5) * res
            pose.pose.position.z = 0.0
            pose.pose.orientation = Quaternion(0, 0, 0, 1)
            poses.append(pose)
        path_msg.poses = poses
        self.pub_path.publish(path_msg)
        rospy.loginfo(f"Published A* path with {len(poses)} points.")

if __name__ == "__main__":
    try:
        AStarPlanner()
    except rospy.ROSInterruptException:
        pass
