"""Speed limit from Nav2's costmap: slow down near obstacles on the path ahead, stop short of them."""
import math
from typing import NamedTuple

import numpy as np

# Defaults, under the names the controller declares them as parameters
GOVERNOR_PARAMS = {
    'use_costmap': True,
    'slow_distance': 2.0,          # m: full speed beyond this clearance
    'stop_distance': 0.35,         # m: zero at or below it
    'obstacle_creep_speed': 0.15,  # m/s: the obstacle limit's floor short of stop_distance
    'clearance_lookahead': 3.0,    # m of path checked
    'costmap_timeout': 1.0,        # s: older than this counts as no costmap
    'crawl_speed': 0.3,            # m/s: the cap without a usable costmap
    'goal_decel': 0.8,             # m/s^2: braking for the end of the path (End of Path)
    'goal_creep_speed': 0.15,      # m/s: the taper never goes below this
}

# Nav2's OccupancyGrid: 100 lethal, 99 inscribed, 1-98 inflation, 0 free, -1 unknown
INSCRIBED = 99

# Spacing of the points checked along the path, about one costmap cell
PATH_STEP = 0.1

# How much closer (m) the path has to bring the robot to a cell for that cell to count
CLOSER_BY = 0.01


class Costmap(NamedTuple):
    cells: np.ndarray   # (height, width) int8; row = y, column = x
    resolution: float   # m per cell
    origin: tuple       # (x, y) of the outer corner of cell [0, 0], in `frame`
    frame: str

    @classmethod
    def from_msg(cls, msg):
        """From a nav_msgs/OccupancyGrid. Nav2 never rotates the origin, so its orientation is ignored."""
        info = msg.info
        cells = np.asarray(msg.data, dtype=np.int8).reshape(info.height, info.width)
        return cls(cells, info.resolution, (info.origin.position.x, info.origin.position.y),
                   msg.header.frame_id)


def governor_param_problem(use_costmap, slow_distance, stop_distance, obstacle_creep_speed,
                           clearance_lookahead, costmap_timeout, crawl_speed, goal_decel,
                           goal_creep_speed):
    """Return why these settings are unusable, or None when they are fine."""
    if not obstacle_creep_speed > 0.0:
        return (f'obstacle_creep_speed must be positive (or the robot never reaches stop_distance), '
                f'got {obstacle_creep_speed}')
    if not goal_decel > 0.0:
        return f'goal_decel must be positive, got {goal_decel}'
    if not goal_creep_speed > 0.0:
        return f'goal_creep_speed must be positive (or the robot never reaches the goal), got {goal_creep_speed}'
    if not stop_distance >= 0.0:
        return f'stop_distance must be >= 0 m, got {stop_distance}'
    if not slow_distance > stop_distance:
        return f'slow_distance must be greater than stop_distance ({stop_distance} m), got {slow_distance}'
    if not clearance_lookahead > 0.0:
        return f'clearance_lookahead must be positive, got {clearance_lookahead}'
    if not costmap_timeout > 0.0:
        return f'costmap_timeout must be positive, got {costmap_timeout}'
    if not crawl_speed >= 0.0:
        return f'crawl_speed must be >= 0 m/s, got {crawl_speed}'
    return None


def speed_for_clearance(clearance, max_speed, slow_distance, stop_distance, creep_speed):
    """max_speed beyond slow_distance, then linearly down towards 0 at stop_distance but never
    below creep_speed (nor above max_speed); 0 at or below stop_distance."""
    if clearance <= stop_distance:
        return 0.0
    if clearance >= slow_distance:
        return float(max_speed)
    ramp = max_speed * (clearance - stop_distance) / (slow_distance - stop_distance)
    return float(min(max(ramp, creep_speed), max_speed))


def to_frame(points, frame_pose):
    """Map-frame points (N, 2) into a frame whose pose in map is (x, y, yaw)."""
    x, y, yaw = frame_pose
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(points, dtype=float) - (x, y)
    return np.column_stack((c * d[:, 0] + s * d[:, 1], c * d[:, 1] - s * d[:, 0]))


def path_progress(position, path_xy, path_s, start_idx):
    """Arc length along the path of `position`'s projection onto it. Like get_local_reference,
    the projection is searched from `start_idx`, the progress so far."""
    position = np.asarray(position, dtype=float)[:2]
    end = min(len(path_xy), start_idx + 50)
    i = start_idx + int(np.argmin(np.linalg.norm(path_xy[start_idx:end] - position, axis=1)))
    best, s0 = np.linalg.norm(path_xy[i] - position), float(path_s[i])
    for j in (i - 1, i):   # the segments either side of the closest point
        if 0 <= j < len(path_xy) - 1:
            a, v = path_xy[j], path_xy[j + 1] - path_xy[j]
            seg2 = float(v @ v)
            if seg2 > 0.0:
                t = min(max(float((position - a) @ v) / seg2, 0.0), 1.0)
                d = np.linalg.norm(a + t * v - position)
                if d < best:
                    best, s0 = d, float(path_s[j] + t * math.sqrt(seg2))
    return s0


def path_ahead(position, path_xy, path_s, start_idx, length, step=PATH_STEP):
    """`position`, then points every `step` m along the path, from the position's projection
    onto the path to `length` m further on (or the end of the path)."""
    position = np.asarray(position, dtype=float)[:2]
    s0 = path_progress(position, path_xy, path_s, start_idx)
    s1 = min(s0 + length, path_s[-1])
    s = np.append(np.arange(s0, s1, step), s1)
    ahead = np.column_stack((np.interp(s, path_s, path_xy[:, 0]), np.interp(s, path_s, path_xy[:, 1])))
    return np.vstack((position, ahead))


def obstacle_clearance(costmap, points, stop_distance):
    """Distance from points[0] (the robot) to the nearest inscribed or lethal cell that the path
    points[1:] passes within stop_distance of and brings the robot closer to. All in the
    costmap's frame. inf when there is none."""
    res = costmap.resolution
    ox, oy = costmap.origin
    height, width = costmap.cells.shape
    # Only cells within stop_distance of a path point can count
    lo = points.min(axis=0) - stop_distance
    hi = points.max(axis=0) + stop_distance
    c0, c1 = max(int((lo[0] - ox) // res), 0), min(int((hi[0] - ox) // res) + 1, width)
    r0, r1 = max(int((lo[1] - oy) // res), 0), min(int((hi[1] - oy) // res) + 1, height)
    if c0 >= c1 or r0 >= r1:
        return math.inf
    rows, cols = np.nonzero(costmap.cells[r0:r1, c0:c1] >= INSCRIBED)
    if rows.size == 0:
        return math.inf
    centres = np.column_stack((ox + (cols + c0 + 0.5) * res, oy + (rows + r0 + 0.5) * res))
    # To the nearest point of each cell, not its centre: (cells, points)
    gap = np.maximum(np.abs(centres[:, None, :] - points[None, :, :]) - res / 2, 0.0)
    dist = np.hypot(gap[..., 0], gap[..., 1])
    now = dist[:, 0]
    ahead = dist[:, 1:].min(axis=1) if len(points) > 1 else now
    counts = (ahead < stop_distance) & (ahead < now - CLOSER_BY)
    return float(now[counts].min()) if counts.any() else math.inf


def speed_limit(costmap, age, points, costmap_pose, max_speed, slow_distance, stop_distance,
                creep_speed, costmap_timeout, crawl_speed):
    """The speed limit for a robot at points[0] about to follow points[1:] (map frame).

    costmap: a Costmap or None; age: s since it arrived; costmap_pose: its frame's pose in map
    as (x, y, yaw), or None if unknown. Returns (limit m/s, clearance m or None, status), status
    one of 'clear', 'slow', 'stop' (fresh costmap) or 'missing', 'unplaced', 'stale' (capped at
    crawl_speed; a stale costmap still stops the robot for what it shows)."""
    crawl = min(crawl_speed, max_speed)
    if costmap is None:
        return crawl, None, 'missing'
    if costmap_pose is None:
        return crawl, None, 'unplaced'
    clearance = obstacle_clearance(costmap, to_frame(points, costmap_pose), stop_distance)
    limit = speed_for_clearance(clearance, max_speed, slow_distance, stop_distance, creep_speed)
    if age > costmap_timeout:
        return min(limit, crawl), clearance, 'stale'
    return limit, clearance, 'stop' if limit <= 0.0 else 'slow' if limit < max_speed else 'clear'

def end_taper_speed(remaining, decel, creep_speed):
    """The fastest the robot can go `remaining` m (arc length) before the end of the path and
    still stop there braking at `decel`: sqrt(2 decel remaining), but never below creep_speed."""
    return max(creep_speed, math.sqrt(2.0 * decel * max(remaining, 0.0)))
