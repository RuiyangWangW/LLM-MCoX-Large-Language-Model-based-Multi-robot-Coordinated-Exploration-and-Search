# aep_strategy.py
"""
AEP Waypoint Strategy (RACER-compatible skeleton)

Key properties (MATCHES USER TEMPLATE EXACTLY):
-----------------------------------------------
1) coord_time:
   - fuse all robot local maps
   - compute DVC (Voronoi) over FREE ∪ UNKNOWN
   - for each robot:
       * sample FREE cells in its DVC
       * compute AEP gain g(x)
       * use cache + current samples
       * assign FREE cell with maximum g(x)

2) not coord_time:
   - if some robots have no waypoints:
       * for each such robot:
           - try sampling FREE cells in PREVIOUS DVC
           - if no FREE cell exists:
               fallback to FULL LOCAL MAP
           - compute g(x)
           - use cache + samples
           - assign waypoint
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from collections import deque


# ============================================================
# -------------------- DVC / VORONOI -------------------------
# ============================================================

def compute_voronoi_owner_map(
    occ: np.ndarray,
    robot_positions_rc: List[Tuple[int, int]],
    occupied_value: int = 1,
    use_8nbr: bool = True,
) -> np.ndarray:
    """
    Geodesic Voronoi over FREE ∪ UNKNOWN.
    OCCUPIED blocks propagation.
    """
    H, W = occ.shape
    owner = -np.ones((H, W), dtype=np.int32)
    dist = np.full((H, W), 10**9, dtype=np.int32)

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if use_8nbr:
        nbrs += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    q = deque()

    for ridx, (r, c) in enumerate(robot_positions_rc):
        r, c = int(r), int(c)
        if 0 <= r < H and 0 <= c < W and occ[r, c] != occupied_value:
            owner[r, c] = ridx
            dist[r, c] = 0
            q.append((r, c, ridx))

    while q:
        r, c, ridx = q.popleft()
        d0 = dist[r, c]

        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if occ[nr, nc] == occupied_value:
                continue

            nd = d0 + 1
            if nd < dist[nr, nc]:
                dist[nr, nc] = nd
                owner[nr, nc] = ridx
                q.append((nr, nc, ridx))

    return owner


# ============================================================
# --------------------- AEP GAIN -----------------------------
# ============================================================
def raycast_unknown_count(
    occ: np.ndarray,
    origin: Tuple[int, int],
    theta: float,
    max_range: float,
    step: float = 1.0,
) -> int:
    """
    Cast a ray from origin and count UNKNOWN (-1) cells until hit OCCUPIED (1) or exceed range.
    FREE (0) passes through, UNKNOWN contributes gain but does NOT block.
    """
    H, W = occ.shape
    r0, c0 = float(origin[0]), float(origin[1])
    dr, dc = math.sin(theta), math.cos(theta)

    gain = 0
    t = 0.0
    last = None

    while t <= max_range:
        r = int(round(r0 + dr * t))
        c = int(round(c0 + dc * t))
        if not (0 <= r < H and 0 <= c < W):
            break

        if (r, c) != last:
            if occ[r, c] == 1:     # OCCUPIED blocks
                break
            if occ[r, c] == -1:    # UNKNOWN adds gain
                gain += 1
            last = (r, c)

        t += step

    return gain

def compute_aep_gain_2d(
    occ: np.ndarray,
    pos: Tuple[int, int],
    sensor_range: float,
    sensor_fov: float,
    n_angles: int = 9,
    ray_step: float = 1.0,
) -> int:
    """
    AEP gain in 2D:
      - compute unknown-count gain per angle via ray casting
      - choose best yaw by max sliding-window sum matching FoV
    """
    angles = [2 * math.pi * k / n_angles for k in range(n_angles)]
    per_angle = [
        raycast_unknown_count(occ, pos, th, sensor_range, ray_step)
        for th in angles
    ]

    # FoV window over 360°
    window = max(1, int(round((sensor_fov / (2 * math.pi)) * n_angles)))
    if window >= n_angles:
        return int(sum(per_angle))

    per2 = per_angle + per_angle[:window - 1]
    s = sum(per2[:window])
    best = s
    for i in range(1, n_angles):
        s += per2[i + window - 1] - per2[i - 1]
        if s > best:
            best = s

    return int(best)


# ============================================================
# --------------------- A* COST ------------------------------
# ============================================================

def astar_cost(
    occ: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    free_mask: np.ndarray,
) -> Optional[int]:
    H, W = occ.shape
    sr, sc = start
    gr, gc = goal

    if not free_mask[gr, gc]:
        return None

    nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (-1, 1), (1, -1), (1, 1)]

    import heapq
    pq = [(0, sr, sc)]
    dist = {(sr, sc): 0}

    while pq:
        d, r, c = heapq.heappop(pq)
        if (r, c) == (gr, gc):
            return d

        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not free_mask[nr, nc]:
                continue
            nd = d + 1
            if nd < dist.get((nr, nc), 1e9):
                dist[(nr, nc)] = nd
                heapq.heappush(pq, (nd, nr, nc))

    return None


# ============================================================
# --------------------- SAMPLING -----------------------------
# ============================================================

def sample_free_cells(region_free: np.ndarray, n: int) -> List[Tuple[int, int]]:
    cells = list(zip(*np.where(region_free)))
    if not cells:
        return []
    if len(cells) <= n:
        return cells
    return random.sample(cells, n)


# ============================================================
# ---------------------- AEP CORE ----------------------------
# ============================================================

@dataclass
class CachePoint:
    rc: Tuple[int, int]
    gain: int


class AEPStrategy:
    def __init__(
        self,
        T_coord: int = 20,
        n_samples: int = 100,
        lambda_dist: float = 0.08,
        g_zero: int = 10,
        cache_gain_threshold: int = 20,
        cache_max_size: int = 200,
        min_astar_cost: int = 5,
    ):
        self.T_coord = T_coord
        self.n_samples = n_samples
        self.lambda_dist = lambda_dist
        self.g_zero = g_zero
        self.cache_gain_threshold = cache_gain_threshold
        self.cache_max_size = cache_max_size
        self.min_astar_cost = min_astar_cost

        self.initialized = False
        self.region_all: Dict[int, np.ndarray] = {}
        self.region_free: Dict[int, np.ndarray] = {}
        self.caches: Dict[int, List[CachePoint]] = {}

    def reset(self):
        self.initialized = False
        self.region_all.clear()
        self.region_free.clear()
        self.caches.clear()

    def _fuse_maps(self, robots):
        occ = robots[0].local_map.copy()
        for r in robots[1:]:
            occ = np.maximum(occ, r.local_map)
        return occ

    def _select_waypoint(self, occ, robot, ridx, region_free, region_all):
        candidates = sample_free_cells(region_free, self.n_samples)
        best_rc = None
        best_score = -1
        best_gain = 0
        free_mask = (occ == 0)
        for rc in candidates:
            g = compute_aep_gain_2d(
                occ, rc, robot.sensor_range, robot.sensor_fov
            )

            if g >= self.cache_gain_threshold:
                self.caches[ridx].append(CachePoint(rc, g))
                self.caches[ridx] = self.caches[ridx][-self.cache_max_size:]

            c = astar_cost(
                occ, tuple(map(int, robot.position)), rc, free_mask
            )
            if c is None:
                continue
            if c < self.min_astar_cost:
                continue

            s = g * math.exp(-self.lambda_dist * c)
            if s > best_score:
                best_score = s
                best_rc = rc
                best_gain = g
        
        # Cache fallback: if best gain is low, try cached points
        # IMPORTANT: Rescore cached points with current map before using
        if best_gain < self.g_zero:
            cache_candidates = []
            for cp in self.caches.get(ridx, []):
                r, c = cp.rc
                # Only consider cached points still in the region
                if not region_all[r, c]:
                    continue

                # Rescore with current occupancy map
                g_new = compute_aep_gain_2d(
                    occ, cp.rc, robot.sensor_range, robot.sensor_fov
                )

                # Only keep if still has reasonable gain
                if g_new >= self.g_zero:
                    c = astar_cost(
                        occ, tuple(map(int, robot.position)), cp.rc, free_mask
                    )
                    if c is not None:
                        if c < self.min_astar_cost:
                            continue
                        s = g_new * math.exp(-self.lambda_dist * c)
                        cache_candidates.append((s, cp.rc, g_new))

            # Select best rescored cached point
            if cache_candidates:
                cache_candidates.sort(reverse=True, key=lambda x: x[0])
                _, best_cached_rc, best_cached_gain = cache_candidates[0]

                # Update cache with new gain value                
                # print(f"Robot {ridx}: Cache fallback used with gain {best_cached_gain} (previous best {best_gain})")
                self.caches[ridx] = [CachePoint(rc, g) for _, rc, g in cache_candidates]
                self.caches[ridx] = self.caches[ridx][-self.cache_max_size:]
                if best_cached_gain > best_gain:
                    return best_cached_rc
        else:
            return best_rc
        
        return None

    # ========================================================
    # ---------------- ASSIGN WAYPOINTS ----------------------
    # ========================================================

    def assign_waypoints(self, robots, step_count: int) -> Dict[int, List[Tuple[int, int]]]:
        waypoints = {}

        coord_time = (not self.initialized) or (step_count % self.T_coord == 0)

        # ---------------- COORD TIME ----------------
        if coord_time:
            occ = self._fuse_maps(robots)
            robot_positions = [tuple(map(int, r.position)) for r in robots]
            for robot in robots:
                robot.local_map = occ.copy()
            owner = compute_voronoi_owner_map(occ, robot_positions)

            for i in range(len(robots)):
                self.region_all[i] = (owner == i)
                self.region_free[i] = self.region_all[i] & (occ == 0)
                self.caches.setdefault(i, [])

                # Clean cache: rescore and remove stale points during coordination
                # This prevents cache from filling up with obsolete waypoints
                if self.caches[i]:
                    cleaned_cache = []
                    for cp in self.caches.get(i, []):
                        r, c = cp.rc
                        if not self.region_all[i][r, c]:
                            continue
                        g_new = compute_aep_gain_2d(occ, cp.rc, robots[i].sensor_range, robots[i].sensor_fov)
                        if g_new >= self.g_zero:
                            cleaned_cache.append(CachePoint(cp.rc, g_new))
                    self.caches[i] = cleaned_cache[-self.cache_max_size:]

            self.initialized = True

            for i, robot in enumerate(robots):
                wp = self._select_waypoint(
                    occ, robot, i, self.region_free[i], self.region_all[i]
                )
                if wp is not None:
                    waypoints[i] = [wp]
            return waypoints

        # ---------------- NON-COORD TIME ----------------
        for i, robot in enumerate(robots):
            if robot.current_waypoint_idx < len(robot.waypoints):
                continue

            occ = robot.local_map
            region_free = self.region_free.get(i, None)
            region_all = self.region_all.get(i, None)

            if region_free is not None and np.any(region_free):
                wp = self._select_waypoint(
                    occ, robot, i, region_free, region_all
                )
            else:
                full_free = (occ == 0)
                full_all = (occ != 1)
                wp = self._select_waypoint(
                    occ, robot, i, full_free, full_all
                )

            if wp is not None:
                waypoints[i] = [wp]

        return waypoints
