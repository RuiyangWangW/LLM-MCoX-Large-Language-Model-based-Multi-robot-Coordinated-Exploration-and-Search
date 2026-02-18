"""
Robot class for navigation in semantic grid maps.

Features:
- 2D point robot on grid map
- Single sensor with configurable FoV and range
- Local map building from observations
- A* navigation to waypoints
"""

import numpy as np
from collections import deque
import heapq
from typing import Tuple, List, Optional, Set, Dict



def extract_frontier_clusters(
    global_map: np.ndarray,
    unknown_value: int = -1,
    free_value: int = 0,
    max_cluster_distance: int = 10,
) -> List[Dict]:
    """
    Extract planning-level frontier clusters from a grid map.

    Frontier cell:
        FREE cell adjacent to UNKNOWN

    Frontier cluster:
        Frontier cells connected via FREE space
        (BFS with depth limit)

    Returns a list of frontier cluster dicts with:
        - id
        - rep (representative frontier cell)
        - centroid
        - size
        - cells
    """

    UNKNOWN = unknown_value
    FREE = free_value

    H, W = global_map.shape

    # --------------------------------------------------
    # Step 1: identify frontier cells
    # --------------------------------------------------
    frontier_mask = np.zeros((H, W), dtype=bool)

    for x in range(H):
        for y in range(W):
            if global_map[x, y] != FREE:
                continue

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if global_map[nx, ny] == UNKNOWN:
                        frontier_mask[x, y] = True
                        break

    # --------------------------------------------------
    # Step 2: cluster frontier cells via BFS through FREE
    # --------------------------------------------------
    frontier_component = -np.ones((H, W), dtype=int)
    component_id = 0

    for x in range(H):
        for y in range(W):
            if not frontier_mask[x, y]:
                continue
            if frontier_component[x, y] != -1:
                continue

            queue = [(x, y, 0)]
            frontier_component[x, y] = component_id
            visited = {(x, y): 0}

            while queue:
                cx, cy, dist = queue.pop(0)

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if not (0 <= nx < H and 0 <= ny < W):
                        continue
                    if (nx, ny) in visited:
                        continue

                    nd = dist + 1
                    if nd > max_cluster_distance:
                        continue

                    # frontier → include
                    if frontier_mask[nx, ny] and frontier_component[nx, ny] == -1:
                        frontier_component[nx, ny] = component_id
                        visited[(nx, ny)] = nd
                        queue.append((nx, ny, nd))

                    # free → expand BFS
                    elif global_map[nx, ny] == FREE:
                        visited[(nx, ny)] = nd
                        queue.append((nx, ny, nd))

            component_id += 1

    # --------------------------------------------------
    # Step 3: group frontier cells by cluster id
    # --------------------------------------------------
    clusters: Dict[int, List[Tuple[int, int]]] = {}

    for x in range(H):
        for y in range(W):
            if frontier_mask[x, y]:
                cid = frontier_component[x, y]
                clusters.setdefault(cid, []).append((x, y))

    # --------------------------------------------------
    # Step 4: build structured frontier clusters
    # --------------------------------------------------
    frontier_clusters = []

    for cid, cells in clusters.items():
        if not cells:
            continue

        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

        # choose an actual frontier cell closest to centroid
        rep = min(
            cells,
            key=lambda c: (c[0] - centroid_x) ** 2 + (c[1] - centroid_y) ** 2
        )

        frontier_clusters.append({
            "id": cid,
            "rep": rep,                          # (x, y) waypoint
            "size": len(cells),
            "cells": cells,
        })

    return frontier_clusters

class Robot:
    """
    2D point robot with sensor and local mapping capabilities.

    The robot maintains:
    - Position and orientation on a grid
    - A sensor with limited FoV and range
    - A local occupancy map built from observations
    - A* path planner for waypoint navigation
    """

    def __init__(
        self,
        position: Tuple[int, int],
        orientation: float = 0.0,
        sensor_fov: float = np.pi / 3,  # Field of view in radians (60 degrees)
        sensor_range: int = 10,  # Range in grid cells
        map_size: Optional[Tuple[int, int]] = None

    ):
        """
        Initialize robot.

        Args:
            position: Initial (z, x) position on grid
            orientation: Initial orientation in radians (0 = +x direction)
            sensor_fov: Sensor field of view in radians
            sensor_range: Sensor range in grid cells
            map_size: (nz, nx) size of environment map (if known)
        """
        self.position = np.array(position, dtype=np.int32)
        self.orientation = orientation

        # Sensor parameters
        self.sensor_fov = sensor_fov
        self.sensor_range = sensor_range

        # Local map: -1=unknown, 0=free, 1=occupied (unified encoding with LLM)
        # Initialize with unknown cells
        if map_size is not None:
            self.local_map = np.full(map_size, -1, dtype=np.int8)
        else:
            # Will be initialized when first observation is made
            self.local_map = None

        # Navigation
        self.current_waypoint_idx = 0
        self.max_replan_failures = 5        # parameter
        self.replan_failure_count = 0       # counter for current waypoint
        self.waypoints: List[Tuple[int, int]] = []
        self.current_path: List[Tuple[int, int]] = []

        # Fallback waypoint (cached until reached or new waypoints assigned)
        self._fallback_waypoint: Optional[Tuple[int, int]] = None

    def set_waypoints(self, waypoints: List[Tuple[int, int]]):
        """
        Set target waypoints for navigation.

        Args:
            waypoints: List of (z, x) waypoint positions
        """
        self.waypoints = waypoints
        self.current_waypoint_idx = 0
        self.replan_failure_count = 0       # counter for current waypoint
        self.current_path = []
        # Clear fallback waypoint when strategy assigns new waypoints
        self._fallback_waypoint = None

    def get_current_waypoint(self) -> Optional[Tuple[int, int]]:
        """
        Get the current target waypoint.

        Priority:
        1. Strategy-assigned waypoints (from RACER or LLM)
        2. Cached fallback waypoint (persists until reached or new waypoints assigned)
        3. Generate new fallback waypoint from updated local map

        Fallback waypoint is cached to ensure robot commits to a target and doesn't
        constantly switch between frontiers. It's cleared when:
        - Reached (via advance_waypoint)
        - New strategy waypoints assigned (via set_waypoints)
        """
        # Priority 1: Strategy-assigned waypoints
        if self.current_waypoint_idx < len(self.waypoints):
            return self.waypoints[self.current_waypoint_idx]

        # Priority 2: Return cached fallback if available
        if self._fallback_waypoint is not None:
            return self._fallback_waypoint

        # Priority 3: Generate new fallback waypoint from current local map
        # This only happens once until the waypoint is reached
        self._fallback_waypoint = self._get_local_frontier_fallback()
        return self._fallback_waypoint

    def _get_local_frontier_fallback(self) -> Optional[Tuple[int, int]]:
        """
        Find the nearest frontier in the robot's local map as a fallback waypoint.

        Returns:
            Nearest frontier position (z, x) or None if no frontiers found
        """
        if self.local_map is None:
            return None

        # Extract frontiers from local map
        try:
            frontier_clusters = extract_frontier_clusters(
                self.local_map,
                unknown_value=-1,
                free_value=0,
            )

            # Find nearest frontier to robot's current position
            robot_pos = np.array(self.position, dtype=float)
            nearest_frontier = min(
                frontier_clusters,
                key=lambda f: np.linalg.norm(np.array(f["rep"]) - robot_pos)
            )

            return tuple(nearest_frontier["rep"])

        except Exception as e:
            # Log exception for debugging
            return None

    def advance_waypoint(self):
        """
        Move to the next waypoint.

        Clears fallback waypoint when reached, allowing robot to find a new target.
        """
        self.current_waypoint_idx += 1
        self.replan_failure_count = 0
        self.current_path = []
        # Clear fallback waypoint when reached
        self._fallback_waypoint = None

    def initialize_local_map(self, map_size: Tuple[int, int]):
        """Initialize local map if not already done."""
        if self.local_map is None:
            self.local_map = np.full(map_size, -1, dtype=np.int8)

    def get_sensor_cells(self) -> Set[Tuple[int, int]]:
        """
        Get all cells visible by the sensor given current position and orientation.

        Returns:
            Set of (z, x) cell coordinates within sensor range and FoV
        """
        visible_cells = set()

        z0, x0 = self.position

        # Scan cells within range
        for dz in range(-self.sensor_range, self.sensor_range + 1):
            for dx in range(-self.sensor_range, self.sensor_range + 1):
                # Check if within range
                dist = np.sqrt(dz**2 + dx**2)
                if dist > self.sensor_range or dist < 0.5:
                    continue

                # Check if within FoV
                angle_to_cell = np.arctan2(dz, dx)
                angle_diff = self._normalize_angle(angle_to_cell - self.orientation)

                if abs(angle_diff) <= self.sensor_fov / 2:
                    z, x = z0 + dz, x0 + dx
                    visible_cells.add((z, x))

        return visible_cells

    def ray_cast(self, target_z: int, target_x: int) -> List[Tuple[int, int]]:
        """
        Perform ray casting from robot position to target cell.
        Uses Bresenham's line algorithm.

        Args:
            target_z, target_x: Target cell coordinates

        Returns:
            List of (z, x) cells along the ray
        """
        z0, x0 = self.position
        z1, x1 = target_z, target_x

        cells = []

        dz = abs(z1 - z0)
        dx = abs(x1 - x0)

        sz = 1 if z0 < z1 else -1
        sx = 1 if x0 < x1 else -1

        err = dz - dx
        z, x = z0, x0

        while True:
            cells.append((z, x))

            if z == z1 and x == x1:
                break

            e2 = 2 * err

            if e2 > -dx:
                err -= dx
                z += sz

            if e2 < dz:
                err += dz
                x += sx

        return cells

    def update_local_map(self, ground_truth_map: np.ndarray):
        """
        Update local map based on sensor observation of ground truth map.

        In addition to sensor FoV, robot always knows a 2-cell radius around itself.
        This prevents the robot's position from becoming a frontier.

        Args:
            ground_truth_map: Ground truth occupancy grid (0=free, 1=occupied)
        """
        if self.local_map is None:
            self.initialize_local_map(ground_truth_map.shape)

        # First, mark a 2-cell radius around robot as known
        # This ensures robot's immediate area is observed
        GUARANTEED_RADIUS = 2
        robot_z, robot_x = self.position

        for dz in range(-GUARANTEED_RADIUS, GUARANTEED_RADIUS + 1):
            for dx in range(-GUARANTEED_RADIUS, GUARANTEED_RADIUS + 1):
                # Check if within circular radius
                if dz**2 + dx**2 > GUARANTEED_RADIUS**2:
                    continue

                z, x = robot_z + dz, robot_x + dx

                # Check bounds
                if not (0 <= z < ground_truth_map.shape[0] and
                        0 <= x < ground_truth_map.shape[1]):
                    continue

                # Directly observe this cell (no ray casting needed for nearby area)
                self.local_map[z, x] = ground_truth_map[z, x]

        # Get visible cells from sensor
        visible_cells = self.get_sensor_cells()

        # Ray cast to each visible cell to handle occlusions
        for target_z, target_x in visible_cells:
            # Check bounds
            if not (0 <= target_z < ground_truth_map.shape[0] and
                    0 <= target_x < ground_truth_map.shape[1]):
                continue

            # Ray cast from robot to target
            ray = self.ray_cast(target_z, target_x)

            # Update cells along ray
            for i, (z, x) in enumerate(ray):
                if not (0 <= z < ground_truth_map.shape[0] and
                        0 <= x < ground_truth_map.shape[1]):
                    break

                # Mark as observed
                if ground_truth_map[z, x] == 1:
                    # Hit obstacle
                    self.local_map[z, x] = 1  # occupied
                    break  # Stop ray (occluded)
                else:
                    # Free space
                    self.local_map[z, x] = 0  # free

    def plan_path_astar(
        self,
        goal: Tuple[int, int],
        other_robot_positions: Optional[List[Tuple[int, int]]] = None
    ) -> List[Tuple[int, int]]:
        """
        Plan path to goal using A* on local map, considering other robots as obstacles.

        Args:
            goal: Target (z, x) position
            other_robot_positions: List of other robots' positions to avoid (treated as temporary obstacles)

        Returns:
            List of (z, x) waypoints from current position to goal
            Empty list if no path found
        """
        if self.local_map is None:
            return []

        start = tuple(self.position)
        goal = tuple(goal)

        # Create dictionary of robot proximity costs (distance-based penalty)
        # Closer to robots = higher cost (but not blocked)
        robot_proximity_cost = {}
        if other_robot_positions:
            for rpos in other_robot_positions:
                rz, rx = rpos
                # Add costs in 5x5 area around each robot
                for dz in range(-2, 3):
                    for dx in range(-2, 3):
                        cell = (rz + dz, rx + dx)
                        dist = max(abs(dz), abs(dx))  # Chebyshev distance
                        if dist == 0:
                            # Robot's exact position - very high cost (effectively blocked)
                            robot_proximity_cost[cell] = 100.0
                        elif dist <= 2:
                            # Close to robot - add penalty inversely proportional to distance
                            # dist=1 → cost=5.0, dist=2 → cost=2.0
                            cost = 10.0 / (dist + 1)
                            robot_proximity_cost[cell] = max(robot_proximity_cost.get(cell, 0), cost)

        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, start))

        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}

        closed_set = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            closed_set.add(current)

            # Check neighbors (8-connected)
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Check if passable in local map
                z, x = neighbor
                if not (0 <= z < self.local_map.shape[0] and
                        0 <= x < self.local_map.shape[1]):
                    continue

                # Occupied cells are not passable
                if self.local_map[z, x] == 1:  # occupied
                    continue

                # Calculate base move cost
                if self.local_map[z, x] == -1:  # unknown
                    # Allow moving through unknown space but with higher cost
                    base_cost = 2.0
                else:  # free (0)
                    base_cost = 1.0

                # Add robot proximity cost (prefer paths away from other robots)
                proximity_penalty = robot_proximity_cost.get(neighbor, 0.0)
                move_cost = base_cost + proximity_penalty

                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)

                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return []

    def move_to(self, target: Tuple[int, int]):
        """
        Move robot to target position and update orientation.

        Args:
            target: Target (z, x) position
        """
        dz = target[0] - self.position[0]
        dx = target[1] - self.position[1]

        # Update orientation to face movement direction
        if dz != 0 or dx != 0:
            self.orientation = np.arctan2(dz, dx)

        # Update position
        self.position = np.array(target, dtype=np.int32)

    def step_towards_waypoint(
        self,
        ground_truth_map: np.ndarray,
        other_robot_positions: Optional[List[Tuple[int, int]]] = None
    ) -> bool:
        """
        Take one step towards current waypoint with active replanning.

        Planning and validation based on robot's local map (updated every step):
        - A* planning uses local map + distance-based costs for robot proximity
        - Step validation checks local map (not ground truth) for passability
        - Robot collision avoidance with 1×1 footprint (exact position only)

        Replans immediately if:
        - Path blocked by newly discovered obstacles
        - Path blocked by other robots
        - Next step invalid in local map

        Args:
            ground_truth_map: Ground truth occupancy map for sensor observations
            other_robot_positions: List of other robots' positions to avoid

        Returns:
            True if waypoint reached, False otherwise
        """
        # Update observation FIRST - may discover new obstacles and frontiers
        self.update_local_map(ground_truth_map)

        # Then get waypoint (so fallback uses updated map with new observations)
        waypoint = self.get_current_waypoint()
        if waypoint is None:
            return False

        # Active replanning: Check if we need to replan due to:
        # 1. No current path
        # 2. Path blocked by newly discovered obstacles in local map
        # 3. Robot collisions in the planned path
        no_path = not self.current_path
        path_blocked = self._is_path_blocked(self.current_path) if self.current_path else False
        robot_blocked = self._is_path_blocked_by_robots(self.current_path, other_robot_positions) if self.current_path else False

        needs_replan = no_path or path_blocked or robot_blocked

        if needs_replan:
            if no_path:
                pass  # Don't print for initial planning
            elif path_blocked:
                self.current_path = []  # Clear invalid path
            elif robot_blocked:
                self.current_path = []  # Clear invalid path
            # Replan with robot positions as temporary obstacles
            self.current_path = self.plan_path_astar(waypoint, other_robot_positions)

            if not self.current_path:
                # No path found - wait and try again next step
                self.replan_failure_count += 1
                if self.replan_failure_count >= self.max_replan_failures:
                    self.current_waypoint_idx += 1
                    print(f"[WARN] Robot at {self.position} skipping unreachable waypoint {waypoint}.")
                    self.replan_failure_count = 0
                return False

            self.replan_failure_count = 0
        # Execute next step along path
        if len(self.current_path) > 1:
            next_pos = self.current_path[1]
            nz, nx = next_pos

            # Triple validation before moving:
            # 1. Bounds check
            # 2. Local map free (uses updated local map, not ground truth)
            # 3. Clear of other robots (exact position check)
            if (0 <= nz < self.local_map.shape[0] and
                0 <= nx < self.local_map.shape[1] and
                self.local_map[nz, nx] == 0 and  # Free in local map (updated knowledge)
                self._is_position_clear_of_robots(next_pos, other_robot_positions)):

                # Valid move - execute it
                self.move_to(next_pos)
                self.current_path = self.current_path[1:]
            else:
                self.current_path = []
                return False

        # Check if reached waypoint
        dist_to_waypoint = np.linalg.norm(self.position - np.array(waypoint))
        if dist_to_waypoint < 1.5:
            self.advance_waypoint()
            return True

        return False

    def _is_path_blocked(self, path: List[Tuple[int, int]]) -> bool:
        """Check if current path is blocked by newly discovered obstacles in local map."""
        if not path:
            return True

        for z, x in path[1:]:  # Skip current position
            if self.local_map[z, x] == 1:  # occupied
                return True

        return False

    def _is_path_blocked_by_robots(
        self,
        path: List[Tuple[int, int]],
        other_robot_positions: Optional[List[Tuple[int, int]]] = None
    ) -> bool:
        """Check if current path would collide with other robots (exact position check)."""
        if not path or not other_robot_positions:
            return False

        # Check each position in the path (skip current position)
        for pos in path[1:]:
            if not self._is_position_clear_of_robots(pos, other_robot_positions):
                return True

        return False

    def _get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors of a cell."""
        z, x = cell
        return [
            (z - 1, x),
            (z + 1, x),
            (z, x - 1),
            (z, x + 1),
            (z - 1, x - 1),
            (z - 1, x + 1),
            (z + 1, x - 1),
            (z + 1, x + 1)
        ]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic for A*."""
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def _is_position_clear_of_robots(
        self,
        position: Tuple[int, int],
        other_robot_positions: Optional[List[Tuple[int, int]]] = None
    ) -> bool:
        """
        Check if position is occupied by another robot.

        Each robot occupies a single grid cell (1×1).
        This prevents robots from moving to the exact same cell.

        Args:
            position: Target position to check
            other_robot_positions: List of other robots' positions

        Returns:
            True if position is clear, False if occupied by another robot
        """
        if other_robot_positions is None or len(other_robot_positions) == 0:
            return True

        # Check if any other robot is at this exact position
        return position not in other_robot_positions

    def get_status(self) -> dict:
        """Get current robot status."""
        return {
            "position": tuple(self.position),
            "orientation": self.orientation,
            "current_waypoint": self.get_current_waypoint(),
            "waypoint_idx": self.current_waypoint_idx,
            "total_waypoints": len(self.waypoints),
            "local_map_coverage": np.sum(self.local_map >= 0) / self.local_map.size if self.local_map is not None else 0.0,
            "path_length": len(self.current_path)
        }
