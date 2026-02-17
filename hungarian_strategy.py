# hungarian_strategy.py
"""
Hungarian Assignment Waypoint Strategy

Key properties:
1) coord_time (every T_coord steps):
   - Fuse all robot local maps
   - Extract frontier clusters from fused map
   - Build cost matrix (robot-to-frontier distances)
   - Use Hungarian algorithm for optimal assignment
   - Assign each robot to its assigned frontier

2) not coord_time:
   - Robots continue to their assigned waypoints
   - If robot completes waypoint before next coordination:
     * Robot's local controller takes over
     * Goes to nearest frontier in its local map
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment


def extract_frontier_clusters(
    occupancy_map: np.ndarray,
    unknown_value: int = -1,
    free_value: int = 0,
    max_cluster_distance: int = 10,
) -> List[Dict]:
    """
    Extract frontier clusters from an occupancy map.

    Frontier cell:
        FREE cell adjacent to UNKNOWN

    Frontier cluster:
        Frontier cells connected via FREE space (BFS with depth limit)

    This matches the implementation in racer_strategy.py and llm_world_builder.py.

    Returns:
        List of frontier clusters, each containing:
        - 'id': cluster id
        - 'rep': representative cell (actual frontier cell closest to centroid)
        - 'size': number of cells in cluster
        - 'cells': list of (row, col) tuples
    """
    from typing import Dict as DictType

    UNKNOWN = unknown_value
    FREE = free_value

    H, W = occupancy_map.shape

    # --------------------------------------------------
    # Step 1: identify frontier cells
    # --------------------------------------------------
    frontier_mask = np.zeros((H, W), dtype=bool)

    for x in range(H):
        for y in range(W):
            if occupancy_map[x, y] != FREE:
                continue

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < H and 0 <= ny < W:
                    if occupancy_map[nx, ny] == UNKNOWN:
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
                    elif occupancy_map[nx, ny] == FREE:
                        visited[(nx, ny)] = nd
                        queue.append((nx, ny, nd))

            component_id += 1

    # --------------------------------------------------
    # Step 3: group frontier cells by cluster id
    # --------------------------------------------------
    clusters: DictType[int, List[Tuple[int, int]]] = {}

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

        # Choose an actual frontier cell closest to centroid
        # (centroid itself might not be a frontier cell)
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


def compute_robot_frontier_costs(
    robot_positions: List[Tuple[int, int]],
    frontier_reps: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Compute cost matrix for robot-to-frontier assignment.

    Cost = Euclidean distance from robot to frontier representative.

    Args:
        robot_positions: List of (row, col) for each robot
        frontier_reps: List of (row, col) for each frontier representative

    Returns:
        cost_matrix: Shape (num_robots, num_frontiers)
    """
    num_robots = len(robot_positions)
    num_frontiers = len(frontier_reps)

    cost_matrix = np.zeros((num_robots, num_frontiers))

    for i, robot_pos in enumerate(robot_positions):
        for j, frontier_pos in enumerate(frontier_reps):
            dist = np.linalg.norm(
                np.array(robot_pos) - np.array(frontier_pos)
            )
            cost_matrix[i, j] = dist

    return cost_matrix


def solve_tsp_2opt(start_pos: np.ndarray, points: List[np.ndarray], max_iterations: int = 1000) -> List[np.ndarray]:
    """
    Solve TSP using 2-opt local search with robot start position as fixed beginning.

    Args:
        start_pos: Robot's current position (fixed start of tour)
        points: List of waypoints to visit
        max_iterations: Maximum number of 2-opt iterations

    Returns:
        Optimized route starting from start_pos
    """
    if not points:
        return []

    if len(points) == 1:
        return [points[0]]

    # Start with nearest neighbor heuristic
    route = []
    remaining = list(points)
    current = start_pos

    while remaining:
        nearest_idx = min(range(len(remaining)), key=lambda i: np.linalg.norm(current - remaining[i]))
        nearest = remaining.pop(nearest_idx)
        route.append(nearest)
        current = nearest

    # 2-opt improvement
    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        for i in range(len(route) - 1):
            for j in range(i + 2, len(route)):
                # Calculate current distance
                if i == 0:
                    d_before_i = np.linalg.norm(start_pos - route[i])
                else:
                    d_before_i = np.linalg.norm(route[i - 1] - route[i])

                d_i_to_i1 = np.linalg.norm(route[i] - route[i + 1])

                if j == len(route) - 1:
                    d_j_to_after = 0
                else:
                    d_j_to_after = np.linalg.norm(route[j] - route[j + 1])

                d_j1_to_j = np.linalg.norm(route[j - 1] - route[j])

                current_dist = d_i_to_i1 + d_j1_to_j

                # Calculate distance after swap
                d_i_to_j = np.linalg.norm(route[i] - route[j])
                d_i1_to_j1 = np.linalg.norm(route[i + 1] - route[j - 1]) if i + 1 != j else 0

                new_dist = d_i_to_j + d_i1_to_j1

                if new_dist < current_dist:
                    # Perform 2-opt swap
                    route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                    improved = True
                    break

            if improved:
                break

    return route


class HungarianStrategy:
    """
    Hungarian Assignment Strategy for Multi-Robot Frontier Exploration.

    Uses the Hungarian algorithm (via scipy.optimize.linear_sum_assignment)
    to optimally assign robots to frontiers based on distance costs.
    """

    def __init__(
        self,
        T_coord: int = 20,
        max_cluster_distance: int = 10,
    ):
        """
        Args:
            T_coord: Coordination interval (steps between global assignments)
            max_cluster_distance: BFS depth limit for frontier clustering (matches RACER/LLM)
        """
        self.T_coord = T_coord
        self.max_cluster_distance = max_cluster_distance
        self.initialized = False

    def reset(self):
        """Reset strategy state for new episode."""
        self.initialized = False

    def _fuse_maps(self, robots) -> np.ndarray:
        """Fuse all robot local maps using element-wise maximum."""
        if not robots:
            return None

        fused = robots[0].local_map.copy()
        for robot in robots[1:]:
            fused = np.maximum(fused, robot.local_map)

        return fused

    def assign_waypoints(
        self,
        robots,
        step_count: int
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Assign waypoints to robots based on Hungarian algorithm.

        Args:
            robots: List of Robot objects
            step_count: Current simulation step

        Returns:
            Dictionary mapping robot_id to list of waypoints [(row, col), ...]
        """
        waypoints = {}

        coord_time = (not self.initialized) or (step_count % self.T_coord == 0)

        # ========== COORDINATION TIME ==========
        if coord_time:
            # Fuse all robot maps
            fused_map = self._fuse_maps(robots)
            for robot in robots:
                robot.local_map = fused_map.copy()
            if fused_map is None:
                return waypoints

            # Extract frontier clusters (matches RACER/LLM implementation)
            frontier_clusters = extract_frontier_clusters(
                fused_map,
                unknown_value=-1,
                free_value=0,
                max_cluster_distance=self.max_cluster_distance,
            )

            if not frontier_clusters:
                # No frontiers found - exploration may be complete
                self.initialized = True
                return waypoints

            # Get robot positions and frontier representatives
            robot_positions = [tuple(map(int, r.position)) for r in robots]
            frontier_reps = [cluster['rep'] for cluster in frontier_clusters]

            num_robots = len(robots)
            num_frontiers = len(frontier_reps)

            # Build cost matrix (robot x frontier distances)
            cost_matrix = compute_robot_frontier_costs(
                robot_positions, frontier_reps
            )

            # ========== RECURSIVE HUNGARIAN ASSIGNMENT ==========
            # Assign ALL frontiers to robots (like LLM planner does)
            assigned_frontiers = {rid: [] for rid in range(num_robots)}
            remaining = set(range(num_frontiers))

            while remaining:
                remaining_list = list(remaining)
                sub_cost = cost_matrix[:, remaining_list]

                row_ind, col_ind = linear_sum_assignment(sub_cost)

                assigned_this_round = set()

                for r_i, f_j_sub in zip(row_ind, col_ind):
                    f_j = remaining_list[f_j_sub]
                    assigned_frontiers[r_i].append(f_j)
                    assigned_this_round.add(f_j)

                if not assigned_this_round:
                    break

                remaining -= assigned_this_round

            # ========== TSP OPTIMIZATION PER ROBOT ==========
            # Order frontiers for each robot using 2-opt TSP
            frontier_positions = np.array(frontier_reps, dtype=float)

            for rid, frontier_idxs in assigned_frontiers.items():
                if not frontier_idxs:
                    continue

                start = np.array(robots[rid].position, dtype=float)
                pts = [frontier_positions[i] for i in frontier_idxs]

                # Solve TSP to order waypoints optimally
                route = solve_tsp_2opt(start, pts)

                # Convert to integer waypoints
                waypoints[rid] = [tuple(np.round(p).astype(int)) for p in route]

            self.initialized = True
            return waypoints

        # ========== NON-COORDINATION TIME ==========
        # Robots continue to their assigned waypoints
        # If they finish early, the robot's local controller will take over
        # and assign them to nearest frontier from their local map
        return waypoints
