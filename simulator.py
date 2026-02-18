"""
Grid Map Simulator
Simple binary occupancy map-based multi-robot exploration simulator.

Notes:
- Maps are binary: 0 = free, 1 = wall/occupied
- Robot local maps use unified encoding: -1 unknown / 0 free / 1 occupied
- Global aggregated map uses the same encoding
- No semantic information (no room types, no doors)
"""

import io
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from robot import Robot, extract_frontier_clusters
from waypoint_strategy import WaypointStrategy, RacerWaypointStrategy, AEPWaypointStrategy, HungarianWaypointStrategy


class GridMapSimulator:
    def __init__(
        self,
        maps_dir: str = "maps",
        waypoint_strategy: Optional[WaypointStrategy] = None
    ):
        """
        Args:
            maps_dir: Directory containing map size subdirectories
                      (e.g., maps/small_map/, maps/medium_map/, maps/large_map/)
            waypoint_strategy: Strategy for assigning waypoints to robots
        """
        self.maps_dir = Path(maps_dir)

        # Scene state
        self.scene_id: Optional[str] = None       # e.g. "small_map"
        self.map_variant: Optional[str] = None    # e.g. "generated_small_map_3"
        self.occupancy_map: Optional[np.ndarray] = None  # (H, W) 0 free / 1 occ

        # Robots
        self.robots: List[Robot] = []
        self.robot_trajectories: List[List[Tuple[int, int]]] = []

        # Simulation
        self.step_count: int = 0

        # Waypoint assignment strategy (pluggable)
        self.waypoint_strategy: Optional[WaypointStrategy] = waypoint_strategy

        # Store frontiers for visualization (strategy-dependent)
        self.latest_frontiers: List[Tuple[int, int]] = []

        # GIF frames
        self._frames: List[Image.Image] = []

        # Coordination calculation times (for waypoint assignment)
        self.coordination_times: List[float] = []

    # ======================================================
    # Scene loading
    # ======================================================

    def load_scene(
        self,
        scene_id: str,
        map_variant: Optional[str] = None,
    ) -> None:
        """
        Load a binary occupancy map from maps/<scene_id>/.

        Args:
            scene_id: Subdirectory name, e.g. "small_map", "medium_map", "large_map"
            map_variant: Specific .npy filename (without extension) to load.
                         If None, loads 'generated_map.npy' (the base map).
                         Examples: "generated_small_map_3", "generated_large_map_0"
        """
        scene_dir = self.maps_dir / scene_id
        if not scene_dir.exists():
            raise ValueError(f"Scene directory not found: {scene_dir}")

        if map_variant is None:
            npy_path = scene_dir / "generated_map.npy"
        else:
            npy_path = scene_dir / f"{map_variant}.npy"

        if not npy_path.exists():
            raise FileNotFoundError(f"Map file not found: {npy_path}")

        raw = np.load(npy_path)

        # Ensure 2-D binary map: 0 = free, 1 = occupied
        if raw.ndim == 3:
            # If stored as (H, W, C), collapse to 2-D using first channel
            raw = raw[..., 0]

        # Normalise to strict 0/1
        self.occupancy_map = (raw != 0).astype(np.uint8)

        self.scene_id = scene_id
        self.map_variant = map_variant or "generated_map"

        if self.waypoint_strategy is not None:
            self.waypoint_strategy.reset()

        self.step_count = 0
        self._frames = []
        self.coordination_times = []

        print(
            f"[INFO] Loaded scene '{scene_id}' variant '{self.map_variant}' "
            f"| grid={self.occupancy_map.shape}"
        )

    def set_waypoint_strategy(self, strategy: WaypointStrategy) -> None:
        """Change the waypoint assignment strategy at runtime."""
        self.waypoint_strategy = strategy
        self.waypoint_strategy.reset()
        print(f"[INFO] Waypoint strategy changed to {strategy.__class__.__name__}")

    # ======================================================
    # Task completion checks
    # ======================================================

    def compute_coverage_ratio(self) -> float:
        """
        Compute coverage ratio: (known free cells) / (total free cells in ground truth).

        Aggregates all robot local maps (for checking only; does not share maps).

        Returns:
            Coverage ratio [0.0, 1.0]
        """
        if not self.robots or self.occupancy_map is None:
            return 0.0

        aggregated_map = self.robots[0].local_map.copy()
        for robot in self.robots[1:]:
            aggregated_map = np.maximum(aggregated_map, robot.local_map)

        total_free_cells = int(np.sum(self.occupancy_map == 0))
        if total_free_cells == 0:
            return 1.0

        known_free_cells = int(
            np.sum((aggregated_map == 0) & (self.occupancy_map == 0))
        )
        return known_free_cells / total_free_cells

    def get_avg_coordination_time(self) -> float:
        """Average coordination calculation time in seconds."""
        if not self.coordination_times:
            return 0.0
        return sum(self.coordination_times) / len(self.coordination_times)

    def check_target_discovered(self, target_position: Tuple[int, int]) -> bool:
        """
        Check if target position has been observed by any robot.

        A cell is discovered if it is marked as known (0 or 1, not -1)
        in any robot's local map.

        Args:
            target_position: (row, col) grid position of the target

        Returns:
            True if any robot has discovered the target
        """
        if not self.robots:
            return False

        target_r, target_c = target_position
        for robot in self.robots:
            if robot.local_map[target_r, target_c] != -1:
                return True
        return False

    # ======================================================
    # Robot spawning
    # ======================================================

    @staticmethod
    def _far_enough(
        p: Tuple[int, int],
        others: List[Tuple[int, int]],
        min_dist: float
    ) -> bool:
        for q in others:
            if np.linalg.norm(np.array(p) - np.array(q)) < min_dist:
                return False
        return True

    def get_entrance_positions(
        self,
        num_positions: int,
        min_separation: int = 5,
        max_tries: int = 5000,
    ) -> List[Tuple[int, int]]:
        """
        Sample random free-cell positions separated by at least `min_separation` cells.
        """
        if self.occupancy_map is None:
            raise ValueError("Load a scene first")

        candidates = np.argwhere(self.occupancy_map == 0)
        if len(candidates) == 0:
            raise ValueError("No free cells found for robot spawn")

        chosen: List[Tuple[int, int]] = []
        tries = 0

        while len(chosen) < num_positions and tries < max_tries:
            tries += 1
            z, x = candidates[np.random.choice(len(candidates))]
            pos = (int(z), int(x))
            if self._far_enough(pos, chosen, min_separation):
                chosen.append(pos)

        if len(chosen) < num_positions:
            raise RuntimeError(
                f"Could not find {num_positions} spawn positions "
                f"with separation >= {min_separation} cells after {max_tries} tries"
            )

        return chosen

    def spawn_robots(
        self,
        num_robots: int = 3,
        initial_positions: Optional[List[Tuple[int, int]]] = None
    ) -> None:
        """
        Spawn robots at specified positions or random free cells.

        Args:
            num_robots: Number of robots to spawn
            initial_positions: Optional list of (row, col) positions.
                               If None, random free cells are used.
        """
        if self.occupancy_map is None:
            raise ValueError("Load a scene first")

        self.robots = []
        self.robot_trajectories = []

        if initial_positions is not None:
            positions = initial_positions[:num_robots]
        else:
            positions = self.get_entrance_positions(num_robots)

        map_size = self.occupancy_map.shape

        for pos in positions:
            r = Robot(
                position=pos,
                orientation=float(np.random.uniform(-np.pi, np.pi)),
                sensor_fov=np.pi / 3,
                sensor_range=10,
                map_size=map_size
            )
            r.update_local_map(self.occupancy_map)
            self.robots.append(r)
            self.robot_trajectories.append([tuple(pos)])

        print(f"[INFO] Spawned {len(self.robots)} robots at {positions}")

    # ======================================================
    # Frontier visualization helper
    # ======================================================

    def _update_frontier_visualization(self) -> None:
        """Extract frontiers from the aggregated robot map for visualization."""
        self.latest_frontiers = []

        if not self.robots:
            return

        # Aggregate all robot local maps
        global_map = self.robots[0].local_map.copy()
        for robot in self.robots[1:]:
            global_map = np.maximum(global_map, robot.local_map)

        frontier_clusters = extract_frontier_clusters(
            global_map,
            unknown_value=-1,
            free_value=0,
        )
        self.latest_frontiers = [tuple(f["rep"]) for f in frontier_clusters]

    # ======================================================
    # Simulation step / reset
    # ======================================================

    def reset(self) -> None:
        """Reset counters, trajectories, and frames. Does not respawn robots."""
        self.step_count = 0
        self._frames = []
        self.coordination_times = []
        self.robot_trajectories = [
            [tuple(map(int, r.position))] for r in self.robots
        ]
        for r in self.robots:
            r.current_waypoint_idx = 0
            r.current_path = []

    def step(self) -> None:
        """Execute one simulation timestep (robots move + update local maps)."""
        if self.occupancy_map is None:
            raise ValueError("Load a scene first")

        # Collect all robot positions for collision avoidance
        all_pos = [tuple(map(int, r.position)) for r in self.robots]

        for i, robot in enumerate(self.robots):
            others = [p for j, p in enumerate(all_pos) if j != i]
            robot.step_towards_waypoint(
                ground_truth_map=self.occupancy_map,
                other_robot_positions=others
            )
            self.robot_trajectories[i].append(tuple(map(int, robot.position)))

        # Waypoint assignment using pluggable strategy
        if self.waypoint_strategy is None:
            return

        if self.waypoint_strategy.should_replan(self.step_count):
            try:
                import time
                coord_start = time.time()
                wp_dict = self.waypoint_strategy.assign_waypoints(
                    robots=self.robots,
                    step_count=self.step_count,
                )
                self.coordination_times.append(time.time() - coord_start)

                for ridx, wps in wp_dict.items():
                    self.robots[ridx].set_waypoints(wps)
                    print(
                        f"[INFO] Assigned waypoints to Robot {ridx} "
                        f"at step {self.step_count} "
                        f"at position {tuple(map(int, self.robots[ridx].position))}"
                    )

                self._update_frontier_visualization()

            except Exception as e:
                print(
                    f"[WARN] Waypoint assignment failed at step {self.step_count}: {e}"
                )
        
        self.step_count += 1
    # ======================================================
    # Visualization
    # ======================================================

    def _plot_ground_truth(self, ax) -> None:
        """Plot the ground-truth binary occupancy map."""
        if self.occupancy_map is None:
            return

        # White = free, black = wall
        ax.imshow(self.occupancy_map, cmap="gray_r", origin="lower", vmin=0, vmax=1)

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.robots))))
        for i, r in enumerate(self.robots):
            z, x = map(int, r.position)
            ax.plot(x, z, "o", color=colors[i], markersize=8)

        if self.latest_frontiers:
            fz, fx = zip(*self.latest_frontiers)
            ax.scatter(fx, fz, c="#66ccff", s=15)

        variant = self.map_variant or ""
        ax.set_title(f"{self.scene_id} ({variant}) | Step {self.step_count}")
        ax.axis("off")

    def _plot_aggregate_local_map(self, ax) -> None:
        """Plot the aggregated observed map across all robots."""
        if not self.robots:
            return

        agg = self.robots[0].local_map.copy()
        for robot in self.robots[1:]:
            agg = np.maximum(agg, robot.local_map)

        H, W = agg.shape
        viz = np.zeros((H, W, 3), dtype=np.float32)

        viz[agg == -1] = [0.3, 0.3, 0.3]   # unknown -> dark gray
        viz[agg == 0] = [1.0, 1.0, 1.0]    # free    -> white
        viz[agg == 1] = [0.0, 0.0, 0.0]    # wall    -> black

        ax.imshow(viz, origin="lower")

        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.robots))))
        for i, r in enumerate(self.robots):
            z, x = map(int, r.position)
            ax.plot(x, z, "o", color=colors[i], markersize=8)

        if self.latest_frontiers:
            fz, fx = zip(*self.latest_frontiers)
            ax.scatter(
                fx, fz, c="#66ccff", s=30, marker="o",
                edgecolors="white", linewidths=1
            )

        ax.set_title(f"Aggregated Local Map | Step {self.step_count}")
        ax.axis("off")

    def _plot_trajectories(self, ax) -> None:
        """Plot robot trajectories on the occupancy map."""
        if self.occupancy_map is None:
            return

        ax.imshow(self.occupancy_map, cmap="gray_r", origin="lower", alpha=0.3)
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(self.robots))))

        for i, traj in enumerate(self.robot_trajectories):
            if len(traj) > 1:
                zs, xs = zip(*traj)
                ax.plot(xs, zs, "-", color=colors[i], linewidth=1.5)
                ax.plot(xs[-1], zs[-1], "o", color=colors[i], markersize=8)

        ax.set_title("Robot Trajectories")
        ax.axis("off")

    def visualize(self, save_path: Optional[str] = None) -> None:
        """Visualize ground truth, aggregated local map, and trajectories."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        self._plot_ground_truth(axes[0])
        self._plot_aggregate_local_map(axes[1])
        self._plot_trajectories(axes[2])

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    # ======================================================
    # GIF capture
    # ======================================================

    def _capture_frame(self) -> Image.Image:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        self._plot_ground_truth(axes[0])
        self._plot_aggregate_local_map(axes[1])
        self._plot_trajectories(axes[2])

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        frame = Image.open(buf).copy()
        buf.close()
        plt.close(fig)
        return frame

    @staticmethod
    def _create_gif(
        frames: List[Image.Image],
        output_path: str,
        duration_ms: int = 100
    ) -> None:
        if not frames:
            raise ValueError("No frames to write.")
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0
        )

    # ======================================================
    # Run
    # ======================================================

    def run(
        self,
        max_steps: int = 300,
        visualize_every: int = 10,
        create_gif: bool = True,
        gif_path: str = "simulation.gif",
        gif_duration_ms: int = 100,
        verbose: bool = False,
        completion_check_fn=None,
    ) -> Dict:
        """
        Run the simulation.

        Args:
            max_steps: Maximum number of simulation steps (horizon)
            visualize_every: Capture a GIF frame every N steps (if create_gif=True)
            create_gif: Whether to capture frames and write a GIF
            gif_path: Output path for the GIF file
            gif_duration_ms: Per-frame duration in milliseconds
            verbose: Print progress every 25 steps
            completion_check_fn: Optional callable(simulator, step) -> bool
                                  Returns True to stop the simulation early

        Returns:
            Dict with keys:
                'completed': bool
                'completion_step': int or None
                'total_steps': int
        """
        if self.occupancy_map is None:
            raise ValueError("Call load_scene() first.")
        if not self.robots:
            raise ValueError("Call spawn_robots() first.")

        if create_gif:
            self._frames = []
            self._frames.append(self._capture_frame())  # frame at step 0

        completion_step = None

        for t in range(max_steps):
            self.step()

            if verbose and (t % 25 == 0):
                print(f"[INFO] Step {self.step_count}/{max_steps}")

            if create_gif and visualize_every > 0 and (
                self.step_count % visualize_every == 0
            ):
                self._frames.append(self._capture_frame())

            if completion_check_fn is not None and completion_check_fn(
                self, self.step_count
            ):
                completion_step = self.step_count
                break

        if create_gif:
            self._create_gif(self._frames, str(gif_path), duration_ms=gif_duration_ms)
            print(f"[INFO] Saved GIF: {gif_path}")

        return {
            "completed": completion_step is not None,
            "completion_step": completion_step,
            "total_steps": self.step_count,
        }


def main():
    from waypoint_strategy import HungarianWaypointStrategy

    sim = GridMapSimulator(
        maps_dir="maps",
        waypoint_strategy=HungarianWaypointStrategy(T_coord=20)
    )
    sim.load_scene("small_map")
    sim.spawn_robots(num_robots=2)
    sim.run(max_steps=300, visualize_every=10, create_gif=True, gif_path="simulation.gif")


if __name__ == "__main__":
    main()
