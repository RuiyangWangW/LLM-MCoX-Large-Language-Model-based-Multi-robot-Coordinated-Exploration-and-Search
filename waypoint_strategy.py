"""
Waypoint Assignment Strategy Interface

Provides a pluggable interface for different waypoint assignment methods.
Each strategy can decide:
- When to replan (every step, every N steps, etc.)
- How to assign waypoints to robots
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class WaypointStrategy(ABC):
    """
    Base class for waypoint assignment strategies.
    """

    def __init__(self, replan_interval: int = 1):
        """
        Args:
            replan_interval: How often to replan (in simulation steps)
                            1 = every step, 20 = every 20 steps, etc.
        """
        self.replan_interval = replan_interval
        self.last_plan_step = -1

    def should_replan(self, step_count: int) -> bool:
        """
        Determine if replanning should occur at this step.

        Args:
            step_count: Current simulation step

        Returns:
            True if replanning should occur
        """
        # Always plan at step 1, then based on interval
        if step_count == 1:
            return True

        if self.replan_interval <= 0:
            return False

        return step_count % self.replan_interval == 0

    @abstractmethod
    def assign_waypoints(
        self,
        robots: List,
        step_count: int,
        **kwargs
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Assign waypoints to robots.

        Args:
            robots: List of Robot objects
            step_count: Current simulation step
            **kwargs: Additional strategy-specific parameters
                     (e.g., llm_input for LLM strategy, semantic info for others)

        Returns:
            Dict[robot_index] -> list of (x, y) waypoints
        """
        pass

    def reset(self):
        """
        Reset strategy state (called when loading new scene).
        """
        self.last_plan_step = -1


class LLMStrategy(WaypointStrategy):
    """
    LLM-based semantic planning strategy.
    Uses GPT-4o to assign robots to rooms, then uses Hungarian algorithm
    for frontier assignment within rooms.

    Neighbors are computed using grid-based adjacency (4-connected cells).
    This is more reliable than door-based connectivity.

    Ablation study flags (all enabled by default):
    - use_neighbors: Include neighbor room information in LLM input (default: True)
    - use_task_info: Include task information (target room type for search) (default: True)

    By default, all features are ENABLED for optimal performance.
    Only disable for ablation studies to measure component contributions.
    """

    def __init__(
        self,
        replan_interval: int = 50,
        mission: str = "explore",
        model: str = "gpt-4o",
        use_neighbors: bool = True,
        use_task_info: bool = True
    ):
        super().__init__(replan_interval)
        self.mission = mission
        self.model = model

        # Ablation flags
        self.use_neighbors = use_neighbors
        self.use_task_info = use_task_info

    def assign_waypoints(
        self,
        robots: List,
        step_count: int,
        **kwargs
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Use LLM to assign robots to rooms, then compute waypoints.

        Required kwargs:
            llm_input: World state from LLMWorldBuilder (graph structure)
            visualization_data: Detailed room/frontier/door data
        """
        from llm_planner import plan_with_llm

        # Get llm_input and visualization_data from kwargs
        llm_input = kwargs.get('llm_input')
        visualization_data = kwargs.get('visualization_data')

        if llm_input is None:
            raise ValueError("LLMStrategy requires 'llm_input' in kwargs")
        if visualization_data is None:
            raise ValueError("LLMStrategy requires 'visualization_data' in kwargs")

        fused = robots[0].local_map.copy()
        for robot in robots[1:]:
            fused = np.maximum(fused, robot.local_map)
        for robot in robots:
            robot.local_map = fused.copy()
            
        waypoints = plan_with_llm(
            llm_input=llm_input,
            visualization_data=visualization_data,
            robots=robots,
            num_robots=len(robots),
            mission=self.mission,
            model=self.model,
            use_task_info=self.use_task_info  # Ablation flag for task info
        )

        # Store plan summary for next round

        print(f"[INFO] LLM assigned waypoints at step {step_count}: {waypoints}")

        return waypoints

    def reset(self):
        super().reset()


class LLMWaypointStrategy(WaypointStrategy):
    """
    GPT-4o centralized planner for multi-robot exploration / search.

    Wrapper for llm_strategy.LLMStrategy.

    Ablation flags:
    - use_vis:          Include aggregated map image in the LLM prompt.
    - use_skeleton:     Include skeleton (medial axis) overlay in the map visualization.
    - use_frontiers:    Include frontier cluster information (text + visualization markers).
    - use_prior_info:   Include a natural-language environment description.
    """

    def __init__(
        self,
        T_coord: int = 50,
        mission: str = "explore",
        model: str = "gpt-4o",
        max_cluster_distance: int = 10,
        use_vis: bool = True,
        use_skeleton: bool = True,
        use_frontiers: bool = True,
        use_prior_info: bool = True,
        prior_info=None,  # str | dict[str, str] | None
    ):
        super().__init__(replan_interval=1)  # LLMStrategy manages timing via T_coord

        self.T_coord = T_coord

        from llm_strategy import LLMStrategy

        self.llm = LLMStrategy(
            T_coord=T_coord,
            mission=mission,
            model=model,
            max_cluster_distance=max_cluster_distance,
            use_vis=use_vis,
            use_skeleton=use_skeleton,
            use_frontiers=use_frontiers,
            use_prior_info=use_prior_info,
            prior_info=prior_info,
        )

    def assign_waypoints(
        self,
        robots: List,
        step_count: int,
        **kwargs
    ) -> Dict[int, List[Tuple[int, int]]]:
        try:
            waypoints = self.llm.assign_waypoints(robots, step_count)
            if waypoints is None:
                print(f"[ERROR] LLM returned None at step {step_count}")
                return {}
            return waypoints
        except Exception as e:
            print(f"[ERROR] LLM exception at step {step_count}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def reset(self):
        super().reset()
        self.llm.reset()


class RacerWaypointStrategy(WaypointStrategy):
    """
    RACER (Rapid Adaptive Coverage Exploration for Robots) strategy.

    Wrapper for the improved racer_strategy.RacerStrategy implementation.
    Uses HGrid partitioning, CVRP coordination, and frontier-guided exploration.

    Note: RACER manages its own coordination timing internally via T_coord.
    The wrapper always returns should_replan=True to let RACER decide when to coordinate.
    """

    def __init__(
        self,
        L: int = 5,                 # HGrid levels
        alpha_u: float = 0.7,       # Subdivision threshold
        delta_u: int = 5,           # Pruning threshold
        alpha_capacity: float = 0.55,  # Capacity factor for CVRP
        T_coord: int = 50,          # RACER's coordination interval
    ):
        super().__init__(replan_interval=1)  # Always replan (RACER manages coordination internally)

        # Store T_coord for external access
        self.T_coord = T_coord

        # Import and create the actual RACER implementation
        from racer_strategy import RacerStrategy

        self.racer = RacerStrategy(
            L=L,
            alpha_u=alpha_u,
            delta_u=delta_u,
            T_coord=T_coord,
            alpha_capacity=alpha_capacity,
        )

    def assign_waypoints(
        self,
        robots: List,
        step_count: int,
        **kwargs
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Delegate to the RACER implementation.

        Note: RACER doesn't use llm_input, it builds its own frontier detection.
        """
        try:
            # Call the improved RACER implementation
            waypoints = self.racer.assign_waypoints(robots, step_count)

            # Ensure we always return a dict (never None)
            if waypoints is None:
                print(f"[ERROR] RACER returned None at step {step_count}")
                return {}

            return waypoints
        except Exception as e:
            print(f"[ERROR] RACER exception at step {step_count}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def reset(self):
        super().reset()
        # Reset RACER state
        self.racer.initialized = False
        self.racer.robot_cells.clear()

class AEPWaypointStrategy(WaypointStrategy):
    """
    AEP (Active Exploration Planning) strategy.

    Wrapper for aep_strategy.AEPStrategy implementation.
    Uses Voronoi partitioning, information gain estimation, and cache-based planning.

    Note: AEP manages its own coordination timing internally via T_coord.
    The wrapper always returns should_replan=True to let AEP decide when to coordinate.
    """

    def __init__(
        self,
        n_samples: int = 80,         # Number of candidate samples per robot
        lambda_dist: float = 0.08,   # Distance penalty factor
        g_zero: int = 5,             # Minimum gain threshold
        cache_gain_threshold: int = 10,  # Minimum gain to cache a point
        cache_max_size: int = 200,   # Maximum cache size per robot
        T_coord: int = 50,           # AEP's coordination interval
    ):
        super().__init__(replan_interval=1)  # Always replan (AEP manages coordination internally)

        # Store T_coord for external access
        self.T_coord = T_coord

        # Import and create the AEP implementation
        from aep_strategy import AEPStrategy

        self.aep = AEPStrategy(
            T_coord=T_coord,
            n_samples=n_samples,
            lambda_dist=lambda_dist,
            g_zero=g_zero,
            cache_gain_threshold=cache_gain_threshold,
            cache_max_size=cache_max_size,
        )

    def assign_waypoints(
        self,
        robots: List,
        step_count: int,
        **kwargs
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Delegate to the AEP implementation.

        Note: AEP doesn't use llm_input, it uses Voronoi partitioning and gain estimation.
        """
        try:
            # Call the AEP implementation
            waypoints = self.aep.assign_waypoints(robots, step_count)

            # Ensure we always return a dict (never None)
            if waypoints is None:
                print(f"[ERROR] AEP returned None at step {step_count}")
                return {}

            return waypoints
        except Exception as e:
            print(f"[ERROR] AEP exception at step {step_count}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def reset(self):
        super().reset()
        # Reset AEP state
        self.aep.reset()


class HungarianWaypointStrategy(WaypointStrategy):
    """
    Hungarian Assignment strategy for frontier-based exploration.

    Wrapper for hungarian_strategy.HungarianStrategy implementation.
    Uses Hungarian algorithm to optimally assign robots to frontiers.

    Key features:
    - Global coordination at fixed intervals (T_coord)
    - Fuses all robot maps at coordination time
    - Extracts frontiers and assigns robots optimally
    - Between coordination, robots use local controller for fallback

    Note: Hungarian manages its own coordination timing internally via T_coord.
    The wrapper always returns should_replan=True to let Hungarian decide when to coordinate.
    """

    def __init__(
        self,
        max_cluster_distance: int = 10, # BFS depth limit for frontier clustering
        T_coord: int = 50,              # Hungarian's coordination interval
    ):
        super().__init__(replan_interval=1)  # Always replan (Hungarian manages coordination internally)

        # Store T_coord for external access
        self.T_coord = T_coord

        from hungarian_strategy import HungarianStrategy

        self.hungarian = HungarianStrategy(
            T_coord=T_coord,
            max_cluster_distance=max_cluster_distance,
        )

    def assign_waypoints(self, robots, step_count, **kwargs):
        """
        Assign waypoints using Hungarian algorithm.

        At coordination time:
        - Fuse all robot maps
        - Extract frontiers
        - Compute optimal robot-frontier assignment

        Between coordination:
        - Robots continue to assigned waypoints
        - If waypoint reached early, local controller takes over
        """
        try:
            waypoints = self.hungarian.assign_waypoints(robots, step_count)
            if waypoints is None:
                print(f"[ERROR] Hungarian returned None at step {step_count}")
                return {}
            return waypoints
        except Exception as e:
            print(f"[ERROR] Hungarian exception at step {step_count}: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def reset(self):
        super().reset()
        self.hungarian.reset()
