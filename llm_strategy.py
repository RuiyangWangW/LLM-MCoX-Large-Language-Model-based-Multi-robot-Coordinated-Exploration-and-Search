"""
LLM-based Centralized Waypoint Planning Strategy

Uses GPT-4o as a centralized planner that:
1. Receives an aggregated map visualization (image) + structured text context
2. Outputs per-robot waypoint lists as (row, col) grid coordinates

The LLM outputs grid coordinates directly — not frontier cluster IDs.
Frontier clusters are provided as candidate suggestions in the prompt, but the
LLM is free to output any valid (row, col) position on the map.

Ablation flags:
- use_vis:          Include the aggregated map image in the LLM input
- use_skeleton:     Include skeleton (medial axis) overlay in the map visualization
- use_frontiers:    Include frontier cluster information (text + visualization markers)
- use_prior_info:   Include a natural-language description of the environment

Replan interval (T_coord) controls how often the LLM is called.
Between calls, robots continue on their previously assigned waypoints.
"""

import base64
import io
import json
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from hungarian_strategy import extract_frontier_clusters


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXPLORE_TEMPLATE = """\
You are coordinating a team of {n} robots exploring an unknown environment \
represented as a 2-D grid (row, col).
MISSION: Exploration — cover as much of the map as possible as efficiently as possible.
For each robot, output an ordered list of (row, col) grid waypoints to visit.
Use the suggested frontier positions and assign waypoints towards frontiers with larger size based on the attached image map if available.{skeleton_hint}
Spread robots across different regions; avoid assigning the same position to multiple robots.
Never assign waypoints that are already known to be free or occupied; focus on the unknown parts of the map!
"""

SEARCH_TEMPLATE = """\
You are coordinating a team of {n} robots searching an unknown environment \
represented as a 2-D grid (row, col) for a hidden target.
MISSION: Search — find the target as quickly as possible.
For each robot, output an ordered list of (row, col) grid waypoints to visit.
Use the suggested frontier positions and assign waypoints towards frontiers with larger size based on the attached image map if available.{skeleton_hint}
If task prior information provided, prioritize the areas accordingly and assign waypoints to it based on the image map and your guess.
Spread robots across different regions; avoid assigning the same position to multiple robots.
Never assign waypoints that are already known to be free or occupied; focus on the unknown parts of the map!
"""


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

ROBOT_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]

def save_debug_image(png_bytes, prefix="llm_input"):
    os.makedirs("debug_images", exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"debug_images/{prefix}_{timestamp}.png"

    with open(filename, "wb") as f:
        f.write(png_bytes)

    print(f"[DEBUG] Saved image to {filename}")
    return filename


def build_map_image(robots, frontier_clusters: List[Dict], use_skeleton: bool = True, use_frontiers: bool = True) -> bytes:
    """
    Render the aggregated local map with per-robot colours and frontier markers.

    - Dark gray  = unknown
    - White      = free
    - Near-black = wall
    - Cyan dots  = frontier cluster representative positions (row, col labelled) (if use_frontiers=True)
    - Coloured circles = robots (labelled R0, R1, …)
    - Yellow lines = skeleton (medial axis) overlay (if use_skeleton=True)

    Parameters
    ----------
    robots : list
        List of robot objects with local_map and position attributes.
    frontier_clusters : list
        List of frontier cluster dictionaries.
    use_skeleton : bool
        Whether to include skeleton (medial axis) overlay in the visualization.
    use_frontiers : bool
        Whether to include frontier cluster markers in the visualization.

    Returns
    -------
    bytes
        PNG bytes suitable for base64 encoding.
    """
    agg = robots[0].local_map.copy()
    for r in robots[1:]:
        agg = np.maximum(agg, r.local_map)

    H, W = agg.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)
    rgb[agg == -1] = [0.35, 0.35, 0.35]
    rgb[agg ==  0] = [1.00, 1.00, 1.00]
    rgb[agg ==  1] = [0.10, 0.10, 0.10]

    # ---------------------------------------------------------
    # Skeleton overlay (exposes corridor backbone) - optional
    # ---------------------------------------------------------
    if use_skeleton:
        free_mask = (agg == 0)
        if np.sum(free_mask) > 0:
            skeleton = skeletonize(free_mask)
            # Strong yellow backbone
            rgb[skeleton] = np.array([1.0, 0.85, 0.0])

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.imshow(rgb, origin="lower")

    # Frontier cluster markers — label with (row, col) so the LLM can read coords (if use_frontiers=True)
    if use_frontiers:
        for cluster in frontier_clusters:
            rr, cc = cluster["rep"]
            ax.plot(cc, rr, "c^", markersize=5, zorder=3)
            ax.text(cc + 1, rr + 1, f"({rr},{cc})",
                    color="cyan", fontsize=5, zorder=4)

    # Robot markers
    legend_patches = []
    for i, robot in enumerate(robots):
        color = ROBOT_COLORS[i % len(ROBOT_COLORS)]
        rr, cc = map(int, robot.position)
        ax.plot(cc, rr, "o", color=color, markersize=10,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ax.text(cc + 1.5, rr + 1.5, f"R{i}",
                color=color, fontsize=7, fontweight="bold", zorder=6)
        legend_patches.append(
            mpatches.Patch(color=color, label=f"Robot {i} @ ({rr},{cc})")
        )

    ax.legend(handles=legend_patches, loc="upper right",
              fontsize=6, framealpha=0.7)
    ax.set_title("Aggregated Local Map  (row increases downward, col rightward)",
                 fontsize=8)
    ax.axis("off")
    plt.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    png_bytes = buf.read()
    #save_debug_image(png_bytes, prefix="aggregated_map")
    buf.close()
    plt.close(fig)
    return png_bytes


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_gpt4o(messages: List[Dict], model: str = "gpt-4o") -> str:
    """Call the OpenAI chat completion API and return the raw text response."""
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1024,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def _parse_llm_response(
    text: str,
    num_robots: int,
    map_shape: Tuple[int, int],
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Parse the LLM JSON response into {robot_id: [(row, col), ...]}.

    Expected JSON schema:
    {
      "assignments": {
        "0": [[row, col], [row, col], ...],
        "1": [[row, col], ...],
        ...
      }
    }

    Validation:
    - Robot IDs outside [0, num_robots) are dropped.
    - Waypoints with row/col outside the map grid are dropped.
    - Malformed waypoint entries (not length-2 lists) are dropped.
    """
    start = text.find("{")
    end   = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in LLM response:\n{text}")

    data = json.loads(text[start:end])

    raw_assignments = data.get("assignments", {})

    H, W = map_shape
    assignments: Dict[int, List[Tuple[int, int]]] = {
        rid: [] for rid in range(num_robots)
    }

    for key, wps in raw_assignments.items():
        rid = int(key)
        if rid not in assignments:
            continue
        valid = []
        for wp in wps:
            if not (isinstance(wp, (list, tuple)) and len(wp) == 2):
                continue
            r, c = int(wp[0]), int(wp[1])
            if 0 <= r < H and 0 <= c < W:
                valid.append((r, c))
        assignments[rid] = valid

    return assignments


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_messages(
    robots,
    frontier_clusters: List[Dict],
    map_shape: Tuple[int, int],
    mission: str,
    use_skeleton: bool,
    use_frontiers: bool,
    use_prior_info: bool,
    prior_info: str,
    png_bytes: Optional[bytes],
) -> List[Dict]:
    """Assemble the OpenAI messages list."""
    num_robots = len(robots)
    H, W = map_shape

    # Skeleton hint (only included if there's an image AND use_skeleton is True)
    skeleton_hint = ""
    if png_bytes is not None and use_skeleton:
        skeleton_hint = (
            "Robots should prioritize:\n"
            "1. Junction nodes in the skeleton (room entrances) lead to frontiers\n"
            "2. Separate branches among robots to reduce overlap"
        )

    # Mission template
    if mission == "explore":
        mission_text = EXPLORE_TEMPLATE.format(n=num_robots, skeleton_hint=skeleton_hint)
    else:
        mission_text = SEARCH_TEMPLATE.format(n=num_robots, skeleton_hint=skeleton_hint)

    # Optional blocks
    prior_block = ""
    if use_prior_info and prior_info:
        prior_block = f"\nEnvironment description:\n{prior_info}\n"

    # Robot positions
    robot_lines = "\n".join(
        f"  Robot {i}: ({int(r.position[0])}, {int(r.position[1])})"
        for i, r in enumerate(robots)
    )

    # Frontier cluster suggestions (only included if use_frontiers is True)
    frontier_block = ""
    if use_frontiers:
        frontier_lines = "\n".join(
            f"  ({c['rep'][0]}, {c['rep'][1]})  [{c['size']} cells]"
            for c in frontier_clusters
        )
        frontier_block = f"\nFrontier positions (row, col) [cluster size]:\n{frontier_lines}\n"

    # Output schema
    robot_id_list = list(range(num_robots))
    schema_rows = "\n".join(
        f'    "{rid}": [[row, col], ...],'
        for rid in robot_id_list
    )
    output_instruction = f"""\
    Grid size: {H} rows × {W} cols.  Coordinates are (row, col), both 0-indexed.
    Respond with a single JSON object — no markdown, no extra text:
    {{
    "assignments": {{
    {schema_rows}
    }}
    }}
    Each waypoint must be a valid [row, col] pair within the grid.
    Waypoints foreach robot must form a spatially coherent path.
    Consecutive waypoints should lie along the same corridor or branch.
    Avoid jumping between disconnected regions."""

    text_body = (
        mission_text
        + prior_block
        + f"\nRobot positions (row, col):\n{robot_lines}\n"
        + frontier_block
        + "\n"
        + output_instruction
    )

    if png_bytes is not None:
        b64 = base64.b64encode(png_bytes).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": text_body},
                ],
            }
        ]
    else:
        messages = [{"role": "user", "content": text_body}]

    return messages


# ---------------------------------------------------------------------------
# Core strategy
# ---------------------------------------------------------------------------

class LLMStrategy:
    """
    GPT-4o centralized planner for multi-robot exploration / search.

    The LLM receives frontier cluster positions as suggestions but outputs
    (row, col) grid coordinates directly as waypoints for each robot.

    Parameters
    ----------
    T_coord : int
        Replan interval — how many steps between LLM calls.
    mission : str
        "explore" or "search".
    model : str
        OpenAI model name (default "gpt-4o").
    max_cluster_distance : int
        BFS depth limit for frontier clustering used to build suggestions.
    use_vis : bool
        Include the aggregated map image in the LLM prompt.
    use_skeleton : bool
        Include skeleton (medial axis) overlay in the map visualization.
    use_frontiers : bool
        Include frontier cluster information (text + visualization markers).
    use_prior_info : bool
        Include a natural-language environment description.
    prior_info : str or dict
        The environment description used when use_prior_info=True.
        Can be a plain string (applied to both tasks) or a dict with
        keys "explore" and/or "search" for task-specific descriptions.
        Example:
            prior_info = {
                "explore": "A medium indoor office with long corridors.",
                "search":  "Target is likely in a dead-end room away from the entrance.",
            }
    """

    def __init__(
        self,
        T_coord: int = 20,
        mission: str = "explore",
        model: str = "gpt-4o",
        max_cluster_distance: int = 10,
        use_vis: bool = True,
        use_skeleton: bool = True,
        use_frontiers: bool = True,
        use_prior_info: bool = True,
        prior_info=None,   # str | dict[str, str] | None
    ):
        self.T_coord = T_coord
        self.mission = mission
        self.model = model
        self.max_cluster_distance = max_cluster_distance
        self.use_vis = use_vis
        self.use_skeleton = use_skeleton
        self.use_frontiers = use_frontiers
        self.use_prior_info = use_prior_info
        # Resolve to the mission-appropriate string once at init time
        self.prior_info = self._resolve_prior_info(prior_info, mission)

        self.initialized: bool = False

    @staticmethod
    def _resolve_prior_info(prior_info, mission: str) -> str:
        """Return the prior_info string appropriate for the given mission."""
        if prior_info is None:
            return ""
        if isinstance(prior_info, dict):
            return prior_info.get(mission, prior_info.get("explore", ""))
        return str(prior_info)

    def reset(self):
        self.initialized = False

    def _fuse_maps(self, robots) -> np.ndarray:
        fused = robots[0].local_map.copy()
        for r in robots[1:]:
            fused = np.maximum(fused, r.local_map)
        for r in robots:
            r.local_map = fused.copy()
        return fused

    def assign_waypoints(
        self,
        robots,
        step_count: int,
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        Assign grid-coordinate waypoints to robots via GPT-4o.

        Pipeline at each coordination step:
          1. Fuse all robot local maps
          2. Extract frontier clusters (used as prompt suggestions + image markers)
          3. Optionally render the aggregated map image
          4. Call GPT-4o — receives image (optional) + text prompt
          5. Parse (row, col) waypoint lists from JSON response
          7. Return {robot_id: [(row, col), ...]}

        Between coordination steps, returns {} so robots keep existing waypoints.
        """
        coord_time = (not self.initialized) or (step_count % self.T_coord == 0)
        if not coord_time:
            return {}

        waypoints: Dict[int, List[Tuple[int, int]]] = {}

        # 1. Fuse maps
        fused_map = self._fuse_maps(robots)
        map_shape = fused_map.shape

        # 2. Extract frontier clusters (for suggestions + visualisation)
        frontier_clusters = extract_frontier_clusters(
            fused_map,
            unknown_value=-1,
            free_value=0,
            max_cluster_distance=self.max_cluster_distance,
        )

        if not frontier_clusters:
            self.initialized = True
            return waypoints

        # 3. Render image (if use_vis)
        png_bytes = None
        if self.use_vis:
            try:
                png_bytes = build_map_image(robots, frontier_clusters, use_skeleton=self.use_skeleton, use_frontiers=self.use_frontiers)
            except Exception as e:
                print(f"[WARN] Map image rendering failed: {e}")

        # 4. Build prompt and call LLM
        messages = _build_messages(
            robots=robots,
            frontier_clusters=frontier_clusters,
            map_shape=map_shape,
            mission=self.mission,
            use_skeleton=self.use_skeleton,
            use_frontiers=self.use_frontiers,
            use_prior_info=self.use_prior_info,
            prior_info=self.prior_info,
            png_bytes=png_bytes,
        )

        try:
            response_text = _call_gpt4o(messages, model=self.model)
            print(f"[LLM] Raw response at step {step_count}:\n{response_text}\n")
        except Exception as e:
            print(f"[WARN] LLM call failed at step {step_count}: {e}")
            self.initialized = True
            return waypoints

        # 5. Parse (row, col) waypoints from response
        try:
            assignments = _parse_llm_response(
                response_text, len(robots), map_shape
            )
        except Exception as e:
            print(f"[WARN] LLM response parse failed at step {step_count}: {e}")
            self.initialized = True
            return waypoints

        print(f"[LLM] Assignments:  {assignments}")

        for rid, wps in assignments.items():
            if not wps:
                continue
            pts   = [np.array(wp, dtype=float) for wp in wps]
            waypoints[rid] = [(int(round(p[0])), int(round(p[1]))) for p in pts]

        self.initialized = True
        print(f"[LLM] Final waypoints at step {step_count}: {waypoints}")
        return waypoints
