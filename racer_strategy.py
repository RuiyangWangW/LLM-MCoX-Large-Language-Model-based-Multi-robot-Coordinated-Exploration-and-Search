import math
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
from dataclasses import dataclass
# OR-Tools for CVRP
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
import heapq
from robot import extract_frontier_clusters





@dataclass
class HGridCell:
    level: int
    min_x: int
    min_y: int
    size: int

    unknown_count: int
    centroid: Tuple[float, float]



class HGridBuilder:
    """
    Paper-faithful HGrid maintenance (Alg. 1 / Sec. IV-B).

    - Maintains an *active* leaf-cell list L_act provided by the caller (per robot).
    - Updates only cells overlapping updated-map AABBs B_m.
    - Applies:
        (1) remove fully known cells,
        (2) subdivide if enough space is known (unknown_ratio < 1 - alpha_u),
        (3) prune tiny unknown at finest level (unknown_count < delta_u),
        (4) remove unreachable centroids (w.r.t. the *given robot position*).
    """

    def __init__(self, L: int = 5, alpha_u: float = 0.7, delta_u: int = 5):
        self.L = int(L)
        self.alpha_u = float(alpha_u)
        self.delta_u = int(delta_u)

    # ---------- public API ----------

    def build_initial(self, occ: np.ndarray) -> List[HGridCell]:
        """
        Initial decomposition S1 in the paper: tile the map with level-0 cells.
        Keep only cells that contain unknown.
        """
        H, W = occ.shape
        root_size = 2 ** self.L
        out: List[HGridCell] = []
        for y in range(0, H, root_size):
            for x in range(0, W, root_size):
                c = self._make_cell(occ, level=0, min_x=x, min_y=y, size=root_size)
                if c is not None:
                    out.append(c)
        return out

    def visible_unknown_from_frontier(
        self,
        occ: np.ndarray,
        max_depth: int = 5,
    ) -> np.ndarray:
        """
        Return boolean mask of UNKNOWN cells reachable
        from any frontier within max_depth steps.
        """

        H, W = occ.shape

        # 1️⃣ Extract frontiers once
        frontiers = extract_frontier_clusters(
            occ,
            unknown_value=-1,
            free_value=0,
        )

        visited = np.zeros((H, W), dtype=bool)
        queue = deque()

        # 2️⃣ Seed BFS from UNKNOWN neighbors of frontier reps
        for f in frontiers:
            fy, fx = f["rep"]  # (row, col)
            visited[fy, fx] = True
            queue.append((fy, fx, 1))

        # 3️⃣ Depth-limited BFS through UNKNOWN only
        while queue:
            y, x, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx

                if (
                    0 <= ny < H and
                    0 <= nx < W and
                    (occ[ny, nx] == -1 or occ[ny, nx] == 0) and
                    not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx, depth + 1))

        return visited


    def update_lact(
        self,
        occ: np.ndarray,
        lact: List[HGridCell],
        updated_aabbs: List[Tuple[int, int, int, int]],
        *,
        robot_pos_xy: Tuple[float, float],
        treat_unknown_as_obstacle: bool = False,
    ) -> List[HGridCell]:
        """
        Incremental hgrid update following Alg. 1.

        Args:
            occ: occupancy grid (-1 unknown, 0 free, 1 occupied)
            lact: current active leaf cells for *one* robot
            updated_aabbs: list of updated-map AABBs B_m, each as (min_x, min_y, max_x_excl, max_y_excl)
            robot_pos_xy: (x,y) == (col,row) for reachability pruning
        """
        if not lact:
            lact = []

        # If no updated region known, conservatively treat as whole map.
        H, W = occ.shape
        if not updated_aabbs:
            return lact

        # --------------------------------------------------
        # Compute visible unknown mask once per update
        # --------------------------------------------------

        visible_unknown = set()
        visible_unknown = self.visible_unknown_from_frontier(occ)

        # Step 1: update only cells overlapping any B_m
        new_lact: List[HGridCell] = []
        for c in lact:
            if not self._cell_overlaps_any_aabb(c, updated_aabbs):
                # unchanged cell; keep as-is
                new_lact.append(c)
                continue

            # recompute cell stats from current map
            refreshed = self._make_cell(occ, c.level, c.min_x, c.min_y, c.size, visible_unknown)
            if refreshed is None:
                # Step 2: discard fully known cells
                continue

            # Step 3: subdivide if enough space is known
            if self._should_subdivide(occ, refreshed):
                new_lact.extend(self._subdivide_to_children(occ, refreshed, visible_unknown))
                continue

            # Step 4: prune tiny unknown at finest level
            if refreshed.level == self.L and refreshed.unknown_count < self.delta_u:
                continue

            new_lact.append(refreshed)

        # Step 4 (continued): it is possible new cells are created only where parents existed.
        # That's intended: ownership of children inherits ownership of the parent.

        # Step 5: remove unreachable centroids w.r.t this robot
        new_lact = self._prune_unreachable_by_astar(
            occ,
            robot_pos_xy=robot_pos_xy,
            cells=new_lact,
            treat_unknown_as_obstacle=treat_unknown_as_obstacle,
        )

        return new_lact

    # ---------- internal helpers ----------

    def _cell_overlaps_any_aabb(
        self, c: HGridCell, aabbs: List[Tuple[int, int, int, int]]
    ) -> bool:
        cx0, cy0 = c.min_x, c.min_y
        cx1, cy1 = c.min_x + c.size, c.min_y + c.size
        for ax0, ay0, ax1, ay1 in aabbs:
            if (cx0 < ax1 and ax0 < cx1 and cy0 < ay1 and ay0 < cy1):
                return True
        return False

    def _make_cell(
        self,
        occ: np.ndarray,
        level: int,
        min_x: int,
        min_y: int,
        size: int,
        visible_unknown: Optional[np.ndarray] = None,
    ) -> Optional[HGridCell]:

        H, W = occ.shape
        max_x = min(min_x + size, W)
        max_y = min(min_y + size, H)

        sub = occ[min_y:max_y, min_x:max_x]

        sub_unknown = (sub == -1)

        if visible_unknown is not None:
            sub_visible = visible_unknown[min_y:max_y, min_x:max_x]
            valid_mask = sub_unknown & sub_visible
        else:
            valid_mask = sub_unknown

        unknown_count = np.count_nonzero(valid_mask)

        if unknown_count == 0:
            return None

        ys, xs = np.nonzero(valid_mask)

        centroid_x = float(xs.mean() + min_x)
        centroid_y = float(ys.mean() + min_y)

        return HGridCell(
            level=int(level),
            min_x=int(min_x),
            min_y=int(min_y),
            size=int(size),
            unknown_count=unknown_count,
            centroid=(centroid_x, centroid_y),
        )



    def _should_subdivide(self, occ: np.ndarray, c: HGridCell) -> bool:
        if c.level >= self.L or c.size <= 1:
            return False
        H, W = occ.shape
        max_x = min(c.min_x + c.size, W)
        max_y = min(c.min_y + c.size, H)
        total = max(1, (max_x - c.min_x) * (max_y - c.min_y))
        unknown_ratio = float(c.unknown_count) / float(total)
        return unknown_ratio < (1.0 - self.alpha_u)

    def _subdivide_to_children(
        self,
        occ: np.ndarray,
        c: HGridCell,
        visible_unknown: Optional[np.ndarray] = None,
        ) -> List[HGridCell]:
        half = max(1, c.size // 2)
        out: List[HGridCell] = []
        for dy in (0, half):
            for dx in (0, half):
                child = self._make_cell(
                    occ,
                    level=c.level + 1,
                    min_x=c.min_x + dx,
                    min_y=c.min_y + dy,
                    size=half,
                    visible_unknown=visible_unknown
                )
                if child is None:
                    continue
                # If at finest level, apply pruning threshold immediately
                if child.level == self.L and child.unknown_count < self.delta_u:
                    continue
                # Recursive subdivision is handled by repeated updates as the map changes,
                # matching Alg. 1's incremental nature.
                out.append(child)
        return out

    def _prune_unreachable_by_astar(
        self,
        occ: np.ndarray,
        *,
        robot_pos_xy: Tuple[float, float],
        cells: List[HGridCell],
        treat_unknown_as_obstacle: bool,
    ) -> List[HGridCell]:
        if not cells:
            return cells
        pruned: List[HGridCell] = []
        for c in cells:
            d = astar_path_length(
                occ,
                start_xy=robot_pos_xy,
                goal_xy=c.centroid,
                treat_unknown_as_obstacle=treat_unknown_as_obstacle,
            )
            if d is not None:
                pruned.append(c)
        return pruned


# ============================================================
# Pairwise CVRP solver
# ============================================================


# ============================================================
# Sparse HGrid graph for path searching acceleration (Sec. V-B)
# ============================================================

class SparseHGridGraph:
    """Sparse graph over hgrid cells to approximate inter-cell path lengths.

    Nodes: hgrid leaf cells (identified by (level, min_x, min_y, size))
    Edges: between spatially adjacent cells (touching/overlapping AABB) and
           k-nearest neighbors by centroid (to keep connectivity).
    Edge weight: A* grid path length between cell centroids (collision-free).

    This follows the spirit of RACER Sec. V-B: avoid O(N_h^2) full-map searches
    by searching on a sparse graph. 
    """

    def __init__(
        self,
        occ: np.ndarray,
        cells: List[HGridCell],
        k_nn: int = 6,
        treat_unknown_as_obstacle: bool = False,
    ):
        self.occ = occ
        self.cells = list(cells)
        self.k_nn = max(0, int(k_nn))
        self.treat_unknown_as_obstacle = treat_unknown_as_obstacle

        self._key_to_idx: Dict[Tuple[int, int, int, int], int] = {}
        for i, c in enumerate(self.cells):
            self._key_to_idx[self.cell_key(c)] = i

        # adjacency list: idx -> list[(nbr_idx, weight)]
        self._adj: List[List[Tuple[int, int]]] = [[] for _ in range(len(self.cells))]
        self._build_edges()

        # Dijkstra cache: src_idx -> dict(dst_idx -> dist)
        self._dist_cache: Dict[int, Dict[int, int]] = {}

    @staticmethod
    def cell_key(c: HGridCell) -> Tuple[int, int, int, int]:
        return (c.level, c.min_x, c.min_y, c.size)

    def idx(self, cell: HGridCell) -> Optional[int]:
        return self._key_to_idx.get(self.cell_key(cell), None)

    def _touching_aabb(self, a: HGridCell, b: HGridCell) -> bool:
        # AABBs in (x, y) grid coordinates.
        ax0, ay0 = a.min_x, a.min_y
        ax1, ay1 = a.min_x + a.size, a.min_y + a.size
        bx0, by0 = b.min_x, b.min_y
        bx1, by1 = b.min_x + b.size, b.min_y + b.size

        # Touching/overlapping if intervals overlap (including touching).
        return (ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1)

    def _add_edge(self, i: int, j: int):
        if i == j:
            return
        ci = self.cells[i]
        cj = self.cells[j]
        w = astar_path_length(
            self.occ,
            start_xy=ci.centroid,
            goal_xy=cj.centroid,
            treat_unknown_as_obstacle=self.treat_unknown_as_obstacle,
        )
        if w is None:
            return
        self._adj[i].append((j, int(w)))

    def _build_edges(self):
        n = len(self.cells)
        if n <= 1:
            return

        # 1) adjacency via touching AABBs (sparse but meaningful)
        for i in range(n):
            for j in range(i + 1, n):
                if self._touching_aabb(self.cells[i], self.cells[j]):
                    self._add_edge(i, j)
                    self._add_edge(j, i)

        # 2) kNN augmentation for connectivity
        if self.k_nn > 0:
            cents = np.array([c.centroid for c in self.cells], dtype=float)  # (x, y)
            for i in range(n):
                # squared distances
                d2 = np.sum((cents - cents[i]) ** 2, axis=1)
                nn = np.argsort(d2)
                added = 0
                for j in nn[1:]:
                    if added >= self.k_nn:
                        break
                    self._add_edge(i, int(j))
                    self._add_edge(int(j), i)
                    added += 1

    def _dijkstra_from(self, src: int) -> Dict[int, int]:
        import heapq
        dist = {src: 0}
        pq = [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if d != dist.get(u, None):
                continue
            for v, w in self._adj[u]:
                nd = d + w
                if nd < dist.get(v, 10**18):
                    dist[v] = nd
                    heapq.heappush(pq, (nd, v))
        return dist

    def shortest_path_len_by_idx(self, src: int, dst: int) -> Optional[int]:
        if src == dst:
            return 0
        if src not in self._dist_cache:
            self._dist_cache[src] = self._dijkstra_from(src)
        return self._dist_cache[src].get(dst, None)


def astar_path_length(
    occ,
    start_xy: Tuple[float, float],
    goal_xy: Tuple[float, float],
    *,
    free_value: int = 0,
    unknown_value: int = -1,
    treat_unknown_as_obstacle: bool = False,
    connectivity: int = 4,
) -> Optional[int]:
    """
    Return shortest path length in grid steps using A*.
    """

    H, W = occ.shape

    sx = int(round(start_xy[0]))
    sy = int(round(start_xy[1]))
    gx = int(round(goal_xy[0]))
    gy = int(round(goal_xy[1]))

    sx = max(0, min(W - 1, sx))
    sy = max(0, min(H - 1, sy))
    gx = max(0, min(W - 1, gx))
    gy = max(0, min(H - 1, gy))

    if (sx, sy) == (gx, gy):
        return 0

    def is_free(x, y):
        v = int(occ[y, x])
        if v == free_value:
            return True
        if v == unknown_value:
            return not treat_unknown_as_obstacle
        return False

    if not is_free(sx, sy) or not is_free(gx, gy):
        return None

    if connectivity == 8:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]
        def step_cost(dx, dy):
            return 1.41421356237 if dx != 0 and dy != 0 else 1.0
        def heuristic(x, y):
            dx = abs(x - gx)
            dy = abs(y - gy)
            return (dx + dy) + (1.41421356237 - 2) * min(dx, dy)
    else:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
        def step_cost(dx, dy):
            return 1.0
        def heuristic(x, y):
            return abs(x - gx) + abs(y - gy)

    gbest = {(sx, sy): 0.0}
    pq = [(heuristic(sx, sy), 0.0, sx, sy)]

    while pq:
        f, g, x, y = heapq.heappop(pq)

        if g > gbest.get((x, y), float("inf")):
            continue

        if (x, y) == (gx, gy):
            return int(round(g))

        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < W and 0 <= ny < H):
                continue
            if not is_free(nx, ny):
                continue

            ng = g + step_cost(dx, dy)
            if ng < gbest.get((nx, ny), float("inf")):
                gbest[(nx, ny)] = ng
                heapq.heappush(pq, (ng + heuristic(nx, ny), ng, nx, ny))

    return None

# ============================================================
# Pairwise CVRP solver (Sec. V-A) with consistency + sparse graph
# ============================================================

def solve_pairwise_cvrp(
    occ,
    depots: List[Tuple[float, float]],          # [robot_i_pos(x,y), robot_j_pos(x,y)]
    tasks: List["HGridCell"],                   # union tasks for the pair
    alpha_capacity: float,
    graph: Optional["SparseHGridGraph"] = None, # sparse graph over *tasks* (in same order)
    prev_first_task_keys: Tuple[
        Optional[Tuple[int, int, int, int]],
        Optional[Tuple[int, int, int, int]]
    ] = (None, None),
    beta_con: int = 2000,
    treat_unknown_as_obstacle: bool = False,
    time_limit_s: int = 1,
) -> Tuple[List["HGridCell"], List["HGridCell"]]:
    """
    Paper-aligned pairwise CVRP for RACER:
      - Open paths (no return-to-depot) via dummy end nodes per vehicle
      - Capacity constraint Eq.(6)
      - Robot->cell cost: A* len + consistency penalty (Eq.(4)-(5))
      - Cell->cell cost: sparse-graph shortest path len (Sec.V-B), fallback to A* if needed

    Returns:
      (ordered_cells_for_robot0, ordered_cells_for_robot1)
    """

    if not tasks:
        return [], []

    assert len(depots) == 2, "depots must be [pos0, pos1]"

    # ---------- helpers ----------
    def cell_key(c: "HGridCell") -> Tuple[int, int, int, int]:
        return (c.level, c.min_x, c.min_y, c.size)

    BIG = 10**7

    Nh = len(tasks)

    # Node indices:
    # 0: depot for vehicle 0
    # 1: depot for vehicle 1
    # 2..(2+Nh-1): task nodes
    # end0 = 2+Nh
    # end1 = 2+Nh+1
    end0 = 2 + Nh
    end1 = 2 + Nh + 1
    N = 2 + Nh + 2

    # ---------- demand + capacity (Eq. 6) ----------
    demands = [0, 0] + [int(c.unknown_count) for c in tasks] + [0, 0]
    total_unknown = sum(int(c.unknown_count) for c in tasks)
    cap = max(1, int(alpha_capacity * total_unknown))

    # ---------- cost accessors ----------
    # depot->task uses A* + consistency penalty
    prev0, prev1 = prev_first_task_keys

    def depot_to_task_cost(veh: int, tidx: int) -> int:
        # veh in {0,1}, tidx in [0..Nh-1]
        depot_xy = depots[veh]
        task_xy = tasks[tidx].centroid
        w = astar_path_length(
            occ,
            depot_xy,
            task_xy,
            treat_unknown_as_obstacle=treat_unknown_as_obstacle,
        )
        if w is None:
            return BIG
        # Consistency penalty only applies if this cell is the first-connected one (approximation):
        # We encode it on depot->that-cell edge.
        k = cell_key(tasks[tidx])
        if veh == 0 and prev0 is not None and k != prev0:
            w += int(beta_con)

        if veh == 1 and prev1 is not None and k != prev1:
            w += int(beta_con)

        return int(w)

    def task_to_task_cost(i: int, j: int) -> int:
        # i,j are task indices [0..Nh-1]
        if i == j:
            return 0
        if graph is not None:
            d = graph.shortest_path_len_by_idx(i, j)
            if d is not None:
                return int(d)
        # fallback: direct A*
        w = astar_path_length(
            occ,
            tasks[i].centroid,
            tasks[j].centroid,
            treat_unknown_as_obstacle=treat_unknown_as_obstacle,
        )
        if w is None:
            return BIG
        return int(w)

    # ---------- OR-Tools model (open routes via dummy ends) ----------
    manager = pywrapcp.RoutingIndexManager(
        N,
        2,
        [0, 1],          # starts at depots
        [end0, end1],    # ends at dummy end nodes (open paths)
    )
    routing = pywrapcp.RoutingModel(manager)

    # Transit (arc) cost callback
    def cost_cb(from_idx: int, to_idx: int) -> int:
        a = manager.IndexToNode(from_idx)
        b = manager.IndexToNode(to_idx)

        # From end nodes: should not go anywhere
        if a == end0 or a == end1:
            return BIG

        # To end nodes: allow termination at 0 cost (from anywhere)
        if b == end0 or b == end1:
            return 0

        # depot <-> depot (rare): discourage
        if a in (0, 1) and b in (0, 1):
            return BIG

        # depot -> task
        if a in (0, 1) and 2 <= b < 2 + Nh:
            veh = a
            tidx = b - 2
            return depot_to_task_cost(veh, tidx)

        # task -> task
        if 2 <= a < 2 + Nh and 2 <= b < 2 + Nh:
            return task_to_task_cost(a - 2, b - 2)

        # task -> depot: discourage (paper uses open CPs)
        if 2 <= a < 2 + Nh and b in (0, 1):
            return BIG

        # any other unexpected transition
        return BIG

    transit_cb = routing.RegisterTransitCallback(cost_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb)

    # Capacity dimension
    def demand_cb(index: int) -> int:
        node = manager.IndexToNode(index)
        return demands[node]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,
        [cap, cap],
        True,
        "Capacity",
    )

    # Search params
    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(search_params)
    if solution is None:
        # fallback: split evenly
        half = Nh // 2
        return tasks[:half], tasks[half:]

    # ---------- Extract ordered routes ----------
    routes: List[List["HGridCell"]] = [[], []]

    for veh in range(2):
        idx = routing.Start(veh)
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            # Only record task nodes
            if 2 <= node < 2 + Nh:
                routes[veh].append(tasks[node - 2])
            idx = solution.Value(routing.NextVar(idx))

    return routes[0], routes[1]


def prune_unreachable_cells(
    occ: np.ndarray,
    start_pos_xy: Tuple[float, float],
    cells: List[HGridCell],
    *,
    treat_unknown_as_obstacle: bool = False,
) -> List[HGridCell]:
    """
    Paper Alg. 1: remove unreachable centroids w.r.t. THIS robot's current position.
    We implement reachability via A* existence check to the cell centroid.
    """
    if not cells:
        return cells
    pruned: List[HGridCell] = []
    for c in cells:
        d = astar_path_length(
            occ,
            start_xy=start_pos_xy,
            goal_xy=c.centroid,
            treat_unknown_as_obstacle=treat_unknown_as_obstacle,
        )
        if d is not None:
            pruned.append(c)
    return pruned


class RacerStrategy:

    def __init__(
        self,
        L: int = 5,
        alpha_u: float = 0.7,
        delta_u: int = 5,
        T_coord: int = 20,
        alpha_capacity: float = 0.55,
        ncp_k: int = 3,
        beta_con: int = 500,  # paper-style: positive penalty
        sparse_k_nn: int = 6,
        treat_unknown_as_obstacle: bool = False,
    ):
        self.hgrid_builder = HGridBuilder(L, alpha_u, delta_u)

        self.T_coord = int(T_coord)
        self.alpha_capacity = float(alpha_capacity)

        # CP (cell visit order) per robot
        self.robot_cells: Dict[int, List[HGridCell]] = {}
        self.robot_cp: Dict[int, List[HGridCell]] = {}


        # Consistency memory: previous CP's first connected cell key per robot (Eq. 5)
        self.prev_first_task_key: Dict[int, Optional[Tuple[int, int, int, int]]] = {}

        # CP-guided planning uses Next Coverage Points (NCP): first ncp_k cells in the CP order
        self.ncp_k = max(1, int(ncp_k))

        # Consistency term magnitude (paper uses penalty to discourage oscillation)
        self.beta_con = int(beta_con)

        # Sparse graph params
        self.sparse_k_nn = max(0, int(sparse_k_nn))
        self.treat_unknown_as_obstacle = bool(treat_unknown_as_obstacle)

        self.initialized = False
        self._prev_fused_occ: Optional[np.ndarray] = None
        self.astar_cache: Dict[Tuple[Tuple[int,int], Tuple[int,int]], Optional[int]] = {}


    def _cell_key(self, c: HGridCell) -> Tuple[int, int, int, int]:
        return (c.level, c.min_x, c.min_y, c.size)

    def _dedup_cells_preserve_order(self, cells: List[HGridCell]) -> List[HGridCell]:
        seen = set()
        out: List[HGridCell] = []
        for c in cells:
            k = self._cell_key(c)
            if k in seen:
                continue
            seen.add(k)
            out.append(c)
        return out
    

    def _frontier_in_cells(self, frontier: dict, cells: List[HGridCell]) -> bool:
        # Frontier rep is (row, col) from extract_frontier_clusters
        frow, fcol = frontier["rep"]
        for cell in cells:
            if (
                cell.min_x <= fcol < cell.min_x + cell.size and  # col check
                cell.min_y <= frow < cell.min_y + cell.size      # row check
            ):
                return True
        return False

    def _cp_equal(self, a: List[HGridCell], b: List[HGridCell]) -> bool:
        if not a or not b:
            return False

        set_a = {self._cell_key(c) for c in a[:self.ncp_k]}
        set_b = {self._cell_key(c) for c in b[:self.ncp_k]}

        return set_a == set_b

    def _frontiers_in_cells(self, frontiers: List[dict], cells: List[HGridCell]) -> List[dict]:
        """Return frontiers whose representative (row,col) lies inside any of the given cells."""
        if not frontiers or not cells:
            return []
        out = []
        for f in frontiers:
            if self._frontier_in_cells(f, cells):
                out.append(f)
        return out

    def _solve_open_tsp(
        self,
        occ: np.ndarray,
        start_xy: Tuple[float, float],   # (x,y) == (col,row)
        end_xy: Tuple[float, float],     # (x,y) == (col,row)
        start_yaw: float,                # robot yaw in radians
        pts_rc: List[Tuple[int, int]],   # [(row,col), ...]
        v_max: float = 1.0,              # cells per second
        yaw_rate_max: float = 1.0,       # rad per second
    ) -> List[Tuple[int, int]]:
        """
        Paper Sec. VI-B1 time lower bound:
            t_lb = max( Len[P]/v_max, min(|Δφ|, 2π-|Δφ|)/yaw_rate_max )
        """
        if not pts_rc:
            return []

        # Dedup while preserving order
        seen = set()
        uniq: List[Tuple[int, int]] = []
        for rc in pts_rc:
            if rc in seen:
                continue
            seen.add(rc)
            uniq.append(rc)

        if len(uniq) == 1:
            return uniq

        n = len(uniq)
        N = n + 2
        START = 0
        END = N - 1
        BIG = 10**7

        # Convert (row,col) -> (x,y)
        pts_xy = [(float(rc[1]), float(rc[0])) for rc in uniq]
        nodes_xy: List[Tuple[float, float]] = [start_xy] + pts_xy + [end_xy]

        def bearing(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            return math.atan2(b[1] - a[1], b[0] - a[0])

        def ang_diff(a: float, b: float) -> float:
            d = abs(a - b)
            return min(d, 2.0 * math.pi - d)

        def trans_time(a: Tuple[float, float], b: Tuple[float, float]) -> float:
            d = astar_path_length(
                occ,
                start_xy=a,
                goal_xy=b,
                treat_unknown_as_obstacle=self.treat_unknown_as_obstacle,
            )
            if d is None:
                return float(BIG)
            return float(d) / max(1e-9, float(v_max))

        # Precompute translation times and geometric bearings for all directed edges
        t_trans = [[float(BIG)] * N for _ in range(N)]
        theta = [[0.0] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    t_trans[i][j] = 0.0
                    theta[i][j] = 0.0
                else:
                    t_trans[i][j] = trans_time(nodes_xy[i], nodes_xy[j])
                    theta[i][j] = bearing(nodes_xy[i], nodes_xy[j])

        # Incoming heading approximation:
        # - exact at START using robot yaw
        # - heuristic elsewhere using bearing from START -> i
        incoming_heading = [0.0] * N
        incoming_heading[START] = float(start_yaw)
        for i in range(1, N):
            incoming_heading[i] = bearing(start_xy, nodes_xy[i])

        # Time-lower-bound cost matrix (scaled to int for OR-Tools)
        SCALE = 1000
        dist = [[BIG] * N for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == j:
                    dist[i][j] = 0
                    continue

                tt = t_trans[i][j]
                dphi = ang_diff(incoming_heading[i], theta[i][j])
                tr = dphi / max(1e-9, float(yaw_rate_max))

                tlb = max(tt, tr)
                dist[i][j] = BIG if tlb >= float(BIG) else int(round(SCALE * tlb))

        manager = pywrapcp.RoutingIndexManager(N, 1, [START], [END])
        routing = pywrapcp.RoutingModel(manager)

        def cost_cb(from_idx: int, to_idx: int) -> int:
            a = manager.IndexToNode(from_idx)
            b = manager.IndexToNode(to_idx)
            return dist[a][b]

        transit = routing.RegisterTransitCallback(cost_cb)
        routing.SetArcCostEvaluatorOfAllVehicles(transit)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.seconds = 1

        sol = routing.SolveWithParameters(search_params)
        if sol is None:
            # fallback: greedy by tlb
            remaining = list(range(1, 1 + n))
            cur = START
            out_rc: List[Tuple[int, int]] = []
            while remaining:
                best = min(remaining, key=lambda j: dist[cur][j])
                remaining.remove(best)
                out_rc.append(uniq[best - 1])
                cur = best
            return out_rc

        # Extract order (skip START/END)
        idx = routing.Start(0)
        order: List[int] = []
        while not routing.IsEnd(idx):
            order.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
        order.append(manager.IndexToNode(idx))  # END

        seq: List[Tuple[int, int]] = []
        for node in order:
            if 1 <= node <= n:
                seq.append(uniq[node - 1])
        return seq


    def _cp_guided_plan_waypoints(
        self,
        occ: np.ndarray,
        robot_pos_xy: Tuple[float, float],
        robot_yaw: float,
        cells_cp: List[HGridCell],
        all_frontiers: List[dict],
    ) -> List[Tuple[int, int]]:
        """
        Paper Sec. VI-B1 (CP-guided exploration):

        1) Take NCP = first ncp_k cells in CP.
        2) Collect frontier *clusters* whose rep lies in NCP.
           If none exist, expand to all cells in CP.
        3) Solve open TSP: start at robot, visit cluster reps, end at centroid of (k+1)-th CP cell
           (or last available cell if shorter).
        4) Return the ordered reps as waypoints.
        """
        if not all_frontiers:
            return []

        if not cells_cp:
            # no CP: fallback to all frontiers, no end constraint
            reps = [f["rep"] for f in all_frontiers]
            return self._solve_open_tsp(occ, robot_pos_xy, robot_pos_xy, robot_yaw, reps)

        ncp = cells_cp[: self.ncp_k]
        cands = self._frontiers_in_cells(all_frontiers, ncp)
        if not cands:
            cands = self._frontiers_in_cells(all_frontiers, cells_cp)
        if not cands:
            return []

        reps = [f["rep"] for f in cands]

        # Choose end: centroid of (k+1)th cell if exists, else last CP cell centroid
        end_cell_idx = min(self.ncp_k, max(0, len(cells_cp) - 1))
        end_xy = cells_cp[end_cell_idx].centroid

        return self._solve_open_tsp(occ, robot_pos_xy, end_xy, robot_yaw, reps)

    def _cp_guided_candidates(self, all_frontiers: List[dict], cells_cp: List[HGridCell]) -> List[dict]:
        if not cells_cp:
            return []
        ncp = cells_cp[: self.ncp_k]
        cands = [f for f in all_frontiers if self._frontier_in_cells(f, ncp)]
        if cands:
            return cands
        # Expand to full CP if NCP empty
        return [f for f in all_frontiers if self._frontier_in_cells(f, cells_cp)]

    # -------------------------------------------------
    # Main entry point
    # -------------------------------------------------
    def assign_waypoints(self, robots: List, step_count: int) -> Dict[int, List[Tuple[int, int]]]:
        """Assign waypoints to robots using RACER-like CVRP partitioning and CP-guided planning."""
        if not robots:
            return {}

        num_robots = len(robots)

        # -------------------------------------------------
        # Initialization: build initial hgrid on fused map
        # -------------------------------------------------
        if not self.initialized:
            occ = robots[0].local_map.copy()
            for robot in robots[1:]:
                occ = np.maximum(occ, robot.local_map)

            # Paper Sec. IV-B: initialize L_act with the coarsest tiling S1 (level-0 cells)
            init_cells = self.hgrid_builder.build_initial(occ)
            print(f"[RACER] Initial HGrid (S1): {len(init_cells)} cells")

            for i in range(num_robots):
                # Each robot maintains its own L_act and prunes unreachable centroids w.r.t itself (Alg. 1)
                cells_i = prune_unreachable_cells(
                    occ,
                    start_pos_xy=robots[i].position,
                    cells=list(init_cells),
                    treat_unknown_as_obstacle=self.treat_unknown_as_obstacle,
                )
                self.robot_cells[i] = cells_i
                self.robot_cp[i] = list(cells_i)  # initial CP order

                self.prev_first_task_key[i] = None
                print(f"[RACER] Robot {i} initial L_act: {len(self.robot_cells[i])} cells")

            self._prev_fused_occ = occ.copy()
            self.initialized = True

        # -------------------------------------------------
        # Coordination cycle (your protocol): fuse maps, rebuild hgrid, then pairwise CVRP
        # -------------------------------------------------
        if step_count % self.T_coord == 0:

            # -------------------------------------------------
            # 1) Fuse maps
            # -------------------------------------------------
            occ = robots[0].local_map.copy()
            for robot in robots[1:]:
                occ = np.maximum(occ, robot.local_map)

            for robot in robots:
                robot.local_map = occ.copy()

            # -------------------------------------------------
            # 2) Compute updated AABB (Alg. 1)
            # -------------------------------------------------
            if self._prev_fused_occ is None:
                H, W = occ.shape
                updated_aabbs = [(0, 0, W, H)]
            else:
                diff = (occ != self._prev_fused_occ)
                if np.any(diff):
                    ys, xs = np.nonzero(diff)
                    min_y, max_y = int(ys.min()), int(ys.max()) + 1
                    min_x, max_x = int(xs.min()), int(xs.max()) + 1
                    updated_aabbs = [(min_x, min_y, max_x, max_y)]
                else:
                    updated_aabbs = []

            # -------------------------------------------------
            # 3) Update each robot's L_act
            # -------------------------------------------------
            for i in range(num_robots):
                lact_i = self.robot_cells.get(i, [])
                self.robot_cells[i] = self.hgrid_builder.update_lact(
                    occ=occ,
                    lact=lact_i,
                    updated_aabbs=updated_aabbs,
                    robot_pos_xy=robots[i].position,
                    treat_unknown_as_obstacle=self.treat_unknown_as_obstacle,
                )
                if i not in self.prev_first_task_key:
                    self.prev_first_task_key[i] = None

            self._prev_fused_occ = occ.copy()

            # -------------------------------------------------
            # 4) Pair robots
            # -------------------------------------------------
            pairs = []
            for i in range(num_robots):
                for j in range(i + 1, num_robots):
                    d = np.linalg.norm(
                        np.array(robots[i].position) - np.array(robots[j].position)
                    )
                    pairs.append((d, i, j))
            pairs.sort(key=lambda x: x[0])

            busy: Set[int] = set()
            cp_changed = {i: False for i in range(num_robots)}

            # -------------------------------------------------
            # 5) Pairwise CVRP
            # -------------------------------------------------
            for _, i, j in pairs:
                if i in busy or j in busy:
                    continue

                pair_dict = {
                    self._cell_key(c): c
                    for c in (self.robot_cells[i] + self.robot_cells[j])
                }
                pair_cells = list(pair_dict.values())
                if not pair_cells:
                    continue

                graph = SparseHGridGraph(
                    occ=occ,
                    cells=pair_cells,
                    k_nn=self.sparse_k_nn,
                    treat_unknown_as_obstacle=self.treat_unknown_as_obstacle,
                )

                ci, cj = solve_pairwise_cvrp(
                    occ,
                    depots=[robots[i].position, robots[j].position],
                    tasks=pair_cells,
                    alpha_capacity=self.alpha_capacity,
                    graph=graph,
                    prev_first_task_keys=(
                        self.prev_first_task_key[i],
                        self.prev_first_task_key[j],
                    ),
                    beta_con=self.beta_con,
                    treat_unknown_as_obstacle=self.treat_unknown_as_obstacle,
                )

                ci = self._dedup_cells_preserve_order(ci)
                cj = self._dedup_cells_preserve_order(cj)

                old_cp_i = self.robot_cp.get(i, [])
                old_cp_j = self.robot_cp.get(j, [])

                changed_i = not self._cp_equal(ci, old_cp_i)
                changed_j = not self._cp_equal(cj, old_cp_j)

                if changed_i:
                    self.robot_cells[i] = ci.copy()
                    self.robot_cp[i] = ci.copy()
                    self.prev_first_task_key[i] = (
                        self._cell_key(ci[0]) if ci else None
                    )
                    cp_changed[i] = True

                if changed_j:
                    self.robot_cells[j] = cj.copy()
                    self.robot_cp[j] = cj.copy()
                    self.prev_first_task_key[j] = (
                        self._cell_key(cj[0]) if cj else None
                    )
                    cp_changed[j] = True

                busy.update({i, j})

            # -------------------------------------------------
            # 6) Replan ONLY for robots whose CP changed
            # -------------------------------------------------
            waypoints: Dict[int, List[Tuple[int, int]]] = {}
            frontiers = extract_frontier_clusters(
                occ,
                unknown_value=-1,
                free_value=0,
            )

            for ridx in range(num_robots):
                if not cp_changed[ridx]:
                    continue   # <<< keep current plan

                cells_cp = self.robot_cp.get(ridx, [])
                wps = self._cp_guided_plan_waypoints(
                    occ=occ,
                    robot_pos_xy=robots[ridx].position,
                    robot_yaw=robots[ridx].orientation,
                    cells_cp=cells_cp,
                    all_frontiers=frontiers,
                )

                waypoints[ridx] = list(wps)

            return waypoints


        # -------------------------------------------------
        # Non-coordination step:
        # keep existing CPs, select next waypoint only if robot finished its list
        # -------------------------------------------------
        waypoints: Dict[int, List[Tuple[int, int]]] = {}

        for ridx in range(num_robots):

            # -------------------------------------------------
            # 1) If robot is still executing current waypoints,
            #    do not overwrite.
            # -------------------------------------------------
            if getattr(robots[ridx], "current_waypoint_idx", 0) < len(getattr(robots[ridx], "waypoints", [])):
                continue

            occ = robots[ridx].local_map
            frontiers = extract_frontier_clusters(
                occ,
                unknown_value=-1,
                free_value=0,
            )

            cells_cp = self.robot_cp.get(ridx, [])

            # -------------------------------------------------
            # 2) Normal CP-guided replanning
            # -------------------------------------------------
            cands = self._cp_guided_candidates(frontiers, cells_cp)

            if cands:
                # Choose nearest frontier inside CP (greedy, no TSP)
                robot_xy = robots[ridx].position
                nearest = min(
                    cands,
                    key=lambda f: np.linalg.norm(
                        np.array(f["rep"]) - np.array(robot_xy)
                    )
                )
                waypoints[ridx] = [nearest["rep"]]
        return waypoints
