"""
maxcal_coverage.py

Coverage Layer 1-C validation controller.

This module implements coverage-only controller on the 20 x 20 region graph used throughout the project.
The paper-level MaxCal kernel is the destination-weighted transition rule

    P_ij = w_ij exp(-lambda_C^j) / Z_i,
    Z_i = sum_{l in N(i)} w_il exp(-lambda_C^l),

with unit edge weights w_ij = 1 on the undirected 8-connected grid. For equal coverage multipliers, 
and in particular for the baseline choice lambda_C^l = 0 for all l, the kernel reduces to the uniform 
random walk over neighbors. By detailed balance the stationary distribution then satisfies

    π_i proportional to deg(i)
    
where deg(i) is the degree of cell i in the region graph (the number of neighboring cells). 
This is the forward Layer 1-C prediction validated in ``maxcal_coverage_validation.py``.

The same reversible form also supports the inverse coverage problem, where we are trying to compute the optimal multipliers given a target stationary distribution.  
Writing b_i = exp(-lambda_C^i), the stationary distribution of the kernel is

    π_i(b) proportional to b_i (A b)_i, where (A b)_i = sum_j A_ij b_j is the neighbor-weighted sum at i,

where A is the adjacency matrix.  The inverse solver uses this relation to recover lambda_C from a prescribed stationary target.  
This is the offline "god-knowledge" design step; the robots themselves still execute only the local transition rule by looking up neighbor multipliers.

Each robot also carries a local cell-wise map storing the most recent known visit time for every cell. 
That map is not part of the Layer 1-C transition kernel itself. It is recorded here so the same controller exposes the
coverage-age observable needed later by the hierarchical architecture.

SETUP:
    pip install numpy matplotlib pillow

RUN:
    python maxcal_coverage.py
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from maxcal_local_maps import (
    RobotWorldMap,
    mean_robot_coverage_age,
    mean_robot_map_record_age,
)


# ============================================================
# PARAMETERS 
# ============================================================

NX = 20                  # grid columns
NY = 20                  # grid rows
CELL_SIZE = 1.0          # metres per cell → world is 20 × 20 m
N_ROBOTS = 50            # number of robots in the swarm   
ROBOT_SPEED  = 0.15      # m / simulation step
T_SIM = 12_000           # total simulation steps
RECORD_EVERY = 60        # steps between c_k(t) snapshots
SNAP_EVERY = 300         # steps between position snapshots
LAMBDA_C_VAL = 0.0       # symmetric multiplier (see §1 note)
SEED = 42
INVERSE_SOLVER_TOL = 1.0e-12
INVERSE_SOLVER_MAX_ITERS = 100_000

# NOTE on LAMBDA_C_VAL = 0:
#   Theory Eq.(1): p*(k2|k1) = w_{k2,k1}·exp(−λ_C^{k2}) / Z(k1)
#   With equal multipliers ∀k, exp(−λ_C) cancels in the ratio,
#   giving a uniform random walk: p*(k2|k1) = 1 / |N(k1)|.
#   The stationary distribution is then π_k ∝ deg(k).
#   The full formula is kept so non-uniform λ_C^k can be computed.


# ============================================================
# 1.  WORLD MODEL
# ============================================================

@dataclass
class World:
    Nx: int
    Ny: int
    K: int
    cell_size: float
    adjacency: List[List[int]]   # 0-indexed; 8-connected


def build_world(Nx: int, Ny: int, cell_size: float) -> World:
    """Build the undirected region graph used."""
    K = Nx * Ny
    adj: List[List[int]] = [[] for _ in range(K)]

    for k in range(K):
        row, col = divmod(k, Nx)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < Ny and 0 <= nc < Nx):
                    continue
                adj[k].append(nr * Nx + nc)

    return World(Nx, Ny, K, cell_size, adj)


def region_center(w: World, k: int) -> Tuple[float, float]:
    """Return the continuous-space centre of region ``k``."""
    row, col = divmod(k, w.Nx)
    x = (col + 0.5) * w.cell_size
    y = (row + 0.5) * w.cell_size
    return x, y


# ============================================================
# 2. MAXCAL TRANSITION PROBABILITIES
# ============================================================
#
# Theory, Eq.(1):
#   p*(k2 | k1) = w_{k2,k1} · exp(−λ_C^{k2}) / Z(k1)
#   Z(k1) = Σ_{k2 ∈ N(k1)} w_{k2,k1} · exp(−λ_C^{k2})
#
# Here w = 1 for all edges (isotropic cost, absorbed into adjacency).
# We pre-compute the full K×K transition matrix so the inner loop is a table look-up.

def build_transition_matrix(w: World, lambda_C: np.ndarray) -> np.ndarray:
    """
    Assemble the transition matrix for the MaxCal kernel from Eq. (1).

        P_ij = A_ij exp(-lambda_C^j) / sum_{l in N(i)} A_il exp(-lambda_C^l).

    Because the graph is unweighted and undirected in this implementation,
    ``A_ij`` is simply the adjacency indicator on the 8-connected grid.
    """
    lambda_C = np.asarray(lambda_C, dtype=np.float64)
    if lambda_C.shape != (w.K,):
        raise ValueError(f"lambda_C must have shape ({w.K},), got {lambda_C.shape}.")
    P = np.zeros((w.K, w.K), dtype=np.float64)
    for k1 in range(w.K):
        neighbors = w.adjacency[k1]
        scores = np.exp(-lambda_C[neighbors])
        Z = scores.sum()
        for i, k2 in enumerate(neighbors):
            P[k1, k2] = scores[i] / Z
    return P


def sample_next_region(w: World, k: int, P: np.ndarray, rng: np.random.Generator) -> int:
    """Sample one Markov transition from the precomputed transition matrix."""
    nb = w.adjacency[k]
    probs = P[k, nb]
    r = rng.random()
    s = 0.0
    for i, p in enumerate(probs):
        s += p
        if s >= r:
            return nb[i]
    return nb[-1]   # fallback for floating-point edge case


@dataclass
class InverseCoverageSolution:
    """
    Result of the inverse Layer 1-C design solve.

    ``lambda_C`` is the multiplier field recovered from the target stationary
    distribution, while the remaining fields document numerical accuracy of the
    solver used to achieve the inverse relation.
    """
    lambda_C: np.ndarray
    stationary: np.ndarray
    iterations: int
    max_abs_stationary_error: float
    l1_stationary_error: float
    converged: bool
    error_history: np.ndarray


@dataclass
class CoverageController:
    """
    Full Layer 1-C coverage controller.

    This keeps together the multiplier field, the precomputed transition
    matrix, the stationary target that motivated those multipliers, and the
    optional inverse-solver diagnostics when the controller was obtained by
    target inversion rather than by a forward baseline choice.
    """
    lambda_C: np.ndarray
    transition: np.ndarray
    target_pi: np.ndarray
    target_mode: str # "degree", "uniform", "custom", or "explicit_lambda".
    # "degree": forward baseline with equal multipliers, stationary distribution proportional to degree.
    # "uniform": uniform stationary distribution.
    # "custom": custom stationary distribution.
    # "explicit_lambda": explicit lambda values.

    # If target_mode is "custom", this field contains the result of solving the inverse MaxCal problem for the specified target_pi. Otherwise, it is None.
    inverse_solution: InverseCoverageSolution | None = None 


def adjacency_matrix(w: World) -> np.ndarray:
    """Return the binary adjacency matrix A of the region graph."""
    adjacency = np.zeros((w.K, w.K), dtype=np.float64)
    for k, neighbors in enumerate(w.adjacency):
        adjacency[k, neighbors] = 1.0
    return adjacency


def uniform_stationary(w: World) -> np.ndarray:
    """Uniform target pi_k = 1 / K used by the inverse coverage validation."""
    return np.full(w.K, 1.0 / w.K, dtype=np.float64)


def stationary_from_weights(w: World, b: np.ndarray) -> np.ndarray:
    """
    Stationary distribution of the coverage kernel written in b_i = exp(-lambda_i).

    For an undirected unit-weight graph,

        P_ij = A_ij b_j / sum_l A_il b_l

    is reversible with stationary distribution

        π_i ∝ b_i sum_j A_ij b_j.

    This is the key inverse-MaxCal relation used to solve lambda_C from a
    desired stationary target.
    """
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (w.K,):
        raise ValueError(f"b must have shape ({w.K},), got {b.shape}.")
    neighbor_mass = np.array([float(np.sum(b[w.adjacency[k]])) for k in range(w.K)], dtype=np.float64)
    raw = b * neighbor_mass
    return raw / raw.sum()


def stationary_from_lambda(w: World, lambda_C: np.ndarray) -> np.ndarray:
    """
    Evaluate the reversible stationary distribution induced by ``lambda_C``.

    The mean shift is removed first because adding the same constant to every
    multiplier leaves all transition probabilities unchanged.
    """
    lambda_C = np.asarray(lambda_C, dtype=np.float64)
    shifted = lambda_C - float(np.mean(lambda_C))
    return stationary_from_weights(w, np.exp(-shifted))


def power_stationary_distribution(
    transition: np.ndarray,
    tol: float = 1.0e-14,
    max_iters: int = 200_000,
) -> np.ndarray:
    """Generic stationary solver used only as a numerical cross-check."""
    pi = np.full(transition.shape[0], 1.0 / transition.shape[0], dtype=np.float64)
    for _ in range(max_iters):
        nxt = pi @ transition
        if float(np.max(np.abs(nxt - pi))) < tol:
            return nxt / nxt.sum()
        pi = nxt
    return pi / pi.sum()


def solve_coverage_multipliers_for_target(
    w: World,
    target_pi: np.ndarray,
    tol: float = INVERSE_SOLVER_TOL,
    max_iters: int = INVERSE_SOLVER_MAX_ITERS,
) -> InverseCoverageSolution:
    """
    Solve the inverse coverage MaxCal problem.

    Input:
        target_pi[k] = desired stationary visit frequency for cell k.

    Output:
        lambda_C[k] values that make the MaxCal kernel's stationary
        distribution match target_pi, up to numerical tolerance.

    The gauge is fixed by enforcing mean(lambda_C) = 0, because adding the same
    constant to every multiplier does not change transition probabilities.

    Numerically, this is solved by symmetric proportional fitting on the
    appendix relation

        π_i(b) proportional to b_i (A b)_i,   b_i = exp(-lambda_C^i),

    rather than by a single closed-form expression for lambda_C. The iteration
    updates underrepresented cells upward in ``b_i`` and overrepresented cells
    downward until the stationary distribution matches the requested target.
    """
    target = np.asarray(target_pi, dtype=np.float64)
    if target.shape != (w.K,):
        raise ValueError(f"target_pi must have shape ({w.K},), got {target.shape}.")
    if np.any(target <= 0.0):
        raise ValueError("The inverse MaxCal solver requires a strictly positive target distribution.")
    target = target / target.sum()

    b = np.ones(w.K, dtype=np.float64)
    error_history: List[float] = []
    converged = False
    iteration = 0

    for iteration in range(max_iters + 1):
        stationary = stationary_from_weights(w, b)
        err = float(np.max(np.abs(stationary - target)))
        error_history.append(err)
        if err < tol:
            converged = True
            break

        # Symmetric proportional fitting for π_i(b) proportional to b_i (A b)_i.
        # Underrepresented cells get larger b_i = exp(-lambda_i), therefore
        # lower lambda_i and higher incoming transition probability.
        b *= np.sqrt(target / np.maximum(stationary, 1.0e-300))
        b /= np.exp(float(np.mean(np.log(b))))

    lambda_C = -np.log(b)
    lambda_C -= float(np.mean(lambda_C))
    stationary = stationary_from_lambda(w, lambda_C)
    return InverseCoverageSolution(
        lambda_C=lambda_C,
        stationary=stationary,
        iterations=int(iteration),
        max_abs_stationary_error=float(np.max(np.abs(stationary - target))),
        l1_stationary_error=float(np.sum(np.abs(stationary - target))),
        converged=bool(converged),
        error_history=np.array(error_history, dtype=np.float64),
    )


def build_coverage_controller(
    w: World,
    target_mode: str = "degree",
    lambda_C_val: float = LAMBDA_C_VAL,
    target_pi: np.ndarray | None = None,
    lambda_C: np.ndarray | None = None,
    tol: float = INVERSE_SOLVER_TOL,
    max_iters: int = INVERSE_SOLVER_MAX_ITERS,
) -> CoverageController:
    """
    Build the MaxCal coverage controller.

    Modes:
        degree
            Forward baseline: equal multipliers, stationary pi proportional to
            cell degree by detailed balance.

        uniform
            Inverse-MaxCal controller: target π_k = 1/K and solve lambda_C.

        custom
            Inverse-MaxCal controller for a user-supplied target_pi.

        explicit_lambda
            Use a supplied lambda_C directly; target_pi is used only as the
            reference distribution for diagnostics.
    """
    if lambda_C is not None:
        lambda_arr = np.asarray(lambda_C, dtype=np.float64)
        if lambda_arr.shape != (w.K,):
            raise ValueError(f"lambda_C must have shape ({w.K},), got {lambda_arr.shape}.")
        reference = stationary_from_lambda(w, lambda_arr) if target_pi is None else np.asarray(target_pi, dtype=np.float64)
        reference = reference / reference.sum()
        return CoverageController(
            lambda_C=lambda_arr.copy(),
            transition=build_transition_matrix(w, lambda_arr),
            target_pi=reference,
            target_mode="explicit_lambda" if target_mode == "degree" else target_mode,
            inverse_solution=None,
        )

    if target_pi is not None or target_mode == "custom":
        if target_pi is None:
            raise ValueError("target_pi is required when target_mode='custom'.")
        solution = solve_coverage_multipliers_for_target(w, target_pi, tol=tol, max_iters=max_iters)
        return CoverageController(
            lambda_C=solution.lambda_C,
            transition=build_transition_matrix(w, solution.lambda_C),
            target_pi=np.asarray(target_pi, dtype=np.float64) / np.sum(target_pi),
            target_mode="custom",
            inverse_solution=solution,
        )

    if target_mode == "uniform":
        target = uniform_stationary(w)
        solution = solve_coverage_multipliers_for_target(w, target, tol=tol, max_iters=max_iters)
        return CoverageController(
            lambda_C=solution.lambda_C,
            transition=build_transition_matrix(w, solution.lambda_C),
            target_pi=target,
            target_mode="uniform",
            inverse_solution=solution,
        )

    if target_mode == "degree":
        lambda_arr = np.full(w.K, lambda_C_val, dtype=np.float64)
        return CoverageController(
            lambda_C=lambda_arr,
            transition=build_transition_matrix(w, lambda_arr),
            target_pi=stationary_from_lambda(w, lambda_arr),
            target_mode="degree",
            inverse_solution=None,
        )

    raise ValueError("target_mode must be one of: 'degree', 'uniform', 'custom', 'explicit_lambda'.")


# ============================================================
# 3. THEORETICAL STATIONARY DISTRIBUTION
# ============================================================
#
# For a reversible MC with unit edge weights and equal λ_C^k,
# detailed balance π_k1 · P[k1,k2] = π_k2 · P[k2,k1] gives:
#
#   π_k1 / deg(k1) = π_k2 / deg(k2)  ⟹  π_k ∝ deg(k)
#
# This is the "ground truth" we compare the simulation against.
#
# Degree census for a 20×20 8-connected grid:
#   Corner cells (4 total):        degree 3
#   Non-corner edge cells (72):    degree 5
#   Interior cells (324):          degree 8

def theoretical_stationary(w: World) -> np.ndarray:
    return stationary_from_lambda(w, np.zeros(w.K, dtype=np.float64))


# ============================================================
# 4. ROBOT
# ============================================================
# Fields
#   from_k  — the Markov-chain state: last region the robot arrived at.
#             This is the variable s_t in the theory.
#   to_k    — target region (already sampled from p*(· | from_k)).
#   (tx,ty) — continuous-space centre of to_k.
#
# DESIGN CHOICE — "sample on arrival":
#   The Markov transition p*(k2|k1) fires ONLY when the robot
#   physically reaches the centre of to_k. Between transitions
#   the Markov state is frozen. Continuous motion is not part
#   of the Markov chain; it is only its physical realisation.
#
# DESIGN CHOICE — capped movement:
#   At each step the robot moves min(speed, dist) toward the target.
#   This prevents the oscillation that arises in discrete-time
#   steering when speed > 2 · tolerance: if we move a fixed step
#   and overshoot, the robot bounces forever just outside the
#   tolerance radius. Capping guarantees arrival in ⌈dist/speed⌉
#   steps exactly, making transit time a deterministic function
#   of geometry — as the theory implicitly assumes.

@dataclass
class Robot:
    """
    Continuous robot state used to physically realize the discrete Markov chain.

    ``from_k`` is the current Markov state. ``to_k`` is the next sampled region.
    The continuous coordinates ``(x, y)`` and target centre ``(tx, ty)`` are
    the motion-layer realization of that discrete state sequence.
    """
    id: int
    x: float
    y: float
    from_k: int     # Markov state
    to_k: int       # next target region
    tx: float       # target centre x
    ty: float       # target centre y
    world_map: RobotWorldMap | None = None


def ensure_robot_map(r: Robot, w: World, t: float = 0.0) -> RobotWorldMap:
    """Allocate the robot's local coverage-age map on first use."""
    if r.world_map is None:
        r.world_map = RobotWorldMap.empty(w.K)
        r.world_map.observe_cell(r.from_k, t)
    return r.world_map


def make_robot(idx: int, w: World, P: np.ndarray, rng: np.random.Generator) -> Robot:
    """Spawn one robot at a random region and sample its first target region."""
    k0 = int(rng.integers(0, w.K))
    cx, cy = region_center(w, k0)
    k1 = sample_next_region(w, k0, P, rng)
    tx, ty = region_center(w, k1)
    world_map = RobotWorldMap.empty(w.K)
    world_map.observe_cell(k0, 0.0)
    return Robot(idx, cx, cy, k0, k1, tx, ty, world_map)


def step_robot(r: Robot, w: World, P: np.ndarray,
               speed: float, markov_visits: np.ndarray,
               rng: np.random.Generator,
               t: int = 0,
               global_last_visit_time: np.ndarray | None = None) -> None:
    """
    Advance one robot by one simulation step.

    The key paper-consistent design choice is that the discrete Markov update is
    triggered only when the robot physically arrives at the centre of its target
    region. Between arrivals the state ``from_k`` is frozen, so convergence is
    measured in Markov time, i.e. number of arrivals, rather than raw simulation
    steps.
    """
    robot_map = ensure_robot_map(r, w)
    dx = r.tx - r.x
    dy = r.ty - r.y
    dist = math.sqrt(dx * dx + dy * dy)

    # ── IN TRANSIT: capped movement ──────────────────────────
    move = min(speed, dist)
    # Move toward target; never overshoot.
    if dist > 1e-10:
        r.x += move * dx / dist
        r.y += move * dy / dist

    # ── ARRIVAL TEST ─────────────────────────────────────────
    # Position matches target to floating-point precision.
    if abs(r.x - r.tx) < 1e-9 and abs(r.y - r.ty) < 1e-9:
        
        # Update Markov state
        r.from_k = r.to_k

        # Record arrival — this is the "visit" counted in c_k(t):
        #   c_k(t) = (# arrivals at k up to Markov-time t) / (total arrivals)
        markov_visits[r.from_k] += 1
        robot_map.observe_cell(r.from_k, float(t))
        if global_last_visit_time is not None:
            global_last_visit_time[r.from_k] = float(t)

        # ── MAXCAL TRANSITION: sample p*(k2 | from_k), Eq.(1) ───
        next_k = sample_next_region(w, r.from_k, P, rng)
        r.to_k = next_k
        r.tx, r.ty = region_center(w, next_k)


# ============================================================
# 5.  MAIN SIMULATION
# ============================================================

@dataclass
class SimResult:
    """Outputs recorded from one Layer 1-C run for theory checks and figures."""
    w: World
    pi_empirical: np.ndarray
    ck_history: List[np.ndarray]                              # c_k(t) snapshots
    markov_step_history: List[int]                            # total arrivals at each snapshot
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]]   # (xs, ys, t)
    lambda_C: np.ndarray | None = None
    target_pi: np.ndarray | None = None
    target_mode: str = "degree"
    inverse_solution: InverseCoverageSolution | None = None
    t_axis: np.ndarray | None = None
    mean_global_coverage_age: np.ndarray | None = None
    mean_local_coverage_age: np.ndarray | None = None
    mean_local_map_record_age: np.ndarray | None = None


def run_simulation(speed: float = ROBOT_SPEED,
                   T: int = T_SIM,
                   lambda_C_val: float = LAMBDA_C_VAL,
                   seed: int = SEED,
                   target_mode: str = "degree",
                   target_pi: np.ndarray | None = None,
                   lambda_C: np.ndarray | None = None,
                   inverse_tol: float = INVERSE_SOLVER_TOL,
                   inverse_max_iters: int = INVERSE_SOLVER_MAX_ITERS) -> SimResult:
    """
    Run the Layer 1-C controller and record both Markov and age observables.

    ``markov_visits`` stores the arrival counts c_k(t) used by the paper-level
    coverage validation. ``global_last_visit_time`` stores the globally observed
    last visit of each cell and is used only to compute the global coverage-age
    diagnostic. The per-robot local maps evolve independently and expose the
    local coverage-age observable needed by the later hierarchical controller.
    """
    rng = np.random.default_rng(seed)
    w = build_world(NX, NY, CELL_SIZE)
    controller = build_coverage_controller(
        w=w,
        target_mode=target_mode,
        lambda_C_val=lambda_C_val,
        target_pi=target_pi,
        lambda_C=lambda_C,
        tol=inverse_tol,
        max_iters=inverse_max_iters,
    )
    P = controller.transition

    robots = [make_robot(i, w, P, rng) for i in range(N_ROBOTS)]
    markov_visits = np.zeros(w.K, dtype=np.int64)
    global_last_visit_time = np.full(w.K, -1.0, dtype=np.float64)
    for robot in robots:
        global_last_visit_time[robot.from_k] = 0.0

    ck_history: List[np.ndarray] = []
    ms_history: List[int] = []
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]] = []
    t_axis: List[int] = []
    global_cov_age_history: List[float] = []
    local_cov_age_history: List[float] = []
    local_map_record_age_history: List[float] = []

    for t in range(1, T + 1):
        for r in robots:
            step_robot(
                r,
                w,
                P,
                speed,
                markov_visits,
                rng,
                t=t,
                global_last_visit_time=global_last_visit_time,
            )

        total = int(markov_visits.sum())
        if t % RECORD_EVERY == 0 and total > 0:
            ck_history.append(markov_visits / total)
            ms_history.append(total)
            maps = [ensure_robot_map(robot, w) for robot in robots]
            global_cov_age = np.where(
                global_last_visit_time >= 0.0,
                float(t) - global_last_visit_time,
                float(t) + 1.0,
            )
            t_axis.append(t)
            global_cov_age_history.append(float(np.mean(global_cov_age)))
            local_cov_age_history.append(mean_robot_coverage_age(maps, float(t)))
            local_map_record_age_history.append(mean_robot_map_record_age(maps, float(t)))

        if t % SNAP_EVERY == 0:
            xs = np.array([r.x for r in robots])
            ys = np.array([r.y for r in robots])
            pos_snapshots.append((xs, ys, t))

    pi_emp = markov_visits / markov_visits.sum()
    return SimResult(
        w,
        pi_emp,
        ck_history,
        ms_history,
        pos_snapshots,
        controller.lambda_C,
        controller.target_pi,
        controller.target_mode,
        controller.inverse_solution,
        np.array(t_axis, dtype=np.int64),
        np.array(global_cov_age_history, dtype=np.float64),
        np.array(local_cov_age_history, dtype=np.float64),
        np.array(local_map_record_age_history, dtype=np.float64),
    )


# ============================================================
# 6. VISUALISATION
# ============================================================

def make_main_figure(res: SimResult):
    w = res.w
    pi_theory = res.target_pi if res.target_pi is not None else theoretical_stationary(w)
    pi_emp = res.pi_empirical

    # Representative regions (one of each degree class)
    k_corner = 0                                  # (0,0) → degree 3
    k_edge = NX // 2                              # bottom-edge middle → degree 5
    k_interior = (NY // 2) * NX + NX // 2        # centre → degree 8

    vmin = min(float(pi_theory.min()), float(pi_emp.min()))
    vmax = max(float(pi_theory.max()), float(pi_emp.max()))
    if math.isclose(vmin, vmax):
        pad = max(abs(vmin) * 0.05, 1.0e-6)
        vmin -= pad
        vmax += pad

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pi_theory.reshape(w.Ny, w.Nx), origin="lower",
                     cmap="viridis", vmin=vmin, vmax=vmax)
    if res.target_mode == "uniform":
        ax1.set_title("(a) Target π̄_k = 1/K")
    elif res.target_mode == "degree":
        ax1.set_title("(a) Target π̄_k ∝ deg(k)")
    else:
        ax1.set_title(f"(a) Target π̄_k ({res.target_mode})")
    ax1.set_xlabel("col"); ax1.set_ylabel("row")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pi_emp.reshape(w.Ny, w.Nx), origin="lower",
                     cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title("(b) Empirical π̂_k")
    ax2.set_xlabel("col"); ax2.set_ylabel("row")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    err = np.abs(pi_emp - pi_theory)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(err.reshape(w.Ny, w.Nx), origin="lower", cmap="Reds")
    ax3.set_title("(c) |π̂_k − π̄_k|")
    ax3.set_xlabel("col"); ax3.set_ylabel("row")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    # Coverage convergence: theory's Eq.(3) predicts rate ∝ 1/(t+1)
    ax4 = fig.add_subplot(gs[1, 0:2])
    ms = res.markov_step_history
    for k, label, col in [
        (k_corner,   "Corner   (deg 3)", "red"),
        (k_edge,     "Edge     (deg 5)", "darkorange"),
        (k_interior, "Interior (deg 8)", "seagreen"),
    ]:
        ck = [snap[k] for snap in res.ck_history]
        ax4.plot(ms, ck, color=col, lw=2, label=label)
        ax4.axhline(pi_theory[k], color=col, ls="--", lw=1.2)
    ax4.set_title("(d) Coverage convergence (dashes = theory)")
    ax4.set_xlabel("Markov steps (arrivals)")
    ax4.set_ylabel("c_k(t)")
    ax4.legend(loc="right")

    ax5 = fig.add_subplot(gs[1, 2])
    xs, ys, t_end = res.pos_snapshots[-1]
    ax5.scatter(xs, ys, s=12, alpha=0.7, color="dodgerblue")
    ax5.set_xlim(0, w.Nx * w.cell_size)
    ax5.set_ylim(0, w.Ny * w.cell_size)
    ax5.set_aspect("equal")
    ax5.set_title(f"(e) Robot positions  t={t_end}")
    ax5.set_xlabel("x (m)"); ax5.set_ylabel("y (m)")

    if res.inverse_solution is None:
        title_suffix = "forward λ_C"
    else:
        title_suffix = (
            "inverse MaxCal "
            f"(iters={res.inverse_solution.iterations}, "
            f"L1={res.inverse_solution.l1_stationary_error:.2e})"
        )
    fig.suptitle(f"MaxCal Coverage — {res.target_mode} target, {title_suffix}")
    fig.tight_layout()
    return fig


def make_phase_figure(T: int = T_SIM):
    """
    Phase diagram: convergence speed vs robot speed.

    Left plot:  Markov time → curves should be speed-independent.
    Right plot: simulation time → curves spread out by speed.
    """
    # Phase diagram: convergence speed vs robot speed.
    #
    # KEY INSIGHT: the theory's convergence rate 1/(t+1) (Eq.3) is
    # in MARKOV time (number of arrivals).  In SIMULATION time the
    # rate scales with robot speed: faster robots = more Markov steps
    # per simulation step = faster apparent convergence.
    # Left plot: Markov time → curves should be speed-independent.
    # Right plot: simulation time → curves spread out by speed.


    w = build_world(NX, NY, CELL_SIZE)
    pi_theory = theoretical_stationary(w)
    k_int = (NY // 2) * NX + NX // 2
    speeds = [0.05, 0.10, 0.20, 0.50, 1.00]

    fig, (ax_m, ax_s) = plt.subplots(1, 2, figsize=(12, 5.2))

    for spd in speeds:
        res = run_simulation(speed=spd, T=T)
        sim_ts = np.arange(1, len(res.ck_history) + 1) * RECORD_EVERY
        errs = np.array([
            max(abs(snap[k_int] - pi_theory[k_int]), 1e-8)
            for snap in res.ck_history
        ])
        ax_m.plot(res.markov_step_history, errs, lw=2, label=f"speed={spd}")
        ax_s.plot(sim_ts, errs, lw=2, label=f"speed={spd}")

    ax_m.set_yscale("log")
    ax_m.set_title("Convergence in Markov time\n(speed-independent prediction)")
    ax_m.set_xlabel("Total Markov steps")
    ax_m.set_ylabel("|c_k(t) − π̄_k|")
    ax_m.legend(loc="upper right")

    ax_s.set_yscale("log")
    ax_s.set_title("Convergence in simulation time\n(faster robot → faster per step)")
    ax_s.set_xlabel("Simulation steps")
    ax_s.set_ylabel("|c_k(t) − π̄_k|")
    ax_s.legend(loc="upper right")

    fig.tight_layout()
    return fig


def make_animation(res: SimResult, fps: int = 12, filename: str = "maxcal_coverage.gif"):
    w = res.w
    pi_theory = theoretical_stationary(w)
    bg = pi_theory.reshape(w.Ny, w.Nx)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bg, origin="lower", cmap="YlOrRd", alpha=0.5,
              extent=(0, w.Nx, 0, w.Ny))
    ax.set_aspect("equal")
    ax.set_xlim(0, w.Nx); ax.set_ylim(0, w.Ny)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    scat = ax.scatter([], [], s=12, alpha=0.8, color="navy")
    title = ax.set_title("")

    def update(frame):
        xs, ys, t = res.pos_snapshots[frame]
        scat.set_offsets(np.column_stack([xs / w.cell_size, ys / w.cell_size]))
        title.set_text(f"t = {t}")
        return scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(res.pos_snapshots), interval=1000 // fps, blit=False
    )

    try:
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Could not save animation ({e}); skipping.")
    plt.close(fig)


# ============================================================
# 7. ENTRY POINT
# ============================================================

def main():
    print("MaxCal Coverage Simulation")
    print(f"  World   : {NX}x{NY} grid, cell={CELL_SIZE} m")
    print(f"  Swarm   : {N_ROBOTS} robots, speed={ROBOT_SPEED} m/step")
    print(f"  Duration: {T_SIM} steps")
    print()

    print("Running baseline coverage simulation (lambda_C = 0)...")
    result = run_simulation(target_mode="degree")
    print(f"  Done. Total Markov steps: {result.markov_step_history[-1]}")
    print(f"  Expected per robot: ~{round(result.markov_step_history[-1] / N_ROBOTS)}")
    print()

    print("Saving main figure → maxcal_coverage_main.png")
    fig_main = make_main_figure(result)
    fig_main.savefig("maxcal_coverage_main.png", dpi=120)
    plt.close(fig_main)

    print("Running inverse-MaxCal uniform-coverage simulation...")
    uniform_result = run_simulation(target_mode="uniform")
    if uniform_result.inverse_solution is not None:
        sol = uniform_result.inverse_solution
        print(
            "  Solver: "
            f"converged={sol.converged}, iterations={sol.iterations}, "
            f"L1={sol.l1_stationary_error:.3e}"
        )

    print("Saving inverse-MaxCal figure → maxcal_coverage_uniform_main.png")
    fig_uniform = make_main_figure(uniform_result)
    fig_uniform.savefig("maxcal_coverage_uniform_main.png", dpi=120)
    plt.close(fig_uniform)

    print("Saving animation → maxcal_coverage.gif")
    make_animation(result)

    print("Saving phase diagram...")
    fig_phase = make_phase_figure()
    fig_phase.savefig("maxcal_coverage_phase.png", dpi=120)
    plt.close(fig_phase)
    print("  Saved maxcal_coverage_phase.png")

    print()
    print("Done. Open the .png files to inspect results.")


if __name__ == "__main__":
    main()