"""
maxcal_info_diffusion.py

MaxCal Coverage + Information Diffusion Controller

The target information spreading rate is converted into an encounter-rate requirement.
Using the mean-field logistic model,

    (1/A) * dn/dt = beta * p_enc * (n/A) * (1 - (n/A)),
    dn/dt = A * beta * p_enc * (n/A) * (1 - (n/A)),

where A is the swarm size, beta is the probability of information transmission
after an encounter, and p_enc is the probability that two randomly selected
robots occupy the same region. The validation summaries use dn/dt, the expected
number of newly informed robots per unit time. Since

    p_enc = sum_k _k^2,

the information observable can be written as Delta f_I(k1, k2) = π̄_k2.
The transition kernel is therefore

    p*(k2 | k1) ∝ w_{k2,k1} exp(-lambda_C^{k2}  -lambda_I  π̄_k_k2).

The circular dependency between p* and π̄ is solved here as a fixed point:
π̄ is the stationary distribution of the transition matrix built from
π̄ itself. Each robot also carries a scalar Age of Information, tau_i,
defined as the time since that robot last exchanged information with another
robot. Successful same-region communication resets tau_i for both robots. A
separate local cell-wise map records last_visit_time[k] and
last_map_record_time[k] so map freshness is available for integration.
No age gate is applied to the Layer 1-I kernel; age is tracked
as an observable for the later hierarchical controller.

The sign prediction follows directly:

    lambda_I < 0   -> high-pi_bar regions are rewarded -> clustering
    lambda_I = 0   -> pure coverage baseline
    lambda_I > 0   -> high-pi_bar regions are penalized -> dispersion

SETUP:
    pip install numpy matplotlib pillow

RUN:
    python maxcal_info_diffusion.py
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

from maxcal_local_maps import (
    RobotWorldMap,
    exchange_maps,
    mean_robot_coverage_age,
    mean_robot_map_record_age,
)


_FIXED_POINT_CACHE: dict[Tuple[int, int, float, float, int, int], Tuple[np.ndarray, np.ndarray, int, float]] = {}


# ============================================================
# PARAMETERS
# ============================================================

NX = 20
NY = 20
CELL_SIZE = 1.0
N_ROBOTS = 100
ROBOT_SPEED = 0.15
T_SIM = 18_000
RECORD_EVERY = 30
SNAP_EVERY = 300
SEED = 42

# Coverage multiplier.
LAMBDA_C_VAL = 0.0

# Signed information multiplier.
#   lambda_I < 0  -> cluster
#   lambda_I >= 0 -> coverage / anti-cluster
LAMBDA_I_VAL = -400.0       # π̄ is O(1/K), so visible Layer 1-I effects need larger multipliers.

# Information-diffusion parameters.
BETA_TRANSMISSION = 0.60
P_ENCOUNTER_GIVEN_COLOCATION = 1.0
INFORMATION_FIELD_MODE = "stationary_fixed_point"  # Paper Layer 1-I field π̄.
USE_AGE_GATE_FOR_CONTROL = False  # Deprecated compatibility flag; paper validation uses explicit AoI maps instead.
INFO_FIXED_POINT_TOL = 1.0e-11
INFO_FIXED_POINT_MAX_ITERS = 400
INFO_FIXED_POINT_DAMPING = 0.50
INFO_FIXED_POINT_PERTURBATION = 1.0e-3

# Optional Layer-2-style gate parameters. The gate is kept available for
# hierarchical experiments, but disabled by default for the paper-exact Layer 1-I
# validation above.
A_HALF = 1.0
R_MEET = 1.0  # Legacy metric radius considered for encounter.
# Sweep used by the standalone script.
LAMBDA_I_SWEEP = (-400.0, -200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0, 400.0)
SWEEP_STALE_AGE = 1.0e6


# ============================================================
# 1. WORLD MODEL
# ============================================================

@dataclass
class World:
    Nx: int
    Ny: int
    K: int
    cell_size: float
    adjacency: List[List[int]]
    centers: np.ndarray


def build_world(Nx: int, Ny: int, cell_size: float) -> World:
    """Build the undirected region graph and continuous cell centres."""
    K = Nx * Ny
    adjacency: List[List[int]] = [[] for _ in range(K)]
    for k in range(K):
        row, col = divmod(k, Nx)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < Ny and 0 <= nc < Nx):
                    continue
                adjacency[k].append(nr * Nx + nc)

    centers = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        row, col = divmod(k, Nx)
        centers[k, 0] = (col + 0.5) * cell_size
        centers[k, 1] = (row + 0.5) * cell_size

    return World(Nx, Ny, K, cell_size, adjacency, centers)


def region_center(w: World, k: int) -> Tuple[float, float]:
    """Return the continuous-space centre of region ``k``."""
    return float(w.centers[k, 0]), float(w.centers[k, 1])


def position_to_cell(w: World, x: float, y: float) -> int:
    """Map a continuous position back to the corresponding grid region."""
    col = min(max(int(x / w.cell_size), 0), w.Nx - 1)
    row = min(max(int(y / w.cell_size), 0), w.Ny - 1)
    return row * w.Nx + col


def theoretical_stationary(w: World) -> np.ndarray:
    """Coverage-only baseline with equal multipliers: π_i proportional to deg(i)."""
    degrees = np.array([len(w.adjacency[k]) for k in range(w.K)], dtype=np.float64)
    return degrees / degrees.sum()


def normalize_probability(values: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    """Normalize a positive vector onto the probability simplex."""
    arr = np.asarray(values, dtype=np.float64)
    arr = np.maximum(arr, floor)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Probability vector must have a finite positive sum.")
    return arr / total


def clustered_seed_distribution(
    w: World,
    anchor_cell: int | None = None,
    sigma_cells: float = 0.75,
    floor: float = 1.0e-6,
) -> np.ndarray:
    """
    Localized positive seed used to follow the clustered fixed-point branch.

    For negative lambda_I the Layer 1-I fixed point can have multiple solutions.
    A single anchored seed breaks the grid symmetry deterministically and picks
    one translated copy of the clustered branch.
    """
    if anchor_cell is None:
        anchor_cell = (w.Ny // 2) * w.Nx + (w.Nx // 2)
    anchor = np.asarray(w.centers[int(anchor_cell)], dtype=np.float64)
    scale = max(float(w.cell_size), 1.0e-12)
    dist2 = np.sum(((w.centers - anchor) / scale) ** 2, axis=1)
    width2 = max(float(sigma_cells) ** 2, 1.0e-12)
    weights = np.exp(-0.5 * dist2 / width2)
    weights[int(anchor_cell)] += 1.0
    weights += float(floor)
    return normalize_probability(weights)


# ============================================================
# 2. STATIC COVERAGE TERM AND DYNAMIC INFORMATION FIELD
# ============================================================

def build_lambda_C(w: World, lambda_C_val: float) -> np.ndarray:
    """Build a homogeneous coverage multiplier field for the Layer 1-I tests."""
    return np.full(w.K, lambda_C_val, dtype=np.float64)


def occupancy_distribution(w: World, robots: Sequence["Robot"]) -> np.ndarray:
    """Empirical occupancy measure occ_t(k) from the current robot positions."""
    occ = np.zeros(w.K, dtype=np.float64)
    if not robots:
        return occ
    for r in robots:
        occ[position_to_cell(w, r.x, r.y)] += 1.0
    return occ / float(len(robots))


def compute_information_field(w: World, robots: Sequence["Robot"]) -> np.ndarray:
    """
    Optional online-density surrogate field.

    This is not the π̄ fixed-point field. It is a local, density-smoothed diagnostic field kept 
    for comparison experiments through ``information_field_mode='online_density'``.

    The Layer 1-I controller used in validation instead sets the information observable equal to the self-consistent stationary field π̄ returned by ``solve_information_fixed_point``.
    """
    occ = occupancy_distribution(w, robots)
    field = np.zeros(w.K, dtype=np.float64)
    for k in range(w.K):
        local = [k] + w.adjacency[k]
        field[k] = float(occ[local].sum())
    max_val = float(field.max())
    if max_val > 0.0:
        field /= max_val
    return field


def encounter_proxy_from_occupancy(occ: np.ndarray) -> float:
    """Empirical same-cell probability computed from the instantaneous occupancy."""
    return float(np.sum(occ ** 2))


def encounter_probability_from_stationary(pi_bar: np.ndarray) -> float:
    """
    Paper Eq. (4) encounter proxy: probability that two independently sampled
    robots occupy the same region, p_enc = sum_k  π̄_k^2.
    """
    pi = np.asarray(pi_bar, dtype=np.float64)
    return float(np.sum(pi ** 2))


def logistic_information_rate(
    informed_fraction: float,
    p_enc: float,
    beta: float = BETA_TRANSMISSION,
    n_agents: int = N_ROBOTS,
) -> float:
    """
    Mean-field prediction from Pinciroli Eq. (4), expressed as dn/dt.

     dn/dt = A * beta * p_enc * (n/A) * (1 - (n/A)).
    """
    f = float(np.clip(informed_fraction, 0.0, 1.0))
    return float(n_agents * beta * p_enc * f * (1.0 - f))


# ============================================================
# 3. MAXCAL TRANSITION KERNEL
# ============================================================

def info_gate(age: float, a_half: float = A_HALF) -> float:
    """Legacy Layer-2-style age gate retained for compatibility experiments."""
    if math.isinf(age):
        return 1.0
    return age / (age + a_half)


def local_transition_probabilities(
    w: World,
    k1: int,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    info_age: float,
    use_age_gate: bool = USE_AGE_GATE_FOR_CONTROL,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Local Layer 1-I transition rule from the MaxCal kernel.

    With the information observable written as ``Delta f_I(j) = info_field[j]``,
    the destination scores are

        exp(-lambda_C^j - lambda_I Delta f_I(j)).

    When ``use_age_gate`` is enabled the information term is multiplied by the
    legacy gate ``g(tau_i)``. That gate is intentionally disabled in the Layer 1-I validation, 
    where AoI is recorded as an observable but does not directly modify the kernel yet.
    """
    neighbors = np.array(w.adjacency[k1], dtype=np.int64)
    gate = info_gate(info_age) if use_age_gate else 1.0
    effective_cost = lambda_C[neighbors] + gate * lambda_I_value * info_field[neighbors]
    shifted = effective_cost - effective_cost.min()
    scores = np.exp(-shifted)
    probs = scores / scores.sum()
    return neighbors, probs


def sample_next_region(
    w: World,
    k1: int,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    info_age: float,
    rng: np.random.Generator,
    use_age_gate: bool = USE_AGE_GATE_FOR_CONTROL,
) -> int:
    """Sample the next region according to the local Layer 1-I transition probabilities."""
    neighbors, probs = local_transition_probabilities(
        w=w,
        k1=k1,
        lambda_C=lambda_C,
        lambda_I_value=lambda_I_value,
        info_field=info_field,
        info_age=info_age,
        use_age_gate=use_age_gate,
    )
    r = rng.random()
    s = 0.0
    for i, p in enumerate(probs):
        s += float(p)
        if s >= r:
            return int(neighbors[i])
    return int(neighbors[-1])


def transition_matrix_from_information_field(
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    use_age_gate: bool = False,
) -> np.ndarray:
    """
    Assemble the full K x K Layer 1-I transition matrix.

    Setting ``info_field`` is the fixed-point stationary field π̄. 
    The resulting matrix is still reversible because the destination-dependent exponential
    weight preserves the same undirected-graph structure as the coverage kernel.
    """
    transition = np.zeros((w.K, w.K), dtype=np.float64)
    for k1 in range(w.K):
        neighbors, probs = local_transition_probabilities(
            w=w,
            k1=k1,
            lambda_C=lambda_C,
            lambda_I_value=lambda_I_value,
            info_field=info_field,
            info_age=math.inf,
            use_age_gate=use_age_gate,
        )
        transition[k1, neighbors] = probs
    return transition


def stationary_distribution_from_information_field(
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
) -> np.ndarray:
    """
    Exact stationary distribution for  Layer 1-I kernel.

    For symmetric unit edge weights and

        P_ij = A_ij b_j / sum_l A_il b_l,
        b_j = exp(-lambda_C_j - lambda_I I_j),

    detailed balance gives

        pi_i proportional to b_i sum_l A_il b_l.

    This is the same derivation used by the inverse coverage solver and avoids
    treating a reversible MaxCal kernel as a generic dense Markov chain.
    """
    costs = lambda_C + lambda_I_value * np.asarray(info_field, dtype=np.float64)
    shifted = costs - float(np.min(costs))
    b = np.exp(-shifted)
    neighbor_sum = np.zeros(w.K, dtype=np.float64)
    for k, neighbors in enumerate(w.adjacency):
        neighbor_sum[k] = float(np.sum(b[neighbors]))
    pi = b * neighbor_sum
    return pi / pi.sum()


def stationary_distribution_from_transition(
    transition: np.ndarray,
    tol: float = 1.0e-13,
    max_iters: int = 100_000,
) -> np.ndarray:
    """
    Generic stationary solve for a stochastic matrix.

    The Layer 1-I implementation normally uses the closed reversible form in
    ``stationary_distribution_from_information_field``. This routine is kept as
    a defensive numerical fallback and diagnostic cross-check.
    """
    del tol, max_iters
    n = transition.shape[0]
    system = transition.T - np.eye(n, dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    system[-1, :] = 1.0
    rhs[-1] = 1.0
    try:
        pi = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError:
        pi = np.full(n, 1.0 / n, dtype=np.float64)
        for _ in range(20_000):
            nxt = pi @ transition
            if float(np.max(np.abs(nxt - pi))) < 1.0e-13:
                pi = nxt
                break
            pi = nxt
    pi = np.maximum(np.real(pi), 0.0)
    return pi / pi.sum()


def solve_information_fixed_point(
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    seed: int = SEED,
    tol: float = INFO_FIXED_POINT_TOL,
    max_iters: int = INFO_FIXED_POINT_MAX_ITERS,
    damping: float = INFO_FIXED_POINT_DAMPING,
    perturbation: float = INFO_FIXED_POINT_PERTURBATION,
    initial_pi: np.ndarray | None = None,
    use_cache: bool | None = None,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """
    Solve Layer 1-I fixed point:

        π̄ = stationary_distribution(P(π̄)).

    Existence follows from the fixed-point map on the probability simplex, but
    the paper does not provide a closed-form solution for π̄. This
    implementation therefore uses a damped fixed-point iteration:

        pi^(m+1) = (1 - alpha) pi^(m) + alpha F(pi^(m)),

    where ``F`` is the stationary distribution induced by the current field.

    For ``lambda_I < 0`` the fixed point can be non-unique because multiple
    clustered branches may coexist with a coverage-like branch. The optional
    ``initial_pi`` argument, the default small perturbation applied for negative
    multipliers, and the continuation logic used by the inverse solver are
    numerical branch-selection devices only; they do not change the Layer 1-I
    equations themselves.
    """
    if use_cache is None:
        use_cache = initial_pi is None

    cache_key = (
        w.Nx,
        w.Ny,
        float(w.cell_size),
        float(lambda_I_value),
        hash(lambda_C.tobytes()),
        int(seed),
        float(tol),
        int(max_iters),
        float(damping),
        float(perturbation),
    )
    if use_cache and cache_key in _FIXED_POINT_CACHE:
        pi_cached, transition_cached, iteration_cached, err_cached = _FIXED_POINT_CACHE[cache_key]
        return pi_cached.copy(), transition_cached.copy(), iteration_cached, err_cached

    if initial_pi is not None:
        pi = normalize_probability(initial_pi)
    else:
        rng = np.random.default_rng(seed)
        pi = theoretical_stationary(w).copy()
        if lambda_I_value < 0.0 and perturbation > 0.0:
            pi = normalize_probability(pi + perturbation * rng.random(w.K))

    err = float("inf")
    transition = transition_matrix_from_information_field(w, lambda_C, lambda_I_value, pi)
    for iteration in range(1, max_iters + 1):
        transition = transition_matrix_from_information_field(w, lambda_C, lambda_I_value, pi)
        stationary = stationary_distribution_from_information_field(w, lambda_C, lambda_I_value, pi)
        updated = normalize_probability((1.0 - damping) * pi + damping * stationary)
        err = float(np.max(np.abs(updated - pi)))
        pi = updated
        if err < tol:
            break

    transition = transition_matrix_from_information_field(w, lambda_C, lambda_I_value, pi)
    pi = stationary_distribution_from_information_field(w, lambda_C, lambda_I_value, pi)
    if use_cache:
        _FIXED_POINT_CACHE[cache_key] = (pi.copy(), transition.copy(), iteration, err)
    return pi, transition, iteration, err


def evaluate_information_rate_constraint(
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    informed_fraction: float,
    beta: float = BETA_TRANSMISSION,
    n_agents: int = N_ROBOTS,
    seed: int = SEED,
    initial_pi: np.ndarray | None = None,
    return_pi: bool = False,
) -> dict[str, float | int]:
    """
    Evaluate scalar information-rate constraint for one lambda_I.

    The Layer 1-I inverse problem is not a spatial target-matching problem like
    coverage. The constrained quantity is the mean-field spreading rate

        dn/dt = A beta p_enc (n/A) (1 - (n/A)),

    with p_enc = sum_k  π̄_k^2 . The stationary field π̄ is
    obtained from the same fixed point used by the controller.
    """
    pi_bar, _, iterations, fixed_point_error = solve_information_fixed_point(
        w,
        lambda_C,
        lambda_I_value,
        seed=seed,
        initial_pi=initial_pi,
        use_cache=initial_pi is None,
    )
    p_enc = encounter_probability_from_stationary(pi_bar)
    rate = logistic_information_rate(
        informed_fraction=informed_fraction,
        p_enc=p_enc,
        beta=beta,
        n_agents=n_agents,
    )
    result: dict[str, float | int | np.ndarray] = {
        "lambda_I": float(lambda_I_value),
        "p_enc": float(p_enc),
        "information_spreading_rate": float(rate),
        "fixed_point_iterations": int(iterations),
        "fixed_point_error": float(fixed_point_error),
    }
    if return_pi:
        result["pi_bar"] = pi_bar.copy()
    return result


def continuation_lambda_grid(start: float, end: float, step: float) -> List[float]:
    """Regular lambda_I grid used to continue one fixed-point branch."""
    if step <= 0.0:
        raise ValueError("Continuation step must be positive.")
    values = [float(start)]
    if end >= start:
        current = float(start)
        while current + step < end - 1.0e-12:
            current += step
            values.append(float(current))
    else:
        current = float(start)
        while current - step > end + 1.0e-12:
            current -= step
            values.append(float(current))
    if abs(values[-1] - end) > 1.0e-12:
        values.append(float(end))
    return values


def strip_fixed_point_sample(sample: dict[str, float | int | np.ndarray]) -> dict[str, float | int]:
    """Drop the full stationary field so summaries stay compact and JSON-safe."""
    return {
        key: value
        for key, value in sample.items()
        if key != "pi_bar"
    }


def trace_information_rate_branch(
    w: World,
    lambda_C: np.ndarray,
    lambda_values: Sequence[float],
    informed_fraction: float,
    branch_name: str,
    beta: float = BETA_TRANSMISSION,
    n_agents: int = N_ROBOTS,
    seed: int = SEED,
) -> List[dict[str, float | int | np.ndarray]]:
    """
    Follow one Layer 1-I fixed-point branch by continuation in ``lambda_I``.

    Each solve is initialized from the previous branch point so the algorithm
    stays on the same self-consistent family of stationary fields whenever
    possible. This is especially important for negative ``lambda_I``, where the
    clustered solutions are not unique up to grid translations.
    """
    if branch_name == "coverage":
        current_pi = theoretical_stationary(w).copy()
    elif branch_name == "clustered":
        current_pi = None
    else:
        raise ValueError(f"Unknown information branch '{branch_name}'.")

    samples: List[dict[str, float | int | np.ndarray]] = []
    for lambda_value in lambda_values:
        sample = evaluate_information_rate_constraint(
            w=w,
            lambda_C=lambda_C,
            lambda_I_value=float(lambda_value),
            informed_fraction=informed_fraction,
            beta=beta,
            n_agents=n_agents,
            seed=seed,
            initial_pi=None if current_pi is None else current_pi,
            return_pi=True,
        )
        sample["branch"] = branch_name
        current_pi = normalize_probability(np.asarray(sample["pi_bar"], dtype=np.float64))
        samples.append(sample)
    return samples


def find_branch_bracket(
    samples: Sequence[dict[str, float | int | np.ndarray]],
    target_rate: float,
) -> tuple[int, int] | None:
    """Locate two consecutive branch samples whose rates bracket the target."""
    for idx in range(len(samples) - 1):
        left = float(samples[idx]["information_spreading_rate"]) - target_rate
        right = float(samples[idx + 1]["information_spreading_rate"]) - target_rate
        if left == 0.0:
            return idx, idx
        if left * right < 0.0 or right == 0.0:
            return idx, idx + 1
    return None


def bisect_information_rate_on_branch(
    w: World,
    lambda_C: np.ndarray,
    target_rate: float,
    informed_fraction: float,
    left_sample: dict[str, float | int | np.ndarray],
    right_sample: dict[str, float | int | np.ndarray],
    beta: float = BETA_TRANSMISSION,
    n_agents: int = N_ROBOTS,
    seed: int = SEED,
    relative_tolerance: float = 1.0e-3,
    lambda_tolerance: float = 1.0e-3,
    max_bisection_iters: int = 40,
) -> tuple[dict[str, float | int | np.ndarray], List[dict[str, float | int | np.ndarray]]]:
    """
    Refine the inverse Layer 1-I solve on one already-chosen fixed-point branch.

    The midpoint initialization is formed from the endpoint stationary fields so
    the bisection step remains on the same branch instead of restarting from a
    generic symmetric guess.
    """
    if float(left_sample["lambda_I"]) > float(right_sample["lambda_I"]):
        left_sample, right_sample = right_sample, left_sample

    target = float(target_rate)
    scale = max(abs(target), 1.0e-12)
    tolerance = relative_tolerance * scale
    history: List[dict[str, float | int | np.ndarray]] = []

    best = min(
        (left_sample, right_sample),
        key=lambda item: abs(float(item["information_spreading_rate"]) - target),
    )

    if float(left_sample["lambda_I"]) == float(right_sample["lambda_I"]):  # pragma: no cover - defensive
        return best, history

    left = dict(left_sample)
    right = dict(right_sample)
    for _ in range(max_bisection_iters):
        left_lambda = float(left["lambda_I"])
        right_lambda = float(right["lambda_I"])
        if abs(right_lambda - left_lambda) <= lambda_tolerance:
            break

        mid_lambda = 0.5 * (left_lambda + right_lambda)
        mid_initial = normalize_probability(
            0.5 * np.asarray(left["pi_bar"], dtype=np.float64)
            + 0.5 * np.asarray(right["pi_bar"], dtype=np.float64)
        )
        mid = evaluate_information_rate_constraint(
            w=w,
            lambda_C=lambda_C,
            lambda_I_value=mid_lambda,
            informed_fraction=informed_fraction,
            beta=beta,
            n_agents=n_agents,
            seed=seed,
            initial_pi=mid_initial,
            return_pi=True,
        )
        mid["branch"] = left.get("branch", right.get("branch", "unknown"))
        history.append(mid)

        if abs(float(mid["information_spreading_rate"]) - target) < abs(float(best["information_spreading_rate"]) - target):
            best = mid
        if abs(float(mid["information_spreading_rate"]) - target) <= tolerance:
            best = mid
            break

        left_sign = float(left["information_spreading_rate"]) - target
        mid_sign = float(mid["information_spreading_rate"]) - target
        if left_sign == 0.0:
            best = left
            break
        if left_sign * mid_sign <= 0.0:
            right = mid
        else:
            left = mid

    return best, history


def solve_lambda_I_for_information_rate(
    w: World,
    lambda_C: np.ndarray,
    target_rate: float,
    informed_fraction: float,
    beta: float = BETA_TRANSMISSION,
    n_agents: int = N_ROBOTS,
    seed: int = SEED,
    candidate_lambdas: Sequence[float] | None = None,
    relative_tolerance: float = 1.0e-3,
    branch: str = "auto",
    clustered_lambda_min: float = -400.0,
    coverage_lambda_max: float = 400.0,
    continuation_step: float = 10.0,
    lambda_tolerance: float = 1.0e-3,
    max_bisection_iters: int = 40,
) -> dict[str, object]:
    """
    Inverse Layer 1-I solve for the scalar multiplier lambda_I.

    The inverse condition is solved on a chosen fixed-point branch.

    For lambda_I < 0 the Layer 1-I fixed point can be non-unique, so the solver
    follows either a coverage-like branch or a clustered branch by continuation:
    each solve is initialized from the previous fixed point on the same branch.
    Once a target rate is bracketed on that branch, bisection refines lambda_I.

    The legacy candidate scan is kept only as a fallback if no chosen branch
    brackets the target. This means the implementation stays faithful to the scalar constraint
    while still documenting the numerical choice made when the fixed point is not unique.
    """
    target = float(target_rate)
    scale = max(abs(target), 1.0e-12)
    tolerance = relative_tolerance * scale

    zero_sample = evaluate_information_rate_constraint(
        w=w,
        lambda_C=lambda_C,
        lambda_I_value=0.0,
        informed_fraction=informed_fraction,
        beta=beta,
        n_agents=n_agents,
        seed=seed,
        initial_pi=theoretical_stationary(w).copy(),
        return_pi=True,
    )
    zero_sample["branch"] = "coverage"
    zero_rate = float(zero_sample["information_spreading_rate"])

    if abs(zero_rate - target) <= tolerance:
        target_p_enc = target / max(float(n_agents) * beta * informed_fraction * (1.0 - informed_fraction), 1.0e-12)
        zero_public = strip_fixed_point_sample(zero_sample)
        return {
            "target_information_spreading_rate": target,
            "target_p_enc": float(target_p_enc),
            "informed_fraction": float(informed_fraction),
            "beta": float(beta),
            "n_agents": int(n_agents),
            "lambda_I": 0.0,
            "predicted_information_spreading_rate": zero_rate,
            "predicted_p_enc": float(zero_sample["p_enc"]),
            "absolute_rate_error": abs(zero_rate - target),
            "relative_rate_error": abs(zero_rate - target) / scale,
            "target_within_scanned_range": True,
            "target_within_branch_range": True,
            "converged": True,
            "solver_method": "exact_zero_baseline",
            "selected_branch": "coverage",
            "used_bisection": False,
            "bracket_found": True,
            "fixed_point_iterations": int(zero_sample["fixed_point_iterations"]),
            "fixed_point_error": float(zero_sample["fixed_point_error"]),
            "evaluations": [zero_public],
            "branch_evaluations": {
                "coverage": [zero_public],
                "clustered": [],
            },
        }

    coverage_trace = trace_information_rate_branch(
        w=w,
        lambda_C=lambda_C,
        lambda_values=continuation_lambda_grid(0.0, coverage_lambda_max, continuation_step),
        informed_fraction=informed_fraction,
        branch_name="coverage",
        beta=beta,
        n_agents=n_agents,
        seed=seed,
    )
    clustered_trace = trace_information_rate_branch(
        w=w,
        lambda_C=lambda_C,
        lambda_values=continuation_lambda_grid(clustered_lambda_min, 0.0, continuation_step),
        informed_fraction=informed_fraction,
        branch_name="clustered",
        beta=beta,
        n_agents=n_agents,
        seed=seed,
    )

    branch_samples = {
        "coverage": coverage_trace,
        "clustered": clustered_trace,
    }
    branch_public = {
        name: [strip_fixed_point_sample(sample) for sample in samples]
        for name, samples in branch_samples.items()
    }

    def rate_error(item: dict[str, float | int | np.ndarray]) -> float:
        return abs(float(item["information_spreading_rate"]) - target)

    branch_info: dict[str, dict[str, object]] = {}
    for name, samples in branch_samples.items():
        rates = [float(item["information_spreading_rate"]) for item in samples]
        bracket = find_branch_bracket(samples, target)
        near_target = [item for item in samples if rate_error(item) <= tolerance]
        branch_info[name] = {
            "samples": samples,
            "min_rate": min(rates),
            "max_rate": max(rates),
            "near_target": near_target,
            "bracket": bracket,
            "target_within_range": bool(min(rates) - tolerance <= target <= max(rates) + tolerance),
        }

    if branch == "auto":
        preferred_branch = "clustered" if target > zero_rate else "coverage"
        fallback_branch = "coverage" if preferred_branch == "clustered" else "clustered"
        ordered_branches = [preferred_branch, fallback_branch]
    else:
        if branch not in branch_samples:
            raise ValueError(f"Unknown branch '{branch}'. Expected one of auto, clustered, coverage.")
        other_branch = "coverage" if branch == "clustered" else "clustered"
        ordered_branches = [branch, other_branch]

    selected_branch_name: str | None = None
    selected_sample: dict[str, float | int | np.ndarray] | None = None
    bisection_history: List[dict[str, float | int | np.ndarray]] = []
    used_bisection = False
    bracket_found = False

    for branch_name in ordered_branches:
        info = branch_info[branch_name]
        near_target = info["near_target"]
        if near_target:
            if branch_name == "clustered":
                selected_sample = max(near_target, key=lambda item: float(item["lambda_I"]))
            else:
                selected_sample = min(near_target, key=lambda item: abs(float(item["lambda_I"])))
            selected_branch_name = branch_name
            break

        bracket = info["bracket"]
        if bracket is not None and bracket[0] != bracket[1]:
            left_sample = info["samples"][bracket[0]]
            right_sample = info["samples"][bracket[1]]
            selected_sample, bisection_history = bisect_information_rate_on_branch(
                w=w,
                lambda_C=lambda_C,
                target_rate=target,
                informed_fraction=informed_fraction,
                left_sample=left_sample,
                right_sample=right_sample,
                beta=beta,
                n_agents=n_agents,
                seed=seed,
                relative_tolerance=relative_tolerance,
                lambda_tolerance=lambda_tolerance,
                max_bisection_iters=max_bisection_iters,
            )
            selected_branch_name = branch_name
            used_bisection = True
            bracket_found = True
            break
        if bracket is not None and bracket[0] == bracket[1]:
            selected_sample = info["samples"][bracket[0]]
            selected_branch_name = branch_name
            bracket_found = True
            break

    solver_method = "branch_continuation_bisection"
    if selected_sample is None:
        if candidate_lambdas is None:
            candidate_lambdas = (
                -400.0,
                -350.0,
                -300.0,
                -250.0,
                -225.0,
                -210.0,
                -200.0,
                -190.0,
                -180.0,
                -175.0,
                -170.0,
                -165.0,
                -160.0,
                -150.0,
                -125.0,
                -100.0,
                -50.0,
                0.0,
                50.0,
                100.0,
                200.0,
                400.0,
            )
        fallback_samples = [
            evaluate_information_rate_constraint(
                w=w,
                lambda_C=lambda_C,
                lambda_I_value=float(lambda_value),
                informed_fraction=informed_fraction,
                beta=beta,
                n_agents=n_agents,
                seed=seed,
                return_pi=True,
            )
            for lambda_value in candidate_lambdas
        ]
        selected_sample = min(
            fallback_samples,
            key=lambda item: (rate_error(item), abs(float(item["lambda_I"]))),
        )
        selected_branch_name = "fallback_scan"
        branch_public["fallback_scan"] = [strip_fixed_point_sample(sample) for sample in fallback_samples]
        solver_method = "legacy_candidate_scan_fallback"

    assert selected_sample is not None
    assert selected_branch_name is not None

    selected_rate = float(selected_sample["information_spreading_rate"])
    selected_p_enc = float(selected_sample["p_enc"])
    target_p_enc = target / max(float(n_agents) * beta * informed_fraction * (1.0 - informed_fraction), 1.0e-12)
    chosen_branch_public = list(branch_public.get(selected_branch_name, []))
    bisection_public = [strip_fixed_point_sample(sample) for sample in bisection_history]
    evaluations = chosen_branch_public + bisection_public
    evaluations.sort(key=lambda item: float(item["lambda_I"]))

    selected_branch_info = branch_info.get(selected_branch_name)
    if selected_branch_info is not None:
        target_within_range = bool(selected_branch_info["target_within_range"])
    else:
        target_within_range = False

    return {
        "target_information_spreading_rate": target,
        "target_p_enc": float(target_p_enc),
        "informed_fraction": float(informed_fraction),
        "beta": float(beta),
        "n_agents": int(n_agents),
        "lambda_I": float(selected_sample["lambda_I"]),
        "predicted_information_spreading_rate": selected_rate,
        "predicted_p_enc": selected_p_enc,
        "absolute_rate_error": abs(selected_rate - target),
        "relative_rate_error": abs(selected_rate - target) / scale,
        "target_within_scanned_range": bool(target_within_range),
        "target_within_branch_range": bool(target_within_range),
        "converged": bool(abs(selected_rate - target) <= tolerance),
        "fixed_point_iterations": int(selected_sample["fixed_point_iterations"]),
        "fixed_point_error": float(selected_sample["fixed_point_error"]),
        "solver_method": solver_method,
        "selected_branch": selected_branch_name,
        "used_bisection": bool(used_bisection),
        "bracket_found": bool(bracket_found),
        "evaluations": evaluations,
        "branch_evaluations": branch_public,
        "selected_pi_bar": np.asarray(selected_sample["pi_bar"], dtype=np.float64).copy(),
    }


# ============================================================
# 4. ROBOT
# ============================================================

@dataclass
class Robot:
    """
    Robot state for the Layer 1-I physical realization.

    ``info_age`` is the paper-level AoI observable tau_i: time since the robot
    last completed a successful information exchange with another robot.
    ``world_map`` stores local cell-wise coverage records and is distinct from
    tau_i.
    """
    id: int
    x: float
    y: float
    from_k: int
    to_k: int
    tx: float
    ty: float
    info_age: float = 0.0
    last_meet: int = 0
    world_map: RobotWorldMap | None = None


def ensure_robot_map(r: Robot, w: World, t: float = 0.0) -> RobotWorldMap:
    """Allocate the robot's local map the first time it is needed."""
    if r.world_map is None:
        r.world_map = RobotWorldMap.empty(w.K)
        r.world_map.observe_cell(r.from_k, t)
    return r.world_map


def make_robot(
    idx: int,
    w: World,
    rng: np.random.Generator,
    fixed_initial_info_age: float | None = None,
) -> Robot:
    """
    Spawn one robot with an optional prescribed initial paper-level AoI.

    The robot begins at the centre of a random region and observes that region
    in its local map immediately.
    """
    k0 = int(rng.integers(0, w.K))
    cx, cy = region_center(w, k0)
    if fixed_initial_info_age is None:
        age0 = float(rng.uniform(0.0, A_HALF))
    else:
        age0 = float(fixed_initial_info_age)
    world_map = RobotWorldMap.empty(w.K)
    world_map.observe_cell(k0, 0.0)
    return Robot(
        id=idx,
        x=cx,
        y=cy,
        from_k=k0,
        to_k=k0,
        tx=cx,
        ty=cy,
        info_age=age0,
        last_meet=0,
        world_map=world_map,
    )


def initialise_targets(
    robots: Sequence[Robot],
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    rng: np.random.Generator,
    info_field: np.ndarray | None = None,
    use_age_gate: bool = USE_AGE_GATE_FOR_CONTROL,
) -> None:
    """Sample the first destination for every robot from the Layer 1-I kernel."""
    if info_field is None:
        info_field = compute_information_field(w, robots)
    for r in robots:
        next_k = sample_next_region(
            w=w,
            k1=r.from_k,
            lambda_C=lambda_C,
            lambda_I_value=lambda_I_value,
            info_field=info_field,
            info_age=r.info_age,
            rng=rng,
            use_age_gate=use_age_gate,
        )
        r.to_k = next_k
        r.tx, r.ty = region_center(w, next_k)


def step_robot(
    r: Robot,
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    speed: float,
    markov_visits: np.ndarray,
    coverage_age_field: np.ndarray,
    t: int,
    rng: np.random.Generator,
    freeze_info_age: bool = False,
    use_age_gate: bool = USE_AGE_GATE_FOR_CONTROL,
) -> None:
    """
    Advance one Layer 1-I robot by one simulation step.

    As in Layer 1-C, the Markov state updates only on physical arrival at the
    centre of the destination region. The global coverage-age field and the
    robot's local map are updated at that instant. AoI evolves independently and
    is reset only by successful information exchange events.
    """
    robot_map = ensure_robot_map(r, w)
    if not freeze_info_age:
        r.info_age += 1.0

    dx = r.tx - r.x
    dy = r.ty - r.y
    dist = math.sqrt(dx * dx + dy * dy)

    move = min(speed, dist)
    if dist > 1e-10:
        r.x += move * dx / dist
        r.y += move * dy / dist

    if abs(r.x - r.tx) < 1e-9 and abs(r.y - r.ty) < 1e-9:
        r.from_k = r.to_k
        markov_visits[r.from_k] += 1
        coverage_age_field[r.from_k] = t
        robot_map.observe_cell(r.from_k, float(t))

        next_k = sample_next_region(
            w=w,
            k1=r.from_k,
            lambda_C=lambda_C,
            lambda_I_value=lambda_I_value,
            info_field=info_field,
            info_age=r.info_age,
            rng=rng,
            use_age_gate=use_age_gate,
        )
        r.to_k = next_k
        r.tx, r.ty = region_center(w, next_k)


def detect_meetings(robots: List[Robot], r_meet: float, t: int) -> int:
    """Legacy radius diagnostic; paper validation uses same-region exchange below."""
    n = len(robots)
    n_pairs = 0
    xs = np.array([r.x for r in robots])
    ys = np.array([r.y for r in robots])
    for i in range(n):
        dx = xs[i + 1:] - xs[i]
        dy = ys[i + 1:] - ys[i]
        hits = np.where(dx * dx + dy * dy <= r_meet * r_meet)[0]
        for j_off in hits:
            j = i + 1 + int(j_off)
            robots[i].info_age = 0.0
            robots[j].info_age = 0.0
            robots[i].last_meet = t
            robots[j].last_meet = t
            n_pairs += 1
    return n_pairs


def same_region_pairs(
    robots: Sequence[Robot],
    w: World,
) -> List[Tuple[int, int, int, float, float]]:
    """Enumerate co-located robot pairs, matching the region-level encounters."""
    by_cell: List[List[int]] = [[] for _ in range(w.K)]
    for idx, robot in enumerate(robots):
        by_cell[position_to_cell(w, robot.x, robot.y)].append(idx)

    pairs: List[Tuple[int, int, int, float, float]] = []
    for cell, members in enumerate(by_cell):
        if len(members) < 2:
            continue
        cx, cy = region_center(w, cell)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.append((members[a], members[b], cell, cx, cy))
    return pairs


def exchange_robot_world_maps(robots: List[Robot], w: World, i: int, j: int, t: int) -> int:
    """Exchange the local cell-wise maps of two robots after a successful contact."""
    first_map = ensure_robot_map(robots[i], w)
    second_map = ensure_robot_map(robots[j], w)
    first_updates, second_updates = exchange_maps(first_map, second_map, float(t))
    return int(first_updates + second_updates)


def perform_information_exchanges(
    robots: List[Robot],
    w: World,
    rng: np.random.Generator,
    t: int,
    informed: np.ndarray | None = None,
    beta: float = BETA_TRANSMISSION,
    p_encounter_given_colocation: float = P_ENCOUNTER_GIVEN_COLOCATION,
) -> Tuple[int, int, int, int]:
    """
    Paper-faithful communication diagnostic.

    A potential encounter occurs when two robots occupy the same region. The
    actual encounter is Bernoulli with probability p_encounter_given_colocation
    (default 1 because the region co-occupancy already realizes p_enc), and a
    successful communication/transmission is Bernoulli with probability beta.
    Successful communication resets AoI. If an informed-state vector is supplied,
    one-informed/one-uninformed successful communications spread the message.
    """
    encounters = 0
    communications = 0
    new_transmissions = 0
    map_records_received = 0
    for i, j, _, _, _ in same_region_pairs(robots, w):
        if rng.random() > p_encounter_given_colocation:
            continue
        encounters += 1
        if rng.random() > beta:
            continue
        communications += 1
        robots[i].info_age = 0.0
        robots[j].info_age = 0.0
        robots[i].last_meet = t
        robots[j].last_meet = t
        map_records_received += exchange_robot_world_maps(robots, w, i, j, t)
        if informed is not None:
            before_i = bool(informed[i])
            before_j = bool(informed[j])
            if before_i or before_j:
                informed[i] = True
                informed[j] = True
            if before_i != before_j:
                new_transmissions += 1
    return encounters, communications, new_transmissions, map_records_received


# ============================================================
# 5. SIMULATION
# ============================================================

@dataclass
class SimResult:
    w: World
    lambda_I_value: float
    pi_empirical: np.ndarray
    mean_info_field: np.ndarray
    final_info_field: np.ndarray
    ck_history: List[np.ndarray]
    markov_step_history: List[int]
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]]
    t_axis: np.ndarray
    dispersion: np.ndarray
    mean_cov_age: np.ndarray
    mean_info_age: np.ndarray
    mean_local_cov_age: np.ndarray
    mean_map_record_age: np.ndarray
    meetings_per_snap: np.ndarray
    encounter_proxy: np.ndarray
    coverage_l1: np.ndarray


def run_simulation(
    speed: float = ROBOT_SPEED,
    T: int = T_SIM,
    lambda_C_val: float = LAMBDA_C_VAL,
    lambda_I_value: float = LAMBDA_I_VAL,
    seed: int = SEED,
    n_robots: int = N_ROBOTS,
    fixed_initial_info_age: float | None = None,
    freeze_info_age: bool = False,
    enable_meetings: bool = True,
    information_field_mode: str = INFORMATION_FIELD_MODE,
    use_age_gate: bool = USE_AGE_GATE_FOR_CONTROL,
) -> SimResult:
    """
    Run one standalone Layer 1-I simulation.

    ``information_field_mode='stationary_fixed_point'``: the robots move under the self-consistent stationary field (π̄).
    ``'online_density'`` is retained only as an exploratory comparison mode.
    The recorded outputs separate paper-level AoI, global coverage age, local
    coverage age, and local map-record age.
    """
    rng = np.random.default_rng(seed)
    w = build_world(NX, NY, CELL_SIZE)
    lambda_C = build_lambda_C(w, lambda_C_val)
    pi_theory = theoretical_stationary(w)
    if information_field_mode == "stationary_fixed_point":
        # Paper-faithful Layer 1-I controller: use the self-consistent field pi_bar.
        control_info_field, _, _, _ = solve_information_fixed_point(
            w=w,
            lambda_C=lambda_C,
            lambda_I_value=lambda_I_value,
            seed=seed,
        )
    elif information_field_mode == "online_density":
        # Optional diagnostic mode, not the paper-level Layer 1-I controller.
        control_info_field = np.full(w.K, 1.0 / w.K, dtype=np.float64)
    else:
        raise ValueError(
            "information_field_mode must be 'stationary_fixed_point' or 'online_density'"
        )

    robots = [
        make_robot(
            idx=i,
            w=w,
            rng=rng,
            fixed_initial_info_age=fixed_initial_info_age,
        )
        for i in range(n_robots)
    ]
    initialise_targets(
        robots=robots,
        w=w,
        lambda_C=lambda_C,
        lambda_I_value=lambda_I_value,
        rng=rng,
        info_field=control_info_field,
        use_age_gate=use_age_gate,
    )

    markov_visits = np.zeros(w.K, dtype=np.int64)
    coverage_age_field = np.full(w.K, -1.0, dtype=np.float64)
    for robot in robots:
        coverage_age_field[robot.from_k] = 0.0
    info_field_accumulator = np.zeros(w.K, dtype=np.float64)

    ck_history: List[np.ndarray] = []
    ms_history: List[int] = []
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]] = []

    t_log: List[int] = []
    disp_log: List[float] = []
    mean_cov_log: List[float] = []
    mean_info_log: List[float] = []
    mean_local_cov_log: List[float] = []
    mean_map_record_log: List[float] = []
    meet_log: List[int] = []
    encounter_log: List[float] = []
    coverage_l1_log: List[float] = []
    meet_accumulator = 0

    final_info_field = compute_information_field(w, robots)

    for t in range(1, T + 1):
        observed_info_field = compute_information_field(w, robots)
        info_field = control_info_field if information_field_mode == "stationary_fixed_point" else observed_info_field
        info_field_accumulator += info_field
        final_info_field = info_field

        for r in robots:
            step_robot(
                r=r,
                w=w,
                lambda_C=lambda_C,
                lambda_I_value=lambda_I_value,
                info_field=info_field,
                speed=speed,
                markov_visits=markov_visits,
                coverage_age_field=coverage_age_field,
                t=t,
                rng=rng,
                freeze_info_age=freeze_info_age,
                use_age_gate=use_age_gate,
            )

        if enable_meetings:
            encounters, _, _, _ = perform_information_exchanges(robots, w, rng, t)
            meet_accumulator += encounters

        if t % RECORD_EVERY == 0:
            xs = np.array([r.x for r in robots])
            ys = np.array([r.y for r in robots])
            occ = occupancy_distribution(w, robots)
            total_arrivals = int(markov_visits.sum())
            ck = markov_visits / max(total_arrivals, 1)

            t_log.append(t)
            disp_log.append(float(np.var(xs) + np.var(ys)))
            maps = [ensure_robot_map(robot, w) for robot in robots]
            global_cov_age = np.where(
                coverage_age_field >= 0.0,
                float(t) - coverage_age_field,
                float(t) + 1.0,
            )
            mean_cov_log.append(float(np.mean(global_cov_age)))
            mean_info_log.append(float(np.mean([robot.info_age for robot in robots])))
            mean_local_cov_log.append(mean_robot_coverage_age(maps, float(t)))
            mean_map_record_log.append(mean_robot_map_record_age(maps, float(t)))
            meet_log.append(meet_accumulator)
            encounter_log.append(encounter_proxy_from_occupancy(occ))
            coverage_l1_log.append(float(np.sum(np.abs(ck - pi_theory))))
            meet_accumulator = 0

            if total_arrivals > 0:
                ck_history.append(ck.copy())
                ms_history.append(total_arrivals)

        if t % SNAP_EVERY == 0:
            xs = np.array([r.x for r in robots])
            ys = np.array([r.y for r in robots])
            pos_snapshots.append((xs, ys, t))

    pi_emp = markov_visits / max(int(markov_visits.sum()), 1)
    mean_info_field = info_field_accumulator / float(max(T, 1))

    return SimResult(
        w=w,
        lambda_I_value=lambda_I_value,
        pi_empirical=pi_emp,
        mean_info_field=mean_info_field,
        final_info_field=final_info_field,
        ck_history=ck_history,
        markov_step_history=ms_history,
        pos_snapshots=pos_snapshots,
        t_axis=np.array(t_log, dtype=np.int64),
        dispersion=np.array(disp_log, dtype=np.float64),
        mean_cov_age=np.array(mean_cov_log, dtype=np.float64),
        mean_info_age=np.array(mean_info_log, dtype=np.float64),
        mean_local_cov_age=np.array(mean_local_cov_log, dtype=np.float64),
        mean_map_record_age=np.array(mean_map_record_log, dtype=np.float64),
        meetings_per_snap=np.array(meet_log, dtype=np.float64),
        encounter_proxy=np.array(encounter_log, dtype=np.float64),
        coverage_l1=np.array(coverage_l1_log, dtype=np.float64),
    )


# ============================================================
# 6. DIAGNOSTICS
# ============================================================

def dominant_frequency(
    t_axis: np.ndarray,
    signal: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if len(signal) < 8:
        return 0.0, 0.0, np.array([]), np.array([])
    dt = float(t_axis[1] - t_axis[0])
    centered = signal - signal.mean()
    window = np.hanning(len(centered))
    spectrum = np.fft.rfft(centered * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    if len(power) <= 2:
        return 0.0, 0.0, freqs, power
    idx = int(np.argmax(power[1:])) + 1
    prominence_floor = float(np.median(power[1:]))
    prominence = float(power[idx] / prominence_floor) if prominence_floor > 0 else float("inf")
    return float(freqs[idx]), prominence, freqs, power


# ============================================================
# 7. VISUALISATION
# ============================================================

def make_main_figure(res: SimResult):
    w = res.w

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(res.mean_info_field.reshape(w.Ny, w.Nx), origin="lower", cmap="magma")
    ax1.set_title("(a) Mean information field π̄(k)")
    ax1.set_xlabel("col")
    ax1.set_ylabel("row")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(res.pi_empirical.reshape(w.Ny, w.Nx), origin="lower", cmap="viridis")
    ax2.set_title("(b) Empirical visit distribution π̂(k)")
    ax2.set_xlabel("col")
    ax2.set_ylabel("row")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    xs, ys, t_end = res.pos_snapshots[-1]
    ax3.scatter(xs, ys, s=14, alpha=0.8, color="dodgerblue")
    ax3.set_xlim(0, w.Nx * w.cell_size)
    ax3.set_ylim(0, w.Ny * w.cell_size)
    ax3.set_aspect("equal")
    ax3.set_title(f"(c) Robot positions at t={t_end}")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.plot(res.t_axis, res.dispersion, color="navy", lw=1.4, label="S(t) dispersion")
    ax4.set_xlabel("simulation step")
    ax4.set_ylabel("S(t)", color="navy")
    ax4.tick_params(axis="y", labelcolor="navy")
    ax4b = ax4.twinx()
    ax4b.plot(res.t_axis, res.mean_info_age, color="crimson", lw=1.2, alpha=0.85, label="mean AoI tau")
    ax4b.plot(res.t_axis, res.mean_cov_age, color="seagreen", lw=1.2, alpha=0.85, label="mean coverage age")
    ax4b.set_ylabel("age (steps)")
    ax4.set_title("(d) Dispersion and ages")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(res.t_axis, res.encounter_proxy, color="darkorange", lw=1.3, label="encounter proxy")
    ax5.set_xlabel("simulation step")
    ax5.set_ylabel("sum_k occ_k^2", color="darkorange")
    ax5.tick_params(axis="y", labelcolor="darkorange")
    ax5b = ax5.twinx()
    ax5b.plot(res.t_axis, res.coverage_l1, color="slateblue", lw=1.2, alpha=0.85, label="coverage L1")
    ax5b.set_ylabel("||c(.,t) - π̄(.)||_1")
    ax5.set_title("(e) Clustering vs. coverage error")
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle(
        "MaxCal Coverage + Information Diffusion "
        f"(lambda_I = {res.lambda_I_value:+.2f})"
    )
    fig.tight_layout()
    return fig


def make_age_plane_figure(res: SimResult):
    fig, ax = plt.subplots(figsize=(6, 6))
    cov = res.mean_cov_age
    info = res.mean_info_age
    n0 = max(1, len(cov) // 10)
    sc = ax.scatter(cov[n0:], info[n0:], c=res.t_axis[n0:], cmap="viridis", s=8)
    ax.plot(cov[n0:], info[n0:], color="grey", lw=0.4, alpha=0.6)
    ax.set_xlabel("mean coverage age")
    ax.set_ylabel("mean AoI tau")
    ax.set_title("Age-plane trajectory")
    fig.colorbar(sc, ax=ax, label="simulation step")
    fig.tight_layout()
    return fig


def make_constraint_sweep_figure(
    values: Sequence[float] = LAMBDA_I_SWEEP,
    T: int = 9_000,
):
    results = [
        run_simulation(
            lambda_I_value=value,
            T=T,
            fixed_initial_info_age=SWEEP_STALE_AGE,
            freeze_info_age=True,
            enable_meetings=False,
        )
        for value in values
    ]
    means_disp = np.array([float(res.dispersion.mean()) for res in results], dtype=np.float64)
    means_enc = np.array([float(res.encounter_proxy.mean()) for res in results], dtype=np.float64)
    final_cov_l1 = np.array([float(res.coverage_l1[-1]) for res in results], dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)

    zero_idx = int(np.where(np.isclose(values_arr, 0.0))[0][0])
    baseline_disp = means_disp[zero_idx]
    baseline_enc = means_enc[zero_idx]

    summary = []
    for value, res, mean_disp, mean_enc, cov_l1 in zip(values_arr, results, means_disp, means_enc, final_cov_l1):
        if mean_disp < baseline_disp and mean_enc > baseline_enc:
            regime = "cluster"
        elif mean_disp >= baseline_disp and mean_enc <= baseline_enc:
            regime = "coverage"
        else:
            regime = "mixed"
        f_star, prominence, _, _ = dominant_frequency(res.t_axis, res.dispersion)
        summary.append(
            (
                float(value),
                float(mean_disp),
                float(mean_enc),
                float(cov_l1),
                float(f_star),
                float(prominence),
                regime,
            )
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    axes[0, 0].plot(values_arr, means_disp, marker="o", lw=2.0, color="navy")
    axes[0, 0].axvline(0.0, color="grey", ls=":")
    axes[0, 0].set_title("Mean dispersion vs lambda_I")
    axes[0, 0].set_xlabel("lambda_I")
    axes[0, 0].set_ylabel("mean S(t)")

    axes[0, 1].plot(values_arr, means_enc, marker="o", lw=2.0, color="darkorange")
    axes[0, 1].axvline(0.0, color="grey", ls=":")
    axes[0, 1].set_title("Encounter proxy vs lambda_I")
    axes[0, 1].set_xlabel("lambda_I")
    axes[0, 1].set_ylabel("mean sum_k occ_k^2")

    axes[1, 0].plot(values_arr, final_cov_l1, marker="o", lw=2.0, color="slateblue")
    axes[1, 0].axvline(0.0, color="grey", ls=":")
    axes[1, 0].set_title("Final coverage error vs lambda_I")
    axes[1, 0].set_xlabel("lambda_I")
    axes[1, 0].set_ylabel("final ||c(.,t) - π̄(.)||_1")

    colors = ["crimson" if value < 0 else "seagreen" for value in values_arr]
    axes[1, 1].scatter(means_disp, means_enc, c=colors, s=55)
    for value, x, y in zip(values_arr, means_disp, means_enc):
        axes[1, 1].annotate(f"{value:+.0f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1, 1].set_title("Regime map")
    axes[1, 1].set_xlabel("mean S(t)")
    axes[1, 1].set_ylabel("mean sum_k occ_k^2")

    fig.suptitle("Signed lambda_I sweep (isolated stale-information test)")
    fig.tight_layout()
    return fig, summary


def make_animation(res: SimResult, fps: int = 12, filename: str = "maxcal_info_diffusion.gif"):
    w = res.w
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        res.mean_info_field.reshape(w.Ny, w.Nx),
        origin="lower",
        cmap="magma",
        alpha=0.45,
        extent=(0, w.Nx, 0, w.Ny),
    )
    ax.set_aspect("equal")
    ax.set_xlim(0, w.Nx)
    ax.set_ylim(0, w.Ny)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    scat = ax.scatter([], [], s=14, alpha=0.8, color="navy")
    title = ax.set_title("")

    def update(frame: int):
        xs, ys, t = res.pos_snapshots[frame]
        scat.set_offsets(np.column_stack([xs / w.cell_size, ys / w.cell_size]))
        title.set_text(f"t = {t}")
        return scat, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(res.pos_snapshots),
        interval=1000 // fps,
        blit=False,
    )
    try:
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        print(f"Saved {filename}")
    except Exception as exc:
        print(f"Could not save animation ({exc}); skipping.")
    plt.close(fig)


# ============================================================
# 8. ENTRY POINT
# ============================================================

def main():
    print("MaxCal Coverage + Information Diffusion Simulation")
    print(f"  World     : {NX}x{NY} grid, cell={CELL_SIZE} m")
    print(f"  Swarm     : {N_ROBOTS} robots, speed={ROBOT_SPEED} m/step")
    print(f"  Duration  : {T_SIM} steps")
    print(f"  lambda_C  : {LAMBDA_C_VAL}")
    print(f"  lambda_I  : {LAMBDA_I_VAL}")
    print(f"  A_half    : {A_HALF}")
    print(f"  R_meet    : {R_MEET}")
    print()
    print("Pinciroli sign interpretation implemented here:")
    print("  lambda_I < 0  -> attraction to high-information cells -> clustering")
    print("  lambda_I = 0  -> pure coverage baseline")
    print("  lambda_I > 0  -> repulsion from high-information cells -> coverage priority")
    print()

    print("Running simulation...")
    result = run_simulation()
    f_star, prominence, _, _ = dominant_frequency(result.t_axis, result.dispersion)
    print(f"  Total Markov arrivals : {result.markov_step_history[-1]}")
    print(f"  Mean dispersion S     : {result.dispersion.mean():.3f}")
    print(f"  Mean encounter proxy  : {result.encounter_proxy.mean():.5f}")
    print(f"  Final coverage L1     : {result.coverage_l1[-1]:.5f}")
    if f_star > 0:
        print(f"  Dominant frequency f* : {f_star:.4e} (period ~ {1.0 / f_star:.0f} steps)")
    else:
        print("  Dominant frequency f* : none detected")
    print(f"  Spectral prominence   : {prominence:.2f}")
    print()

    print("Saving main figure -> maxcal_info_diffusion_main.png")
    fig_main = make_main_figure(result)
    fig_main.savefig("maxcal_info_diffusion_main.png", dpi=120)
    plt.close(fig_main)

    print("Saving age-plane figure -> maxcal_info_diffusion_age_plane.png")
    fig_age = make_age_plane_figure(result)
    fig_age.savefig("maxcal_info_diffusion_age_plane.png", dpi=120)
    plt.close(fig_age)

    print("Saving animation -> maxcal_info_diffusion.gif")
    make_animation(result)

    print("Running signed lambda_I sweep (isolated stale-information test)...")
    fig_sweep, summary = make_constraint_sweep_figure()
    fig_sweep.savefig("maxcal_info_diffusion_sweep.png", dpi=120)
    plt.close(fig_sweep)
    print("  Saved maxcal_info_diffusion_sweep.png")
    print("  Sweep summary (lambda_I, mean S, mean encounter, final L1, f*, prom, regime):")
    for row in summary:
        print(
            "    "
            f"{row[0]:+5.1f}  "
            f"S={row[1]:7.3f}  "
            f"E={row[2]:.5f}  "
            f"L1={row[3]:.5f}  "
            f"f*={row[4]:.4e}  "
            f"prom={row[5]:6.2f}  "
            f"-> {row[6]}"
        )


if __name__ == "__main__":
    main()
