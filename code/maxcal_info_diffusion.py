"""
RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Layer 1-I information-diffusion controller and inverse rate solve.

The information objective is not a target spatial density; it is a spreading-rate constraint.  
For ``n`` informed robots out of ``A`` robots, with ``f=n/A``, the mean-field model is

    dn/dt = A beta p_enc f(1-f).

The encounter probability is induced by the stationary control field
``pi_bar``:

    p_enc = sum_k pi_bar_k^2.

The information kernel uses that same field as the destination observable,

    P_ij(pi_bar) = A_ij exp[-lambda_C,j - lambda_I pi_bar_j]
                   / sum_l A_il exp[-lambda_C,l - lambda_I pi_bar_l].

Therefore the controller must solve the self-consistency equation

    pi_bar = stationary(P(pi_bar)).

Negative ``lambda_I`` rewards high-``pi_bar`` destinations and creates the
clustered branch; nonnegative ``lambda_I`` recovers or favors the
coverage-like branch.  The inverse solver chooses the multiplier that matches
the requested rate, breaking ties by the minimum change from the neutral
controller ``lambda_I=0``.
"""

from __future__ import annotations

import math
import os
import tempfile
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import maxcal_core as core
from maxcal_local_maps import (
    RobotWorldMap,
    exchange_maps,
    mean_robot_coverage_age,
    mean_robot_map_record_age,
)


NX = 20
NY = 20
CELL_SIZE = 1.0
N_ROBOTS = 50
ROBOT_SPEED = 0.15
T_SIM = 18_000
RECORD_EVERY = 30
SNAP_EVERY = 300
SEED = 42

LAMBDA_C_VAL = 0.0
LAMBDA_I_VAL = -400.0
BETA_TRANSMISSION = 0.75
P_ENCOUNTER_GIVEN_COLOCATION = 1.0
INFORMATION_FIELD_MODE = "stationary_fixed_point"
USE_AGE_GATE_FOR_CONTROL = False
INFO_FIXED_POINT_TOL = 1.0e-11
INFO_FIXED_POINT_MAX_ITERS = 400
INFO_FIXED_POINT_DAMPING = 0.50
INFO_FIXED_POINT_PERTURBATION = 1.0e-3
A_HALF = 1.0
R_MEET = 1.0
LAMBDA_I_SWEEP = (-400.0, -200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0, 400.0)
SWEEP_STALE_AGE = 1.0e6

World = core.World
_FIXED_POINT_CACHE: dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, int, float]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Layer 1-I MaxCal information-diffusion controller.")
    parser.add_argument("--outdir", type=str, default="maxcal_info_diffusion", help="Directory for figures and summary JSON.")
    parser.add_argument("--T", type=int, default=T_SIM, help="Main simulation horizon.")
    parser.add_argument("--sweep-T", type=int, default=9_000, help="Horizon for the lambda_I sweep.")
    parser.add_argument("--lambda-I", type=float, default=LAMBDA_I_VAL, help="Information multiplier for the main run.")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    parser.add_argument("--no-animation", action="store_true", help="Skip GIF generation.")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@dataclass
class Robot:
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


def build_world(Nx: int, Ny: int, cell_size: float) -> World:
    return core.build_grid_world(Nx, Ny, cell_size)


def region_center(w: World, k: int) -> Tuple[float, float]:
    return core.region_center(w, k)


def position_to_cell(w: World, x: float, y: float) -> int:
    return core.position_to_cell(w, x, y)


def theoretical_stationary(w: World) -> np.ndarray:
    degrees = np.array([len(neighbors) for neighbors in w.adjacency], dtype=np.float64)
    return degrees / float(degrees.sum())


def normalize_probability(values: np.ndarray, floor: float = 1.0e-300) -> np.ndarray:
    return core.normalize_probability(values, floor=floor)


def clustered_seed_distribution(
    w: World,
    anchor_cell: int | None = None,
    sigma_cells: float = 0.75,
    floor: float = 1.0e-6,
) -> np.ndarray:
    """Localized seed used to follow one clustered fixed-point branch."""
    if anchor_cell is None:
        anchor_cell = (w.Ny // 2) * w.Nx + (w.Nx // 2)
    anchor = w.centers[int(anchor_cell)]
    dist2 = np.sum(((w.centers - anchor) / max(w.cell_size, 1.0e-12)) ** 2, axis=1)
    weights = np.exp(-0.5 * dist2 / max(sigma_cells * sigma_cells, 1.0e-12))
    weights[int(anchor_cell)] += 1.0
    return normalize_probability(weights + floor)


def build_lambda_C(w: World, lambda_C_val: float) -> np.ndarray:
    return np.full(w.K, float(lambda_C_val), dtype=np.float64)


def occupancy_distribution(w: World, robots: Sequence[Robot]) -> np.ndarray:
    positions = [(robot.x, robot.y) for robot in robots]
    return core.occupancy_distribution(w, positions)


def compute_information_field(w: World, robots: Sequence[Robot]) -> np.ndarray:
    """Normalized local occupancy density over each cell and its neighbors."""
    occ = occupancy_distribution(w, robots)
    field = np.zeros(w.K, dtype=np.float64)
    for cell, neighbors in enumerate(w.adjacency):
        field[cell] = occ[[cell, *neighbors]].sum()
    max_val = float(field.max())
    return field / max_val if max_val > 0.0 else field


def encounter_proxy_from_occupancy(occ: np.ndarray) -> float:
    return float(np.sum(np.asarray(occ, dtype=np.float64) ** 2))


def encounter_probability_from_stationary(pi_bar: np.ndarray) -> float:
    """Return ``p_enc = sum_k pi_bar_k^2`` for same-cell encounters."""
    pi = np.asarray(pi_bar, dtype=np.float64)
    return float(np.sum(pi * pi))


def logistic_information_rate(
    informed_fraction: float,
    p_enc: float,
    beta: float = BETA_TRANSMISSION,
    n_agents: int = N_ROBOTS,
) -> float:
    """Mean-field information-spreading rate ``A beta p_enc f(1-f)``."""
    f = float(np.clip(informed_fraction, 0.0, 1.0))
    return float(n_agents * beta * p_enc * f * (1.0 - f))


def info_gate(age: float, a_half: float = A_HALF) -> float:
    if math.isinf(age):
        return 1.0
    return float(age) / (float(age) + float(a_half))


def local_transition_probabilities(
    w: World,
    k1: int,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    info_age: float,
    use_age_gate: bool = USE_AGE_GATE_FOR_CONTROL,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the local Layer 1-I probabilities out of cell ``k1``.

    The destination cost is ``lambda_C[j] + g(tau) lambda_I I[j]``.  The age
    gate ``g(tau)`` is disabled in the validations, giving the fixed-point kernel above.
    """
    neighbors = np.array(w.adjacency[k1], dtype=np.int64)
    gate = info_gate(info_age) if use_age_gate else 1.0
    cost = np.asarray(lambda_C, dtype=np.float64)[neighbors] + gate * float(lambda_I_value) * np.asarray(info_field)[neighbors]
    weights = np.exp(-(cost - float(cost.min())))
    return neighbors, weights / float(weights.sum())


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
    neighbors, probs = local_transition_probabilities(w, k1, lambda_C, lambda_I_value, info_field, info_age, use_age_gate)
    return int(rng.choice(neighbors, p=probs))


def _information_cost(lambda_C: np.ndarray, lambda_I_value: float, info_field: np.ndarray) -> np.ndarray:
    return np.asarray(lambda_C, dtype=np.float64) + float(lambda_I_value) * np.asarray(info_field, dtype=np.float64)


def transition_matrix_from_information_field(
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    use_age_gate: bool = False,
) -> np.ndarray:
    """Build ``P(pi_bar)`` from a supplied information field."""
    if use_age_gate:
        transition = np.zeros((w.K, w.K), dtype=np.float64)
        for k in range(w.K):
            neighbors, probs = local_transition_probabilities(w, k, lambda_C, lambda_I_value, info_field, math.inf, False)
            transition[k, neighbors] = probs
        return transition
    return core.destination_weight_transition(w, _information_cost(lambda_C, lambda_I_value, info_field))


def stationary_distribution_from_information_field(
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
) -> np.ndarray:
    """Stationary distribution of the Layer 1-I kernel for fixed field."""
    return core.reversible_stationary_from_cost(w, _information_cost(lambda_C, lambda_I_value, info_field))


def stationary_distribution_from_transition(
    transition: np.ndarray,
    tol: float = 1.0e-13,
    max_iters: int = 100_000,
) -> np.ndarray:
    return core.power_stationary_distribution(transition, tol=tol, max_iters=max_iters)


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
    """Solve the Layer 1-I self-consistency equation.

    The information observable is the stationary field itself: 
    ``pi_bar = stationary(P(pi_bar))``.  Negative ``lambda_I`` rewards
    high-probability cells and can create the clustered branch used in the Stage I-C inverse-rate validation.
    """
    if use_cache is None:
        use_cache = initial_pi is None
    cache_key = (w.Nx, w.Ny, float(w.cell_size), float(lambda_I_value), lambda_C.tobytes(), int(seed))
    if use_cache and cache_key in _FIXED_POINT_CACHE:
        pi, transition, iters, err = _FIXED_POINT_CACHE[cache_key]
        return pi.copy(), transition.copy(), iters, err

    if initial_pi is None:
        pi = theoretical_stationary(w)
        if lambda_I_value < 0.0 and perturbation > 0.0:
            rng = np.random.default_rng(seed)
            pi = normalize_probability(pi + perturbation * rng.random(w.K))
    else:
        pi = normalize_probability(initial_pi)

    err = float("inf")
    transition = transition_matrix_from_information_field(w, lambda_C, lambda_I_value, pi)
    for iteration in range(1, max_iters + 1):
        stationary = stationary_distribution_from_information_field(w, lambda_C, lambda_I_value, pi)
        updated = normalize_probability((1.0 - damping) * pi + damping * stationary)
        err = float(np.max(np.abs(updated - pi)))
        pi = updated
        if err < tol:
            break
    else:
        iteration = max_iters

    pi = stationary_distribution_from_information_field(w, lambda_C, lambda_I_value, pi)
    transition = transition_matrix_from_information_field(w, lambda_C, lambda_I_value, pi)
    if use_cache:
        _FIXED_POINT_CACHE[cache_key] = (pi.copy(), transition.copy(), int(iteration), err)
    return pi, transition, int(iteration), float(err)


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
) -> dict[str, float | int | np.ndarray]:
    """Evaluate one candidate ``lambda_I`` for the inverse-rate problem.

    The returned rate is obtained in two steps: solve
    ``pi_bar = stationary(P(pi_bar))``, then insert
    ``p_enc=sum_k pi_bar_k^2`` into ``dn/dt=A beta p_enc f(1-f)``.
    """
    pi_bar, _, iterations, err = solve_information_fixed_point(
        w, lambda_C, lambda_I_value, seed=seed, initial_pi=initial_pi, use_cache=initial_pi is None
    )
    p_enc = encounter_probability_from_stationary(pi_bar)
    result: dict[str, float | int | np.ndarray] = {
        "lambda_I": float(lambda_I_value),
        "p_enc": p_enc,
        "information_spreading_rate": logistic_information_rate(informed_fraction, p_enc, beta, n_agents),
        "fixed_point_iterations": int(iterations),
        "fixed_point_error": float(err),
    }
    if return_pi:
        result["pi_bar"] = pi_bar.copy()
    return result


def continuation_lambda_grid(start: float, end: float, step: float) -> List[float]:
    """Inclusive monotone grid used to trace a fixed-point branch."""
    if step <= 0.0:
        raise ValueError("Continuation step must be positive.")
    sign = 1.0 if end >= start else -1.0
    values = [float(start)]
    current = float(start)
    while sign * (end - current) > step + 1.0e-12:
        current += sign * step
        values.append(float(current))
    if abs(values[-1] - end) > 1.0e-12:
        values.append(float(end))
    return values


def strip_fixed_point_sample(sample: dict[str, float | int | np.ndarray]) -> dict[str, float | int]:
    return {key: value for key, value in sample.items() if key != "pi_bar"}  # type: ignore[return-value]


def trace_information_rate_branch(
    w: World,
    lambda_C: np.ndarray,
    lambda_values: Sequence[float],
    informed_fraction: float,
    branch_name: str,
    beta: float = BETA_TRANSMISSION,
    n_agents: int = N_ROBOTS,
    seed: int = SEED,
    restart_each_lambda: bool = False,
) -> List[dict[str, float | int | np.ndarray]]:
    """Follow one fixed-point branch by continuation in ``lambda_I``.

    ``coverage`` starts from the unbiased stationary law.  ``clustered`` starts
    from a localized seed, which lets the solver remain on the high-encounter
    branch when it exists.
    """
    if branch_name == "coverage":
        branch_seed: np.ndarray | None = theoretical_stationary(w)
    elif branch_name == "clustered":
        branch_seed = clustered_seed_distribution(w)
    else:
        raise ValueError("branch_name must be 'coverage' or 'clustered'.")

    current_pi = branch_seed
    samples: list[dict[str, float | int | np.ndarray]] = []
    for lambda_value in lambda_values:
        initial_pi = branch_seed if restart_each_lambda else current_pi
        sample = evaluate_information_rate_constraint(
            w=w,
            lambda_C=lambda_C,
            lambda_I_value=float(lambda_value),
            informed_fraction=informed_fraction,
            beta=beta,
            n_agents=n_agents,
            seed=seed,
            initial_pi=initial_pi,
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
    for idx in range(len(samples) - 1):
        left = float(samples[idx]["information_spreading_rate"]) - target_rate
        right = float(samples[idx + 1]["information_spreading_rate"]) - target_rate
        if left == 0.0:
            return idx, idx
        if right == 0.0 or left * right < 0.0:
            return idx, idx + 1
    return None


def information_rate_solution_score(
    sample: dict[str, float | int | np.ndarray],
    target_rate: float,
    relative_tolerance: float,
    reference_lambda_I: float,
) -> tuple[int, float, float]:
    """Rank inverse-rate candidates by target match, then distance from reference.

    The scalar Layer 1-I inverse problem can have a flat or nearly flat set of
    solutions when the fixed point is saturated.  In that case the useful
    controller is the minimum-change multiplier relative to the neutral
    baseline, not the extreme endpoint of the scanned range.
    """
    rate_error = abs(float(sample["information_spreading_rate"]) - float(target_rate))
    tolerance = float(relative_tolerance) * max(abs(float(target_rate)), 1.0e-12)
    lambda_distance = abs(float(sample["lambda_I"]) - float(reference_lambda_I))
    if rate_error <= tolerance:
        return (0, lambda_distance, rate_error)
    return (1, rate_error, lambda_distance)


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
    reference_lambda_I: float = 0.0,
) -> tuple[dict[str, float | int | np.ndarray], List[dict[str, float | int | np.ndarray]]]:
    if float(left_sample["lambda_I"]) > float(right_sample["lambda_I"]):
        left_sample, right_sample = right_sample, left_sample

    target = float(target_rate)
    left = dict(left_sample)
    right = dict(right_sample)
    history: list[dict[str, float | int | np.ndarray]] = []
    best = min(
        (left, right),
        key=lambda item: information_rate_solution_score(item, target, relative_tolerance, reference_lambda_I),
    )

    for _ in range(max_bisection_iters):
        if abs(float(right["lambda_I"]) - float(left["lambda_I"])) <= lambda_tolerance:
            break
        mid_lambda = 0.5 * (float(left["lambda_I"]) + float(right["lambda_I"]))
        mid_seed = normalize_probability(0.5 * np.asarray(left["pi_bar"]) + 0.5 * np.asarray(right["pi_bar"]))
        mid = evaluate_information_rate_constraint(
            w, lambda_C, mid_lambda, informed_fraction, beta, n_agents, seed, initial_pi=mid_seed, return_pi=True
        )
        mid["branch"] = left.get("branch", right.get("branch", "unknown"))
        history.append(mid)
        if information_rate_solution_score(mid, target, relative_tolerance, reference_lambda_I) < information_rate_solution_score(
            best, target, relative_tolerance, reference_lambda_I
        ):
            best = mid

        if (float(left["information_spreading_rate"]) - target) * (float(mid["information_spreading_rate"]) - target) <= 0.0:
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
    reference_lambda_I: float = 0.0,
) -> dict[str, object]:
    """Choose ``lambda_I`` on a fixed-point branch to match a rate target.

    The report follows Pinciroli by converting information diffusion into
    ``dn/dt = A beta p_enc f (1-f)``.  Once a candidate ``lambda_I`` gives a
    fixed point ``pi_bar``, the encounter proxy is ``p_enc = sum_k pi_bar[k]^2``.
    This function searches the coverage-like fixed point and a localized
    clustered fixed point at each candidate multiplier, then bisects a bracket
    when the sampled branch is locally continuous.
    If several multipliers match the same rate, it selects the minimum-change
    solution closest to ``reference_lambda_I``; the neutral default is
    ``lambda_I = 0``.
    """
    target = float(target_rate)
    scale = max(abs(target), 1.0e-12)
    tolerance = relative_tolerance * scale
    target_within_selected_branch = False

    zero = evaluate_information_rate_constraint(
        w, lambda_C, 0.0, informed_fraction, beta, n_agents, seed, initial_pi=theoretical_stationary(w), return_pi=True
    )
    zero["branch"] = "coverage"
    if abs(float(zero["information_spreading_rate"]) - target) <= tolerance:
        selected = zero
        branch_public = {"coverage": [strip_fixed_point_sample(zero)], "clustered": []}
        solver_method = "exact_zero_baseline"
        selected_branch = "coverage"
        used_bisection = False
        bracket_found = True
        evaluations = [strip_fixed_point_sample(zero)]
        target_within_range = True
        target_within_selected_branch = True
    else:
        coverage_trace = trace_information_rate_branch(
            w, lambda_C, continuation_lambda_grid(0.0, coverage_lambda_max, continuation_step),
            informed_fraction, "coverage", beta, n_agents, seed
        )
        clustered_values = sorted(
            set(
                continuation_lambda_grid(clustered_lambda_min, 0.0, continuation_step)
                + [-50.0, -40.0, -30.0, -25.0, -20.0, -17.5, -15.0, -12.5, -10.0, -5.0, 0.0]
            )
        )
        clustered_anchors = {
            "corner": 0,
            "edge": w.Nx // 2,
            "center": (w.Ny // 2) * w.Nx + (w.Nx // 2),
        }
        clustered_trace: list[dict[str, float | int | np.ndarray]] = []
        for anchor_name, anchor_cell in clustered_anchors.items():
            branch_seed = clustered_seed_distribution(w, anchor_cell)
            for lambda_value in clustered_values:
                sample = evaluate_information_rate_constraint(
                    w,
                    lambda_C,
                    float(lambda_value),
                    informed_fraction,
                    beta,
                    n_agents,
                    seed,
                    initial_pi=branch_seed,
                    return_pi=True,
                )
                sample["branch"] = "clustered"
                sample["branch_seed"] = anchor_name
                clustered_trace.append(sample)
        clustered_trace.sort(key=lambda item: (float(item["lambda_I"]), str(item.get("branch_seed", ""))))
        traces = {"coverage": coverage_trace, "clustered": clustered_trace}
        branch_public = {name: [strip_fixed_point_sample(sample) for sample in samples] for name, samples in traces.items()}

        preferred = "clustered" if target > float(zero["information_spreading_rate"]) else "coverage"
        ordered = [preferred, "coverage" if preferred == "clustered" else "clustered"] if branch == "auto" else [branch]
        if any(name not in traces for name in ordered):
            raise ValueError("branch must be 'auto', 'coverage', or 'clustered'.")

        selected = None
        selected_branch = ordered[0]
        bisection_history: list[dict[str, float | int | np.ndarray]] = []
        used_bisection = False
        bracket_found = False
        target_within_range = False
        target_within_any_branch = False
        target_within_selected_branch = False
        for name in ordered:
            samples = traces[name]
            rates = np.array([float(sample["information_spreading_rate"]) for sample in samples], dtype=np.float64)
            branch_in_range = bool(rates.min() - tolerance <= target <= rates.max() + tolerance)
            target_within_any_branch = target_within_any_branch or branch_in_range
            near = [sample for sample in samples if abs(float(sample["information_spreading_rate"]) - target) <= tolerance]
            if near:
                selected = min(
                    near,
                    key=lambda item: information_rate_solution_score(item, target, relative_tolerance, reference_lambda_I),
                )
                selected_branch = name
                target_within_selected_branch = branch_in_range
                break
            bracket_idx = find_branch_bracket(samples, target)
            if bracket_idx is not None:
                selected_branch = name
                bracket_found = True
                target_within_selected_branch = branch_in_range
                if bracket_idx[0] == bracket_idx[1]:
                    selected = samples[bracket_idx[0]]
                else:
                    selected, bisection_history = bisect_information_rate_on_branch(
                        w, lambda_C, target, informed_fraction, samples[bracket_idx[0]], samples[bracket_idx[1]],
                        beta, n_agents, seed, relative_tolerance, lambda_tolerance, max_bisection_iters, reference_lambda_I
                    )
                    used_bisection = True
                break
        target_within_range = target_within_any_branch

        solver_method = "branch_continuation_bisection"
        if selected is None:
            if candidate_lambdas is None:
                candidate_lambdas = (
                    -400.0,
                    -300.0,
                    -200.0,
                    -100.0,
                    -50.0,
                    -40.0,
                    -30.0,
                    -25.0,
                    -20.0,
                    -17.5,
                    -15.0,
                    -12.5,
                    -10.0,
                    -5.0,
                    0.0,
                    50.0,
                    100.0,
                    200.0,
                    400.0,
                )
            fallback: list[dict[str, float | int | np.ndarray]] = []
            fallback_initials = {
                "coverage": theoretical_stationary(w),
                "clustered_corner": clustered_seed_distribution(w, 0),
                "clustered_edge": clustered_seed_distribution(w, w.Nx // 2),
                "clustered_center": clustered_seed_distribution(w, (w.Ny // 2) * w.Nx + (w.Nx // 2)),
            }
            for value in candidate_lambdas:
                for fallback_branch, initial_pi in fallback_initials.items():
                    sample = evaluate_information_rate_constraint(
                        w, lambda_C, value, informed_fraction, beta, n_agents, seed, initial_pi=initial_pi, return_pi=True
                    )
                    sample["branch"] = "coverage" if fallback_branch == "coverage" else "clustered"
                    sample["branch_seed"] = fallback_branch
                    fallback.append(sample)
            selected = min(
                fallback,
                key=lambda item: information_rate_solution_score(item, target, relative_tolerance, reference_lambda_I),
            )
            selected_branch = "fallback_scan"
            branch_public[selected_branch] = [strip_fixed_point_sample(sample) for sample in fallback]
            solver_method = "candidate_scan_fallback"
            bisection_history = []
            target_within_range = False
            target_within_selected_branch = False
        evaluations = branch_public.get(selected_branch, []) + [strip_fixed_point_sample(sample) for sample in bisection_history]

    assert selected is not None
    selected_rate = float(selected["information_spreading_rate"])
    selected_p_enc = float(selected["p_enc"])
    target_p_enc = target / max(float(n_agents) * beta * informed_fraction * (1.0 - informed_fraction), 1.0e-12)
    return {
        "target_information_spreading_rate": target,
        "target_p_enc": float(target_p_enc),
        "informed_fraction": float(informed_fraction),
        "beta": float(beta),
        "n_agents": int(n_agents),
        "reference_lambda_I": float(reference_lambda_I),
        "selection_rule": "minimize |lambda_I - reference_lambda_I| among target-matching fixed-point solutions",
        "lambda_I": float(selected["lambda_I"]),
        "predicted_information_spreading_rate": selected_rate,
        "predicted_p_enc": selected_p_enc,
        "absolute_rate_error": abs(selected_rate - target),
        "relative_rate_error": abs(selected_rate - target) / scale,
        "target_within_scanned_range": bool(target_within_range),
        "target_within_branch_range": bool(target_within_selected_branch),
        "converged": bool(abs(selected_rate - target) <= tolerance),
        "fixed_point_iterations": int(selected["fixed_point_iterations"]),
        "fixed_point_error": float(selected["fixed_point_error"]),
        "solver_method": solver_method,
        "selected_branch": selected_branch,
        "used_bisection": bool(used_bisection),
        "bracket_found": bool(bracket_found),
        "evaluations": evaluations,
        "branch_evaluations": branch_public,
        "selected_pi_bar": np.asarray(selected["pi_bar"], dtype=np.float64).copy(),
    }


def ensure_robot_map(r: Robot, w: World, t: float = 0.0) -> RobotWorldMap:
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
    k0 = int(rng.integers(0, w.K))
    x, y = region_center(w, k0)
    age0 = float(rng.uniform(0.0, A_HALF)) if fixed_initial_info_age is None else float(fixed_initial_info_age)
    robot_map = RobotWorldMap.empty(w.K)
    robot_map.observe_cell(k0, 0.0)
    return Robot(idx, x, y, k0, k0, x, y, age0, 0, robot_map)


def initialise_targets(
    robots: Sequence[Robot],
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    rng: np.random.Generator,
    info_field: np.ndarray | None = None,
    use_age_gate: bool = USE_AGE_GATE_FOR_CONTROL,
) -> None:
    field = compute_information_field(w, robots) if info_field is None else info_field
    for robot in robots:
        robot.to_k = sample_next_region(w, robot.from_k, lambda_C, lambda_I_value, field, robot.info_age, rng, use_age_gate)
        robot.tx, robot.ty = region_center(w, robot.to_k)


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
    if not freeze_info_age:
        r.info_age += 1.0
    r.x, r.y = core.move_toward(r.x, r.y, r.tx, r.ty, speed)
    if abs(r.x - r.tx) >= 1.0e-9 or abs(r.y - r.ty) >= 1.0e-9:
        return
    r.from_k = r.to_k
    markov_visits[r.from_k] += 1
    coverage_age_field[r.from_k] = float(t)
    ensure_robot_map(r, w).observe_cell(r.from_k, float(t))
    r.to_k = sample_next_region(w, r.from_k, lambda_C, lambda_I_value, info_field, r.info_age, rng, use_age_gate)
    r.tx, r.ty = region_center(w, r.to_k)


def detect_meetings(robots: List[Robot], r_meet: float, t: int) -> int:
    count = 0
    for i, first in enumerate(robots):
        for second in robots[i + 1:]:
            if (first.x - second.x) ** 2 + (first.y - second.y) ** 2 <= r_meet * r_meet:
                first.info_age = second.info_age = 0.0
                first.last_meet = second.last_meet = t
                count += 1
    return count


def same_region_pairs(robots: Sequence[Robot], w: World) -> List[Tuple[int, int, int, float, float]]:
    by_cell: list[list[int]] = [[] for _ in range(w.K)]
    for idx, robot in enumerate(robots):
        by_cell[position_to_cell(w, robot.x, robot.y)].append(idx)
    pairs: list[Tuple[int, int, int, float, float]] = []
    for cell, members in enumerate(by_cell):
        cx, cy = region_center(w, cell)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.append((members[a], members[b], cell, cx, cy))
    return pairs


def exchange_robot_world_maps(robots: List[Robot], w: World, i: int, j: int, t: int) -> int:
    first_updates, second_updates = exchange_maps(ensure_robot_map(robots[i], w), ensure_robot_map(robots[j], w), float(t))
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
    encounters = communications = transmissions = map_updates = 0
    for i, j, _, _, _ in same_region_pairs(robots, w):
        if rng.random() > p_encounter_given_colocation:
            continue
        encounters += 1
        if rng.random() > beta:
            continue
        communications += 1
        robots[i].info_age = robots[j].info_age = 0.0
        robots[i].last_meet = robots[j].last_meet = t
        map_updates += exchange_robot_world_maps(robots, w, i, j, t)
        if informed is not None:
            before_i, before_j = bool(informed[i]), bool(informed[j])
            if before_i or before_j:
                informed[i] = True
                informed[j] = True
            transmissions += int(before_i != before_j)
    return encounters, communications, transmissions, map_updates


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
    """Run the Layer 1-I physical simulation.

    In ``stationary_fixed_point`` mode the robots move under the solved
    ``pi_bar`` field, which isolates Pinciroli's theoretical prediction.  In
    ``online_density`` mode the field is recomputed from current occupancy and
    is retained as an exploratory diagnostic, not the report baseline.
    """
    rng = np.random.default_rng(seed)
    w = build_world(NX, NY, CELL_SIZE)
    lambda_C = build_lambda_C(w, lambda_C_val)
    coverage_reference = theoretical_stationary(w)
    if information_field_mode == "stationary_fixed_point":
        control_field, _, _, _ = solve_information_fixed_point(w, lambda_C, lambda_I_value, seed=seed)
    elif information_field_mode == "online_density":
        control_field = np.full(w.K, 1.0 / w.K, dtype=np.float64)
    else:
        raise ValueError("information_field_mode must be 'stationary_fixed_point' or 'online_density'.")

    robots = [make_robot(i, w, rng, fixed_initial_info_age) for i in range(n_robots)]
    initialise_targets(robots, w, lambda_C, lambda_I_value, rng, control_field, use_age_gate)
    markov_visits = np.zeros(w.K, dtype=np.int64)
    last_visit = np.full(w.K, -1.0, dtype=np.float64)
    for robot in robots:
        last_visit[robot.from_k] = 0.0

    ck_history: list[np.ndarray] = []
    markov_history: list[int] = []
    snapshots: list[Tuple[np.ndarray, np.ndarray, int]] = []
    t_axis: list[int] = []
    dispersion_log: list[float] = []
    cov_age_log: list[float] = []
    info_age_log: list[float] = []
    local_cov_age_log: list[float] = []
    map_age_log: list[float] = []
    meetings_log: list[int] = []
    encounter_log: list[float] = []
    coverage_l1_log: list[float] = []
    field_acc = np.zeros(w.K, dtype=np.float64)
    final_field = control_field.copy()
    meetings_since_log = 0

    for t in range(1, T + 1):
        observed_field = compute_information_field(w, robots)
        field = control_field if information_field_mode == "stationary_fixed_point" else observed_field
        field_acc += field
        final_field = field
        for robot in robots:
            step_robot(robot, w, lambda_C, lambda_I_value, field, speed, markov_visits, last_visit, t, rng, freeze_info_age, use_age_gate)

        if enable_meetings:
            encounters, _, _, _ = perform_information_exchanges(robots, w, rng, t)
            meetings_since_log += encounters

        if t % RECORD_EVERY == 0:
            xs = np.array([robot.x for robot in robots], dtype=np.float64)
            ys = np.array([robot.y for robot in robots], dtype=np.float64)
            occ = occupancy_distribution(w, robots)
            arrivals = max(int(markov_visits.sum()), 1)
            ck = markov_visits / arrivals
            maps = [ensure_robot_map(robot, w) for robot in robots]
            t_axis.append(t)
            dispersion_log.append(float(np.var(xs) + np.var(ys)))
            cov_age_log.append(float(core.coverage_age(last_visit, t).mean()))
            info_age_log.append(float(np.mean([robot.info_age for robot in robots])))
            local_cov_age_log.append(mean_robot_coverage_age(maps, float(t)))
            map_age_log.append(mean_robot_map_record_age(maps, float(t)))
            meetings_log.append(meetings_since_log)
            encounter_log.append(encounter_proxy_from_occupancy(occ))
            coverage_l1_log.append(core.l1_error(ck, coverage_reference))
            if int(markov_visits.sum()) > 0:
                ck_history.append(ck.copy())
                markov_history.append(int(markov_visits.sum()))
            meetings_since_log = 0

        if t % SNAP_EVERY == 0:
            snapshots.append((
                np.array([robot.x for robot in robots], dtype=np.float64),
                np.array([robot.y for robot in robots], dtype=np.float64),
                t,
            ))

    arrivals = max(int(markov_visits.sum()), 1)
    return SimResult(
        w=w,
        lambda_I_value=float(lambda_I_value),
        pi_empirical=markov_visits / arrivals,
        mean_info_field=field_acc / float(max(T, 1)),
        final_info_field=final_field,
        ck_history=ck_history,
        markov_step_history=markov_history,
        pos_snapshots=snapshots,
        t_axis=np.array(t_axis, dtype=np.int64),
        dispersion=np.array(dispersion_log, dtype=np.float64),
        mean_cov_age=np.array(cov_age_log, dtype=np.float64),
        mean_info_age=np.array(info_age_log, dtype=np.float64),
        mean_local_cov_age=np.array(local_cov_age_log, dtype=np.float64),
        mean_map_record_age=np.array(map_age_log, dtype=np.float64),
        meetings_per_snap=np.array(meetings_log, dtype=np.float64),
        encounter_proxy=np.array(encounter_log, dtype=np.float64),
        coverage_l1=np.array(coverage_l1_log, dtype=np.float64),
    )


def dominant_frequency(t_axis: np.ndarray, signal: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if len(signal) < 8:
        return 0.0, 0.0, np.array([]), np.array([])
    dt = float(t_axis[1] - t_axis[0])
    centered = signal - signal.mean()
    power = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    if len(power) <= 2:
        return 0.0, 0.0, freqs, power
    idx = int(np.argmax(power[1:])) + 1
    floor = float(np.median(power[1:]))
    return float(freqs[idx]), float(power[idx] / floor) if floor > 0.0 else float("inf"), freqs, power


def make_main_figure(res: SimResult) -> plt.Figure:
    w = res.w
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3)
    panels = [
        (res.mean_info_field, "mean information field", "magma"),
        (res.pi_empirical, "empirical visits", "viridis"),
        (res.final_info_field, "final control field", "magma"),
    ]
    for col, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data.reshape(w.Ny, w.Nx), origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, :2])
    ax.plot(res.t_axis, res.dispersion, label="dispersion", color="navy")
    ax2 = ax.twinx()
    ax2.plot(res.t_axis, res.mean_info_age, label="mean AoI", color="crimson")
    ax2.plot(res.t_axis, res.mean_cov_age, label="coverage age", color="seagreen")
    ax.set_xlabel("simulation step")
    ax.set_ylabel("dispersion")
    ax2.set_ylabel("age")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(res.t_axis, res.encounter_proxy, color="darkorange", label="encounter proxy")
    if len(res.coverage_l1):
        ax2 = ax.twinx()
        ax2.plot(res.t_axis, res.coverage_l1, color="slateblue", alpha=0.8, label="coverage L1")
    ax.set_xlabel("simulation step")
    ax.set_title("encounters and coverage error")
    fig.suptitle(f"MaxCal information diffusion (lambda_I={res.lambda_I_value:+.1f})")
    fig.tight_layout()
    return fig


def make_age_plane_figure(res: SimResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 6))
    if len(res.t_axis):
        sc = ax.scatter(res.mean_cov_age, res.mean_info_age, c=res.t_axis, cmap="viridis", s=8)
        ax.plot(res.mean_cov_age, res.mean_info_age, color="grey", lw=0.5, alpha=0.6)
        fig.colorbar(sc, ax=ax, label="simulation step")
    ax.set_xlabel("mean coverage age")
    ax.set_ylabel("mean AoI")
    ax.set_title("age-plane trajectory")
    fig.tight_layout()
    return fig


def make_constraint_sweep_figure(values: Sequence[float] = LAMBDA_I_SWEEP, T: int = 9_000):
    results = [
        run_simulation(lambda_I_value=value, T=T, fixed_initial_info_age=SWEEP_STALE_AGE, freeze_info_age=True, enable_meetings=False)
        for value in values
    ]
    values_arr = np.array(values, dtype=np.float64)
    mean_disp = np.array([res.dispersion.mean() for res in results], dtype=np.float64)
    mean_enc = np.array([res.encounter_proxy.mean() for res in results], dtype=np.float64)
    final_l1 = np.array([res.coverage_l1[-1] if len(res.coverage_l1) else np.nan for res in results], dtype=np.float64)
    zero_idx = int(np.where(np.isclose(values_arr, 0.0))[0][0])
    summary = []
    for value, res, disp, enc, l1 in zip(values_arr, results, mean_disp, mean_enc, final_l1):
        regime = "cluster" if disp < mean_disp[zero_idx] and enc > mean_enc[zero_idx] else "coverage"
        freq, prom, _, _ = dominant_frequency(res.t_axis, res.dispersion)
        summary.append((float(value), float(disp), float(enc), float(l1), float(freq), float(prom), regime))

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes[0, 0].plot(values_arr, mean_disp, marker="o")
    axes[0, 0].set_title("mean dispersion")
    axes[0, 1].plot(values_arr, mean_enc, marker="o", color="darkorange")
    axes[0, 1].set_title("mean encounter proxy")
    axes[1, 0].plot(values_arr, final_l1, marker="o", color="slateblue")
    axes[1, 0].set_title("final coverage L1")
    axes[1, 1].scatter(mean_disp, mean_enc, c=["crimson" if v < 0 else "seagreen" for v in values_arr])
    for v, x, y in zip(values_arr, mean_disp, mean_enc):
        axes[1, 1].annotate(f"{v:+.0f}", (x, y), fontsize=8)
    axes[1, 1].set_title("regime map")
    for ax in axes.flat:
        ax.set_xlabel("lambda_I" if ax is not axes[1, 1] else "mean dispersion")
    axes[1, 1].set_ylabel("mean encounter proxy")
    fig.tight_layout()
    return fig, summary


def make_animation(res: SimResult, fps: int = 12, filename: str = "maxcal_info_diffusion.gif") -> None:
    if not res.pos_snapshots:
        return
    w = res.w
    extent = (0.0, w.Nx * w.cell_size, 0.0, w.Ny * w.cell_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        res.mean_info_field.reshape(w.Ny, w.Nx),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="magma",
        alpha=0.45,
    )
    scat = ax.scatter([], [], s=14, color="navy", alpha=0.8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    title = ax.set_title("")

    def update(frame: int):
        xs, ys, t = res.pos_snapshots[frame]
        scat.set_offsets(np.column_stack([xs, ys]))
        title.set_text(f"t = {t}")
        return scat, title

    anim = animation.FuncAnimation(fig, update, frames=len(res.pos_snapshots), interval=1000 // fps, blit=False)
    try:
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
    finally:
        plt.close(fig)


def sweep_rows_to_dicts(summary: Sequence[tuple[float, float, float, float, float, float, str]]) -> list[dict[str, Any]]:
    return [
        {
            "lambda_I": float(row[0]),
            "mean_dispersion": float(row[1]),
            "mean_encounter_proxy": float(row[2]),
            "final_coverage_l1": float(row[3]),
            "dominant_frequency": float(row[4]),
            "spectral_prominence": float(row[5]),
            "regime": row[6],
        }
        for row in summary
    ]


def run_summary(res: SimResult, fixed_point: dict[str, float | int | np.ndarray]) -> dict[str, Any]:
    return {
        "lambda_I": res.lambda_I_value,
        "total_markov_arrivals": int(res.markov_step_history[-1]) if res.markov_step_history else 0,
        "fixed_point": {
            "p_enc": float(fixed_point["p_enc"]),
            "predicted_information_spreading_rate_at_one_seed": float(fixed_point["information_spreading_rate"]),
            "iterations": int(fixed_point["fixed_point_iterations"]),
            "error": float(fixed_point["fixed_point_error"]),
        },
        "simulation": {
            "mean_dispersion": float(res.dispersion.mean()) if len(res.dispersion) else float("nan"),
            "mean_encounter_proxy": float(res.encounter_proxy.mean()) if len(res.encounter_proxy) else float("nan"),
            "mean_coverage_age": float(res.mean_cov_age.mean()) if len(res.mean_cov_age) else float("nan"),
            "mean_information_age": float(res.mean_info_age.mean()) if len(res.mean_info_age) else float("nan"),
            "final_coverage_l1": float(res.coverage_l1[-1]) if len(res.coverage_l1) else float("nan"),
        },
    }


def write_summary(
    outdir: Path,
    result: SimResult,
    fixed_point: dict[str, float | int | np.ndarray],
    sweep_rows: list[dict[str, Any]],
    figures: list[str],
) -> dict[str, Any]:
    zero = next((row for row in sweep_rows if abs(row["lambda_I"]) < 1.0e-12), None)
    strongest_negative = min(sweep_rows, key=lambda row: row["lambda_I"])
    strongest_positive = max(sweep_rows, key=lambda row: row["lambda_I"])
    checks = {
        "fixed_point_converged": bool(float(fixed_point["fixed_point_error"]) < 1.0e-8),
        "negative_lambda_increases_encounter_over_zero": bool(
            zero is not None and strongest_negative["mean_encounter_proxy"] > zero["mean_encounter_proxy"]
        ),
        "positive_lambda_does_not_exceed_negative_encounter": bool(
            strongest_positive["mean_encounter_proxy"] <= strongest_negative["mean_encounter_proxy"]
        ),
        "main_run_completed": bool(result.markov_step_history),
    }
    payload = {
        "paper_alignment": {
            "layer": "Layer 1-I information diffusion",
            "claim": "The information multiplier reshapes the stationary field to tune encounter probability, which controls the mean-field information spreading rate.",
            "maxcal_kernel": "P[i,j] proportional to exp(-lambda_C[j] - lambda_I pi_bar[j]) with pi_bar = stationary(P).",
            "rate_model": "dn/dt = A beta p_enc f (1 - f)",
        },
        "environment": {
            "Nx": NX,
            "Ny": NY,
            "K": result.w.K,
            "cell_size": CELL_SIZE,
            "robots": N_ROBOTS,
            "beta_transmission": BETA_TRANSMISSION,
            "record_every": RECORD_EVERY,
            "information_field_mode": INFORMATION_FIELD_MODE,
        },
        "main_run": run_summary(result, fixed_point),
        "lambda_I_sweep": sweep_rows,
        "checks": checks,
        "information_layer_ready": bool(all(checks.values())),
        "figures": figures,
    }
    with open(outdir / "maxcal_info_diffusion_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    return payload


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    result = run_simulation(T=args.T, lambda_I_value=args.lambda_I, seed=args.seed)
    world = build_world(NX, NY, CELL_SIZE)
    lambda_c = build_lambda_C(world, LAMBDA_C_VAL)
    fixed_point = evaluate_information_rate_constraint(
        world,
        lambda_c,
        args.lambda_I,
        informed_fraction=1.0 / float(N_ROBOTS),
        seed=args.seed,
        return_pi=False,
    )

    figures = [
        "maxcal_info_diffusion_main.png",
        "maxcal_info_diffusion_age_plane.png",
        "maxcal_info_diffusion_sweep.png",
    ]
    fig = make_main_figure(result)
    fig.savefig(outdir / figures[0], dpi=140)
    plt.close(fig)
    fig = make_age_plane_figure(result)
    fig.savefig(outdir / figures[1], dpi=140)
    plt.close(fig)
    if not args.no_animation:
        make_animation(result, filename=str(outdir / "maxcal_info_diffusion.gif"))
        figures.append("maxcal_info_diffusion.gif")
    fig, summary = make_constraint_sweep_figure(T=args.sweep_T)
    fig.savefig(outdir / "maxcal_info_diffusion_sweep.png", dpi=140)
    plt.close(fig)
    sweep_rows = sweep_rows_to_dicts(summary)
    payload = write_summary(outdir, result, fixed_point, sweep_rows, figures)

    print("MaxCal Information Diffusion (Layer 1-I)")
    print(f"  Output directory       : {outdir}")
    print(f"  lambda_I               : {args.lambda_I:+.3f}")
    print(f"  Fixed-point p_enc      : {fixed_point['p_enc']:.6f}")
    print(f"  Predicted dn/dt        : {fixed_point['information_spreading_rate']:.6f}")
    print(f"  Mean dispersion        : {payload['main_run']['simulation']['mean_dispersion']:.3f}")
    print(f"  Mean encounter proxy   : {payload['main_run']['simulation']['mean_encounter_proxy']:.5f}")
    print(f"  Information layer ready: {payload['information_layer_ready']}")
    print("  Saved summary          : maxcal_info_diffusion_summary.json")


if __name__ == "__main__":
    main()
