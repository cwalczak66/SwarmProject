"""RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Hierarchical MaxCal controller.

Layer 1 supplies two fixed behavior kernels:
* coverage: the inverse coverage controller from ``maxcal_coverage.py``;
* diffusion: the fixed-point/inverse-rate controller from ``maxcal_info_diffusion.py``.

Layer 2 chooses the active behavior by another MaxCal softmax over modes ``m in {coverage, diffusion}``:

    Pr(m | x_t) = exp[-L_m(x_t)] / sum_q exp[-L_q(x_t)].

The state ``x_t`` is the pair of macroscopic ages used in the report:
coverage age ``A_C`` and Age of Information ``A_I``.  The linear baseline uses

    L_m = lambda_C a_C^+(m) + lambda_I a_I^+(m) + lambda_switch 1[m != previous],

where ``a_C^+`` and ``a_I^+`` are predicted next ages after one decision interval.  

The quadratic controller adds

    + lambda_C2 (a_C^+(m))^2 + lambda_I2 (a_I^+(m))^2,

which makes the mode log-odds state-dependent and is the intended oscillatory
mechanism tested in Stage III.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Literal, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import maxcal_core as core
import maxcal_coverage as coverage_layer
import maxcal_info_diffusion as info_layer
from maxcal_local_maps import RobotWorldMap, exchange_maps, mean_robot_map_record_age


Mode = Literal["coverage", "diffusion"]
ConstraintForm = Literal["linear", "quadratic"]
World = core.World


@dataclass
class HierarchicalConfig:
    controller_form: ConstraintForm = "linear"
    Nx: int = 20
    Ny: int = 20
    cell_size: float = 1.0
    n_robots: int = 50
    speed: float = 0.15
    T_sim: int = 18_000
    record_every: int = 30
    snap_every: int = 300
    seed: int = 42

    coverage_target_mode: str = "uniform"
    diffusion_layer1_mode: str = "inverse_rate"
    lambda_I_layer1: float = -400.0
    diffusion_target_rate: float | None = None
    diffusion_target_p_enc: float = 0.50
    diffusion_inverse_branch: str = "clustered"
    diffusion_continuation_step: float = 10.0
    beta_transmission: float = 0.80
    p_encounter_given_colocation: float = 1.0

    mode_decision_interval: int = 180
    coverage_age_scale: float = 120.0
    information_age_scale: float = 70.0
    coverage_rate: float | None = None
    information_rate: float | None = None
    auto_calibrate_layer2_rates: bool = True
    rate_calibration_T: int = 2_400
    lambda_coverage_linear: float = 1.0
    lambda_information_linear: float = 1.0
    lambda_coverage_quadratic: float = 2.0
    lambda_information_quadratic: float = 2.0
    lambda_switch: float = 0.0
    initial_mode: Mode = "coverage"
    initial_info_age_max: float = 8.0
    stem: str = "maxcal_hierarchical"


@dataclass
class Robot:
    id: int
    x: float
    y: float
    from_k: int
    to_k: int
    tx: float
    ty: float
    info_age: float
    last_meet: int = 0
    world_map: RobotWorldMap | None = None


@dataclass
class Layer1Kernels:
    coverage_transition: np.ndarray
    diffusion_transition: np.ndarray
    coverage_reference: np.ndarray
    diffusion_reference: np.ndarray
    coverage_lambda_C: np.ndarray
    diffusion_target_mode: str
    diffusion_lambda_I: float
    diffusion_solver_method: str
    diffusion_selected_branch: str
    diffusion_target_rate: float | None
    diffusion_target_p_enc: float | None
    diffusion_predicted_rate: float | None
    diffusion_predicted_p_enc: float | None
    diffusion_converged: bool
    coverage_solver_iterations: int | None
    coverage_solver_l1_error: float | None
    diffusion_fixed_point_iterations: int
    diffusion_fixed_point_error: float


@dataclass
class SwitchEvent:
    t: int
    new_mode: Mode
    p_coverage: float
    p_diffusion: float
    mean_info_age: float
    mean_cov_age: float
    encounter_proxy: float
    meetings_since_switch: int


@dataclass
class SimResult:
    config: HierarchicalConfig
    forced_mode: Mode | None
    w: World
    kernels: Layer1Kernels
    total_arrivals: int
    pi_empirical: np.ndarray
    pi_reference: np.ndarray
    mean_coverage_field: np.ndarray
    mean_information_field: np.ndarray
    final_coverage_field: np.ndarray
    final_information_field: np.ndarray
    t_axis: np.ndarray
    dispersion: np.ndarray
    mean_cov_age: np.ndarray
    mean_info_age: np.ndarray
    mean_local_map_record_age: np.ndarray
    encounter_proxy: np.ndarray
    meetings_per_log: np.ndarray
    communications_per_log: np.ndarray
    coverage_reference_l1: np.ndarray
    mode_indicator: np.ndarray
    mode_prob_coverage: np.ndarray
    mode_prob_diffusion: np.ndarray
    mode_names: List[Mode]
    switch_events: List[SwitchEvent]
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int, Mode]]


def build_world(Nx: int, Ny: int, cell_size: float) -> World:
    return core.build_grid_world(Nx, Ny, cell_size)


def region_center(w: World, k: int) -> Tuple[float, float]:
    return core.region_center(w, k)


def position_to_cell(w: World, x: float, y: float) -> int:
    return core.position_to_cell(w, x, y)


def mode_to_indicator(mode: Mode) -> int:
    return 0 if mode == "coverage" else 1


def build_layer1_kernels(cfg: HierarchicalConfig) -> Layer1Kernels:
    """Build the two Layer 1 kernels used by the Layer 2 selector.

    Coverage mode uses the inverse coverage kernel by default.
    Diffusion mode uses either a fixed ``lambda_I`` or the inverse-rate solve from ``maxcal_info_diffusion.py``.
    """
    w = build_world(cfg.Nx, cfg.Ny, cfg.cell_size)
    coverage_controller = coverage_layer.build_coverage_controller(w, target_mode=cfg.coverage_target_mode)
    lambda_C_info = info_layer.build_lambda_C(w, info_layer.LAMBDA_C_VAL)

    target_rate: float | None = None
    target_p_enc: float | None = None
    predicted_rate: float | None = None
    predicted_p_enc: float | None = None
    solver_method = "fixed_lambda"
    selected_branch = "manual"
    converged = True

    if cfg.diffusion_layer1_mode == "inverse_rate":
        informed_fraction = 1.0 / max(float(cfg.n_robots), 1.0)
        target_p_enc = float(cfg.diffusion_target_p_enc)
        if cfg.diffusion_target_rate is None:
            target_rate = info_layer.logistic_information_rate(
                informed_fraction, target_p_enc, cfg.beta_transmission, cfg.n_robots
            )
        else:
            target_rate = float(cfg.diffusion_target_rate)
            denom = max(float(cfg.n_robots) * cfg.beta_transmission * informed_fraction * (1.0 - informed_fraction), 1.0e-12)
            target_p_enc = target_rate / denom

        inverse = info_layer.solve_lambda_I_for_information_rate(
            w=w,
            lambda_C=lambda_C_info,
            target_rate=target_rate,
            informed_fraction=informed_fraction,
            beta=cfg.beta_transmission,
            n_agents=cfg.n_robots,
            seed=cfg.seed,
            branch=cfg.diffusion_inverse_branch,
            continuation_step=cfg.diffusion_continuation_step,
        )
        lambda_I = float(inverse["lambda_I"])
        pi_info = np.asarray(inverse["selected_pi_bar"], dtype=np.float64)
        transition = info_layer.transition_matrix_from_information_field(w, lambda_C_info, lambda_I, pi_info)
        solver_method = str(inverse["solver_method"])
        selected_branch = str(inverse["selected_branch"])
        predicted_rate = float(inverse["predicted_information_spreading_rate"])
        predicted_p_enc = float(inverse["predicted_p_enc"])
        converged = bool(inverse["converged"])
        info_iters = int(inverse["fixed_point_iterations"])
        info_err = float(inverse["fixed_point_error"])
    else:
        lambda_I = float(cfg.lambda_I_layer1)
        pi_info, transition, info_iters, info_err = info_layer.solve_information_fixed_point(
            w, lambda_C_info, lambda_I, seed=cfg.seed
        )

    coverage_solution = coverage_controller.inverse_solution
    return Layer1Kernels(
        coverage_transition=coverage_controller.transition,
        diffusion_transition=transition,
        coverage_reference=coverage_controller.target_pi,
        diffusion_reference=pi_info,
        coverage_lambda_C=coverage_controller.lambda_C,
        diffusion_target_mode=cfg.diffusion_layer1_mode,
        diffusion_lambda_I=lambda_I,
        diffusion_solver_method=solver_method,
        diffusion_selected_branch=selected_branch,
        diffusion_target_rate=target_rate,
        diffusion_target_p_enc=target_p_enc,
        diffusion_predicted_rate=predicted_rate,
        diffusion_predicted_p_enc=predicted_p_enc,
        diffusion_converged=converged,
        coverage_solver_iterations=None if coverage_solution is None else coverage_solution.iterations,
        coverage_solver_l1_error=None if coverage_solution is None else coverage_solution.l1_stationary_error,
        diffusion_fixed_point_iterations=info_iters,
        diffusion_fixed_point_error=info_err,
    )


def make_robot(idx: int, w: World, rng: np.random.Generator, initial_info_age_max: float) -> Robot:
    k0 = int(rng.integers(0, w.K))
    x, y = region_center(w, k0)
    robot_map = RobotWorldMap.empty(w.K)
    robot_map.observe_cell(k0, 0.0)
    return Robot(idx, x, y, k0, k0, x, y, float(rng.uniform(0.0, initial_info_age_max)), 0, robot_map)


def ensure_robot_map(robot: Robot, w: World, t: float = 0.0) -> RobotWorldMap:
    if robot.world_map is None:
        robot.world_map = RobotWorldMap.empty(w.K)
        robot.world_map.observe_cell(robot.from_k, t)
    return robot.world_map


def occupancy_distribution(w: World, robots: Sequence[Robot]) -> np.ndarray:
    return core.occupancy_distribution(w, [(robot.x, robot.y) for robot in robots])


def _normalize_field(field: np.ndarray) -> np.ndarray:
    max_val = float(field.max())
    return field / max_val if max_val > 0.0 else field


def coverage_age_array(coverage_last_visit: np.ndarray, t: int) -> np.ndarray:
    """Global coverage-age field ``kappa_k(t)=t-T_k``."""
    return core.coverage_age(coverage_last_visit, float(t))


def compute_coverage_field(w: World, coverage_last_visit: np.ndarray, t: int) -> np.ndarray:
    """Normalized local average of coverage age over each cell neighborhood."""
    cell_age = coverage_age_array(coverage_last_visit, t)
    field = np.zeros(w.K, dtype=np.float64)
    for cell, neighbors in enumerate(w.adjacency):
        field[cell] = float(np.mean(cell_age[[cell, *neighbors]]))
    return _normalize_field(field)


def compute_information_field(w: World, robots: Sequence[Robot]) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized local occupancy density used as the online information field."""
    occ = occupancy_distribution(w, robots)
    field = np.zeros(w.K, dtype=np.float64)
    for cell, neighbors in enumerate(w.adjacency):
        field[cell] = float(occ[[cell, *neighbors]].sum())
    return _normalize_field(field), occ


def encounter_proxy_from_occupancy(occ: np.ndarray) -> float:
    return float(np.sum(np.asarray(occ, dtype=np.float64) ** 2))


def sample_next_region_from_transition(w: World, k1: int, transition: np.ndarray, rng: np.random.Generator) -> int:
    return core.sample_from_transition(w, transition, k1, rng)


def transition_for_mode(kernels: Layer1Kernels, mode: Mode) -> np.ndarray:
    return kernels.coverage_transition if mode == "coverage" else kernels.diffusion_transition


def initialise_targets(
    w: World,
    robots: Sequence[Robot],
    kernels: Layer1Kernels,
    mode: Mode,
    rng: np.random.Generator,
) -> None:
    transition = transition_for_mode(kernels, mode)
    for robot in robots:
        robot.to_k = sample_next_region_from_transition(w, robot.from_k, transition, rng)
        robot.tx, robot.ty = region_center(w, robot.to_k)


def same_region_pairs(w: World, robots: Sequence[Robot]) -> List[Tuple[int, int, int]]:
    by_cell: list[list[int]] = [[] for _ in range(w.K)]
    for idx, robot in enumerate(robots):
        by_cell[position_to_cell(w, robot.x, robot.y)].append(idx)
    pairs: list[Tuple[int, int, int]] = []
    for cell, members in enumerate(by_cell):
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                pairs.append((members[a], members[b], cell))
    return pairs


def perform_information_exchanges(
    cfg: HierarchicalConfig,
    w: World,
    robots: List[Robot],
    rng: np.random.Generator,
    t: int,
) -> Tuple[int, int, int]:
    """Apply Bernoulli communication and reset paper AoI on success."""
    encounters = communications = map_records = 0
    for i, j, _ in same_region_pairs(w, robots):
        if rng.random() > cfg.p_encounter_given_colocation:
            continue
        encounters += 1
        if rng.random() > cfg.beta_transmission:
            continue
        communications += 1
        robots[i].info_age = robots[j].info_age = 0.0
        robots[i].last_meet = robots[j].last_meet = t
        first, second = exchange_maps(ensure_robot_map(robots[i], w), ensure_robot_map(robots[j], w), float(t))
        map_records += int(first + second)
    return encounters, communications, map_records


def observables(
    t: int,
    coverage_last_visit: np.ndarray,
    robots: Sequence[Robot],
    occ: np.ndarray,
) -> Tuple[float, float, float]:
    """Return ``(mean coverage age, mean AoI, encounter proxy)``."""
    return (
        float(np.mean(coverage_age_array(coverage_last_visit, t))),
        float(np.mean([robot.info_age for robot in robots])),
        encounter_proxy_from_occupancy(occ),
    )


def local_map_record_age(w: World, robots: Sequence[Robot], t: int) -> float:
    return mean_robot_map_record_age([ensure_robot_map(robot, w) for robot in robots], float(t))


def predicted_next_ages(
    cfg: HierarchicalConfig,
    mean_cov_age: float,
    mean_info_age: float,
    mode: Mode,
) -> Tuple[float, float]:
    """One-step mean-age model from Pinciroli Appendix C.

    Coverage mode reduces coverage age at rate ``rC`` and lets AoI grow;
    diffusion mode reduces AoI at rate ``rI`` and lets coverage age grow.  The
    report estimates or provides these rates before evaluating Layer 2:

        coverage:  A_C^+ = max(0, A_C + dt - r_C),  A_I^+ = A_I + dt
        diffusion: A_C^+ = A_C + dt,                A_I^+ = max(0, A_I + dt - r_I)
    """
    dt = float(cfg.mode_decision_interval)
    r_c = float(cfg.coverage_rate if cfg.coverage_rate is not None else dt)
    r_i = float(cfg.information_rate if cfg.information_rate is not None else dt)
    if mode == "coverage":
        return max(0.0, mean_cov_age + dt - r_c), mean_info_age + dt
    return mean_cov_age + dt, max(0.0, mean_info_age + dt - r_i)


def mode_cost(
    cfg: HierarchicalConfig,
    mean_cov_age: float,
    mean_info_age: float,
    mode: Mode,
    previous_mode: Mode | None = None,
) -> float:
    """Layer 2 mode cost ``L_m`` from Pinciroli Appendix C.

    Linear form: cost is affine in the predicted next coverage age and AoI.
    Quadratic form: adds squared age terms, making the coverage/diffusion
    log-odds state dependent.
    """
    next_cov, next_info = predicted_next_ages(cfg, mean_cov_age, mean_info_age, mode)
    cov = next_cov / max(cfg.coverage_age_scale, 1.0e-12)
    info = next_info / max(cfg.information_age_scale, 1.0e-12)
    cost = cfg.lambda_coverage_linear * cov + cfg.lambda_information_linear * info
    if cfg.controller_form == "quadratic":
        cost += cfg.lambda_coverage_quadratic * cov * cov
        cost += cfg.lambda_information_quadratic * info * info
    if previous_mode is not None and previous_mode != mode:
        cost += cfg.lambda_switch
    return float(cost)


def mode_probabilities(
    cfg: HierarchicalConfig,
    mean_cov_age: float,
    mean_info_age: float,
    previous_mode: Mode | None = None,
) -> Tuple[float, float]:
    """Convert costs to ``Pr(m)=exp(-L_m)/sum_q exp(-L_q)``."""
    costs = np.array(
        [
            mode_cost(cfg, mean_cov_age, mean_info_age, "coverage", previous_mode),
            mode_cost(cfg, mean_cov_age, mean_info_age, "diffusion", previous_mode),
        ],
        dtype=np.float64,
    )
    weights = np.exp(-(costs - float(costs.min())))
    probs = weights / float(weights.sum())
    return float(probs[0]), float(probs[1])


def choose_mode(
    cfg: HierarchicalConfig,
    mean_cov_age: float,
    mean_info_age: float,
    rng: np.random.Generator,
    previous_mode: Mode | None = None,
) -> Tuple[Mode, float, float]:
    p_cov, p_diff = mode_probabilities(cfg, mean_cov_age, mean_info_age, previous_mode)
    return ("coverage" if rng.random() < p_cov else "diffusion"), p_cov, p_diff


def estimate_effective_age_rate(t_axis: np.ndarray, age_series: np.ndarray) -> float:
    if len(t_axis) < 2 or len(age_series) < 2:
        return 0.0
    dt = np.diff(np.asarray(t_axis, dtype=np.float64))
    ages = np.asarray(age_series, dtype=np.float64)
    if len(dt) != len(ages) - 1:
        return 0.0
    return float(np.maximum(ages[:-1] + dt - ages[1:], 0.0).mean())


def auto_calibrate_layer2_rates(cfg: HierarchicalConfig, kernels: Layer1Kernels) -> None:
    if not cfg.auto_calibrate_layer2_rates:
        return
    if cfg.coverage_rate is not None and cfg.information_rate is not None:
        return
    preview_T = max(int(cfg.rate_calibration_T), int(4 * cfg.mode_decision_interval))
    preview_cfg = replace(
        cfg,
        T_sim=preview_T,
        record_every=int(cfg.mode_decision_interval),
        snap_every=max(int(cfg.snap_every), preview_T + 1),
        coverage_rate=0.0 if cfg.coverage_rate is None else cfg.coverage_rate,
        information_rate=0.0 if cfg.information_rate is None else cfg.information_rate,
        auto_calibrate_layer2_rates=False,
    )
    coverage_preview = run_simulation(preview_cfg, forced_mode="coverage", prebuilt_kernels=kernels, calibrate_rates=False)
    diffusion_preview = run_simulation(preview_cfg, forced_mode="diffusion", prebuilt_kernels=kernels, calibrate_rates=False)
    if cfg.coverage_rate is None:
        cfg.coverage_rate = max(estimate_effective_age_rate(coverage_preview.t_axis, coverage_preview.mean_cov_age), 1.0e-6)
    if cfg.information_rate is None:
        cfg.information_rate = max(estimate_effective_age_rate(diffusion_preview.t_axis, diffusion_preview.mean_info_age), 1.0e-6)


def step_robot(
    cfg: HierarchicalConfig,
    w: World,
    robot: Robot,
    transition: np.ndarray,
    markov_visits: np.ndarray,
    coverage_last_visit: np.ndarray,
    t: int,
    rng: np.random.Generator,
) -> None:
    robot.info_age += 1.0
    robot.x, robot.y = core.move_toward(robot.x, robot.y, robot.tx, robot.ty, cfg.speed)
    if abs(robot.x - robot.tx) >= 1.0e-9 or abs(robot.y - robot.ty) >= 1.0e-9:
        return
    robot.from_k = robot.to_k
    markov_visits[robot.from_k] += 1
    coverage_last_visit[robot.from_k] = float(t)
    ensure_robot_map(robot, w).observe_cell(robot.from_k, float(t))
    robot.to_k = sample_next_region_from_transition(w, robot.from_k, transition, rng)
    robot.tx, robot.ty = region_center(w, robot.to_k)


def run_simulation(
    cfg: HierarchicalConfig,
    forced_mode: Mode | None = None,
    prebuilt_kernels: Layer1Kernels | None = None,
    calibrate_rates: bool = True,
) -> SimResult:
    """Run the full hierarchical simulation or a forced fixed-mode baseline."""
    rng = np.random.default_rng(cfg.seed)
    w = build_world(cfg.Nx, cfg.Ny, cfg.cell_size)
    kernels = prebuilt_kernels if prebuilt_kernels is not None else build_layer1_kernels(cfg)
    if calibrate_rates:
        auto_calibrate_layer2_rates(cfg, kernels)

    robots = [make_robot(i, w, rng, cfg.initial_info_age_max) for i in range(cfg.n_robots)]
    coverage_last_visit = np.full(w.K, -1.0, dtype=np.float64)
    for robot in robots:
        coverage_last_visit[robot.from_k] = 0.0
    markov_visits = np.zeros(w.K, dtype=np.int64)

    mode: Mode = forced_mode if forced_mode is not None else cfg.initial_mode
    initialise_targets(w, robots, kernels, mode, rng)
    switch_events: list[SwitchEvent] = []
    meetings_since_switch = meetings_since_log = communications_since_log = 0
    p_cov_current, p_diff_current = mode_probabilities(cfg, 0.0, 0.0, mode)

    coverage_field_acc = np.zeros(w.K, dtype=np.float64)
    info_field_acc = np.zeros(w.K, dtype=np.float64)
    final_coverage_field = np.zeros(w.K, dtype=np.float64)
    final_information_field = np.zeros(w.K, dtype=np.float64)

    t_log: list[int] = []
    disp_log: list[float] = []
    cov_age_log: list[float] = []
    info_age_log: list[float] = []
    map_record_log: list[float] = []
    enc_log: list[float] = []
    meet_log: list[float] = []
    comm_log: list[float] = []
    l1_log: list[float] = []
    mode_log: list[int] = []
    p_cov_log: list[float] = []
    p_diff_log: list[float] = []
    mode_name_log: list[Mode] = []
    snapshots: list[Tuple[np.ndarray, np.ndarray, int, Mode]] = []

    for t in range(1, cfg.T_sim + 1):
        transition = transition_for_mode(kernels, mode)
        for robot in robots:
            step_robot(cfg, w, robot, transition, markov_visits, coverage_last_visit, t, rng)

        encounters, communications, _ = perform_information_exchanges(cfg, w, robots, rng, t)
        meetings_since_log += encounters
        communications_since_log += communications
        meetings_since_switch += encounters

        coverage_field = compute_coverage_field(w, coverage_last_visit, t)
        information_field, occ = compute_information_field(w, robots)
        coverage_field_acc += coverage_field
        info_field_acc += information_field
        final_coverage_field = coverage_field
        final_information_field = information_field
        mean_cov_age, mean_info_age, encounter = observables(t, coverage_last_visit, robots, occ)

        if forced_mode is None and t % cfg.mode_decision_interval == 0:
            new_mode, p_cov_current, p_diff_current = choose_mode(cfg, mean_cov_age, mean_info_age, rng, mode)
            if new_mode != mode:
                mode = new_mode
                switch_events.append(
                    SwitchEvent(t, mode, p_cov_current, p_diff_current, mean_info_age, mean_cov_age, encounter, meetings_since_switch)
                )
                meetings_since_switch = 0
        elif forced_mode is not None:
            mode = forced_mode
            p_cov_current = 1.0 if mode == "coverage" else 0.0
            p_diff_current = 1.0 - p_cov_current

        if t % cfg.record_every == 0:
            xs = np.array([robot.x for robot in robots], dtype=np.float64)
            ys = np.array([robot.y for robot in robots], dtype=np.float64)
            arrivals = max(int(markov_visits.sum()), 1)
            ck = markov_visits / arrivals
            t_log.append(t)
            disp_log.append(float(np.var(xs) + np.var(ys)))
            cov_age_log.append(mean_cov_age)
            info_age_log.append(mean_info_age)
            map_record_log.append(local_map_record_age(w, robots, t))
            enc_log.append(encounter)
            meet_log.append(float(meetings_since_log))
            comm_log.append(float(communications_since_log))
            l1_log.append(core.l1_error(ck, kernels.coverage_reference))
            mode_log.append(mode_to_indicator(mode))
            p_cov_log.append(p_cov_current)
            p_diff_log.append(p_diff_current)
            mode_name_log.append(mode)
            meetings_since_log = communications_since_log = 0

        if t % cfg.snap_every == 0:
            snapshots.append((
                np.array([robot.x for robot in robots], dtype=np.float64),
                np.array([robot.y for robot in robots], dtype=np.float64),
                t,
                mode,
            ))

    total_arrivals = max(int(markov_visits.sum()), 1)
    return SimResult(
        config=cfg,
        forced_mode=forced_mode,
        w=w,
        kernels=kernels,
        total_arrivals=int(markov_visits.sum()),
        pi_empirical=markov_visits / total_arrivals,
        pi_reference=kernels.coverage_reference,
        mean_coverage_field=coverage_field_acc / float(max(cfg.T_sim, 1)),
        mean_information_field=info_field_acc / float(max(cfg.T_sim, 1)),
        final_coverage_field=final_coverage_field,
        final_information_field=final_information_field,
        t_axis=np.array(t_log, dtype=np.int64),
        dispersion=np.array(disp_log, dtype=np.float64),
        mean_cov_age=np.array(cov_age_log, dtype=np.float64),
        mean_info_age=np.array(info_age_log, dtype=np.float64),
        mean_local_map_record_age=np.array(map_record_log, dtype=np.float64),
        encounter_proxy=np.array(enc_log, dtype=np.float64),
        meetings_per_log=np.array(meet_log, dtype=np.float64),
        communications_per_log=np.array(comm_log, dtype=np.float64),
        coverage_reference_l1=np.array(l1_log, dtype=np.float64),
        mode_indicator=np.array(mode_log, dtype=np.int64),
        mode_prob_coverage=np.array(p_cov_log, dtype=np.float64),
        mode_prob_diffusion=np.array(p_diff_log, dtype=np.float64),
        mode_names=mode_name_log,
        switch_events=switch_events,
        pos_snapshots=snapshots,
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


def phase_loop_area(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    xn = (x - x.mean()) / max(float(x.std()), 1.0e-9)
    yn = (y - y.mean()) / max(float(y.std()), 1.0e-9)
    return float(0.5 * abs(np.dot(xn, np.roll(yn, -1)) - np.dot(yn, np.roll(xn, -1))))


def mode_window_summary(res: SimResult) -> Dict[str, Dict[str, float]]:
    modes = np.array(res.mode_names)
    summary: dict[str, dict[str, float]] = {}
    for mode in ("coverage", "diffusion"):
        mask = modes == mode
        if not np.any(mask):
            summary[mode] = {key: float("nan") for key in (
                "mean_dispersion", "mean_encounter_proxy", "mean_cov_age", "mean_info_age",
                "mean_meetings_per_log", "mean_communications_per_log",
            )}
            summary[mode]["window_fraction"] = 0.0
            continue
        summary[mode] = {
            "window_fraction": float(mask.mean()),
            "mean_dispersion": float(res.dispersion[mask].mean()),
            "mean_encounter_proxy": float(res.encounter_proxy[mask].mean()),
            "mean_cov_age": float(res.mean_cov_age[mask].mean()),
            "mean_info_age": float(res.mean_info_age[mask].mean()),
            "mean_meetings_per_log": float(res.meetings_per_log[mask].mean()),
            "mean_communications_per_log": float(res.communications_per_log[mask].mean()),
        }
    return summary


def _mean_or_nan(values: np.ndarray) -> float:
    return float(values.mean()) if len(values) else float("nan")


def result_summary(res: SimResult) -> Dict[str, object]:
    freq, prominence, _, _ = dominant_frequency(res.t_axis, res.dispersion)
    return {
        "controller_form": res.config.controller_form,
        "forced_mode": res.forced_mode,
        "total_arrivals": res.total_arrivals,
        "n_switches": len(res.switch_events),
        "dominant_frequency": freq,
        "spectral_prominence": prominence,
        "phase_loop_area": phase_loop_area(res.mean_cov_age, res.mean_info_age),
        "mean_dispersion": _mean_or_nan(res.dispersion),
        "mean_encounter_proxy": _mean_or_nan(res.encounter_proxy),
        "mean_cov_age": _mean_or_nan(res.mean_cov_age),
        "mean_info_age": _mean_or_nan(res.mean_info_age),
        "mean_local_map_record_age": _mean_or_nan(res.mean_local_map_record_age),
        "mean_meetings_per_log": _mean_or_nan(res.meetings_per_log),
        "mean_communications_per_log": _mean_or_nan(res.communications_per_log),
        "mean_mode_probability_coverage": _mean_or_nan(res.mode_prob_coverage),
        "mean_mode_probability_diffusion": _mean_or_nan(res.mode_prob_diffusion),
        "final_reference_l1": float(res.coverage_reference_l1[-1]) if len(res.coverage_reference_l1) else float("nan"),
        "layer1": {
            "coverage_target": res.config.coverage_target_mode,
            "coverage_solver_iterations": res.kernels.coverage_solver_iterations,
            "coverage_solver_l1_error": res.kernels.coverage_solver_l1_error,
            "diffusion_target_mode": res.kernels.diffusion_target_mode,
            "diffusion_lambda_I": res.kernels.diffusion_lambda_I,
            "diffusion_solver_method": res.kernels.diffusion_solver_method,
            "diffusion_selected_branch": res.kernels.diffusion_selected_branch,
            "diffusion_target_rate": res.kernels.diffusion_target_rate,
            "diffusion_target_p_enc": res.kernels.diffusion_target_p_enc,
            "diffusion_predicted_rate": res.kernels.diffusion_predicted_rate,
            "diffusion_predicted_p_enc": res.kernels.diffusion_predicted_p_enc,
            "diffusion_converged": res.kernels.diffusion_converged,
            "diffusion_fixed_point_iterations": res.kernels.diffusion_fixed_point_iterations,
            "diffusion_fixed_point_error": res.kernels.diffusion_fixed_point_error,
        },
        "layer2_rates": {
            "mode_decision_interval": res.config.mode_decision_interval,
            "coverage_rate": res.config.coverage_rate,
            "information_rate": res.config.information_rate,
            "auto_calibrated": res.config.auto_calibrate_layer2_rates,
            "rate_calibration_T": res.config.rate_calibration_T,
            "coverage_age_scale": res.config.coverage_age_scale,
            "information_age_scale": res.config.information_age_scale,
            "lambda_switch": res.config.lambda_switch,
        },
        "mode_windows": mode_window_summary(res),
        "switch_events": [asdict(event) for event in res.switch_events],
    }


def mode_shading(ax: plt.Axes, res: SimResult, alpha: float = 0.08) -> None:
    if len(res.t_axis) == 0:
        return
    modes = np.array(res.mode_names)
    start = 0
    for i in range(1, len(res.t_axis) + 1):
        if i < len(res.t_axis) and modes[i] == modes[start]:
            continue
        color = "seagreen" if modes[start] == "coverage" else "crimson"
        ax.axvspan(res.t_axis[start], res.t_axis[i - 1], color=color, alpha=alpha, lw=0)
        start = i


def _safe_last_snapshot(res: SimResult) -> Tuple[np.ndarray, np.ndarray, int, Mode]:
    if res.pos_snapshots:
        return res.pos_snapshots[-1]
    mode = res.mode_names[-1] if res.mode_names else res.config.initial_mode
    return np.array([]), np.array([]), 0, mode


def make_main_figure(res: SimResult) -> plt.Figure:
    w = res.w
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 3)
    panels = [
        (res.mean_coverage_field, "mean coverage-age field", "YlGnBu"),
        (res.mean_information_field, "mean meeting-density field", "magma"),
    ]
    for col, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(data.reshape(w.Ny, w.Nx), origin="lower", cmap=cmap)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)
    ax = fig.add_subplot(gs[0, 2])
    xs, ys, t_end, mode_end = _safe_last_snapshot(res)
    if len(xs):
        ax.scatter(xs, ys, s=12, alpha=0.8)
    ax.set_xlim(0, w.Nx * w.cell_size)
    ax.set_ylim(0, w.Ny * w.cell_size)
    ax.set_aspect("equal")
    ax.set_title(f"positions t={t_end} ({mode_end})")

    ax = fig.add_subplot(gs[1, :2])
    mode_shading(ax, res)
    ax.plot(res.t_axis, res.dispersion, color="navy", label="dispersion")
    ax2 = ax.twinx()
    ax2.plot(res.t_axis, res.encounter_proxy, color="darkorange", label="encounter proxy")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, fontsize=8)
    ax.set_title("dispersion and encounter proxy")

    ax = fig.add_subplot(gs[1, 2])
    mode_shading(ax, res)
    ax.plot(res.t_axis, res.mode_indicator, color="black", label="mode")
    ax.plot(res.t_axis, res.mode_prob_diffusion, color="crimson", label="P(diffusion)")
    ax.set_yticks([0, 1], labels=["coverage", "diffusion"])
    ax.legend(fontsize=8)
    ax.set_title("mode selector")

    ax = fig.add_subplot(gs[2, :2])
    mode_shading(ax, res)
    ax.plot(res.t_axis, res.mean_cov_age, color="seagreen", label="coverage age")
    ax.plot(res.t_axis, res.mean_info_age, color="crimson", label="AoI")
    ax.plot(res.t_axis, res.communications_per_log, color="slateblue", label="communications/log")
    ax.legend(fontsize=8)
    ax.set_title("paper observables")

    ax = fig.add_subplot(gs[2, 2])
    ax.plot(res.t_axis, res.coverage_reference_l1, color="purple")
    ax.set_title("coverage target L1")

    title = f"Hierarchical MaxCal ({res.config.controller_form})"
    if res.forced_mode is not None:
        title += f" - {res.forced_mode} only"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def make_age_plane_figure(res: SimResult) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5.8))
    colors = np.where(np.array(res.mode_names) == "coverage", "seagreen", "crimson")
    ax.scatter(res.mean_cov_age, res.mean_info_age, c=colors, s=12, alpha=0.7)
    ax.plot(res.mean_cov_age, res.mean_info_age, color="grey", lw=0.5, alpha=0.7)
    ax.set_xlabel("mean coverage age")
    ax.set_ylabel("mean AoI")
    ax.set_title("age-plane trajectory")
    fig.tight_layout()
    return fig


def make_psd_figure(res: SimResult) -> plt.Figure:
    freq, prominence, freqs, power = dominant_frequency(res.t_axis, res.dispersion)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    if len(freqs):
        ax.plot(freqs[1:], power[1:], color="navy")
        if freq > 0.0:
            ax.axvline(freq, color="crimson", ls=":", label=f"f={freq:.3e}, prom={prominence:.1f}")
            ax.legend(fontsize=8)
    ax.set_xlabel("frequency (1/step)")
    ax.set_ylabel("power")
    ax.set_title("dispersion power spectrum")
    fig.tight_layout()
    return fig


def make_animation(res: SimResult, filename: Path, fps: int = 12) -> None:
    if not res.pos_snapshots:
        return
    w = res.w
    extent = (0.0, w.Nx * w.cell_size, 0.0, w.Ny * w.cell_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    scat = ax.scatter([], [], s=14, alpha=0.8)
    title = ax.set_title("")

    def update(frame: int):
        xs, ys, t, mode = res.pos_snapshots[frame]
        scat.set_offsets(np.column_stack([xs, ys]))
        title.set_text(f"t = {t} ({mode})")
        return scat, title

    anim = animation.FuncAnimation(fig, update, frames=len(res.pos_snapshots), interval=1000 // fps, blit=False)
    try:
        anim.save(str(filename), writer=animation.PillowWriter(fps=fps))
    finally:
        plt.close(fig)


def save_result_bundle(res: SimResult, outdir: Path) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = res.config.stem
    figure_names = [
        f"{stem}_main.png",
        f"{stem}_age_plane.png",
        f"{stem}_psd.png",
        f"{stem}.gif",
    ]
    for suffix, maker in [
        ("main", make_main_figure),
        ("age_plane", make_age_plane_figure),
        ("psd", make_psd_figure),
    ]:
        fig = maker(res)
        fig.savefig(outdir / f"{stem}_{suffix}.png", dpi=140)
        plt.close(fig)
    make_animation(res, outdir / f"{stem}.gif")
    summary = result_summary(res)
    with open(outdir / f"{stem}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "paper_alignment": {
                    "layer": "Hierarchical MaxCal supervisor",
                    "claim": "A higher-level MaxCal selector arbitrates between coverage and information-diffusion kernels using macroscopic coverage-age and AoI observables; quadratic constraints are the intended oscillatory mechanism.",
                    "validated_outputs": [
                        "mode timeline",
                        "coverage-age/AoI phase plane",
                        "dispersion spectrum",
                        "coverage and diffusion fields",
                    ],
                },
                "config": asdict(res.config),
                "result": summary,
                "figures": figure_names,
            },
            handle,
            indent=2,
        )
    return summary


def _finite(value: float) -> bool:
    return bool(np.isfinite(value))


def validator_checks(integrated: SimResult, coverage_only: SimResult, diffusion_only: SimResult) -> Dict[str, object]:
    """Shared validation checks for linear and quadratic hierarchical runs."""
    int_sum = result_summary(integrated)
    cov_sum = result_summary(coverage_only)
    diff_sum = result_summary(diffusion_only)
    modes = int_sum["mode_windows"]
    cov_fraction = float(modes["coverage"]["window_fraction"])
    diff_fraction = float(modes["diffusion"]["window_fraction"])
    diffusion_more_clustered = bool(
        _finite(float(modes["diffusion"]["mean_dispersion"]))
        and _finite(float(modes["coverage"]["mean_dispersion"]))
        and modes["diffusion"]["mean_dispersion"] < modes["coverage"]["mean_dispersion"]
        and modes["diffusion"]["mean_encounter_proxy"] > modes["coverage"]["mean_encounter_proxy"]
    )
    diffusion_more_comms = bool(
        _finite(float(modes["diffusion"]["mean_communications_per_log"]))
        and _finite(float(modes["coverage"]["mean_communications_per_log"]))
        and modes["diffusion"]["mean_communications_per_log"] >= modes["coverage"]["mean_communications_per_log"]
    )
    return {
        "integrated": int_sum,
        "coverage_only": cov_sum,
        "diffusion_only": diff_sum,
        "checks": {
            "mode_probabilities_are_valid": bool(
                np.all(integrated.mode_prob_coverage >= -1.0e-12)
                and np.all(integrated.mode_prob_coverage <= 1.0 + 1.0e-12)
                and np.all(integrated.mode_prob_diffusion >= -1.0e-12)
                and np.all(integrated.mode_prob_diffusion <= 1.0 + 1.0e-12)
                and np.allclose(integrated.mode_prob_coverage + integrated.mode_prob_diffusion, 1.0)
            ),
            "at_least_two_switches": bool(int_sum["n_switches"] >= 2),
            "both_modes_occupy_nontrivial_fraction": bool(cov_fraction >= 0.05 and diff_fraction >= 0.05),
            "integrated_encounter_between_fixed_mode_baselines": bool(
                diff_sum["mean_encounter_proxy"] >= int_sum["mean_encounter_proxy"] >= cov_sum["mean_encounter_proxy"]
            ),
            "diffusion_windows_are_more_clustered_than_coverage_windows": diffusion_more_clustered,
            "diffusion_windows_generate_more_communications": diffusion_more_comms,
            "positive_phase_loop_area": bool(float(int_sum["phase_loop_area"]) > 0.5),
            "spectral_prominence_above_floor": bool(float(int_sum["spectral_prominence"]) > 4.0),
        },
    }


def make_validation_figure(integrated: SimResult, coverage_only: SimResult, diffusion_only: SimResult, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, attr, title, ylabel in [
        (axes[0, 0], "dispersion", "dispersion comparison", "S(t)"),
        (axes[0, 1], "encounter_proxy", "encounter comparison", "sum occ^2"),
    ]:
        ax.plot(integrated.t_axis, getattr(integrated, attr), label="integrated", color="navy")
        ax.plot(coverage_only.t_axis, getattr(coverage_only, attr), label="coverage-only", color="seagreen", alpha=0.8)
        ax.plot(diffusion_only.t_axis, getattr(diffusion_only, attr), label="diffusion-only", color="crimson", alpha=0.8)
        mode_shading(ax, integrated)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
    axes[1, 0].plot(integrated.mean_cov_age, integrated.mean_info_age, label="integrated", color="navy")
    axes[1, 0].plot(coverage_only.mean_cov_age, coverage_only.mean_info_age, label="coverage-only", color="seagreen", alpha=0.8)
    axes[1, 0].plot(diffusion_only.mean_cov_age, diffusion_only.mean_info_age, label="diffusion-only", color="crimson", alpha=0.8)
    axes[1, 0].set_title("age-plane comparison")
    axes[1, 0].set_xlabel("coverage age")
    axes[1, 0].set_ylabel("AoI")
    axes[1, 0].legend(fontsize=8)
    modes = mode_window_summary(integrated)
    x = np.arange(2)
    axes[1, 1].bar(x, [modes["coverage"]["mean_dispersion"], modes["diffusion"]["mean_dispersion"]], color=["seagreen", "crimson"])
    axes[1, 1].set_xticks(x, labels=["coverage", "diffusion"])
    axes[1, 1].set_title("mode-wise dispersion")
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def make_quadratic_comparison_figure(linear_res: SimResult, quadratic_res: SimResult, outpath: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, attr, title in [
        (axes[0, 0], "dispersion", "dispersion"),
        (axes[0, 1], "encounter_proxy", "encounter proxy"),
    ]:
        ax.plot(linear_res.t_axis, getattr(linear_res, attr), label="linear", color="navy")
        ax.plot(quadratic_res.t_axis, getattr(quadratic_res, attr), label="quadratic", color="crimson")
        ax.set_title(title)
        ax.legend(fontsize=8)
    axes[1, 0].plot(linear_res.mean_cov_age, linear_res.mean_info_age, label="linear", color="navy")
    axes[1, 0].plot(quadratic_res.mean_cov_age, quadratic_res.mean_info_age, label="quadratic", color="crimson")
    axes[1, 0].set_title("age-plane")
    axes[1, 0].legend(fontsize=8)
    lin = mode_window_summary(linear_res)
    quad = mode_window_summary(quadratic_res)
    x = np.arange(2)
    axes[1, 1].bar(x - 0.16, [lin["coverage"]["mean_dispersion"], lin["diffusion"]["mean_dispersion"]], width=0.32, label="linear")
    axes[1, 1].bar(x + 0.16, [quad["coverage"]["mean_dispersion"], quad["diffusion"]["mean_dispersion"]], width=0.32, label="quadratic")
    axes[1, 1].set_xticks(x, labels=["coverage", "diffusion"])
    axes[1, 1].set_title("mode-wise dispersion")
    axes[1, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140)
    plt.close(fig)


def validation_json(outpath: Path, payload: Dict[str, object]) -> None:
    with open(outpath, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def build_arg_parser(description: str, default_outdir: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--outdir", type=str, default=default_outdir, help="Output directory.")
    parser.add_argument("--T", type=int, default=18_000, help="Simulation length.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--diffusion-layer1-mode", choices=("inverse_rate", "fixed_lambda"), default="inverse_rate")
    parser.add_argument("--diffusion-lambda-I", type=float, default=-400.0)
    parser.add_argument("--diffusion-target-rate", type=float, default=None)
    parser.add_argument("--diffusion-target-p-enc", type=float, default=0.50)
    parser.add_argument("--diffusion-branch", choices=("auto", "clustered", "coverage"), default="clustered")
    parser.add_argument("--diffusion-continuation-step", type=float, default=10.0)
    parser.add_argument("--coverage-rate", type=float, default=None)
    parser.add_argument("--information-rate", type=float, default=None)
    parser.add_argument("--rate-calibration-T", type=int, default=2_400)
    parser.add_argument("--lambda-switch", type=float, default=0.0)
    return parser
