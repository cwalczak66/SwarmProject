"""
RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)  
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------


Layer 1-C coverage controller and inverse MaxCal solve.

With coverage multiplier ``lambda_C`` the local controller is

    P_ij = A_ij exp[-lambda_C,j] / sum_l A_il exp[-lambda_C,l].

For constant ``lambda_C`` this is the unbiased neighbor random walk, whose
stationary distribution on the finite 8-connected grid is degree-proportional.
For a requested target coverage density ``pi_bar`` the inverse problem uses the
reversible stationary formula

    pi_i(b) = b_i (A b)_i / sum_m b_m (A b)_m,
    b_i = exp[-lambda_C,i],

and solves ``pi(b)=pi_bar`` by symmetric proportional fitting.  The additive gauge
of ``lambda_C`` is fixed by enforcing ``mean(lambda_C)=0``.

The physical simulation advances the Markov chain only when a robot reaches
its sampled destination, so convergence is reported in arrival count rather
than raw time steps.
"""

from __future__ import annotations

import math
import os
import tempfile
import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation

import maxcal_core as core
from maxcal_local_maps import RobotWorldMap, mean_robot_coverage_age, mean_robot_map_record_age


NX = 20
NY = 20
CELL_SIZE = 1.0
N_ROBOTS = 50
ROBOT_SPEED = 0.15
T_SIM = 12_000
RECORD_EVERY = 60
SNAP_EVERY = 300
LAMBDA_C_VAL = 0.0
SEED = 42
INVERSE_SOLVER_TOL = 1.0e-12
INVERSE_SOLVER_MAX_ITERS = 100_000

World = core.World


@dataclass
class InverseCoverageSolution:
    lambda_C: np.ndarray
    stationary: np.ndarray
    iterations: int
    max_abs_stationary_error: float
    l1_stationary_error: float
    converged: bool
    error_history: np.ndarray


@dataclass
class CoverageController:
    lambda_C: np.ndarray
    transition: np.ndarray
    target_pi: np.ndarray
    target_mode: str
    inverse_solution: InverseCoverageSolution | None = None


@dataclass
class Robot:
    id: int
    x: float
    y: float
    from_k: int
    to_k: int
    tx: float
    ty: float
    world_map: RobotWorldMap | None = None


@dataclass
class SimResult:
    w: World
    pi_empirical: np.ndarray
    ck_history: List[np.ndarray]
    markov_step_history: List[int]
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]]
    lambda_C: np.ndarray | None = None
    target_pi: np.ndarray | None = None
    target_mode: str = "degree"
    inverse_solution: InverseCoverageSolution | None = None
    t_axis: np.ndarray | None = None
    mean_global_coverage_age: np.ndarray | None = None
    mean_local_coverage_age: np.ndarray | None = None
    mean_local_map_record_age: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Layer 1-C MaxCal coverage controller.")
    parser.add_argument("--outdir", type=str, default="maxcal_coverage", help="Directory for figures and summary JSON.")
    parser.add_argument("--T", type=int, default=T_SIM, help="Simulation horizon.")
    parser.add_argument("--speed", type=float, default=ROBOT_SPEED, help="Robot speed in cells per step.")
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


def build_world(Nx: int, Ny: int, cell_size: float) -> World:
    return core.build_grid_world(Nx, Ny, cell_size)


def region_center(w: World, k: int) -> Tuple[float, float]:
    return core.region_center(w, k)


def adjacency_matrix(w: World) -> np.ndarray:
    return core.adjacency_matrix(w)


def uniform_stationary(w: World) -> np.ndarray:
    return np.full(w.K, 1.0 / w.K, dtype=np.float64)


def build_transition_matrix(w: World, lambda_C: np.ndarray) -> np.ndarray:
    return core.destination_weight_transition(w, np.asarray(lambda_C, dtype=np.float64))


def sample_next_region(w: World, k: int, P: np.ndarray, rng: np.random.Generator) -> int:
    return core.sample_from_transition(w, P, k, rng)


def stationary_from_weights(w: World, b: np.ndarray) -> np.ndarray:
    """Evaluate ``pi_i(b)=b_i(A b)_i / sum_m b_m(A b)_m``."""
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (w.K,):
        raise ValueError(f"b must have shape ({w.K},), got {b.shape}.")
    raw = b * np.array([b[neighbors].sum() for neighbors in w.adjacency], dtype=np.float64)
    return core.normalize_probability(raw)


def stationary_from_lambda(w: World, lambda_C: np.ndarray) -> np.ndarray:
    """Stationary law induced by coverage multipliers, with gauge removed."""
    lambda_C = np.asarray(lambda_C, dtype=np.float64)
    shifted = lambda_C - float(lambda_C.mean())
    return stationary_from_weights(w, np.exp(-shifted))


def power_stationary_distribution(
    transition: np.ndarray,
    tol: float = 1.0e-14,
    max_iters: int = 200_000,
) -> np.ndarray:
    return core.power_stationary_distribution(transition, tol=tol, max_iters=max_iters)


def theoretical_stationary(w: World) -> np.ndarray:
    """Equal-multiplier coverage baseline: pi_k proportional to degree(k)."""
    return stationary_from_lambda(w, np.zeros(w.K, dtype=np.float64))


def solve_coverage_multipliers_for_target(
    w: World,
    target_pi: np.ndarray,
    tol: float = INVERSE_SOLVER_TOL,
    max_iters: int = INVERSE_SOLVER_MAX_ITERS,
) -> InverseCoverageSolution:
    """Solve the inverse coverage problem for a strictly positive target.

    For the destination-weighted coverage kernel, detailed balance gives 
    ``pi_i proportional to b_i (A b)_i`` with ``b_i = exp(-lambda_C[i])``.
    The uniform-target test is this positive matrix-scaling problem on the 8-connected grid.

    Each iteration applies the symmetric proportional-fitting update

        b_i <- b_i sqrt(pi_bar_i / pi_i(b)).

    If a cell is underrepresented, ``b_i`` rises and therefore
    ``lambda_C,i=-log b_i`` falls, making that destination more attractive.
    """
    target = core.normalize_probability(np.asarray(target_pi, dtype=np.float64))
    if target.shape != (w.K,):
        raise ValueError(f"target_pi must have shape ({w.K},), got {target.shape}.")
    if np.any(target <= 0.0):
        raise ValueError("The inverse coverage solve requires a strictly positive target.")

    b = np.ones(w.K, dtype=np.float64)
    history: list[float] = []
    converged = False
    iteration = 0

    for iteration in range(max_iters + 1):
        stationary = stationary_from_weights(w, b)
        err = float(np.max(np.abs(stationary - target)))
        history.append(err)
        if err < tol:
            converged = True
            break
        b *= np.sqrt(target / np.maximum(stationary, 1.0e-300))
        b /= math.exp(float(np.mean(np.log(b))))

    lambda_C = -np.log(b)
    lambda_C -= float(lambda_C.mean())
    stationary = stationary_from_lambda(w, lambda_C)
    return InverseCoverageSolution(
        lambda_C=lambda_C,
        stationary=stationary,
        iterations=int(iteration),
        max_abs_stationary_error=float(np.max(np.abs(stationary - target))),
        l1_stationary_error=core.l1_error(stationary, target),
        converged=converged,
        error_history=np.array(history, dtype=np.float64),
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
    """Build the Layer 1-C controller for a forward or inverse target.

    ``target_mode='degree'`` is the constant-multiplier paper baseline.
    ``target_mode='uniform'`` or a supplied ``target_pi`` invokes the inverse
    MaxCal solve above and then uses the solved multipliers in the same local
    transition rule.
    """
    if lambda_C is not None:
        lambda_arr = np.asarray(lambda_C, dtype=np.float64)
        if lambda_arr.shape != (w.K,):
            raise ValueError(f"lambda_C must have shape ({w.K},), got {lambda_arr.shape}.")
        reference = stationary_from_lambda(w, lambda_arr) if target_pi is None else core.normalize_probability(target_pi)
        mode = "explicit_lambda" if target_mode == "degree" else target_mode
        return CoverageController(lambda_arr.copy(), build_transition_matrix(w, lambda_arr), reference, mode)

    if target_pi is not None or target_mode == "custom":
        if target_pi is None:
            raise ValueError("target_pi is required when target_mode='custom'.")
        solution = solve_coverage_multipliers_for_target(w, target_pi, tol=tol, max_iters=max_iters)
        target = core.normalize_probability(target_pi)
        return CoverageController(solution.lambda_C, build_transition_matrix(w, solution.lambda_C), target, "custom", solution)

    if target_mode == "uniform":
        target = uniform_stationary(w)
        solution = solve_coverage_multipliers_for_target(w, target, tol=tol, max_iters=max_iters)
        return CoverageController(solution.lambda_C, build_transition_matrix(w, solution.lambda_C), target, "uniform", solution)

    if target_mode == "degree":
        lambda_arr = np.full(w.K, lambda_C_val, dtype=np.float64)
        return CoverageController(lambda_arr, build_transition_matrix(w, lambda_arr), stationary_from_lambda(w, lambda_arr), "degree")

    raise ValueError("target_mode must be one of: degree, uniform, custom, explicit_lambda.")


def ensure_robot_map(r: Robot, w: World, t: float = 0.0) -> RobotWorldMap:
    if r.world_map is None:
        r.world_map = RobotWorldMap.empty(w.K)
        r.world_map.observe_cell(r.from_k, t)
    return r.world_map


def make_robot(idx: int, w: World, P: np.ndarray, rng: np.random.Generator) -> Robot:
    k0 = int(rng.integers(0, w.K))
    x, y = region_center(w, k0)
    k1 = sample_next_region(w, k0, P, rng)
    tx, ty = region_center(w, k1)
    robot_map = RobotWorldMap.empty(w.K)
    robot_map.observe_cell(k0, 0.0)
    return Robot(idx, x, y, k0, k1, tx, ty, robot_map)


def step_robot(
    r: Robot,
    w: World,
    P: np.ndarray,
    speed: float,
    markov_visits: np.ndarray,
    rng: np.random.Generator,
    t: int = 0,
    global_last_visit_time: np.ndarray | None = None,
) -> None:
    """Move a robot and sample a new Markov transition only on arrival."""
    r.x, r.y = core.move_toward(r.x, r.y, r.tx, r.ty, speed)
    if abs(r.x - r.tx) >= 1.0e-9 or abs(r.y - r.ty) >= 1.0e-9:
        return

    r.from_k = r.to_k
    markov_visits[r.from_k] += 1
    ensure_robot_map(r, w).observe_cell(r.from_k, float(t))
    if global_last_visit_time is not None:
        global_last_visit_time[r.from_k] = float(t)

    r.to_k = sample_next_region(w, r.from_k, P, rng)
    r.tx, r.ty = region_center(w, r.to_k)


def run_simulation(
    speed: float = ROBOT_SPEED,
    T: int = T_SIM,
    lambda_C_val: float = LAMBDA_C_VAL,
    seed: int = SEED,
    target_mode: str = "degree",
    target_pi: np.ndarray | None = None,
    lambda_C: np.ndarray | None = None,
    inverse_tol: float = INVERSE_SOLVER_TOL,
    inverse_max_iters: int = INVERSE_SOLVER_MAX_ITERS,
) -> SimResult:
    """Run the physical Layer 1-C simulation.

    The controller samples a next cell from ``P`` only after each arrival.  The
    empirical coverage ``c_k(n)`` is therefore normalized by the total number of
    Markov arrivals ``n`` rather than by the number of raw simulation ticks.
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

    robots = [make_robot(i, w, controller.transition, rng) for i in range(N_ROBOTS)]
    markov_visits = np.zeros(w.K, dtype=np.int64)
    last_visit = np.full(w.K, -1.0, dtype=np.float64)
    for robot in robots:
        last_visit[robot.from_k] = 0.0

    ck_history: list[np.ndarray] = []
    markov_history: list[int] = []
    snapshots: list[Tuple[np.ndarray, np.ndarray, int]] = []
    t_axis: list[int] = []
    global_age: list[float] = []
    local_age: list[float] = []
    map_age: list[float] = []

    for t in range(1, T + 1):
        for robot in robots:
            step_robot(robot, w, controller.transition, speed, markov_visits, rng, t=t, global_last_visit_time=last_visit)

        arrivals = int(markov_visits.sum())
        if t % RECORD_EVERY == 0 and arrivals > 0:
            ck_history.append(markov_visits / arrivals)
            markov_history.append(arrivals)
            maps = [ensure_robot_map(robot, w) for robot in robots]
            t_axis.append(t)
            global_age.append(float(core.coverage_age(last_visit, t).mean()))
            local_age.append(mean_robot_coverage_age(maps, float(t)))
            map_age.append(mean_robot_map_record_age(maps, float(t)))

        if t % SNAP_EVERY == 0:
            snapshots.append((
                np.array([robot.x for robot in robots], dtype=np.float64),
                np.array([robot.y for robot in robots], dtype=np.float64),
                t,
            ))

    total = max(int(markov_visits.sum()), 1)
    return SimResult(
        w=w,
        pi_empirical=markov_visits / total,
        ck_history=ck_history,
        markov_step_history=markov_history,
        pos_snapshots=snapshots,
        lambda_C=controller.lambda_C,
        target_pi=controller.target_pi,
        target_mode=controller.target_mode,
        inverse_solution=controller.inverse_solution,
        t_axis=np.array(t_axis, dtype=np.int64),
        mean_global_coverage_age=np.array(global_age, dtype=np.float64),
        mean_local_coverage_age=np.array(local_age, dtype=np.float64),
        mean_local_map_record_age=np.array(map_age, dtype=np.float64),
    )


def make_main_figure(res: SimResult) -> plt.Figure:
    w = res.w
    target = res.target_pi if res.target_pi is not None else theoretical_stationary(w)
    err = np.abs(res.pi_empirical - target)
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3)

    for ax, data, title, cmap in [
        (fig.add_subplot(gs[0, 0]), target, "target stationary π̄", "viridis"),
        (fig.add_subplot(gs[0, 1]), res.pi_empirical, "empirical visits", "viridis"),
        (fig.add_subplot(gs[0, 2]), err, "absolute error", "Reds"),
    ]:
        im = ax.imshow(data.reshape(w.Ny, w.Nx), origin="lower", cmap=cmap)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = fig.add_subplot(gs[1, :2])
    if res.ck_history:
        reps = [(0, "corner"), (NX // 2, "edge"), ((NY // 2) * NX + NX // 2, "interior")]
        for cell, label in reps:
            ax.plot(res.markov_step_history, [snap[cell] for snap in res.ck_history], label=label)
            ax.axhline(target[cell], ls=":", lw=1.0)
    ax.set_xlabel("Markov arrivals")
    ax.set_ylabel("coverage fraction")
    ax.set_title("representative convergence")
    ax.legend(fontsize=8)

    ax_pos = fig.add_subplot(gs[1, 2])
    if res.pos_snapshots:
        xs, ys, t_end = res.pos_snapshots[-1]
        ax_pos.scatter(xs, ys, s=12, alpha=0.75)
        ax_pos.set_title(f"robot positions at t={t_end}")
    else:
        ax_pos.set_title("robot positions")
    ax_pos.set_xlim(0, w.Nx * w.cell_size)
    ax_pos.set_ylim(0, w.Ny * w.cell_size)
    ax_pos.set_aspect("equal")
    ax_pos.set_xlabel("x")
    ax_pos.set_ylabel("y")

    suffix = "forward" if res.inverse_solution is None else f"inverse, L1={res.inverse_solution.l1_stationary_error:.2e}"
    fig.suptitle(f"MaxCal coverage ({res.target_mode}, {suffix})")
    fig.tight_layout()
    return fig


def make_phase_figure(T: int = T_SIM) -> plt.Figure:
    w = build_world(NX, NY, CELL_SIZE)
    target = theoretical_stationary(w)
    speeds = [0.05, 0.10, 0.20, 0.50, 1.00]
    fig, (ax_m, ax_t) = plt.subplots(1, 2, figsize=(11, 4.8))
    for speed in speeds:
        res = run_simulation(speed=speed, T=T)
        if not res.ck_history:
            continue
        errors = np.array([core.l1_error(snap, target) for snap in res.ck_history], dtype=np.float64)
        sim_time = np.arange(1, len(errors) + 1) * RECORD_EVERY
        ax_m.plot(res.markov_step_history, errors, label=f"v={speed:g}")
        ax_t.plot(sim_time, errors, label=f"v={speed:g}")
    for ax, xlabel in [(ax_m, "Markov arrivals"), (ax_t, "simulation step")]:
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("L1 error")
        ax.legend(fontsize=8)
    ax_m.set_title("collapse in Markov time")
    ax_t.set_title("speed effect in simulation time")
    fig.tight_layout()
    return fig


def make_animation(res: SimResult, fps: int = 12, filename: str = "maxcal_coverage.gif") -> None:
    if not res.pos_snapshots:
        return
    w = res.w
    extent = (0.0, w.Nx * w.cell_size, 0.0, w.Ny * w.cell_size)
    fig, ax = plt.subplots(figsize=(6, 6))
    background = res.target_pi if res.target_pi is not None else theoretical_stationary(w)
    ax.imshow(
        background.reshape(w.Ny, w.Nx),
        origin="lower",
        extent=extent,
        interpolation="nearest",
        cmap="YlOrRd",
        alpha=0.45,
    )
    scat = ax.scatter([], [], s=12, color="navy", alpha=0.8)
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


def degree_stats(w: World, values: np.ndarray) -> dict[str, dict[str, float | int]]:
    degrees = np.array([len(neighbors) for neighbors in w.adjacency], dtype=np.int64)
    stats: dict[str, dict[str, float | int]] = {}
    for degree in sorted(np.unique(degrees)):
        group = np.asarray(values[degrees == degree], dtype=np.float64)
        stats[str(int(degree))] = {
            "degree": int(degree),
            "n_cells": int(len(group)),
            "mean": float(group.mean()),
            "min": float(group.min()),
            "max": float(group.max()),
            "std": float(group.std()),
        }
    return stats


def representative_summary(res: SimResult, target: np.ndarray) -> dict[str, dict[str, float | int]]:
    cells = {
        "corner": 0,
        "edge": NX // 2,
        "interior": (NY // 2) * NX + NX // 2,
    }
    return {
        label: {
            "index": int(cell),
            "degree": int(len(res.w.adjacency[cell])),
            "target_pi": float(target[cell]),
            "empirical_pi": float(res.pi_empirical[cell]),
            "abs_error": float(abs(res.pi_empirical[cell] - target[cell])),
        }
        for label, cell in cells.items()
    }


def run_summary(res: SimResult, label: str) -> dict[str, Any]:
    target = res.target_pi if res.target_pi is not None else theoretical_stationary(res.w)
    history_l1 = [core.l1_error(snapshot, target) for snapshot in res.ck_history]
    summary: dict[str, Any] = {
        "label": label,
        "target_mode": res.target_mode,
        "total_markov_arrivals": int(res.markov_step_history[-1]) if res.markov_step_history else 0,
        "empirical_l1_error": core.l1_error(res.pi_empirical, target),
        "final_recorded_l1_error": float(history_l1[-1]) if history_l1 else float("nan"),
        "representative_cells": representative_summary(res, target),
        "lambda_C": {
            "mean": float(np.mean(res.lambda_C)) if res.lambda_C is not None else float("nan"),
            "min": float(np.min(res.lambda_C)) if res.lambda_C is not None else float("nan"),
            "max": float(np.max(res.lambda_C)) if res.lambda_C is not None else float("nan"),
        },
    }
    if res.inverse_solution is not None:
        summary["inverse_solver"] = {
            "converged": res.inverse_solution.converged,
            "iterations": res.inverse_solution.iterations,
            "l1_stationary_error": res.inverse_solution.l1_stationary_error,
            "max_abs_stationary_error": res.inverse_solution.max_abs_stationary_error,
            "lambda_by_degree": degree_stats(res.w, res.inverse_solution.lambda_C),
        }
    return summary


def write_summary(outdir: Path, forward: SimResult, uniform: SimResult, figures: list[str]) -> dict[str, Any]:
    forward_summary = run_summary(forward, "equal-multiplier degree baseline")
    uniform_summary = run_summary(uniform, "inverse uniform target")
    checks = {
        "forward_has_markov_samples": bool(forward.markov_step_history),
        "forward_targets_degree_stationary_distribution": bool(forward.target_mode == "degree"),
        "inverse_solver_converged": bool(uniform.inverse_solution and uniform.inverse_solution.converged),
        "inverse_theory_matches_uniform": bool(
            uniform.inverse_solution is not None and uniform.inverse_solution.l1_stationary_error < 1.0e-8
        ),
        "inverse_empirical_run_completed": bool(uniform.markov_step_history),
    }
    payload = {
        "paper_alignment": {
            "layer": "Layer 1-C coverage",
            "claim": "Equal multipliers recover the degree-proportional random walk; inverse MaxCal multipliers reshape the stationary distribution to a requested target.",
            "maxcal_kernel": "P[i,j] proportional to A[i,j] exp(-lambda_C[j])",
        },
        "environment": {
            "Nx": NX,
            "Ny": NY,
            "K": forward.w.K,
            "cell_size": CELL_SIZE,
            "robots": N_ROBOTS,
            "record_every": RECORD_EVERY,
        },
        "forward_degree": forward_summary,
        "inverse_uniform": uniform_summary,
        "checks": checks,
        "coverage_layer_ready": bool(all(checks.values())),
        "figures": figures,
    }
    with open(outdir / "maxcal_coverage_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2)
    return payload


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    result = run_simulation(target_mode="degree", T=args.T, speed=args.speed, seed=args.seed)
    uniform = run_simulation(target_mode="uniform", T=args.T, speed=args.speed, seed=args.seed + 1)

    figures = [
        "maxcal_coverage_main.png",
        "maxcal_coverage_uniform_main.png",
        "maxcal_coverage_phase.png",
    ]

    fig = make_main_figure(result)
    fig.savefig(outdir / figures[0], dpi=140)
    plt.close(fig)

    fig = make_main_figure(uniform)
    fig.savefig(outdir / figures[1], dpi=140)
    plt.close(fig)

    fig = make_phase_figure(T=args.T)
    fig.savefig(outdir / figures[2], dpi=140)
    plt.close(fig)
    if not args.no_animation:
        make_animation(result, filename=str(outdir / "maxcal_coverage.gif"))
        figures.append("maxcal_coverage.gif")

    payload = write_summary(outdir, result, uniform, figures)
    forward = payload["forward_degree"]
    inverse = payload["inverse_uniform"]
    solver = inverse.get("inverse_solver", {})

    print("MaxCal Coverage (Layer 1-C)")
    print(f"  Output directory       : {outdir}")
    print(f"  Forward arrivals       : {forward['total_markov_arrivals']}")
    print(f"  Forward empirical L1   : {forward['empirical_l1_error']:.6f}")
    print(
        "  Uniform inverse solve  : "
        f"converged={solver.get('converged')}, "
        f"iterations={solver.get('iterations')}, "
        f"theory L1={solver.get('l1_stationary_error', float('nan')):.3e}"
    )
    print(f"  Uniform empirical L1   : {inverse['empirical_l1_error']:.6f}")
    print(f"  Coverage layer ready   : {payload['coverage_layer_ready']}")
    print("  Saved summary          : maxcal_coverage_summary.json")


if __name__ == "__main__":
    main()
