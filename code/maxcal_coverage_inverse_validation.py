"""
Inverse-MaxCal validation for uniform coverage on an irregular grid.

The existing coverage validator tests the forward case lambda_C^k = constant:
the MaxCal kernel reduces to a uniform random walk and therefore has
stationary distribution pi_k proportional to the cell degree. This script
tests the complementary inverse problem:

    target first  -> solve lambda_C^k  -> build P  -> verify pi_k = target

For the 20 x 20 8-connected grid, the target is uniform coverage,
rho_k = 1 / K. The MaxCal transition kernel is

    P_ij = A_ij exp(-lambda_j) / sum_l A_il exp(-lambda_l),

where A is the undirected adjacency matrix and unit edge weights are used.
Writing b_j = exp(-lambda_j), reversibility gives the stationary distribution

    pi_i(b) = b_i (A b)_i / sum_m b_m (A b)_m.

Thus the inverse problem is the positive matrix-scaling equation

    b_i (A b)_i proportional to rho_i.

The solved lambda map is global/offline design knowledge. The robots still
execute only the local transition rule by looking up the precomputed
lambda_C value of neighboring cells.

Usage:
    venv/bin/python maxcal_coverage_uniform_target_validation.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import maxcal_coverage as mc


@dataclass
class SolverResult:
    iterations: int
    max_abs_stationary_error: float
    l1_stationary_error: float
    converged: bool


@dataclass
class DegreeClassStats:
    degree: int
    n_cells: int
    mean: float
    minimum: float
    maximum: float
    std: float


@dataclass
class RepresentativeCell:
    label: str
    index: int
    degree: int
    target_pi: float
    baseline_pi: float
    optimized_theory_pi: float
    optimized_empirical_pi: float
    optimized_empirical_abs_error: float
    lambda_C: float


@dataclass
class DegreeTiedFit:
    iterations: int
    final_step: float
    objective_l2: float
    l1_stationary_error: float
    max_abs_stationary_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and validate MaxCal coverage multipliers for a uniform stationary target."
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="maxcal_coverage_uniform_target_validation",
        help="Directory for summary, raw arrays, and figures.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=60_000,
        help=(
            "Simulation length for the empirical optimized-kernel check. "
            "This is longer than the baseline Layer 1-C run because a flat "
            "400-cell target needs more arrivals for a low-noise histogram."
        ),
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=mc.ROBOT_SPEED,
        help="Robot speed in m/step for the empirical check.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=mc.SEED + 1000,
        help="Random seed for the empirical check.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-12,
        help="Max absolute stationary error tolerance for the inverse solver.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=100_000,
        help="Maximum iterations for the inverse solver.",
    )
    return parser.parse_args()


def adjacency_matrix(world: mc.World) -> np.ndarray:
    adjacency = np.zeros((world.K, world.K), dtype=np.float64)
    for k, neighbors in enumerate(world.adjacency):
        adjacency[k, neighbors] = 1.0
    return adjacency


def stationary_from_b(adjacency: np.ndarray, b: np.ndarray) -> np.ndarray:
    raw = b * (adjacency @ b)
    return raw / raw.sum()


def stationary_from_lambda(adjacency: np.ndarray, lambda_c: np.ndarray) -> np.ndarray:
    shifted = lambda_c - float(np.mean(lambda_c))
    b = np.exp(-shifted)
    return stationary_from_b(adjacency, b)


def power_stationary(transition: np.ndarray, tol: float = 1e-14, max_iters: int = 200_000) -> np.ndarray:
    pi = np.full(transition.shape[0], 1.0 / transition.shape[0], dtype=np.float64)
    for _ in range(max_iters):
        nxt = pi @ transition
        if float(np.max(np.abs(nxt - pi))) < tol:
            return nxt / nxt.sum()
        pi = nxt
    return pi / pi.sum()


def solve_uniform_target_multipliers(
    adjacency: np.ndarray,
    target: np.ndarray,
    tol: float,
    max_iters: int,
) -> Tuple[np.ndarray, np.ndarray, SolverResult, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    target = target / target.sum()
    if np.any(target <= 0.0):
        raise ValueError("The inverse scaling solver requires a strictly positive target distribution.")

    b = np.ones_like(target)
    error_history: List[float] = []
    converged = False
    err = math.inf
    it = 0

    for it in range(max_iters + 1):
        pi = stationary_from_b(adjacency, b)
        err = float(np.max(np.abs(pi - target)))
        error_history.append(err)
        if err < tol:
            converged = True
            break

        # Symmetric proportional fitting. If a cell is underrepresented,
        # increase b_i = exp(-lambda_i); if overrepresented, decrease it.
        update = np.sqrt(target / np.maximum(pi, 1e-300))
        b *= update

        # Fix the arbitrary gauge: multiplying every b_i by the same constant
        # leaves every transition probability unchanged.
        b /= np.exp(float(np.mean(np.log(b))))

    lambda_c = -np.log(b)
    lambda_c -= float(np.mean(lambda_c))
    pi = stationary_from_lambda(adjacency, lambda_c)
    solver = SolverResult(
        iterations=int(it),
        max_abs_stationary_error=float(np.max(np.abs(pi - target))),
        l1_stationary_error=float(np.sum(np.abs(pi - target))),
        converged=bool(converged),
    )
    return lambda_c, pi, solver, np.array(error_history, dtype=np.float64)


def transition_checks(world: mc.World, transition: np.ndarray) -> Dict[str, float]:
    row_sums = transition.sum(axis=1)
    off_neighbor_mask = np.ones_like(transition, dtype=bool)
    for k in range(world.K):
        off_neighbor_mask[k, world.adjacency[k]] = False
        off_neighbor_mask[k, k] = False
    return {
        "max_row_sum_error": float(np.max(np.abs(row_sums - 1.0))),
        "max_off_neighbor_probability": float(np.max(np.abs(transition[off_neighbor_mask]))),
        "minimum_positive_neighbor_probability": float(
            min(np.min(transition[k, world.adjacency[k]]) for k in range(world.K))
        ),
        "maximum_neighbor_probability": float(
            max(np.max(transition[k, world.adjacency[k]]) for k in range(world.K))
        ),
    }


def run_simulation_with_lambda(
    lambda_c: np.ndarray,
    speed: float,
    T: int,
    seed: int,
) -> mc.SimResult:
    rng = np.random.default_rng(seed)
    world = mc.build_world(mc.NX, mc.NY, mc.CELL_SIZE)
    transition = mc.build_transition_matrix(world, lambda_c)

    robots = [mc.make_robot(i, world, transition, rng) for i in range(mc.N_ROBOTS)]
    markov_visits = np.zeros(world.K, dtype=np.int64)
    ck_history: List[np.ndarray] = []
    markov_step_history: List[int] = []
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]] = []

    for t in range(1, T + 1):
        for robot in robots:
            mc.step_robot(robot, world, transition, speed, markov_visits, rng)

        total = int(markov_visits.sum())
        if t % mc.RECORD_EVERY == 0 and total > 0:
            ck_history.append(markov_visits / total)
            markov_step_history.append(total)

        if t % mc.SNAP_EVERY == 0:
            xs = np.array([robot.x for robot in robots])
            ys = np.array([robot.y for robot in robots])
            pos_snapshots.append((xs, ys, t))

    pi_empirical = markov_visits / markov_visits.sum()
    return mc.SimResult(world, pi_empirical, ck_history, markov_step_history, pos_snapshots)


def degree_array(world: mc.World) -> np.ndarray:
    return np.array([len(neighbors) for neighbors in world.adjacency], dtype=np.int64)


def stats_by_degree(values: np.ndarray, degrees: np.ndarray) -> Dict[str, DegreeClassStats]:
    result: Dict[str, DegreeClassStats] = {}
    for degree in sorted(int(d) for d in np.unique(degrees)):
        mask = degrees == degree
        selected = values[mask]
        result[str(degree)] = DegreeClassStats(
            degree=degree,
            n_cells=int(mask.sum()),
            mean=float(np.mean(selected)),
            minimum=float(np.min(selected)),
            maximum=float(np.max(selected)),
            std=float(np.std(selected)),
        )
    return result


def representative_cells(
    world: mc.World,
    target: np.ndarray,
    baseline_pi: np.ndarray,
    optimized_pi: np.ndarray,
    empirical_pi: np.ndarray,
    lambda_c: np.ndarray,
) -> Dict[str, RepresentativeCell]:
    reps = {
        "corner": 0,
        "edge": mc.NX // 2,
        "interior": (mc.NY // 2) * mc.NX + mc.NX // 2,
    }
    output: Dict[str, RepresentativeCell] = {}
    for label, idx in reps.items():
        output[label] = RepresentativeCell(
            label=label,
            index=idx,
            degree=len(world.adjacency[idx]),
            target_pi=float(target[idx]),
            baseline_pi=float(baseline_pi[idx]),
            optimized_theory_pi=float(optimized_pi[idx]),
            optimized_empirical_pi=float(empirical_pi[idx]),
            optimized_empirical_abs_error=float(abs(empirical_pi[idx] - target[idx])),
            lambda_C=float(lambda_c[idx]),
        )
    return output


def lambda_from_degree_params(params: np.ndarray, degree_values: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    lambda_c = np.zeros_like(degrees, dtype=np.float64)
    for idx, degree in enumerate(degree_values):
        lambda_c[degrees == degree] = float(params[idx])
    lambda_c -= float(np.mean(lambda_c))
    return lambda_c


def optimise_degree_tied_multipliers(
    adjacency: np.ndarray,
    target: np.ndarray,
    degrees: np.ndarray,
    initial_lambda_c: np.ndarray,
    min_step: float = 1e-8,
    max_iters: int = 20_000,
) -> Tuple[np.ndarray, np.ndarray, DegreeTiedFit]:
    degree_values = np.array(sorted(int(d) for d in np.unique(degrees)), dtype=np.int64)
    params = np.array(
        [float(np.mean(initial_lambda_c[degrees == degree])) for degree in degree_values],
        dtype=np.float64,
    )

    def evaluate(candidate_params: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        candidate_lambda = lambda_from_degree_params(candidate_params, degree_values, degrees)
        candidate_pi = stationary_from_lambda(adjacency, candidate_lambda)
        diff = candidate_pi - target
        return float(np.dot(diff, diff)), candidate_lambda, candidate_pi

    best_obj, best_lambda, best_pi = evaluate(params)
    step = 0.5
    iterations = 0

    while step > min_step and iterations < max_iters:
        improved = False
        for dim in range(len(params)):
            for direction in (1.0, -1.0):
                candidate = params.copy()
                candidate[dim] += direction * step
                obj, candidate_lambda, candidate_pi = evaluate(candidate)
                iterations += 1
                if obj < best_obj:
                    params = candidate
                    best_obj = obj
                    best_lambda = candidate_lambda
                    best_pi = candidate_pi
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5

    fit = DegreeTiedFit(
        iterations=int(iterations),
        final_step=float(step),
        objective_l2=float(best_obj),
        l1_stationary_error=float(np.sum(np.abs(best_pi - target))),
        max_abs_stationary_error=float(np.max(np.abs(best_pi - target))),
    )
    return best_lambda, best_pi, fit


def make_maps_figure(
    outdir: Path,
    world: mc.World,
    baseline_pi: np.ndarray,
    target: np.ndarray,
    optimized_pi: np.ndarray,
    lambda_c: np.ndarray,
    empirical_pi: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2))
    pi_vmin = min(float(baseline_pi.min()), float(target.min()), float(optimized_pi.min()))
    pi_vmax = max(float(baseline_pi.max()), float(target.max()), float(optimized_pi.max()))

    panels = [
        ("(a) lambda_C = 0 stationary pi", baseline_pi, "viridis", pi_vmin, pi_vmax),
        ("(b) Uniform target rho", target, "viridis", pi_vmin, pi_vmax),
        ("(c) Optimized stationary pi", optimized_pi, "viridis", pi_vmin, pi_vmax),
        ("(d) Optimized lambda_C map", lambda_c, "coolwarm", None, None),
        ("(e) Empirical optimized visits", empirical_pi, "viridis", pi_vmin, pi_vmax),
        ("(f) Empirical |pi_hat - rho|", np.abs(empirical_pi - target), "Reds", None, None),
    ]

    for ax, (title, values, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(values.reshape(world.Ny, world.Nx), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Inverse MaxCal coverage: fitted multipliers for uniform stationary coverage")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_maps.png", dpi=150)
    plt.close(fig)


def make_degree_figure(
    outdir: Path,
    degrees: np.ndarray,
    baseline_pi: np.ndarray,
    target: np.ndarray,
    optimized_pi: np.ndarray,
    empirical_pi: np.ndarray,
    lambda_c: np.ndarray,
    tied_pi: np.ndarray,
) -> None:
    degree_values = sorted(int(d) for d in np.unique(degrees))
    labels = [f"deg {degree}" for degree in degree_values]
    x = np.arange(len(degree_values))
    width = 0.16

    def class_means(values: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(values[degrees == degree])) for degree in degree_values])

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(x - 2 * width, class_means(baseline_pi), width=width, label="lambda=0 theory")
    axes[0].bar(x - width, class_means(target), width=width, label="uniform target")
    axes[0].bar(x, class_means(optimized_pi), width=width, label="full lambda theory")
    axes[0].bar(x + width, class_means(empirical_pi), width=width, label="full lambda empirical")
    axes[0].bar(x + 2 * width, class_means(tied_pi), width=width, label="degree-tied optimum")
    axes[0].set_xticks(x, labels=labels)
    axes[0].set_ylabel("mean stationary/empirical probability per cell")
    axes[0].set_title("Coverage probability by degree class")
    axes[0].legend(fontsize=8)

    lambda_means = class_means(lambda_c)
    lambda_min = np.array([float(np.min(lambda_c[degrees == degree])) for degree in degree_values])
    lambda_max = np.array([float(np.max(lambda_c[degrees == degree])) for degree in degree_values])
    yerr = np.vstack([lambda_means - lambda_min, lambda_max - lambda_means])
    axes[1].bar(x, lambda_means, yerr=yerr, capsize=5, color=["firebrick", "darkorange", "seagreen"])
    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].set_xticks(x, labels=labels)
    axes[1].set_ylabel("lambda_C, gauge mean zero")
    axes[1].set_title("Optimized multiplier pattern")

    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_degree_classes.png", dpi=150)
    plt.close(fig)


def make_case_comparison_figure(
    outdir: Path,
    world: mc.World,
    cases: List[Dict[str, object]],
    target: np.ndarray,
) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(12.8, 10.6))
    lambda_abs = max(float(np.max(np.abs(case["lambda_c"]))) for case in cases)
    lambda_abs = max(lambda_abs, 1e-12)
    pi_vmin = min(
        float(target.min()),
        *(float(np.min(case["theory_pi"])) for case in cases),
        *(float(np.min(case["empirical_pi"])) for case in cases),
    )
    pi_vmax = max(
        float(target.max()),
        *(float(np.max(case["theory_pi"])) for case in cases),
        *(float(np.max(case["empirical_pi"])) for case in cases),
    )

    for row, case in enumerate(cases):
        lambda_c = np.asarray(case["lambda_c"], dtype=np.float64)
        theory_pi = np.asarray(case["theory_pi"], dtype=np.float64)
        empirical_pi = np.asarray(case["empirical_pi"], dtype=np.float64)
        label = str(case["label"])

        im0 = axes[row, 0].imshow(
            lambda_c.reshape(world.Ny, world.Nx),
            origin="lower",
            cmap="coolwarm",
            vmin=-lambda_abs,
            vmax=lambda_abs,
        )
        axes[row, 0].set_title(f"{label}: lambda_C")
        axes[row, 0].set_xlabel("col")
        axes[row, 0].set_ylabel("row")
        fig.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        im1 = axes[row, 1].imshow(
            theory_pi.reshape(world.Ny, world.Nx),
            origin="lower",
            cmap="viridis",
            vmin=pi_vmin,
            vmax=pi_vmax,
        )
        axes[row, 1].set_title("theoretical stationary pi")
        axes[row, 1].set_xlabel("col")
        axes[row, 1].set_ylabel("row")
        fig.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        im2 = axes[row, 2].imshow(
            empirical_pi.reshape(world.Ny, world.Nx),
            origin="lower",
            cmap="viridis",
            vmin=pi_vmin,
            vmax=pi_vmax,
        )
        axes[row, 2].set_title("empirical visits")
        axes[row, 2].set_xlabel("col")
        axes[row, 2].set_ylabel("row")
        fig.colorbar(im2, ax=axes[row, 2], fraction=0.046)

    fig.suptitle("Coverage behavior under different MaxCal coverage multipliers")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_cases.png", dpi=150)
    plt.close(fig)


def make_degree_time_figure(
    outdir: Path,
    degrees: np.ndarray,
    target: np.ndarray,
    cases: List[Dict[str, object]],
) -> None:
    degree_values = sorted(int(d) for d in np.unique(degrees))
    colors = {3: "firebrick", 5: "darkorange", 8: "seagreen"}
    fig, axes = plt.subplots(1, len(cases), figsize=(15.0, 4.4), sharey=True)

    for ax, case in zip(axes, cases):
        sim_result: mc.SimResult = case["sim_result"]  # type: ignore[assignment]
        ck_history = np.array(sim_result.ck_history, dtype=np.float64)
        markov_steps = np.array(sim_result.markov_step_history, dtype=np.int64)
        for degree in degree_values:
            mask = degrees == degree
            class_trace = ck_history[:, mask].mean(axis=1)
            ax.plot(markov_steps, class_trace, lw=2.0, color=colors[degree], label=f"degree {degree}")
        ax.axhline(float(target[0]), color="black", ls="--", lw=1.1, label="uniform target")
        ax.set_title(str(case["label"]))
        ax.set_xlabel("Markov steps")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel("mean visit probability per cell")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Corner, edge, and interior coverage under different lambda_C choices")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_degree_traces.png", dpi=150)
    plt.close(fig)


def save_raw_data(
    outdir: Path,
    target: np.ndarray,
    baseline_pi: np.ndarray,
    optimized_pi: np.ndarray,
    empirical_pi: np.ndarray,
    lambda_c: np.ndarray,
    tied_lambda: np.ndarray,
    tied_pi: np.ndarray,
    error_history: np.ndarray,
    case_results: List[Dict[str, object]],
) -> None:
    case_arrays: Dict[str, np.ndarray] = {}
    for case in case_results:
        key = f"case_{case['key']}"
        sim_result: mc.SimResult = case["sim_result"]  # type: ignore[assignment]
        case_arrays[f"{key}_lambda_C"] = np.asarray(case["lambda_c"], dtype=np.float64)
        case_arrays[f"{key}_theory_pi"] = np.asarray(case["theory_pi"], dtype=np.float64)
        case_arrays[f"{key}_empirical_pi"] = np.asarray(case["empirical_pi"], dtype=np.float64)
        case_arrays[f"{key}_markov_steps"] = np.array(sim_result.markov_step_history, dtype=np.int64)
        case_arrays[f"{key}_ck_history"] = np.array(sim_result.ck_history, dtype=np.float64)

    np.savez(
        outdir / "maxcal_coverage_uniform_target_raw_data.npz",
        target_pi=target,
        baseline_lambda_zero_pi=baseline_pi,
        optimized_theory_pi=optimized_pi,
        optimized_lambda_C=lambda_c,
        degree_tied_lambda_C=tied_lambda,
        degree_tied_pi=tied_pi,
        solver_error_history=error_history,
        **case_arrays,
    )


def serialise_stats(stats: Dict[str, DegreeClassStats]) -> Dict[str, Dict[str, float | int]]:
    return {key: asdict(value) for key, value in stats.items()}


def save_summary(
    outdir: Path,
    args: argparse.Namespace,
    world: mc.World,
    solver: SolverResult,
    transition_checks_payload: Dict[str, float],
    baseline_pi: np.ndarray,
    target: np.ndarray,
    optimized_pi: np.ndarray,
    power_pi: np.ndarray,
    empirical_pi: np.ndarray,
    lambda_c: np.ndarray,
    tied_lambda: np.ndarray,
    tied_pi: np.ndarray,
    tied_fit: DegreeTiedFit,
    case_results: List[Dict[str, object]],
) -> None:
    degrees = degree_array(world)
    optimized_case = next(case for case in case_results if case["key"] == "full")
    optimized_empirical_pi = np.asarray(optimized_case["empirical_pi"], dtype=np.float64)
    optimized_sim_result: mc.SimResult = optimized_case["sim_result"]  # type: ignore[assignment]
    reps = representative_cells(world, target, baseline_pi, optimized_pi, optimized_empirical_pi, lambda_c)

    case_payload = []
    for case in case_results:
        case_lambda = np.asarray(case["lambda_c"], dtype=np.float64)
        case_theory = np.asarray(case["theory_pi"], dtype=np.float64)
        case_empirical = np.asarray(case["empirical_pi"], dtype=np.float64)
        case_sim: mc.SimResult = case["sim_result"]  # type: ignore[assignment]
        case_payload.append(
            {
                "key": str(case["key"]),
                "label": str(case["label"]),
                "theory_l1_error_to_uniform_target": float(np.sum(np.abs(case_theory - target))),
                "theory_max_abs_error_to_uniform_target": float(np.max(np.abs(case_theory - target))),
                "empirical_l1_error_to_uniform_target": float(np.sum(np.abs(case_empirical - target))),
                "empirical_max_abs_error_to_uniform_target": float(np.max(np.abs(case_empirical - target))),
                "total_markov_steps": int(case_sim.markov_step_history[-1]),
                "lambda_C_by_degree": serialise_stats(stats_by_degree(case_lambda, degrees)),
                "theory_stationary_by_degree": serialise_stats(stats_by_degree(case_theory, degrees)),
                "empirical_stationary_by_degree": serialise_stats(stats_by_degree(case_empirical, degrees)),
            }
        )

    summary = {
        "config": {
            "Nx": mc.NX,
            "Ny": mc.NY,
            "K": world.K,
            "cell_size_m": mc.CELL_SIZE,
            "robots": mc.N_ROBOTS,
            "simulation_steps": args.T,
            "speed_m_per_step": args.speed,
            "seed": args.seed,
            "target": "uniform coverage, rho_k = 1 / K",
            "lambda_gauge": "mean(lambda_C) = 0; adding a constant does not change P",
        },
        "theory": {
            "kernel": "P_ij = A_ij exp(-lambda_j) / sum_l A_il exp(-lambda_l)",
            "stationary_formula": "pi_i = b_i (A b)_i / sum_m b_m (A b)_m, b_i = exp(-lambda_i)",
            "inverse_condition": "For uniform rho, solve b_i (A b)_i = constant for every cell.",
            "interpretation": (
                "Lower lambda_C means larger exp(-lambda_C), so the cell is made more attractive. "
                "Boundary cells need lower lambda_C to compensate for fewer neighbors."
            ),
        },
        "baseline_lambda_zero": {
            "l1_error_to_uniform_target": float(np.sum(np.abs(baseline_pi - target))),
            "max_abs_error_to_uniform_target": float(np.max(np.abs(baseline_pi - target))),
            "stationary_by_degree": serialise_stats(stats_by_degree(baseline_pi, degrees)),
        },
        "optimized_full_per_cell_lambda": {
            "solver": asdict(solver),
            "transition_checks": transition_checks_payload,
            "power_iteration_l1_error_to_target": float(np.sum(np.abs(power_pi - target))),
            "power_iteration_max_abs_error_to_target": float(np.max(np.abs(power_pi - target))),
            "lambda_C_by_degree": serialise_stats(stats_by_degree(lambda_c, degrees)),
            "exp_minus_lambda_by_degree": serialise_stats(stats_by_degree(np.exp(-lambda_c), degrees)),
            "stationary_by_degree": serialise_stats(stats_by_degree(optimized_pi, degrees)),
            "representative_cells": {label: asdict(rep) for label, rep in reps.items()},
        },
        "degree_tied_optimized_lambda": {
            "note": (
                "This is the best three-class corner/edge/interior fit found by coordinate-pattern "
                "search. It explains the multiplier ordering, but it is not exact because cells with "
                "the same degree can still have different neighbor compositions."
            ),
            "fit": asdict(tied_fit),
            "lambda_C_by_degree": serialise_stats(stats_by_degree(tied_lambda, degrees)),
            "stationary_by_degree": serialise_stats(stats_by_degree(tied_pi, degrees)),
            "l1_error_to_uniform_target": float(np.sum(np.abs(tied_pi - target))),
            "max_abs_error_to_uniform_target": float(np.max(np.abs(tied_pi - target))),
        },
        "case_comparison": {
            "note": (
                "These cases make the effect of different lambda_C choices visible: zero multipliers "
                "recover the degree-biased random walk; degree-tied multipliers partially compensate "
                "corners and edges; full per-cell multipliers solve the uniform stationary target."
            ),
            "cases": case_payload,
        },
        "empirical_optimized_simulation": {
            "total_markov_steps": int(optimized_sim_result.markov_step_history[-1]),
            "l1_error_to_uniform_target": float(np.sum(np.abs(optimized_empirical_pi - target))),
            "max_abs_error_to_uniform_target": float(np.max(np.abs(optimized_empirical_pi - target))),
            "stationary_by_degree": serialise_stats(stats_by_degree(optimized_empirical_pi, degrees)),
        },
    }

    summary["uniform_coverage_inverse_maxcal_validated"] = bool(
        solver.converged
        and solver.max_abs_stationary_error < args.tol * 10.0
        and transition_checks_payload["max_row_sum_error"] < 1e-12
        and transition_checks_payload["max_off_neighbor_probability"] < 1e-12
        and summary["optimized_full_per_cell_lambda"]["power_iteration_max_abs_error_to_target"] < 1e-10
        and summary["baseline_lambda_zero"]["l1_error_to_uniform_target"]
        > summary["optimized_full_per_cell_lambda"]["solver"]["l1_stationary_error"]
    )

    with open(outdir / "maxcal_coverage_uniform_target_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    world = mc.build_world(mc.NX, mc.NY, mc.CELL_SIZE)
    adjacency = adjacency_matrix(world)
    target = np.full(world.K, 1.0 / world.K, dtype=np.float64)
    baseline_pi = mc.theoretical_stationary(world)

    lambda_c, optimized_pi, solver, error_history = solve_uniform_target_multipliers(
        adjacency=adjacency,
        target=target,
        tol=args.tol,
        max_iters=args.max_iters,
    )
    optimized_transition = mc.build_transition_matrix(world, lambda_c)
    checks = transition_checks(world, optimized_transition)
    power_pi = power_stationary(optimized_transition)

    degrees = degree_array(world)
    tied_lambda, tied_pi, tied_fit = optimise_degree_tied_multipliers(
        adjacency=adjacency,
        target=target,
        degrees=degrees,
        initial_lambda_c=lambda_c,
    )
    baseline_lambda = np.zeros(world.K, dtype=np.float64)
    case_specs = [
        ("zero", "lambda_C = 0", baseline_lambda, baseline_pi),
        ("degree_tied", "degree-tied lambda_C", tied_lambda, tied_pi),
        ("full", "full per-cell lambda_C", lambda_c, optimized_pi),
    ]
    case_results: List[Dict[str, object]] = []
    for offset, (key, label, case_lambda, case_theory) in enumerate(case_specs):
        sim_result = run_simulation_with_lambda(
            lambda_c=case_lambda,
            speed=args.speed,
            T=args.T,
            seed=args.seed + offset,
        )
        case_results.append(
            {
                "key": key,
                "label": label,
                "lambda_c": case_lambda,
                "theory_pi": case_theory,
                "empirical_pi": sim_result.pi_empirical,
                "sim_result": sim_result,
            }
        )
    full_empirical_pi = np.asarray(case_results[-1]["empirical_pi"], dtype=np.float64)

    make_maps_figure(outdir, world, baseline_pi, target, optimized_pi, lambda_c, full_empirical_pi)
    make_degree_figure(outdir, degrees, baseline_pi, target, optimized_pi, full_empirical_pi, lambda_c, tied_pi)
    make_case_comparison_figure(outdir, world, case_results, target)
    make_degree_time_figure(outdir, degrees, target, case_results)
    save_raw_data(
        outdir=outdir,
        target=target,
        baseline_pi=baseline_pi,
        optimized_pi=optimized_pi,
        empirical_pi=full_empirical_pi,
        lambda_c=lambda_c,
        tied_lambda=tied_lambda,
        tied_pi=tied_pi,
        error_history=error_history,
        case_results=case_results,
    )
    save_summary(
        outdir=outdir,
        args=args,
        world=world,
        solver=solver,
        transition_checks_payload=checks,
        baseline_pi=baseline_pi,
        target=target,
        optimized_pi=optimized_pi,
        power_pi=power_pi,
        empirical_pi=full_empirical_pi,
        lambda_c=lambda_c,
        tied_lambda=tied_lambda,
        tied_pi=tied_pi,
        tied_fit=tied_fit,
        case_results=case_results,
    )

    lambda_stats = stats_by_degree(lambda_c, degrees)
    print("Inverse MaxCal Coverage Validation: Uniform Target")
    print(f"  Output directory       : {outdir}")
    print(f"  Solver converged       : {solver.converged} in {solver.iterations} iterations")
    print(f"  Theory max abs error   : {solver.max_abs_stationary_error:.3e}")
    print(f"  lambda=0 L1 vs uniform : {np.sum(np.abs(baseline_pi - target)):.6f}")
    print(f"  optimized L1 vs uniform: {solver.l1_stationary_error:.3e}")
    print(f"  degree-tied L1 vs uniform: {tied_fit.l1_stationary_error:.6f}")
    print(f"  empirical L1 vs uniform: {np.sum(np.abs(full_empirical_pi - target)):.6f}")
    print("  Mean lambda_C by degree:")
    for degree in sorted(lambda_stats, key=int):
        stat = lambda_stats[degree]
        print(
            f"    degree {degree}: mean={stat.mean:+.6f}, "
            f"min={stat.minimum:+.6f}, max={stat.maximum:+.6f}"
        )
    print("  Saved summary          : maxcal_coverage_uniform_target_summary.json")


if __name__ == "__main__":
    main()
