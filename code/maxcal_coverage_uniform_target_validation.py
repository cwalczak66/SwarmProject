"""
RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Validate inverse MaxCal coverage for a uniform stationary target.
The forward baseline ``lambda_C=0`` gives the degree-biased stationary law

    pi_i = deg(i) / sum_m deg(m),

which is not uniform on a finite 8-connected grid because corners, edges, and
interior cells have different degrees.  The inverse problem asks for

    pi_bar_i = 1/K,
    pi_i(b) = b_i(A b)_i / sum_m b_m(A b)_m = pi_bar_i,
    b_i = exp[-lambda_C,i].

The solved per-cell ``lambda_C`` is an offline design object; each robot still
executes the same local rule by reading the multiplier values of neighboring
destinations.  The degree-tied controller in this script is only a diagnostic:
it shows the corner/edge/interior ordering, but cannot exactly solve the
uniform target because equal-degree cells can have different neighbor
compositions.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import maxcal_core as core
import maxcal_coverage as mc


FIGURES = [
    "maxcal_coverage_uniform_target_maps.png",
    "maxcal_coverage_uniform_target_cases.png",
    "maxcal_coverage_uniform_target_degree_classes.png",
    "maxcal_coverage_uniform_target_degree_traces.png",
]


@dataclass
class DegreeTiedFit:
    iterations: int
    final_step: float
    objective_l2: float
    l1_stationary_error: float
    max_abs_stationary_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate inverse MaxCal coverage for a uniform target.")
    parser.add_argument("--outdir", type=str, default="maxcal_coverage_uniform_target_validation")
    parser.add_argument("--T", type=int, default=60_000)
    parser.add_argument("--speed", type=float, default=mc.ROBOT_SPEED)
    parser.add_argument("--seed", type=int, default=mc.SEED + 1000)
    parser.add_argument("--tol", type=float, default=1.0e-12)
    parser.add_argument("--max-iters", type=int, default=100_000)
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


def degree_array(world: mc.World) -> np.ndarray:
    """Return the graph degree of every cell."""
    return np.array([len(neighbors) for neighbors in world.adjacency], dtype=np.int64)


def lambda_from_degree_params(params: np.ndarray, degree_values: np.ndarray, degrees: np.ndarray) -> np.ndarray:
    """Expand three degree-class parameters into a mean-zero multiplier field."""
    lambda_c = np.zeros_like(degrees, dtype=np.float64)
    for idx, degree in enumerate(degree_values):
        lambda_c[degrees == degree] = float(params[idx])
    lambda_c -= float(lambda_c.mean())
    return lambda_c


def optimize_degree_tied_multipliers(
    world: mc.World,
    target: np.ndarray,
    initial_lambda: np.ndarray,
    min_step: float = 1.0e-8,
    max_iters: int = 20_000,
) -> tuple[np.ndarray, np.ndarray, DegreeTiedFit]:
    """Best diagnostic fit using one ``lambda_C`` per degree class.

    The full inverse solve has one multiplier per cell.  This reduced fit is
    intentionally not the production controller; it isolates the intuitive
    boundary effect predicted by the stationary formula.
    """
    degrees = degree_array(world)
    degree_values = np.array(sorted(np.unique(degrees)), dtype=np.int64)
    params = np.array([initial_lambda[degrees == degree].mean() for degree in degree_values], dtype=np.float64)

    def evaluate(candidate: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        lambda_c = lambda_from_degree_params(candidate, degree_values, degrees)
        pi = mc.stationary_from_lambda(world, lambda_c)
        diff = pi - target
        return float(np.dot(diff, diff)), lambda_c, pi

    best_obj, best_lambda, best_pi = evaluate(params)
    step = 0.5
    iterations = 0
    while step > min_step and iterations < max_iters:
        improved = False
        for dim in range(len(params)):
            for direction in (1.0, -1.0):
                candidate = params.copy()
                candidate[dim] += direction * step
                obj, lambda_c, pi = evaluate(candidate)
                iterations += 1
                if obj < best_obj:
                    params, best_obj, best_lambda, best_pi = candidate, obj, lambda_c, pi
                    improved = True
                    break
            if improved:
                break
        if not improved:
            step *= 0.5

    fit = DegreeTiedFit(
        iterations=iterations,
        final_step=step,
        objective_l2=best_obj,
        l1_stationary_error=core.l1_error(best_pi, target),
        max_abs_stationary_error=float(np.max(np.abs(best_pi - target))),
    )
    return best_lambda, best_pi, fit


def transition_checks(world: mc.World, transition: np.ndarray) -> dict[str, float]:
    """Check stochastic rows and neighbor-only support of the solved kernel."""
    off_neighbor = np.ones_like(transition, dtype=bool)
    for cell, neighbors in enumerate(world.adjacency):
        off_neighbor[cell, neighbors] = False
        off_neighbor[cell, cell] = False
    positive = np.concatenate([transition[cell, neighbors] for cell, neighbors in enumerate(world.adjacency)])
    return {
        "max_row_sum_error": float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
        "max_off_neighbor_probability": float(np.max(np.abs(transition[off_neighbor]))),
        "minimum_positive_neighbor_probability": float(np.min(positive)),
        "maximum_neighbor_probability": float(np.max(positive)),
    }


def stats_by_degree(values: np.ndarray, degrees: np.ndarray) -> dict[str, dict[str, float | int]]:
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


def representative_cells(
    world: mc.World,
    target: np.ndarray,
    baseline: np.ndarray,
    degree_tied: np.ndarray,
    full: np.ndarray,
    empirical: np.ndarray,
    lambda_c: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    reps = {
        "corner": 0,
        "edge": mc.NX // 2,
        "interior": (mc.NY // 2) * mc.NX + mc.NX // 2,
    }
    return {
        label: {
            "index": int(cell),
            "degree": int(len(world.adjacency[cell])),
            "target_pi": float(target[cell]),
            "baseline_pi": float(baseline[cell]),
            "degree_tied_pi": float(degree_tied[cell]),
            "full_theory_pi": float(full[cell]),
            "full_empirical_pi": float(empirical[cell]),
            "full_empirical_abs_error": float(abs(empirical[cell] - target[cell])),
            "lambda_C": float(lambda_c[cell]),
        }
        for label, cell in reps.items()
    }


def run_with_lambda(lambda_c: np.ndarray, target: np.ndarray, speed: float, T: int, seed: int) -> mc.SimResult:
    return mc.run_simulation(
        speed=speed,
        T=T,
        seed=seed,
        target_mode="explicit_lambda",
        lambda_C=lambda_c,
        target_pi=target,
    )


def save_raw_data(outdir: Path, payload: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {
        "target": payload["target"],
        "baseline_pi": payload["baseline_pi"],
        "degree_tied_pi": payload["degree_tied_pi"],
        "full_pi": payload["full_pi"],
        "full_empirical_pi": payload["empirical_pi"],
        "lambda_C": payload["lambda_C"],
        "degree_tied_lambda_C": payload["degree_tied_lambda_C"],
        "solver_error_history": payload["solution"].error_history,
    }
    for case in payload["cases"]:
        result: mc.SimResult = case["result"]
        prefix = f"case_{case['key']}"
        arrays[f"{prefix}_lambda_C"] = np.asarray(case["lambda_C"], dtype=np.float64)
        arrays[f"{prefix}_theory_pi"] = np.asarray(case["theory_pi"], dtype=np.float64)
        arrays[f"{prefix}_empirical_pi"] = result.pi_empirical
        arrays[f"{prefix}_markov_steps"] = np.asarray(result.markov_step_history, dtype=np.int64)
        arrays[f"{prefix}_ck_history"] = np.asarray(result.ck_history, dtype=np.float64)
    np.savez_compressed(outdir / "maxcal_coverage_uniform_target_raw_data.npz", **arrays)


def make_maps_figure(
    outdir: Path,
    world: mc.World,
    baseline_pi: np.ndarray,
    target: np.ndarray,
    full_pi: np.ndarray,
    lambda_c: np.ndarray,
    empirical_pi: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.2))
    pi_vmin = min(float(baseline_pi.min()), float(target.min()), float(full_pi.min()), float(empirical_pi.min()))
    pi_vmax = max(float(baseline_pi.max()), float(target.max()), float(full_pi.max()), float(empirical_pi.max()))
    lambda_abs = max(float(np.max(np.abs(lambda_c))), 1.0e-12)
    panels = [
        ("(a) lambda_C = 0 stationary π", baseline_pi, "viridis", pi_vmin, pi_vmax),
        ("(b) uniform target π̄", target, "viridis", pi_vmin, pi_vmax),
        ("(c) optimized stationary π", full_pi, "viridis", pi_vmin, pi_vmax),
        ("(d) optimized lambda_C map", lambda_c, "coolwarm", -lambda_abs, lambda_abs),
        ("(e) empirical optimized visits", empirical_pi, "viridis", pi_vmin, pi_vmax),
        ("(f) empirical absolute error", np.abs(empirical_pi - target), "Reds", None, None),
    ]
    for ax, (title, values, cmap, vmin, vmax) in zip(axes.flat, panels):
        im = ax.imshow(values.reshape(world.Ny, world.Nx), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Inverse MaxCal coverage: multipliers for uniform stationary coverage")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_maps.png", dpi=150)
    plt.close(fig)


def make_degree_figure(
    outdir: Path,
    degrees: np.ndarray,
    baseline_pi: np.ndarray,
    target: np.ndarray,
    full_pi: np.ndarray,
    empirical_pi: np.ndarray,
    lambda_c: np.ndarray,
    tied_pi: np.ndarray,
) -> None:
    degree_values = np.array(sorted(np.unique(degrees)), dtype=np.int64)
    labels = [f"deg {int(degree)}" for degree in degree_values]
    x = np.arange(len(degree_values))
    width = 0.16

    def class_means(values: np.ndarray) -> np.ndarray:
        return np.array([float(np.mean(values[degrees == degree])) for degree in degree_values], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    axes[0].bar(x - 2 * width, class_means(baseline_pi), width=width, label="lambda=0 theory")
    axes[0].bar(x - width, class_means(target), width=width, label="uniform target")
    axes[0].bar(x, class_means(full_pi), width=width, label="full lambda theory")
    axes[0].bar(x + width, class_means(empirical_pi), width=width, label="full lambda empirical")
    axes[0].bar(x + 2 * width, class_means(tied_pi), width=width, label="degree-tied theory")
    axes[0].set_xticks(x, labels=labels)
    axes[0].set_ylabel("mean probability per cell")
    axes[0].set_title("coverage probability by degree class")
    axes[0].legend(fontsize=8)

    lambda_means = class_means(lambda_c)
    lambda_min = np.array([float(np.min(lambda_c[degrees == degree])) for degree in degree_values])
    lambda_max = np.array([float(np.max(lambda_c[degrees == degree])) for degree in degree_values])
    yerr = np.vstack([lambda_means - lambda_min, lambda_max - lambda_means])
    axes[1].bar(x, lambda_means, yerr=yerr, capsize=5, color=["firebrick", "darkorange", "seagreen"])
    axes[1].axhline(0.0, color="black", lw=1.0)
    axes[1].set_xticks(x, labels=labels)
    axes[1].set_ylabel("lambda_C, gauge mean zero")
    axes[1].set_title("optimized multiplier by degree")

    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_degree_classes.png", dpi=150)
    plt.close(fig)


def make_case_comparison_figure(outdir: Path, world: mc.World, cases: list[dict[str, Any]], target: np.ndarray) -> None:
    fig, axes = plt.subplots(len(cases), 3, figsize=(12.8, 10.6))
    lambda_abs = max(max(float(np.max(np.abs(case["lambda_C"]))) for case in cases), 1.0e-12)
    pi_vmin = min(float(target.min()), *(float(np.min(case["theory_pi"])) for case in cases))
    pi_vmax = max(float(target.max()), *(float(np.max(case["theory_pi"])) for case in cases))
    for row, case in enumerate(cases):
        result: mc.SimResult = case["result"]
        panels = [
            ("lambda_C", np.asarray(case["lambda_C"]), "coolwarm", -lambda_abs, lambda_abs),
            ("theoretical stationary π̄", np.asarray(case["theory_pi"]), "viridis", pi_vmin, pi_vmax),
            ("empirical visits", result.pi_empirical, "viridis", pi_vmin, pi_vmax),
        ]
        for col, (title, values, cmap, vmin, vmax) in enumerate(panels):
            im = axes[row, col].imshow(values.reshape(world.Ny, world.Nx), origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{case['label']}: {title}" if col == 0 else title)
            axes[row, col].set_xlabel("col")
            axes[row, col].set_ylabel("row")
            fig.colorbar(im, ax=axes[row, col], fraction=0.046)
    fig.suptitle("Coverage behavior under different MaxCal multipliers")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_cases.png", dpi=150)
    plt.close(fig)


def make_degree_trace_figure(outdir: Path, degrees: np.ndarray, target: np.ndarray, cases: list[dict[str, Any]]) -> None:
    degree_values = np.array(sorted(np.unique(degrees)), dtype=np.int64)
    colors = {3: "firebrick", 5: "darkorange", 8: "seagreen"}
    fig, axes = plt.subplots(1, len(cases), figsize=(15.0, 4.4), sharey=True)
    if len(cases) == 1:
        axes = np.array([axes])
    for ax, case in zip(axes, cases):
        result: mc.SimResult = case["result"]
        ck_history = np.asarray(result.ck_history, dtype=np.float64)
        markov_steps = np.asarray(result.markov_step_history, dtype=np.int64)
        if len(ck_history):
            for degree in degree_values:
                mask = degrees == degree
                ax.plot(markov_steps, ck_history[:, mask].mean(axis=1), lw=2.0, color=colors[int(degree)], label=f"degree {int(degree)}")
        ax.axhline(float(target[0]), color="black", ls="--", lw=1.1, label="uniform target")
        ax.set_title(str(case["label"]))
        ax.set_xlabel("Markov arrivals")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean visit probability per cell")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Corner, edge, and interior convergence under different lambda_C choices")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_uniform_target_degree_traces.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    world = mc.build_world(mc.NX, mc.NY, mc.CELL_SIZE)
    target = mc.uniform_stationary(world)
    baseline_pi = mc.theoretical_stationary(world)

    solution = mc.solve_coverage_multipliers_for_target(world, target, tol=args.tol, max_iters=args.max_iters)
    full_pi = solution.stationary
    full_transition = mc.build_transition_matrix(world, solution.lambda_C)
    degree_tied_lambda, degree_tied_pi, degree_fit = optimize_degree_tied_multipliers(world, target, solution.lambda_C)
    case_specs = [
        ("zero", "lambda_C = 0", np.zeros(world.K, dtype=np.float64), baseline_pi),
        ("degree_tied", "degree-tied lambda_C", degree_tied_lambda, degree_tied_pi),
        ("full", "full per-cell lambda_C", solution.lambda_C, full_pi),
    ]
    cases: list[dict[str, Any]] = []
    for offset, (key, label, lambda_c, theory_pi) in enumerate(case_specs):
        result = run_with_lambda(lambda_c, target, args.speed, args.T, args.seed + offset)
        cases.append(
            {
                "key": key,
                "label": label,
                "lambda_C": lambda_c,
                "theory_pi": theory_pi,
                "result": result,
            }
        )

    full_case = next(case for case in cases if case["key"] == "full")
    empirical: mc.SimResult = full_case["result"]
    empirical_l1 = core.l1_error(empirical.pi_empirical, target)

    degrees = degree_array(world)
    payload = {
        "target": target,
        "baseline_pi": baseline_pi,
        "degree_tied_pi": degree_tied_pi,
        "degree_tied_lambda_C": degree_tied_lambda,
        "full_pi": full_pi,
        "empirical_pi": empirical.pi_empirical,
        "lambda_C": solution.lambda_C,
        "solution": solution,
        "cases": cases,
    }
    save_raw_data(outdir, payload)
    make_maps_figure(outdir, world, baseline_pi, target, full_pi, solution.lambda_C, empirical.pi_empirical)
    make_degree_figure(outdir, degrees, baseline_pi, target, full_pi, empirical.pi_empirical, solution.lambda_C, degree_tied_pi)
    make_case_comparison_figure(outdir, world, cases, target)
    make_degree_trace_figure(outdir, degrees, target, cases)

    case_summary = []
    for case in cases:
        result: mc.SimResult = case["result"]
        theory_pi = np.asarray(case["theory_pi"], dtype=np.float64)
        case_summary.append(
            {
                "key": str(case["key"]),
                "label": str(case["label"]),
                "theory_l1_error": core.l1_error(theory_pi, target),
                "theory_max_abs_error": float(np.max(np.abs(theory_pi - target))),
                "empirical_l1_error": core.l1_error(result.pi_empirical, target),
                "empirical_max_abs_error": float(np.max(np.abs(result.pi_empirical - target))),
                "total_markov_arrivals": int(result.markov_step_history[-1]) if result.markov_step_history else 0,
                "lambda_by_degree": stats_by_degree(np.asarray(case["lambda_C"], dtype=np.float64), degrees),
                "theory_by_degree": stats_by_degree(theory_pi, degrees),
                "empirical_by_degree": stats_by_degree(result.pi_empirical, degrees),
            }
        )

    checks = {
        "zero_multiplier_is_not_uniform_on_irregular_grid": bool(core.l1_error(baseline_pi, target) > 1.0e-3),
        "full_inverse_solver_converged": bool(solution.converged),
        "full_inverse_stationary_matches_uniform": bool(solution.l1_stationary_error < 1.0e-8),
        "degree_tied_is_only_a_partial_fit": bool(degree_fit.l1_stationary_error > 10.0 * solution.l1_stationary_error),
        "empirical_run_completed": bool(empirical.markov_step_history),
    }
    summary = {
        "paper_alignment": {
            "layer": "Inverse Layer 1-C coverage",
            "claim": "For a requested coverage density rho, inverse MaxCal chooses destination multipliers so the reversible stationary law pi(lambda_C) matches rho.",
            "validated_outputs": [
                "lambda=0 degree-biased baseline",
                "full per-cell inverse multiplier field",
                "uniform-target stationary residual",
                "finite-run empirical visit histogram",
            ],
        },
        "environment": {
            "Nx": mc.NX,
            "Ny": mc.NY,
            "K": world.K,
            "cell_size": mc.CELL_SIZE,
            "robots": mc.N_ROBOTS,
            "T": args.T,
            "speed": args.speed,
        },
        "zero_multiplier": {
            "l1_stationary_error": core.l1_error(baseline_pi, target),
            "max_abs_stationary_error": float(np.max(np.abs(baseline_pi - target))),
        },
        "degree_tied": asdict(degree_fit),
        "full_per_cell": {
            "iterations": solution.iterations,
            "converged": solution.converged,
            "l1_stationary_error": solution.l1_stationary_error,
            "max_abs_stationary_error": solution.max_abs_stationary_error,
            "transition_checks": transition_checks(world, full_transition),
        },
        "empirical": {
            "total_markov_arrivals": int(empirical.markov_step_history[-1]) if empirical.markov_step_history else 0,
            "l1_error": empirical_l1,
            "max_abs_error": float(np.max(np.abs(empirical.pi_empirical - target))),
        },
        "lambda_by_degree": stats_by_degree(solution.lambda_C, degrees),
        "stationary_by_degree": stats_by_degree(full_pi, degrees),
        "representative_cells": representative_cells(
            world, target, baseline_pi, degree_tied_pi, full_pi, empirical.pi_empirical, solution.lambda_C
        ),
        "case_comparison": {
            "note": "zero multipliers recover degree-biased coverage; degree-tied multipliers partially compensate; full per-cell multipliers solve the uniform target.",
            "cases": case_summary,
        },
        "checks": checks,
        "inverse_coverage_ready": bool(all(checks.values())),
        "figures": FIGURES,
    }
    with open(outdir / "maxcal_coverage_uniform_target_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)

    print("Inverse MaxCal Coverage Validation: Uniform Target")
    print(f"  Output directory        : {outdir}")
    print(f"  Validation ready        : {summary['inverse_coverage_ready']}")
    print(f"  Solver converged        : {solution.converged} in {solution.iterations} iterations")
    print(f"  Theory max abs error    : {solution.max_abs_stationary_error:.3e}")
    print(f"  lambda=0 L1 vs uniform  : {summary['zero_multiplier']['l1_stationary_error']:.6f}")
    print(f"  optimized L1 vs uniform : {solution.l1_stationary_error:.3e}")
    print(f"  degree-tied L1 vs uniform: {degree_fit.l1_stationary_error:.6f}")
    print(f"  empirical L1 vs uniform : {empirical_l1:.6f}")
    print("  Mean lambda_C by degree :")
    for degree, stat in stats_by_degree(solution.lambda_C, degrees).items():
        print(f"    degree {degree}: mean={stat['mean']:+.6f}, min={stat['min']:+.6f}, max={stat['max']:+.6f}")
    print("  Saved summary           : maxcal_coverage_uniform_target_summary.json")


if __name__ == "__main__":
    main()
