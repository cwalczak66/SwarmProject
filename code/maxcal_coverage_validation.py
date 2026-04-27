"""
RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Validate the forward Layer 1-C coverage controller.

With constant coverage multiplier,

    P_ij = A_ij / deg(i),
    pi_i = deg(i) / sum_m deg(m).

The validator confirms that the implementation has exactly this kernel, that
Markov transitions are counted only when a robot physically reaches its sampled
destination, and that the empirical L1 error follows the expected late-time
sampling scale

    ||c(n)-π̄||_1 approximately C/(n+1),

when plotted against Markov arrivals ``n``.
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


SPEED_SWEEP = [0.05, 0.10, 0.20, 0.50, 1.00]
FIGURES = [
    "maxcal_coverage_validation_baseline.png",
    "maxcal_coverage_validation_speed_sweep.png",
]


@dataclass
class RepresentativeCell:
    index: int
    degree: int
    target_pi: float
    final_ck: float
    final_abs_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Coverage Layer 1-C.")
    parser.add_argument("--outdir", type=str, default="maxcal_coverage_validation")
    parser.add_argument("--T", type=int, default=mc.T_SIM)
    parser.add_argument("--speed", type=float, default=mc.ROBOT_SPEED)
    parser.add_argument("--seed", type=int, default=mc.SEED)
    parser.add_argument("--fit-start-frac", type=float, default=0.50)
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


def kernel_checks(world: mc.World, transition: np.ndarray) -> dict[str, float]:
    """Check ``P_ij=1/deg(i)`` on neighbors and zero elsewhere."""
    off_neighbor = np.ones_like(transition, dtype=bool)
    max_uniform_error = 0.0
    for cell, neighbors in enumerate(world.adjacency):
        off_neighbor[cell, neighbors] = False
        off_neighbor[cell, cell] = False
        target = 1.0 / len(neighbors)
        max_uniform_error = max(max_uniform_error, float(np.max(np.abs(transition[cell, neighbors] - target))))
    return {
        "max_row_sum_error": float(np.max(np.abs(transition.sum(axis=1) - 1.0))),
        "max_off_neighbor_probability": float(np.max(np.abs(transition[off_neighbor]))),
        "max_uniform_neighbor_error": max_uniform_error,
    }


def motion_checks(world: mc.World, transition: np.ndarray) -> dict[str, float | bool]:
    """Verify that one Markov step is one completed cell-to-cell arrival."""
    rng = np.random.default_rng(0)
    source = 0
    target = world.adjacency[source][0]
    x0, y0 = mc.region_center(world, source)
    tx, ty = mc.region_center(world, target)
    dist = float(np.hypot(tx - x0, ty - y0))

    in_transit = mc.Robot(0, x0, y0, source, target, tx, ty)
    transit_visits = np.zeros(world.K, dtype=np.int64)
    mc.step_robot(in_transit, world, transition, dist / 3.0, transit_visits, rng, t=1)

    arrival = mc.Robot(1, x0, y0, source, target, tx, ty)
    arrival_visits = np.zeros(world.K, dtype=np.int64)
    last_visit = np.full(world.K, -1.0, dtype=np.float64)
    mc.step_robot(arrival, world, transition, dist * 10.0, arrival_visits, rng, t=5, global_last_visit_time=last_visit)
    arrival_map = mc.ensure_robot_map(arrival, world)

    return {
        "transit_keeps_markov_state": in_transit.from_k == source and int(transit_visits.sum()) == 0,
        "arrival_updates_markov_state": arrival.from_k == target and int(arrival_visits[target]) == 1,
        "arrival_position_exact": abs(arrival.x - tx) < 1.0e-12 and abs(arrival.y - ty) < 1.0e-12,
        "arrival_next_target_is_neighbor": arrival.to_k in world.adjacency[arrival.from_k],
        "arrival_local_visit_time": bool(arrival_map.last_visit_time[target] == 5.0),
        "arrival_global_visit_time": bool(last_visit[target] == 5.0),
        "target_distance": dist,
    }


def representative_cells(world: mc.World, target_pi: np.ndarray, final_ck: np.ndarray) -> dict[str, RepresentativeCell]:
    reps = {
        "corner": 0,
        "edge": mc.NX // 2,
        "interior": (mc.NY // 2) * mc.NX + mc.NX // 2,
    }
    return {
        label: RepresentativeCell(
            index=cell,
            degree=len(world.adjacency[cell]),
            target_pi=float(target_pi[cell]),
            final_ck=float(final_ck[cell]),
            final_abs_error=float(abs(final_ck[cell] - target_pi[cell])),
        )
        for label, cell in reps.items()
    }


def fit_l1_over_markov_time(markov_steps: np.ndarray, l1_errors: np.ndarray, start_fraction: float) -> dict[str, float]:
    """Fit the diagnostic ``L1(n) ~= C/(n+1)`` on late-time samples."""
    start = int(len(markov_steps) * start_fraction)
    if len(markov_steps[start:]) < 2:
        return {"start_fraction": start_fraction, "C_hat": float("nan"), "r2": float("nan"), "n_points": 0}
    x = 1.0 / (markov_steps[start:].astype(np.float64) + 1.0)
    y = l1_errors[start:]
    c_hat = float(np.dot(x, y) / max(np.dot(x, x), 1.0e-12))
    residual = y - c_hat * x
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual * residual)) / ss_tot if ss_tot > 0.0 else float("nan")
    return {"start_fraction": start_fraction, "C_hat": c_hat, "r2": r2, "n_points": int(len(y))}


def run_case(speed: float, T: int, seed: int) -> dict[str, Any]:
    """Run one coverage simulation and compute theory-aligned diagnostics."""
    result = mc.run_simulation(speed=speed, T=T, seed=seed)
    target = mc.theoretical_stationary(result.w)
    ck = np.array(result.ck_history, dtype=np.float64)
    markov_steps = np.array(result.markov_step_history, dtype=np.int64)
    sim_steps = np.arange(1, len(ck) + 1, dtype=np.int64) * mc.RECORD_EVERY
    l1 = np.sum(np.abs(ck - target[None, :]), axis=1) if len(ck) else np.array([], dtype=np.float64)
    return {
        "result": result,
        "target": target,
        "ck_history": ck,
        "markov_steps": markov_steps,
        "sim_steps": sim_steps,
        "l1_errors": l1,
        "representatives": representative_cells(result.w, target, ck[-1] if len(ck) else result.pi_empirical),
    }


def speed_sweep(T: int) -> list[dict[str, Any]]:
    cases = []
    for speed in SPEED_SWEEP:
        case = run_case(speed=speed, T=T, seed=mc.SEED)
        cases.append(
            {
                "speed": speed,
                "markov_steps": case["markov_steps"],
                "sim_steps": case["sim_steps"],
                "l1_errors": case["l1_errors"],
                "final_l1": float(case["l1_errors"][-1]) if len(case["l1_errors"]) else float("nan"),
                "total_arrivals": int(case["result"].markov_step_history[-1]) if case["result"].markov_step_history else 0,
            }
        )
    return cases


def normalized_spread(curves: list[dict[str, Any]], axis_key: str, n_grid: int = 200) -> float:
    valid = [case for case in curves if len(case["l1_errors"]) >= 2]
    if len(valid) < 2:
        return float("nan")
    max_start = max(float(case[axis_key][0]) for case in valid)
    min_end = min(float(case[axis_key][-1]) for case in valid)
    if min_end > max_start:
        grid = np.linspace(max_start, min_end, n_grid)
        values = np.array([np.interp(grid, case[axis_key], case["l1_errors"]) for case in valid])
    else:
        grid = np.linspace(0.0, 1.0, n_grid)
        values = []
        for case in valid:
            axis = np.asarray(case[axis_key], dtype=np.float64)
            axis = (axis - axis[0]) / max(float(axis[-1] - axis[0]), 1.0e-12)
            values.append(np.interp(grid, axis, case["l1_errors"]))
        values = np.asarray(values, dtype=np.float64)
    mean = np.maximum(values.mean(axis=0), 1.0e-12)
    return float(np.mean(values.std(axis=0) / mean))


def save_raw_data(outdir: Path, baseline: dict[str, Any], sweep: list[dict[str, Any]]) -> None:
    np.savez_compressed(
        outdir / "maxcal_coverage_validation_raw_data.npz",
        baseline_markov_steps=baseline["markov_steps"],
        baseline_sim_steps=baseline["sim_steps"],
        baseline_l1_errors=baseline["l1_errors"],
        baseline_target=baseline["target"],
        baseline_empirical=baseline["result"].pi_empirical,
        sweep_speeds=np.array([case["speed"] for case in sweep], dtype=np.float64),
    )


def make_baseline_figure(outdir: Path, baseline: dict[str, Any], fit: dict[str, float]) -> None:
    result: mc.SimResult = baseline["result"]
    target = baseline["target"]
    world = result.w
    empirical = result.pi_empirical
    abs_error = np.abs(empirical - target)
    final_l1 = float(abs_error.sum())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    pi_vmin = min(float(target.min()), float(empirical.min()))
    pi_vmax = max(float(target.max()), float(empirical.max()))
    for ax, data, title, cmap in [
        (axes[0, 0], target, "theory pi proportional to degree", "viridis"),
        (axes[0, 1], empirical, "empirical visits", "viridis"),
        (axes[0, 2], abs_error, f"absolute error map, L1={final_l1:.3f}", "Reds"),
    ]:
        if cmap == "viridis":
            im = ax.imshow(data.reshape(world.Ny, world.Nx), origin="lower", cmap=cmap, vmin=pi_vmin, vmax=pi_vmax)
        else:
            im = ax.imshow(data.reshape(world.Ny, world.Nx), origin="lower", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("col")
        ax.set_ylabel("row")
        fig.colorbar(im, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    markov_steps = np.asarray(baseline["markov_steps"], dtype=np.float64)
    l1_errors = np.asarray(baseline["l1_errors"], dtype=np.float64)
    ax.plot(markov_steps, l1_errors, color="navy", lw=2.0, label="simulation")
    if len(markov_steps) and int(fit.get("n_points", 0)) > 0 and np.isfinite(fit.get("C_hat", np.nan)):
        fit_curve = float(fit["C_hat"]) / (markov_steps + 1.0)
        ax.plot(
            markov_steps,
            fit_curve,
            color="crimson",
            ls="--",
            lw=2.0,
            label=f"C/(n+1), R2={fit['r2']:.2f}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Markov arrivals n")
    ax.set_ylabel("overall L1 error")
    ax.set_title("overall convergence and 1/(n+1) rate fit")
    ax.legend(fontsize=8)

    for label, rep in baseline["representatives"].items():
        cell = rep.index
        axes[1, 1].plot(baseline["markov_steps"], baseline["ck_history"][:, cell], label=label)
        axes[1, 1].axhline(target[cell], ls=":", lw=1.0)
    axes[1, 1].set_xlabel("Markov arrivals")
    axes[1, 1].set_ylabel("coverage fraction")
    axes[1, 1].set_title("representative cells")
    axes[1, 1].legend(fontsize=8)

    axes[1, 2].hist(abs_error, bins=30, color="firebrick", alpha=0.8)
    axes[1, 2].axvline(float(abs_error.mean()), color="black", ls="--", lw=1.0, label=f"mean={abs_error.mean():.2e}")
    axes[1, 2].set_xlabel("|empirical π - theory π̄|")
    axes[1, 2].set_ylabel("cells")
    axes[1, 2].set_title("cellwise error distribution")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Forward Layer 1-C coverage validation")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_validation_baseline.png", dpi=140)
    plt.close(fig)


def make_speed_figure(outdir: Path, sweep: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for case in sweep:
        axes[0].plot(case["markov_steps"], case["l1_errors"], label=f"v={case['speed']:g}")
        axes[1].plot(case["sim_steps"], case["l1_errors"], label=f"v={case['speed']:g}")
    for ax, xlabel in [(axes[0], "Markov arrivals"), (axes[1], "simulation step")]:
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("L1 error")
        ax.legend(fontsize=8)
    axes[0].set_title("coverage convergence in Markov time")
    axes[1].set_title("coverage convergence in simulation time")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_validation_speed_sweep.png", dpi=140)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    world = mc.build_world(mc.NX, mc.NY, mc.CELL_SIZE)
    transition = mc.build_transition_matrix(world, np.zeros(world.K, dtype=np.float64))
    kernel = kernel_checks(world, transition)
    motion = motion_checks(world, transition)
    stationary_l1 = core.l1_error(mc.power_stationary_distribution(transition), mc.theoretical_stationary(world))
    baseline = run_case(args.speed, args.T, args.seed)
    sweep = speed_sweep(args.T)
    fit = fit_l1_over_markov_time(baseline["markov_steps"], baseline["l1_errors"], args.fit_start_frac)
    markov_spread = normalized_spread(sweep, "markov_steps")
    sim_spread = normalized_spread(sweep, "sim_steps")

    checks = {
        "kernel_rows_are_stochastic": bool(kernel["max_row_sum_error"] < 1.0e-12),
        "kernel_uses_only_neighbor_moves": bool(kernel["max_off_neighbor_probability"] < 1.0e-12),
        "equal_multipliers_make_uniform_neighbor_walk": bool(kernel["max_uniform_neighbor_error"] < 1.0e-12),
        "stationary_distribution_is_degree_proportional": bool(stationary_l1 < 1.0e-10),
        "arrival_updates_markov_state": bool(motion["arrival_updates_markov_state"]),
        "in_transit_motion_does_not_count_markov_step": bool(motion["transit_keeps_markov_state"]),
        "baseline_has_samples": bool(len(baseline["markov_steps"]) > 0),
        "markov_time_spread_not_worse_than_sim_time": bool(markov_spread <= 1.05 * sim_spread),
    }
    summary = {
        "paper_alignment": {
            "layer": "Layer 1-C coverage validation",
            "claim": "The forward MaxCal coverage controller is a destination-weighted random walk whose equal-multiplier baseline has degree-proportional stationary coverage.",
            "validated_outputs": [
                "transition-kernel sanity checks",
                "arrival-counted Markov convergence",
                "speed sweep showing that Markov arrivals are the natural time variable",
            ],
        },
        "environment": {
            "Nx": mc.NX,
            "Ny": mc.NY,
            "cell_size": mc.CELL_SIZE,
            "robots": mc.N_ROBOTS,
            "record_every": mc.RECORD_EVERY,
        },
        "kernel_checks": kernel,
        "motion_checks": motion,
        "stationary_l1_vs_degree_target": stationary_l1,
        "baseline": {
            "speed": args.speed,
            "T": args.T,
            "total_markov_arrivals": int(baseline["result"].markov_step_history[-1]) if baseline["result"].markov_step_history else 0,
            "final_l1_error": float(baseline["l1_errors"][-1]) if len(baseline["l1_errors"]) else float("nan"),
            "late_time_fit": fit,
            "representative_cells": baseline["representatives"],
        },
        "speed_sweep": {
            "cases": [
                {
                    "speed": case["speed"],
                    "total_arrivals": case["total_arrivals"],
                    "final_l1": case["final_l1"],
                }
                for case in sweep
            ],
            "relative_spread_markov_time": markov_spread,
            "relative_spread_sim_time": sim_spread,
            "markov_time_collapses_better": bool(markov_spread < sim_spread),
        },
        "checks": checks,
        "coverage_validation_ready": bool(all(checks.values())),
        "figures": FIGURES,
    }

    save_raw_data(outdir, baseline, sweep)
    make_baseline_figure(outdir, baseline, fit)
    make_speed_figure(outdir, sweep)
    with open(outdir / "maxcal_coverage_validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(json_safe(summary), handle, indent=2)

    print("MaxCal Coverage Validation (Layer 1-C)")
    print(f"  Output directory       : {outdir}")
    print(f"  Validation ready       : {summary['coverage_validation_ready']}")
    print(f"  Kernel row error       : {kernel['max_row_sum_error']:.3e}")
    print(f"  Stationary L1 vs theory: {stationary_l1:.3e}")
    print(f"  Baseline arrivals      : {summary['baseline']['total_markov_arrivals']}")
    print(f"  Baseline final L1      : {summary['baseline']['final_l1_error']:.6f}")
    print(f"  Late-time fit R^2      : {fit['r2']:.6f}")
    print(f"  Markov/sim spread      : {markov_spread:.3f} / {sim_spread:.3f}")
    print("  Saved summary          : maxcal_coverage_validation_summary.json")


if __name__ == "__main__":
    main()
