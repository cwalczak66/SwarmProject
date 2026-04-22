"""
Standalone validation harness for Coverage Layer 1-C.

This script keeps the controller in ``maxcal_coverage.py`` unchanged and
computes the checks described in the methodology:

- the lambda_C = 0 kernel is row-stochastic and uniform over neighbors
- the stationary target is pi_bar_k proportional to deg(k)
- the discrete Markov update only fires on arrival
- capped motion prevents overshoot
- the L1 coverage error is measured in Markov time
- representative corner / edge / interior cells approach their targets
- the five-speed sweep is compared in both Markov and simulation time

This validator also defines the reference "pure coverage" regime used by
the diffusion validator when it interprets lambda_I = 0 as the no-information
baseline.

Usage:
    venv/bin/python maxcal_coverage_validation.py
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import maxcal_coverage as mc


SPEED_SWEEP = [0.05, 0.10, 0.20, 0.50, 1.00]


@dataclass
class InverseFit:
    start_fraction: float
    start_index: int
    n_points: int
    C_hat: float
    r2: float


@dataclass
class RepresentativeCell:
    index: int
    degree: int
    target_pi: float
    final_ck: float
    final_abs_error: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Coverage Layer 1-C.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="maxcal_coverage_validation",
        help="Directory for the validation report, figures, and raw data.",
    )
    parser.add_argument(
        "--T",
        type=int,
        default=mc.T_SIM,
        help="Simulation length in simulation steps.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=mc.ROBOT_SPEED,
        help="Baseline robot speed in m/step.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=mc.SEED,
        help="Seed for the baseline validation run.",
    )
    parser.add_argument(
        "--fit-start-frac",
        type=float,
        default=0.50,
        help="Fraction of the trace to discard before the late-time C/(t+1) fit.",
    )
    return parser.parse_args()


def build_baseline_objects():
    world = mc.build_world(mc.NX, mc.NY, mc.CELL_SIZE)
    lambda_c = np.zeros(world.K, dtype=np.float64)
    transition = mc.build_transition_matrix(world, lambda_c)
    pi_theory = mc.theoretical_stationary(world)
    return world, transition, pi_theory


def kernel_checks(world: mc.World, transition: np.ndarray) -> Dict[str, float]:
    row_sums = transition.sum(axis=1)
    max_row_sum_error = float(np.max(np.abs(row_sums - 1.0)))

    off_neighbor_mask = np.ones_like(transition, dtype=bool)
    for k in range(world.K):
        off_neighbor_mask[k, world.adjacency[k]] = False
        off_neighbor_mask[k, k] = False
    max_off_neighbor_probability = float(np.max(np.abs(transition[off_neighbor_mask])))

    max_uniform_neighbor_error = 0.0
    for k in range(world.K):
        neighbors = world.adjacency[k]
        target = 1.0 / len(neighbors)
        max_uniform_neighbor_error = max(
            max_uniform_neighbor_error,
            float(np.max(np.abs(transition[k, neighbors] - target))),
        )

    return {
        "max_row_sum_error": max_row_sum_error,
        "max_off_neighbor_probability": max_off_neighbor_probability,
        "max_uniform_neighbor_error": max_uniform_neighbor_error,
    }


def motion_and_arrival_checks(world: mc.World, transition: np.ndarray) -> Dict[str, float | bool]:
    rng = np.random.default_rng(0)
    from_k = 0
    to_k = world.adjacency[from_k][0]
    x0, y0 = mc.region_center(world, from_k)
    tx, ty = mc.region_center(world, to_k)
    dist = float(np.hypot(tx - x0, ty - y0))

    in_transit = mc.Robot(0, x0, y0, from_k, to_k, tx, ty)
    visits_transit = np.zeros(world.K, dtype=np.int64)
    slow_speed = dist / 3.0
    mc.step_robot(in_transit, world, transition, slow_speed, visits_transit, rng)
    transit_move = float(np.hypot(in_transit.x - x0, in_transit.y - y0))
    transit_remaining = float(np.hypot(tx - in_transit.x, ty - in_transit.y))

    arrival = mc.Robot(1, x0, y0, from_k, to_k, tx, ty)
    visits_arrival = np.zeros(world.K, dtype=np.int64)
    fast_speed = dist * 10.0
    mc.step_robot(arrival, world, transition, fast_speed, visits_arrival, rng)
    arrival_move = float(np.hypot(arrival.x - x0, arrival.y - y0))

    return {
        "transit_from_k_unchanged": in_transit.from_k == from_k,
        "transit_visit_count_unchanged": int(visits_transit.sum()) == 0,
        "transit_move_equals_speed": abs(transit_move - slow_speed) < 1e-12,
        "transit_remaining_distance": transit_remaining,
        "arrival_visit_incremented": int(visits_arrival[to_k]) == 1,
        "arrival_total_visits": int(visits_arrival.sum()),
        "arrival_from_k_updated": arrival.from_k == to_k,
        "arrival_position_exact": abs(arrival.x - tx) < 1e-12 and abs(arrival.y - ty) < 1e-12,
        "arrival_move_capped_to_distance": abs(arrival_move - dist) < 1e-12,
        "arrival_next_target_is_neighbor": arrival.to_k in world.adjacency[arrival.from_k],
        "target_distance": dist,
    }


def inverse_fit(markov_steps: np.ndarray, errors: np.ndarray, start_fraction: float) -> InverseFit:
    start_index = int(len(markov_steps) * start_fraction)
    x = 1.0 / (markov_steps[start_index:] + 1.0)
    y = errors[start_index:]
    c_hat = float(np.dot(x, y) / np.dot(x, x))
    y_hat = c_hat * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return InverseFit(
        start_fraction=start_fraction,
        start_index=start_index,
        n_points=len(y),
        C_hat=c_hat,
        r2=r2,
    )


def representative_cells(world: mc.World, pi_theory: np.ndarray, final_ck: np.ndarray) -> Dict[str, RepresentativeCell]:
    reps = {
        "corner": 0,
        "edge": mc.NX // 2,
        "interior": (mc.NY // 2) * mc.NX + mc.NX // 2,
    }
    result: Dict[str, RepresentativeCell] = {}
    for label, idx in reps.items():
        result[label] = RepresentativeCell(
            index=idx,
            degree=len(world.adjacency[idx]),
            target_pi=float(pi_theory[idx]),
            final_ck=float(final_ck[idx]),
            final_abs_error=float(abs(final_ck[idx] - pi_theory[idx])),
        )
    return result


def simulation_steps_from_history(ck_history: List[np.ndarray]) -> np.ndarray:
    return np.arange(1, len(ck_history) + 1, dtype=np.int64) * mc.RECORD_EVERY


def run_baseline(speed: float, T: int, seed: int, fit_start_fraction: float) -> Dict[str, object]:
    result = mc.run_simulation(speed=speed, T=T, seed=seed)
    pi_theory = mc.theoretical_stationary(result.w)
    ck_history = np.array(result.ck_history)
    markov_steps = np.array(result.markov_step_history, dtype=np.int64)
    sim_steps = simulation_steps_from_history(result.ck_history)
    l1_errors = np.sum(np.abs(ck_history - pi_theory[None, :]), axis=1)
    fit = inverse_fit(markov_steps.astype(np.float64), l1_errors, fit_start_fraction)
    reps = representative_cells(result.w, pi_theory, ck_history[-1])

    return {
        "result": result,
        "pi_theory": pi_theory,
        "ck_history": ck_history,
        "markov_steps": markov_steps,
        "sim_steps": sim_steps,
        "l1_errors": l1_errors,
        "fit": fit,
        "representatives": reps,
    }


def run_speed_sweep(T: int) -> Dict[str, object]:
    sweep_runs = []
    for speed in SPEED_SWEEP:
        result = mc.run_simulation(speed=speed, T=T, seed=mc.SEED)
        pi_theory = mc.theoretical_stationary(result.w)
        ck_history = np.array(result.ck_history)
        markov_steps = np.array(result.markov_step_history, dtype=np.int64)
        sim_steps = simulation_steps_from_history(result.ck_history)
        l1_errors = np.sum(np.abs(ck_history - pi_theory[None, :]), axis=1)
        sweep_runs.append(
            {
                "speed": speed,
                "markov_steps": markov_steps,
                "sim_steps": sim_steps,
                "l1_errors": l1_errors,
                "final_markov_steps": int(markov_steps[-1]),
                "final_l1_error": float(l1_errors[-1]),
            }
        )

    common_markov_low = max(run["markov_steps"][0] for run in sweep_runs)
    common_markov_high = min(run["markov_steps"][-1] for run in sweep_runs)
    common_markov_grid = np.linspace(common_markov_low, common_markov_high, 100)
    markov_stack = np.stack(
        [
            np.interp(common_markov_grid, run["markov_steps"], run["l1_errors"])
            for run in sweep_runs
        ]
    )
    markov_rel_range = np.mean(
        (markov_stack.max(axis=0) - markov_stack.min(axis=0))
        / np.maximum(np.median(markov_stack, axis=0), 1e-12)
    )

    sim_stack = np.stack([run["l1_errors"] for run in sweep_runs])
    sim_rel_range = np.mean(
        (sim_stack.max(axis=0) - sim_stack.min(axis=0))
        / np.maximum(np.median(sim_stack, axis=0), 1e-12)
    )

    return {
        "runs": sweep_runs,
        "common_markov_low": float(common_markov_low),
        "common_markov_high": float(common_markov_high),
        "markov_mean_relative_range": float(markov_rel_range),
        "simulation_mean_relative_range": float(sim_rel_range),
    }


def save_raw_data(outdir: Path, baseline: Dict[str, object], speed_sweep: Dict[str, object]) -> None:
    np.savez(
        outdir / "maxcal_coverage_validation_raw_data.npz",
        baseline_markov_steps=baseline["markov_steps"],
        baseline_sim_steps=baseline["sim_steps"],
        baseline_l1_errors=baseline["l1_errors"],
        baseline_ck_history=baseline["ck_history"],
        baseline_pi_theory=baseline["pi_theory"],
        speed_sweep_speeds=np.array([run["speed"] for run in speed_sweep["runs"]], dtype=np.float64),
        speed_sweep_final_markov_steps=np.array(
            [run["final_markov_steps"] for run in speed_sweep["runs"]],
            dtype=np.int64,
        ),
        speed_sweep_final_l1_errors=np.array(
            [run["final_l1_error"] for run in speed_sweep["runs"]],
            dtype=np.float64,
        ),
    )


def make_baseline_figure(outdir: Path, baseline: Dict[str, object]) -> None:
    result = baseline["result"]
    pi_theory = baseline["pi_theory"]
    ck_history = baseline["ck_history"]
    markov_steps = baseline["markov_steps"]
    l1_errors = baseline["l1_errors"]
    fit: InverseFit = baseline["fit"]
    reps: Dict[str, RepresentativeCell] = baseline["representatives"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    fit_x = markov_steps.astype(np.float64)
    fit_curve = fit.C_hat / (fit_x + 1.0)
    axes[0].plot(fit_x, l1_errors, lw=2.2, color="navy", label="Observed L1 error")
    axes[0].plot(fit_x, fit_curve, lw=1.8, ls="--", color="crimson", label="C/(t+1) fit")
    axes[0].axvline(
        markov_steps[fit.start_index],
        color="gray",
        ls=":",
        lw=1.2,
        label=f"Late-time start ({fit.start_fraction:.2f})",
    )
    axes[0].set_title(f"Baseline L1 convergence, R^2={fit.r2:.3f}")
    axes[0].set_xlabel("Markov steps (arrivals)")
    axes[0].set_ylabel("||c(.,t) - pi_bar(.)||_1")
    axes[0].legend(loc="upper right")

    colors = {"corner": "red", "edge": "darkorange", "interior": "seagreen"}
    for label, rep in reps.items():
        axes[1].plot(
            markov_steps,
            ck_history[:, rep.index],
            lw=2.0,
            color=colors[label],
            label=f"{label.title()} (deg {rep.degree})",
        )
        axes[1].axhline(rep.target_pi, color=colors[label], ls="--", lw=1.2)
    axes[1].set_title("Representative cells by degree class")
    axes[1].set_xlabel("Markov steps (arrivals)")
    axes[1].set_ylabel("c_k(t)")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_validation_baseline.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.imshow(
        np.abs(result.pi_empirical - pi_theory).reshape(result.w.Ny, result.w.Nx),
        origin="lower",
        cmap="Reds",
    )
    ax.set_title("Final |pi_hat_k - pi_bar_k|")
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_validation_final_error_map.png", dpi=140)
    plt.close(fig)


def make_speed_figure(outdir: Path, speed_sweep: Dict[str, object]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for run in speed_sweep["runs"]:
        label = f"v={run['speed']:.2f} m/step"
        axes[0].plot(run["markov_steps"], run["l1_errors"], lw=2.0, label=label)
        axes[1].plot(run["sim_steps"], run["l1_errors"], lw=2.0, label=label)

    axes[0].set_yscale("log")
    axes[0].set_title("L1 error in Markov time")
    axes[0].set_xlabel("Markov steps (arrivals)")
    axes[0].set_ylabel("||c(.,t) - pi_bar(.)||_1")
    axes[0].legend(loc="upper right")

    axes[1].set_yscale("log")
    axes[1].set_title("L1 error in simulation time")
    axes[1].set_xlabel("Simulation steps")
    axes[1].set_ylabel("||c(.,t) - pi_bar(.)||_1")
    axes[1].legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(outdir / "maxcal_coverage_validation_speed_sweep.png", dpi=140)
    plt.close(fig)


def save_summary(
    outdir: Path,
    args: argparse.Namespace,
    kernel: Dict[str, float],
    motion: Dict[str, float | bool],
    baseline: Dict[str, object],
    speed_sweep: Dict[str, object],
) -> None:
    fit: InverseFit = baseline["fit"]
    reps: Dict[str, RepresentativeCell] = baseline["representatives"]

    summary = {
        "config": {
            "Nx": mc.NX,
            "Ny": mc.NY,
            "cell_size_m": mc.CELL_SIZE,
            "robots": mc.N_ROBOTS,
            "lambda_C_value": mc.LAMBDA_C_VAL,
            "stage2_baseline_regime": "pure coverage reference (lambda_I = 0)",
            "baseline_speed_m_per_step": args.speed,
            "simulation_steps": args.T,
            "seed": args.seed,
            "record_every_steps": mc.RECORD_EVERY,
        },
        "kernel_checks": kernel,
        "motion_and_arrival_checks": motion,
        "baseline_run": {
            "total_markov_steps": int(baseline["markov_steps"][-1]),
            "final_l1_error": float(baseline["l1_errors"][-1]),
            "fit": asdict(fit),
            "fit_pass_r2_gt_0_95": bool(fit.r2 > 0.95),
            "representative_cells": {label: asdict(rep) for label, rep in reps.items()},
        },
        "speed_sweep": {
            "speeds_m_per_step": SPEED_SWEEP,
            "markov_mean_relative_range": speed_sweep["markov_mean_relative_range"],
            "simulation_mean_relative_range": speed_sweep["simulation_mean_relative_range"],
            "markov_time_collapse_stronger_than_simulation_time": bool(
                speed_sweep["markov_mean_relative_range"]
                < speed_sweep["simulation_mean_relative_range"]
            ),
            "per_speed_final_values": [
                {
                    "speed": run["speed"],
                    "final_markov_steps": run["final_markov_steps"],
                    "final_l1_error": run["final_l1_error"],
                }
                for run in speed_sweep["runs"]
            ],
        },
    }

    summary["coverage_layer1_ready_for_stage2"] = bool(
        kernel["max_row_sum_error"] < 1e-12
        and kernel["max_off_neighbor_probability"] < 1e-12
        and kernel["max_uniform_neighbor_error"] < 1e-12
        and motion["transit_from_k_unchanged"]
        and motion["transit_visit_count_unchanged"]
        and motion["arrival_visit_incremented"]
        and motion["arrival_from_k_updated"]
        and motion["arrival_position_exact"]
        and motion["arrival_move_capped_to_distance"]
        and summary["baseline_run"]["fit_pass_r2_gt_0_95"]
        and summary["speed_sweep"]["markov_time_collapse_stronger_than_simulation_time"]
    )

    with open(outdir / "maxcal_coverage_validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    world, transition, _ = build_baseline_objects()
    kernel = kernel_checks(world, transition)
    motion = motion_and_arrival_checks(world, transition)
    baseline = run_baseline(
        speed=args.speed,
        T=args.T,
        seed=args.seed,
        fit_start_fraction=args.fit_start_frac,
    )
    speed_sweep = run_speed_sweep(T=args.T)

    save_raw_data(outdir, baseline, speed_sweep)
    make_baseline_figure(outdir, baseline)
    make_speed_figure(outdir, speed_sweep)
    save_summary(outdir, args, kernel, motion, baseline, speed_sweep)

    fit: InverseFit = baseline["fit"]
    print("MaxCal Coverage Validation (Layer 1-C)")
    print(f"  Output directory      : {outdir}")
    print(f"  Baseline markov steps : {int(baseline['markov_steps'][-1])}")
    print(f"  Baseline final L1     : {float(baseline['l1_errors'][-1]):.6f}")
    print(f"  Late-time fit R^2     : {fit.r2:.6f}")
    print(
        "  Markov vs sim spread  : "
        f"{speed_sweep['markov_mean_relative_range']:.3f} vs "
        f"{speed_sweep['simulation_mean_relative_range']:.3f}"
    )
    print("  Saved summary         : maxcal_coverage_validation_summary.json")


if __name__ == "__main__":
    main()
