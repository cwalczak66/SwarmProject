"""
Validation harness for Information Diffusion Layer 1-I.

This validator is written around the Pinciroli sign prediction quoted by the
user from the local draft:

    lambda_I < 0   -> robots cluster
    lambda_I >= 0  -> robots prioritize coverage / avoid clustering

The checks are therefore organized around that exact claim:

1. Local kernel sign check:
   for a fixed positive information field, negative lambda_I must increase the
   probability of moving toward high-information neighbors, zero must recover
   the coverage kernel, and positive lambda_I must do the opposite.

2. Isolated Layer 1-I sign sweep:
   with stale information frozen on, negative lambda_I must produce lower
   dispersion and higher encounter propensity than the lambda_I = 0 baseline,
   while non-negative lambda_I must not increase clustering.

3. Closed-loop preview:
   with age gating and meetings enabled, fresh information behaves like
   coverage, stale information behaves like clustering, and the full loop
   generates meetings while moving between those regimes.

Usage:
    venv/bin/python maxcal_info_diffusion_validation.py
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import maxcal_info_diffusion as mid


SIGN_SWEEP_VALUES = [-10.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 10.0]
PURE_STALE_AGE = 1.0e6
PURE_FRESH_AGE = 0.0


@dataclass
class SignSweepPoint:
    lambda_I_value: float
    mean_dispersion: float
    mean_encounter_proxy: float
    final_coverage_l1: float
    total_markov_arrivals: int
    corner_visit_probability: float
    edge_visit_probability: float
    interior_visit_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate information diffusion Layer 1-I.")
    parser.add_argument(
        "--outdir",
        type=str,
        default="maxcal_info_diffusion_validation",
        help="Directory for validation outputs.",
    )
    parser.add_argument(
        "--sign-sweep-T",
        type=int,
        default=9_000,
        help="Simulation length for the isolated sign sweep.",
    )
    parser.add_argument(
        "--stage2-preview-T",
        type=int,
        default=12_000,
        help="Simulation length for the closed-loop preview runs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=mid.SEED,
        help="Base random seed.",
    )
    return parser.parse_args()


def _robot_at_cell(robot_id: int, w: mid.World, k: int, age: float) -> mid.Robot:
    cx, cy = mid.region_center(w, k)
    return mid.Robot(robot_id, cx, cy, k, k, cx, cy, info_age=age, last_meet=0)


def kernel_sign_checks() -> Dict[str, float | bool]:
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    lambda_c = mid.build_lambda_C(w, mid.LAMBDA_C_VAL)
    sample_k = (mid.NY // 2) * mid.NX + mid.NX // 2
    dense_neighbor = w.adjacency[sample_k][0]
    sparse_neighbor = w.adjacency[sample_k][-1]

    robots = [_robot_at_cell(i, w, dense_neighbor, PURE_STALE_AGE) for i in range(10)]
    robots.extend(
        [
            _robot_at_cell(10, w, 0, PURE_STALE_AGE),
            _robot_at_cell(11, w, w.K - 1, PURE_STALE_AGE),
        ]
    )
    info_field = mid.compute_information_field(w, robots)

    neighbors_neg, probs_neg = mid.local_transition_probabilities(
        w, sample_k, lambda_c, -4.0, info_field, PURE_STALE_AGE
    )
    neighbors_zero, probs_zero = mid.local_transition_probabilities(
        w, sample_k, lambda_c, 0.0, info_field, PURE_STALE_AGE
    )
    neighbors_pos, probs_pos = mid.local_transition_probabilities(
        w, sample_k, lambda_c, 4.0, info_field, PURE_STALE_AGE
    )

    assert np.array_equal(neighbors_neg, neighbors_zero)
    assert np.array_equal(neighbors_neg, neighbors_pos)

    dense_idx = int(np.where(neighbors_neg == dense_neighbor)[0][0])
    sparse_idx = int(np.where(neighbors_neg == sparse_neighbor)[0][0])
    uniform_prob = 1.0 / len(neighbors_neg)

    return {
        "sample_state": int(sample_k),
        "dense_neighbor": int(dense_neighbor),
        "sparse_neighbor": int(sparse_neighbor),
        "dense_neighbor_info_value": float(info_field[dense_neighbor]),
        "sparse_neighbor_info_value": float(info_field[sparse_neighbor]),
        "prob_dense_lambda_negative": float(probs_neg[dense_idx]),
        "prob_dense_lambda_zero": float(probs_zero[dense_idx]),
        "prob_dense_lambda_positive": float(probs_pos[dense_idx]),
        "prob_sparse_lambda_negative": float(probs_neg[sparse_idx]),
        "prob_sparse_lambda_zero": float(probs_zero[sparse_idx]),
        "prob_sparse_lambda_positive": float(probs_pos[sparse_idx]),
        "zero_lambda_recovers_uniform_kernel": bool(np.max(np.abs(probs_zero - uniform_prob)) < 1e-12),
        "negative_lambda_prefers_high_information_neighbor": bool(probs_neg[dense_idx] > probs_zero[dense_idx]),
        "positive_lambda_penalizes_high_information_neighbor": bool(probs_pos[dense_idx] < probs_zero[dense_idx]),
        "negative_vs_positive_ordering_on_dense_neighbor": bool(probs_neg[dense_idx] > probs_zero[dense_idx] > probs_pos[dense_idx]),
    }


def information_field_checks() -> Dict[str, float | bool]:
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    cluster_k = (mid.NY // 2) * mid.NX + mid.NX // 2
    robots = [_robot_at_cell(i, w, cluster_k, PURE_STALE_AGE) for i in range(15)]
    robots.extend(
        [
            _robot_at_cell(15, w, 0, PURE_STALE_AGE),
            _robot_at_cell(16, w, w.K - 1, PURE_STALE_AGE),
        ]
    )
    field = mid.compute_information_field(w, robots)
    global_max_index = int(np.argmax(field))
    cluster_local = [cluster_k] + w.adjacency[cluster_k]

    return {
        "cluster_cell": int(cluster_k),
        "cluster_field_value": float(field[cluster_k]),
        "global_max_index": global_max_index,
        "global_max_value": float(field[global_max_index]),
        "cluster_cell_attains_local_peak": bool(field[cluster_k] >= np.max(field[cluster_local]) - 1e-12),
        "global_peak_stays_near_robot_cluster": bool(global_max_index in cluster_local),
        "field_is_nonnegative": bool(np.all(field >= -1e-12)),
    }


def age_gate_checks() -> Dict[str, float | bool]:
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    lambda_c = mid.build_lambda_C(w, mid.LAMBDA_C_VAL)
    sample_k = (mid.NY // 2) * mid.NX + mid.NX // 2
    dense_neighbor = w.adjacency[sample_k][0]
    robots = [_robot_at_cell(i, w, dense_neighbor, PURE_STALE_AGE) for i in range(10)]
    info_field = mid.compute_information_field(w, robots)

    neighbors_fresh, probs_fresh = mid.local_transition_probabilities(
        w, sample_k, lambda_c, mid.LAMBDA_I_VAL, info_field, PURE_FRESH_AGE
    )
    neighbors_stale, probs_stale = mid.local_transition_probabilities(
        w, sample_k, lambda_c, mid.LAMBDA_I_VAL, info_field, PURE_STALE_AGE
    )
    dense_idx = int(np.where(neighbors_fresh == dense_neighbor)[0][0])
    uniform_prob = 1.0 / len(neighbors_fresh)

    return {
        "gate_at_zero_age": float(mid.info_gate(PURE_FRESH_AGE)),
        "gate_at_stale_age": float(mid.info_gate(PURE_STALE_AGE)),
        "fresh_age_kernel_is_uniform": bool(np.max(np.abs(probs_fresh - uniform_prob)) < 1e-12),
        "stale_age_activates_negative_lambda_clustering_bias": bool(probs_stale[dense_idx] > probs_fresh[dense_idx]),
        "dense_neighbor_prob_fresh": float(probs_fresh[dense_idx]),
        "dense_neighbor_prob_stale": float(probs_stale[dense_idx]),
        "neighbor_ordering_preserved": bool(np.array_equal(neighbors_fresh, neighbors_stale)),
    }


def meeting_reset_checks() -> Dict[str, int | bool]:
    robots = [
        mid.Robot(0, 1.0, 1.0, 0, 0, 1.0, 1.0, info_age=25.0, last_meet=0),
        mid.Robot(1, 1.0 + 0.5 * mid.R_MEET, 1.0, 0, 0, 1.0, 1.0, info_age=40.0, last_meet=0),
        mid.Robot(2, 10.0, 10.0, 0, 0, 10.0, 10.0, info_age=55.0, last_meet=0),
    ]
    n_pairs = mid.detect_meetings(robots, mid.R_MEET, t=7)
    return {
        "pairs_detected": n_pairs,
        "first_robot_reset": bool(robots[0].info_age == 0.0 and robots[0].last_meet == 7),
        "second_robot_reset": bool(robots[1].info_age == 0.0 and robots[1].last_meet == 7),
        "third_robot_unchanged": bool(robots[2].info_age == 55.0 and robots[2].last_meet == 0),
    }


def representative_probabilities(pi_empirical: np.ndarray) -> Dict[str, float]:
    corner = 0
    edge = mid.NX // 2
    interior = (mid.NY // 2) * mid.NX + mid.NX // 2
    return {
        "corner": float(pi_empirical[corner]),
        "edge": float(pi_empirical[edge]),
        "interior": float(pi_empirical[interior]),
    }


def run_sign_case(T: int, lambda_I_value: float, seed: int) -> SignSweepPoint:
    res = mid.run_simulation(
        T=T,
        lambda_I_value=lambda_I_value,
        seed=seed,
        fixed_initial_info_age=PURE_STALE_AGE,
        freeze_info_age=True,
        enable_meetings=False,
    )
    reps = representative_probabilities(res.pi_empirical)
    return SignSweepPoint(
        lambda_I_value=lambda_I_value,
        mean_dispersion=float(res.dispersion.mean()),
        mean_encounter_proxy=float(res.encounter_proxy.mean()),
        final_coverage_l1=float(res.coverage_l1[-1]),
        total_markov_arrivals=int(res.markov_step_history[-1]),
        corner_visit_probability=reps["corner"],
        edge_visit_probability=reps["edge"],
        interior_visit_probability=reps["interior"],
    )


def sign_sweep(T: int, seed: int) -> Dict[str, object]:
    points = [
        run_sign_case(T=T, lambda_I_value=value, seed=seed + i)
        for i, value in enumerate(SIGN_SWEEP_VALUES)
    ]

    zero_point = next(point for point in points if abs(point.lambda_I_value) < 1e-12)
    negative_points = [point for point in points if point.lambda_I_value < 0.0]
    nonnegative_points = [point for point in points if point.lambda_I_value >= 0.0]

    return {
        "points": points,
        "zero_baseline": zero_point,
        "all_negative_reduce_dispersion_vs_zero": bool(
            all(point.mean_dispersion < zero_point.mean_dispersion for point in negative_points)
        ),
        "all_negative_increase_encounter_vs_zero": bool(
            all(point.mean_encounter_proxy > zero_point.mean_encounter_proxy for point in negative_points)
        ),
        "all_negative_raise_coverage_error_vs_zero": bool(
            all(point.final_coverage_l1 > zero_point.final_coverage_l1 for point in negative_points)
        ),
        "all_nonnegative_do_not_increase_encounter_vs_zero": bool(
            all(point.mean_encounter_proxy <= 1.05 * zero_point.mean_encounter_proxy for point in nonnegative_points)
        ),
        "all_nonnegative_keep_dispersion_near_or_above_zero": bool(
            all(point.mean_dispersion >= 0.95 * zero_point.mean_dispersion for point in nonnegative_points)
        ),
        "zero_baseline_preserves_degree_ordering": bool(
            zero_point.interior_visit_probability > zero_point.edge_visit_probability > zero_point.corner_visit_probability
        ),
    }


def stage2_preview(T: int, seed: int) -> Dict[str, object]:
    zero_baseline = mid.run_simulation(
        T=T,
        lambda_I_value=0.0,
        seed=seed,
        fixed_initial_info_age=PURE_STALE_AGE,
        freeze_info_age=True,
        enable_meetings=False,
    )
    fresh_mode = mid.run_simulation(
        T=T,
        lambda_I_value=mid.LAMBDA_I_VAL,
        seed=seed + 1,
        fixed_initial_info_age=PURE_FRESH_AGE,
        freeze_info_age=True,
        enable_meetings=False,
    )
    stale_mode = mid.run_simulation(
        T=T,
        lambda_I_value=mid.LAMBDA_I_VAL,
        seed=seed + 2,
        fixed_initial_info_age=PURE_STALE_AGE,
        freeze_info_age=True,
        enable_meetings=False,
    )
    closed_loop = mid.run_simulation(
        T=T,
        lambda_I_value=mid.LAMBDA_I_VAL,
        seed=seed + 3,
        fixed_initial_info_age=None,
        freeze_info_age=False,
        enable_meetings=True,
    )

    def pack(res: mid.SimResult) -> Dict[str, float]:
        return {
            "mean_dispersion": float(res.dispersion.mean()),
            "mean_encounter_proxy": float(res.encounter_proxy.mean()),
            "final_coverage_l1": float(res.coverage_l1[-1]),
            "mean_info_age": float(res.mean_info_age.mean()),
            "mean_meetings_per_snap": float(res.meetings_per_snap.mean()),
        }

    fresh_matches_zero = (
        abs(float(fresh_mode.encounter_proxy.mean()) - float(zero_baseline.encounter_proxy.mean()))
        <= 0.10 * float(zero_baseline.encounter_proxy.mean())
        and abs(float(fresh_mode.dispersion.mean()) - float(zero_baseline.dispersion.mean()))
        <= 0.10 * float(zero_baseline.dispersion.mean())
    )

    stale_clusters_vs_zero = (
        float(stale_mode.encounter_proxy.mean()) > float(zero_baseline.encounter_proxy.mean())
        and float(stale_mode.dispersion.mean()) < float(zero_baseline.dispersion.mean())
    )

    return {
        "zero_baseline": pack(zero_baseline),
        "fresh_information_mode": pack(fresh_mode),
        "stale_information_mode": pack(stale_mode),
        "closed_loop": pack(closed_loop),
        "fresh_information_recovers_coverage_baseline": bool(fresh_matches_zero),
        "stale_information_activates_clustering": bool(stale_clusters_vs_zero),
        "closed_loop_generates_meetings": bool(float(closed_loop.meetings_per_snap.mean()) > 0.0),
    }


def make_sign_sweep_figure(outdir: Path, sweep: Dict[str, object]) -> None:
    points: List[SignSweepPoint] = sweep["points"]
    values = np.array([point.lambda_I_value for point in points], dtype=np.float64)
    disp = np.array([point.mean_dispersion for point in points], dtype=np.float64)
    encounter = np.array([point.mean_encounter_proxy for point in points], dtype=np.float64)
    coverage_l1 = np.array([point.final_coverage_l1 for point in points], dtype=np.float64)
    arrivals = np.array([point.total_markov_arrivals for point in points], dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    axes[0, 0].plot(values, disp, marker="o", lw=2.0, color="navy")
    axes[0, 0].axvline(0.0, color="grey", ls=":")
    axes[0, 0].set_title("Mean dispersion vs lambda_I")
    axes[0, 0].set_xlabel("lambda_I")
    axes[0, 0].set_ylabel("mean S(t)")

    axes[0, 1].plot(values, encounter, marker="o", lw=2.0, color="darkorange")
    axes[0, 1].axvline(0.0, color="grey", ls=":")
    axes[0, 1].set_title("Encounter proxy vs lambda_I")
    axes[0, 1].set_xlabel("lambda_I")
    axes[0, 1].set_ylabel("mean sum_k occ_k^2")

    axes[1, 0].plot(values, coverage_l1, marker="o", lw=2.0, color="slateblue")
    axes[1, 0].axvline(0.0, color="grey", ls=":")
    axes[1, 0].set_title("Final coverage error vs lambda_I")
    axes[1, 0].set_xlabel("lambda_I")
    axes[1, 0].set_ylabel("final ||c(.,t)-pi_bar(.)||_1")

    axes[1, 1].plot(values, arrivals, marker="o", lw=2.0, color="seagreen")
    axes[1, 1].axvline(0.0, color="grey", ls=":")
    axes[1, 1].set_title("Total Markov arrivals vs lambda_I")
    axes[1, 1].set_xlabel("lambda_I")
    axes[1, 1].set_ylabel("total arrivals")

    fig.tight_layout()
    fig.savefig(outdir / "maxcal_info_diffusion_validation_sign_sweep.png", dpi=140)
    plt.close(fig)


def save_raw_data(outdir: Path, sweep: Dict[str, object]) -> None:
    points: List[SignSweepPoint] = sweep["points"]
    np.savez(
        outdir / "maxcal_info_diffusion_validation_raw_data.npz",
        lambda_I_values=np.array([point.lambda_I_value for point in points], dtype=np.float64),
        mean_dispersion=np.array([point.mean_dispersion for point in points], dtype=np.float64),
        mean_encounter_proxy=np.array([point.mean_encounter_proxy for point in points], dtype=np.float64),
        final_coverage_l1=np.array([point.final_coverage_l1 for point in points], dtype=np.float64),
        total_markov_arrivals=np.array([point.total_markov_arrivals for point in points], dtype=np.int64),
        corner_visit_probability=np.array([point.corner_visit_probability for point in points], dtype=np.float64),
        edge_visit_probability=np.array([point.edge_visit_probability for point in points], dtype=np.float64),
        interior_visit_probability=np.array([point.interior_visit_probability for point in points], dtype=np.float64),
    )


def save_summary(
    outdir: Path,
    args: argparse.Namespace,
    kernel: Dict[str, float | bool],
    field: Dict[str, float | bool],
    gate: Dict[str, float | bool],
    meeting: Dict[str, int | bool],
    sweep: Dict[str, object],
    preview: Dict[str, object],
) -> None:
    points: List[SignSweepPoint] = sweep["points"]
    zero_point: SignSweepPoint = sweep["zero_baseline"]

    summary = {
        "config": {
            "Nx": mid.NX,
            "Ny": mid.NY,
            "cell_size_m": mid.CELL_SIZE,
            "robots": mid.N_ROBOTS,
            "lambda_I_sign_convention": "lambda_I < 0 -> cluster; lambda_I >= 0 -> coverage-prioritizing",
            "lambda_I_sign_sweep": SIGN_SWEEP_VALUES,
            "sign_sweep_T": args.sign_sweep_T,
            "stage2_preview_T": args.stage2_preview_T,
            "seed": args.seed,
            "pure_stale_age": PURE_STALE_AGE,
        },
        "kernel_sign_checks": kernel,
        "information_field_checks": field,
        "age_gate_checks": gate,
        "meeting_reset_checks": meeting,
        "sign_sweep": {
            "points": [asdict(point) for point in points],
            "zero_baseline": asdict(zero_point),
            "all_negative_reduce_dispersion_vs_zero": sweep["all_negative_reduce_dispersion_vs_zero"],
            "all_negative_increase_encounter_vs_zero": sweep["all_negative_increase_encounter_vs_zero"],
            "all_negative_raise_coverage_error_vs_zero": sweep["all_negative_raise_coverage_error_vs_zero"],
            "all_nonnegative_do_not_increase_encounter_vs_zero": sweep["all_nonnegative_do_not_increase_encounter_vs_zero"],
            "all_nonnegative_keep_dispersion_near_or_above_zero": sweep["all_nonnegative_keep_dispersion_near_or_above_zero"],
            "zero_baseline_preserves_degree_ordering": sweep["zero_baseline_preserves_degree_ordering"],
        },
        "stage2_preview": preview,
    }

    summary["layer1_i_ready_for_stage2"] = bool(
        kernel["zero_lambda_recovers_uniform_kernel"]
        and kernel["negative_lambda_prefers_high_information_neighbor"]
        and kernel["positive_lambda_penalizes_high_information_neighbor"]
        and kernel["negative_vs_positive_ordering_on_dense_neighbor"]
        and field["cluster_cell_attains_local_peak"]
        and field["global_peak_stays_near_robot_cluster"]
        and field["field_is_nonnegative"]
        and gate["fresh_age_kernel_is_uniform"]
        and gate["stale_age_activates_negative_lambda_clustering_bias"]
        and meeting["pairs_detected"] >= 1
        and meeting["first_robot_reset"]
        and meeting["second_robot_reset"]
        and meeting["third_robot_unchanged"]
        and sweep["all_negative_reduce_dispersion_vs_zero"]
        and sweep["all_negative_increase_encounter_vs_zero"]
        and sweep["all_negative_raise_coverage_error_vs_zero"]
        and sweep["all_nonnegative_do_not_increase_encounter_vs_zero"]
        and sweep["all_nonnegative_keep_dispersion_near_or_above_zero"]
        and sweep["zero_baseline_preserves_degree_ordering"]
        and preview["fresh_information_recovers_coverage_baseline"]
        and preview["stale_information_activates_clustering"]
        and preview["closed_loop_generates_meetings"]
    )

    with open(outdir / "maxcal_info_diffusion_validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    kernel = kernel_sign_checks()
    field = information_field_checks()
    gate = age_gate_checks()
    meeting = meeting_reset_checks()
    sweep = sign_sweep(T=args.sign_sweep_T, seed=args.seed)
    preview = stage2_preview(T=args.stage2_preview_T, seed=args.seed)

    make_sign_sweep_figure(outdir, sweep)
    save_raw_data(outdir, sweep)
    save_summary(outdir, args, kernel, field, gate, meeting, sweep, preview)

    ready = (
        kernel["zero_lambda_recovers_uniform_kernel"]
        and kernel["negative_lambda_prefers_high_information_neighbor"]
        and kernel["positive_lambda_penalizes_high_information_neighbor"]
        and field["cluster_cell_attains_local_peak"]
        and gate["fresh_age_kernel_is_uniform"]
        and gate["stale_age_activates_negative_lambda_clustering_bias"]
        and sweep["all_negative_reduce_dispersion_vs_zero"]
        and sweep["all_negative_increase_encounter_vs_zero"]
        and sweep["all_nonnegative_do_not_increase_encounter_vs_zero"]
        and preview["fresh_information_recovers_coverage_baseline"]
        and preview["stale_information_activates_clustering"]
        and preview["closed_loop_generates_meetings"]
    )

    print("MaxCal Information Diffusion Validation (Layer 1-I)")
    print(f"  Output directory      : {outdir}")
    print(f"  Layer 1-I ready       : {bool(ready)}")
    print(
        "  Zero-baseline sweep   : "
        f"S={sweep['zero_baseline'].mean_dispersion:.3f}, "
        f"E={sweep['zero_baseline'].mean_encounter_proxy:.5f}, "
        f"L1={sweep['zero_baseline'].final_coverage_l1:.5f}"
    )
    print(
        "  Stage II preview      : "
        f"fresh->coverage={preview['fresh_information_recovers_coverage_baseline']}, "
        f"stale->cluster={preview['stale_information_activates_clustering']}, "
        f"meetings={preview['closed_loop_generates_meetings']}"
    )
    print("  Saved summary         : maxcal_info_diffusion_validation_summary.json")


if __name__ == "__main__":
    main()
