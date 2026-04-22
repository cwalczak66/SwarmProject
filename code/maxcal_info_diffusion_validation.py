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


@dataclass
class InformationSpreadPoint:
    lambda_I_value: float
    final_informed_fraction: float
    time_to_50_percent: int | None
    time_to_90_percent: int | None
    time_to_all_informed: int | None
    total_meetings: int
    meeting_cells_visited: int
    mean_meeting_distance_from_center: float
    total_transmission_events: int
    transmission_cells_visited: int
    mean_transmission_distance_from_center: float
    transmission_efficiency: float


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
        "--spread-T",
        type=int,
        default=6_000,
        help="Simulation length for the informed-robot spread validation.",
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


def _meeting_pairs(robots: Sequence[mid.Robot], r_meet: float) -> List[tuple[int, int, float, float]]:
    pairs: List[tuple[int, int, float, float]] = []
    xs = np.array([r.x for r in robots], dtype=np.float64)
    ys = np.array([r.y for r in robots], dtype=np.float64)
    for i in range(len(robots)):
        dx = xs[i + 1:] - xs[i]
        dy = ys[i + 1:] - ys[i]
        hits = np.where(dx * dx + dy * dy <= r_meet * r_meet)[0]
        for j_off in hits:
            j = i + 1 + int(j_off)
            pairs.append((i, j, 0.5 * (xs[i] + xs[j]), 0.5 * (ys[i] + ys[j])))
    return pairs


def _threshold_time(t_axis: Sequence[int], fractions: Sequence[float], threshold: float) -> int | None:
    for t, frac in zip(t_axis, fractions):
        if frac >= threshold:
            return int(t)
    return None


def run_information_spread_case(
    T: int,
    lambda_I_value: float,
    seed: int,
    n_initial_informed: int = 1,
) -> tuple[InformationSpreadPoint, Dict[str, np.ndarray]]:
    """
    Track information diffusion directly as a robot-level contagion process.

    A small set of initially informed robots carries a binary information flag.
    When an informed robot meets an uninformed robot, the uninformed robot becomes
    informed. This validation is intentionally separate from coverage maps:
    the primary observable is the fraction of informed robots over time, and the
    spatial diagnostic is where transmission meetings occur.
    """
    rng = np.random.default_rng(seed)
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    lambda_c = mid.build_lambda_C(w, mid.LAMBDA_C_VAL)
    robots = [mid.make_robot(i, w, rng, fixed_initial_info_age=None) for i in range(mid.N_ROBOTS)]
    mid.initialise_targets(robots, w, lambda_c, lambda_I_value, rng)

    informed = np.zeros(mid.N_ROBOTS, dtype=bool)
    informed[:n_initial_informed] = True
    markov_visits = np.zeros(w.K, dtype=np.int64)
    coverage_age_field = np.zeros(w.K, dtype=np.float64)

    t_axis: List[int] = []
    informed_fraction: List[float] = []
    cumulative_meetings: List[int] = []
    cumulative_transmissions: List[int] = []
    meetings_per_interval: List[int] = []
    transmissions_per_interval: List[int] = []
    meeting_heatmap = np.zeros(w.K, dtype=np.float64)
    transmission_heatmap = np.zeros(w.K, dtype=np.float64)
    total_meetings = 0
    total_transmissions = 0
    interval_meetings = 0
    interval_transmissions = 0

    for t in range(1, T + 1):
        info_field = mid.compute_information_field(w, robots)
        for r in robots:
            mid.step_robot(
                r=r,
                w=w,
                lambda_C=lambda_c,
                lambda_I_value=lambda_I_value,
                info_field=info_field,
                speed=mid.ROBOT_SPEED,
                markov_visits=markov_visits,
                coverage_age_field=coverage_age_field,
                t=t,
                rng=rng,
                freeze_info_age=False,
            )

        for i, j, mx, my in _meeting_pairs(robots, mid.R_MEET):
            total_meetings += 1
            interval_meetings += 1
            cell = mid.position_to_cell(w, mx, my)
            meeting_heatmap[cell] += 1.0
            before_i = bool(informed[i])
            before_j = bool(informed[j])
            if before_i or before_j:
                informed[i] = True
                informed[j] = True
            if before_i != before_j:
                total_transmissions += 1
                interval_transmissions += 1
                transmission_heatmap[cell] += 1.0
            robots[i].info_age = 0.0
            robots[j].info_age = 0.0
            robots[i].last_meet = t
            robots[j].last_meet = t

        if t % mid.RECORD_EVERY == 0 or t == T:
            t_axis.append(t)
            informed_fraction.append(float(informed.mean()))
            cumulative_meetings.append(total_meetings)
            cumulative_transmissions.append(total_transmissions)
            meetings_per_interval.append(interval_meetings)
            transmissions_per_interval.append(interval_transmissions)
            interval_meetings = 0
            interval_transmissions = 0

    centers = w.centers
    center_xy = np.array([0.5 * w.Nx * w.cell_size, 0.5 * w.Ny * w.cell_size])
    meeting_cells = np.where(meeting_heatmap > 0)[0]
    transmission_cells = np.where(transmission_heatmap > 0)[0]
    if len(meeting_cells) > 0:
        meeting_weights = meeting_heatmap[meeting_cells]
        meeting_distances = np.linalg.norm(centers[meeting_cells] - center_xy, axis=1)
        mean_meeting_distance = float(np.average(meeting_distances, weights=meeting_weights))
    else:
        mean_meeting_distance = float("nan")
    if len(transmission_cells) > 0:
        weights = transmission_heatmap[transmission_cells]
        distances = np.linalg.norm(centers[transmission_cells] - center_xy, axis=1)
        mean_transmission_distance = float(np.average(distances, weights=weights))
    else:
        mean_transmission_distance = float("nan")
    transmission_efficiency = float(total_transmissions / total_meetings) if total_meetings > 0 else 0.0

    point = InformationSpreadPoint(
        lambda_I_value=float(lambda_I_value),
        final_informed_fraction=float(informed_fraction[-1]),
        time_to_50_percent=_threshold_time(t_axis, informed_fraction, 0.50),
        time_to_90_percent=_threshold_time(t_axis, informed_fraction, 0.90),
        time_to_all_informed=_threshold_time(t_axis, informed_fraction, 1.00),
        total_meetings=int(total_meetings),
        meeting_cells_visited=int(len(meeting_cells)),
        mean_meeting_distance_from_center=mean_meeting_distance,
        total_transmission_events=int(total_transmissions),
        transmission_cells_visited=int(len(transmission_cells)),
        mean_transmission_distance_from_center=mean_transmission_distance,
        transmission_efficiency=transmission_efficiency,
    )
    series = {
        "t_axis": np.array(t_axis, dtype=np.int64),
        "informed_fraction": np.array(informed_fraction, dtype=np.float64),
        "cumulative_meetings": np.array(cumulative_meetings, dtype=np.int64),
        "cumulative_transmissions": np.array(cumulative_transmissions, dtype=np.int64),
        "meetings_per_interval": np.array(meetings_per_interval, dtype=np.int64),
        "transmissions_per_interval": np.array(transmissions_per_interval, dtype=np.int64),
        "meeting_heatmap": meeting_heatmap,
        "transmission_heatmap": transmission_heatmap,
    }
    return point, series


def information_spread_validation(T: int, seed: int) -> Dict[str, object]:
    values = [-10.0, 0.0, 10.0]
    points: List[InformationSpreadPoint] = []
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for idx, value in enumerate(values):
        point, value_series = run_information_spread_case(T=T, lambda_I_value=value, seed=seed + 100 + idx)
        points.append(point)
        series[f"{value:+.1f}"] = value_series

    negative = next(point for point in points if point.lambda_I_value < 0.0)
    zero = next(point for point in points if abs(point.lambda_I_value) < 1e-12)
    positive = next(point for point in points if point.lambda_I_value > 0.0)

    def finite_or_large(value: int | None) -> float:
        return float(value) if value is not None else float("inf")

    return {
        "points": points,
        "series": series,
        "negative_reaches_90_percent": bool(negative.time_to_90_percent is not None),
        "all_cases_reach_90_percent": bool(all(point.time_to_90_percent is not None for point in points)),
        "all_cases_reach_all_informed": bool(all(point.time_to_all_informed is not None for point in points)),
        "negative_generates_transmissions": bool(negative.total_transmission_events > 0),
        "transmission_locations_recorded": bool(negative.transmission_cells_visited > 0),
        "negative_generates_more_meetings_than_zero_and_positive": bool(
            negative.total_meetings > zero.total_meetings and negative.total_meetings > positive.total_meetings
        ),
        "negative_transmissions_are_more_spatially_concentrated": bool(
            negative.transmission_cells_visited < zero.transmission_cells_visited
            and negative.transmission_cells_visited < positive.transmission_cells_visited
        ),
        "fastest_fraction_informed_lambda_I": float(
            min(points, key=lambda point: finite_or_large(point.time_to_90_percent)).lambda_I_value
        ),
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


def make_information_spread_figure(outdir: Path, spread: Dict[str, object]) -> None:
    points: List[InformationSpreadPoint] = spread["points"]
    series: Dict[str, Dict[str, np.ndarray]] = spread["series"]

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.5))
    colors = {-10.0: "crimson", 0.0: "grey", 10.0: "seagreen"}

    for point in points:
        key = f"{point.lambda_I_value:+.1f}"
        axes[0, 0].plot(
            series[key]["t_axis"],
            series[key]["informed_fraction"],
            lw=2.0,
            color=colors[point.lambda_I_value],
            label=f"lambda_I={point.lambda_I_value:+.0f}",
        )
    axes[0, 0].set_title("Fraction of informed robots")
    axes[0, 0].set_xlabel("simulation step")
    axes[0, 0].set_ylabel("informed fraction")
    axes[0, 0].set_ylim(-0.02, 1.02)
    axes[0, 0].legend(fontsize=8)

    for point in points:
        key = f"{point.lambda_I_value:+.1f}"
        axes[0, 1].plot(
            series[key]["t_axis"],
            np.maximum(series[key]["cumulative_meetings"], 1),
            lw=2.0,
            color=colors[point.lambda_I_value],
            label=f"lambda_I={point.lambda_I_value:+.0f}",
        )
    axes[0, 1].set_title("Cumulative physical meetings")
    axes[0, 1].set_xlabel("simulation step")
    axes[0, 1].set_ylabel("meetings, log scale")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=8)

    labels = [f"{point.lambda_I_value:+.0f}" for point in points]
    t90 = [
        point.time_to_90_percent if point.time_to_90_percent is not None else np.nan
        for point in points
    ]
    axes[1, 0].bar(labels, t90, color=[colors[point.lambda_I_value] for point in points])
    axes[1, 0].set_title("Time to 90% informed")
    axes[1, 0].set_xlabel("lambda_I")
    axes[1, 0].set_ylabel("simulation step")

    x = np.arange(len(points))
    width = 0.35
    meeting_cells = [point.meeting_cells_visited for point in points]
    transmission_cells = [point.transmission_cells_visited for point in points]
    axes[1, 1].bar(x - width / 2, meeting_cells, width=width, color="steelblue", label="meeting cells")
    axes[1, 1].bar(x + width / 2, transmission_cells, width=width, color="darkorange", label="transmission cells")
    axes[1, 1].set_xticks(x, labels=labels)
    axes[1, 1].set_title("Spatial spread of contacts")
    axes[1, 1].set_xlabel("lambda_I")
    axes[1, 1].set_ylabel("number of cells")
    axes[1, 1].legend(fontsize=8)

    fig.suptitle("Information diffusion validation: informed fraction and meeting locations")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_info_diffusion_validation_spread.png", dpi=140)
    plt.close(fig)


def make_information_meeting_maps_figure(outdir: Path, spread: Dict[str, object]) -> None:
    points: List[InformationSpreadPoint] = spread["points"]
    series: Dict[str, Dict[str, np.ndarray]] = spread["series"]

    fig, axes = plt.subplots(2, len(points), figsize=(13.8, 7.4))
    meeting_max = max(
        float(np.log1p(series[f"{point.lambda_I_value:+.1f}"]["meeting_heatmap"]).max())
        for point in points
    )
    transmission_max = max(
        float(np.log1p(series[f"{point.lambda_I_value:+.1f}"]["transmission_heatmap"]).max())
        for point in points
    )
    meeting_max = max(meeting_max, 1e-12)
    transmission_max = max(transmission_max, 1e-12)

    for col, point in enumerate(points):
        key = f"{point.lambda_I_value:+.1f}"
        meeting_map = np.log1p(series[key]["meeting_heatmap"]).reshape(mid.NY, mid.NX)
        transmission_map = np.log1p(series[key]["transmission_heatmap"]).reshape(mid.NY, mid.NX)

        im0 = axes[0, col].imshow(meeting_map, origin="lower", cmap="Blues", vmin=0.0, vmax=meeting_max)
        axes[0, col].set_title(
            f"lambda_I={point.lambda_I_value:+.0f}: all meetings\n"
            f"{point.total_meetings} meetings, {point.meeting_cells_visited} cells"
        )
        axes[0, col].set_xlabel("col")
        axes[0, col].set_ylabel("row")
        fig.colorbar(im0, ax=axes[0, col], fraction=0.046)

        im1 = axes[1, col].imshow(
            transmission_map,
            origin="lower",
            cmap="magma",
            vmin=0.0,
            vmax=transmission_max,
        )
        axes[1, col].set_title(
            f"lambda_I={point.lambda_I_value:+.0f}: successful transmissions\n"
            f"{point.total_transmission_events} events, {point.transmission_cells_visited} cells"
        )
        axes[1, col].set_xlabel("col")
        axes[1, col].set_ylabel("row")
        fig.colorbar(im1, ax=axes[1, col], fraction=0.046)

    fig.suptitle("Where robots met and where information changed hands, log(1 + count)")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_info_diffusion_validation_meeting_maps.png", dpi=140)
    plt.close(fig)


def save_raw_data(outdir: Path, sweep: Dict[str, object], spread: Dict[str, object]) -> None:
    points: List[SignSweepPoint] = sweep["points"]
    spread_points: List[InformationSpreadPoint] = spread["points"]
    spread_series: Dict[str, Dict[str, np.ndarray]] = spread["series"]
    spread_arrays: Dict[str, np.ndarray] = {}
    for point in spread_points:
        key = f"{point.lambda_I_value:+.1f}".replace("+", "pos_").replace("-", "neg_").replace(".", "_")
        spread_arrays[f"spread_{key}_t_axis"] = spread_series[f"{point.lambda_I_value:+.1f}"]["t_axis"]
        spread_arrays[f"spread_{key}_informed_fraction"] = spread_series[f"{point.lambda_I_value:+.1f}"]["informed_fraction"]
        spread_arrays[f"spread_{key}_cumulative_meetings"] = spread_series[f"{point.lambda_I_value:+.1f}"]["cumulative_meetings"]
        spread_arrays[f"spread_{key}_cumulative_transmissions"] = spread_series[f"{point.lambda_I_value:+.1f}"]["cumulative_transmissions"]
        spread_arrays[f"spread_{key}_meetings_per_interval"] = spread_series[f"{point.lambda_I_value:+.1f}"]["meetings_per_interval"]
        spread_arrays[f"spread_{key}_transmissions_per_interval"] = spread_series[f"{point.lambda_I_value:+.1f}"]["transmissions_per_interval"]
        spread_arrays[f"spread_{key}_meeting_heatmap"] = spread_series[f"{point.lambda_I_value:+.1f}"]["meeting_heatmap"]
        spread_arrays[f"spread_{key}_transmission_heatmap"] = spread_series[f"{point.lambda_I_value:+.1f}"]["transmission_heatmap"]
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
        spread_lambda_I_values=np.array([point.lambda_I_value for point in spread_points], dtype=np.float64),
        spread_final_informed_fraction=np.array([point.final_informed_fraction for point in spread_points], dtype=np.float64),
        spread_time_to_50_percent=np.array([
            -1 if point.time_to_50_percent is None else point.time_to_50_percent for point in spread_points
        ], dtype=np.int64),
        spread_time_to_90_percent=np.array([
            -1 if point.time_to_90_percent is None else point.time_to_90_percent for point in spread_points
        ], dtype=np.int64),
        spread_time_to_all_informed=np.array([
            -1 if point.time_to_all_informed is None else point.time_to_all_informed for point in spread_points
        ], dtype=np.int64),
        spread_total_meetings=np.array([point.total_meetings for point in spread_points], dtype=np.int64),
        spread_meeting_cells_visited=np.array([point.meeting_cells_visited for point in spread_points], dtype=np.int64),
        spread_total_transmission_events=np.array([point.total_transmission_events for point in spread_points], dtype=np.int64),
        spread_transmission_cells_visited=np.array(
            [point.transmission_cells_visited for point in spread_points],
            dtype=np.int64,
        ),
        spread_transmission_efficiency=np.array(
            [point.transmission_efficiency for point in spread_points],
            dtype=np.float64,
        ),
        **spread_arrays,
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
    spread: Dict[str, object],
) -> None:
    points: List[SignSweepPoint] = sweep["points"]
    zero_point: SignSweepPoint = sweep["zero_baseline"]
    spread_points: List[InformationSpreadPoint] = spread["points"]

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
            "spread_T": args.spread_T,
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
        "informed_robot_spread": {
            "points": [asdict(point) for point in spread_points],
            "negative_reaches_90_percent": spread["negative_reaches_90_percent"],
            "all_cases_reach_90_percent": spread["all_cases_reach_90_percent"],
            "all_cases_reach_all_informed": spread["all_cases_reach_all_informed"],
            "negative_generates_transmissions": spread["negative_generates_transmissions"],
            "transmission_locations_recorded": spread["transmission_locations_recorded"],
            "negative_generates_more_meetings_than_zero_and_positive": spread[
                "negative_generates_more_meetings_than_zero_and_positive"
            ],
            "negative_transmissions_are_more_spatially_concentrated": spread[
                "negative_transmissions_are_more_spatially_concentrated"
            ],
            "fastest_fraction_informed_lambda_I": spread["fastest_fraction_informed_lambda_I"],
            "interpretation": (
                "Fraction-informed curves and transmission heatmaps validate information diffusion "
                "more directly than visit maps by measuring robot-level information spread, the number "
                "of physical meetings that make communication possible, and where information-changing "
                "encounters occurred. Cumulative successful transmissions are not used as a primary plot "
                "because they are almost algebraically equivalent to the number of newly informed robots."
            ),
        },
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
        and spread["negative_reaches_90_percent"]
        and spread["all_cases_reach_90_percent"]
        and spread["all_cases_reach_all_informed"]
        and spread["negative_generates_transmissions"]
        and spread["transmission_locations_recorded"]
        and spread["negative_generates_more_meetings_than_zero_and_positive"]
        and spread["negative_transmissions_are_more_spatially_concentrated"]
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
    spread = information_spread_validation(T=args.spread_T, seed=args.seed)

    make_sign_sweep_figure(outdir, sweep)
    make_information_spread_figure(outdir, spread)
    make_information_meeting_maps_figure(outdir, spread)
    save_raw_data(outdir, sweep, spread)
    save_summary(outdir, args, kernel, field, gate, meeting, sweep, preview, spread)

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
        and spread["negative_reaches_90_percent"]
        and spread["all_cases_reach_90_percent"]
        and spread["all_cases_reach_all_informed"]
        and spread["negative_generates_transmissions"]
        and spread["transmission_locations_recorded"]
        and spread["negative_generates_more_meetings_than_zero_and_positive"]
        and spread["negative_transmissions_are_more_spatially_concentrated"]
    )
    negative_spread = next(point for point in spread["points"] if point.lambda_I_value < 0.0)

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
    print(
        "  Informed spread       : "
        f"lambda_I=-10 reaches 90% at t={negative_spread.time_to_90_percent}, "
        f"meetings={negative_spread.total_meetings}, "
        f"transmission cells={negative_spread.transmission_cells_visited}, "
        f"fastest lambda_I={spread['fastest_fraction_informed_lambda_I']:+.0f}"
    )
    print("  Saved summary         : maxcal_info_diffusion_validation_summary.json")


if __name__ == "__main__":
    main()
