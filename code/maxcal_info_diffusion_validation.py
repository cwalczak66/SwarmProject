"""
Validation script for Information Diffusion Layer 1-I.

This validator is written around the Pinciroli sign prediction quoted by the
user from the local draft:

    lambda_I < 0   -> robots cluster
    lambda_I >= 0  -> robots prioritize coverage / avoid clustering

The checks are therefore organized around that exact claim:

1. Local kernel sign check:
   for a fixed positive information field, negative lambda_I must increase the
   probability of moving toward high-information neighbors, zero must recover
   the coverage kernel, and positive lambda_I must do the opposite.

2. Paper fixed-point sign sweep:
   the information observable is π̄_k from Pinciroli's fixed-point equation,
   and p_enc = sum_k π̄_k^2 is recorded as the theoretical encounter
   probability used in the mean-field diffusion equation.

3. Stochastic information sharing:
   robots encounter each other when they occupy the same region. Successful
   communication is Bernoulli with probability beta, and successful exchange
   resets the Age of Information tau_i, i.e. the time since that robot last exchanged information. 
   This tests actual sharing, not only radius-based contact counting.

4. Inverse information-rate solve:
   unlike coverage, which inverts a desired spatial distribution, Layer 1-I
   inverts spreading-rate constraint dn/dt by finding the scalar
   lambda_I whose fixed-point p_enc produces the requested rate. The inverse
   solve follows a chosen fixed-point branch by continuation and refines the
   multiplier by bisection when the target is bracketed.

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

SIGN_SWEEP_VALUES = [-400.0, -200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0, 400.0]
PURE_STALE_AGE = 1.0e6
PURE_FRESH_AGE = 0.0


@dataclass
class SignSweepPoint:
    lambda_I_value: float
    mean_dispersion: float
    mean_encounter_proxy: float
    theoretical_p_enc: float
    final_coverage_l1: float
    total_markov_arrivals: int
    corner_visit_probability: float
    edge_visit_probability: float
    interior_visit_probability: float


@dataclass
class InformationSpreadPoint:
    lambda_I_value: float
    theoretical_p_enc: float
    beta_transmission: float
    predicted_initial_dn_dt: float
    initial_source_cell: int
    final_informed_fraction: float
    time_to_50_percent: int | None
    time_to_90_percent: int | None
    time_to_all_informed: int | None
    total_meetings: int
    total_successful_communications: int
    total_map_records_received: int
    meeting_cells_visited: int
    mean_meeting_distance_from_center: float
    total_transmission_events: int
    transmission_cells_visited: int
    mean_transmission_distance_from_center: float
    transmission_efficiency: float
    final_mean_coverage_age: float
    final_mean_aoi: float
    final_mean_local_coverage_age: float
    final_mean_map_record_age: float


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
        help="Simulation length for the isolated sign sweep. Use 9000 for the report-scale run.",
    )
    parser.add_argument(
        "--stage2-preview-T",
        type=int,
        default=0,
        help="Optional simulation length for legacy age-observable preview runs; 0 skips this diagnostic.",
    )
    parser.add_argument(
        "--spread-T",
        type=int,
        default=6_000,
        help="Simulation length for the informed-robot spread validation. Use 6000 for the report-scale run.",
    )
    parser.add_argument(
        "--inverse-target-rate",
        type=float,
        default=None,
        help=(
            "Target dn/dt for the inverse information-rate validation. If omitted, "
            "--inverse-target-p-enc is converted to a rate using Pinciroli's "
            "mean-field equation at one initially informed robot."
        ),
    )
    parser.add_argument(
        "--inverse-target-p-enc",
        type=float,
        default=0.50,
        help="Encounter-probability target used to derive the inverse target rate when no rate is supplied.",
    )
    parser.add_argument(
        "--inverse-branch",
        type=str,
        choices=("auto", "clustered", "coverage"),
        default="auto",
        help=(
            "Fixed-point branch used by the inverse Layer 1-I solve. "
            "'auto' selects clustered for targets above the zero-lambda rate and "
            "coverage for targets below it."
        ),
    )
    parser.add_argument(
        "--inverse-continuation-step",
        type=float,
        default=10.0,
        help="Lambda_I step size used to trace each fixed-point branch before local bisection.",
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
    robot = mid.Robot(robot_id, cx, cy, k, k, cx, cy, info_age=age, last_meet=0)
    robot.world_map = mid.RobotWorldMap.empty(w.K)
    robot.world_map.observe_cell(k, 0.0)
    return robot


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


def local_map_age_checks() -> Dict[str, float | bool]:
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    first_cell = 0
    second_cell = w.K - 1
    first = _robot_at_cell(0, w, first_cell, PURE_FRESH_AGE)
    second = _robot_at_cell(1, w, second_cell, PURE_FRESH_AGE)

    first_map = mid.ensure_robot_map(first, w)
    second_map = mid.ensure_robot_map(second, w)
    first_cov_before = float(first_map.coverage_age(20.0)[second_cell])
    first_record_age_before = float(first_map.map_record_age(20.0)[second_cell])

    updates = mid.exchange_robot_world_maps([first, second], w, 0, 1, t=20)
    first_cov_after = float(first_map.coverage_age(20.0)[second_cell])
    first_record_age_after = float(first_map.map_record_age(20.0)[second_cell])
    second_cov_after = float(second_map.coverage_age(20.0)[first_cell])
    second_record_age_after = float(second_map.map_record_age(20.0)[first_cell])

    return {
        "first_unknown_second_cell_coverage_age_before": first_cov_before,
        "first_unknown_second_cell_map_record_age_before": first_record_age_before,
        "map_records_received": int(updates),
        "first_received_second_cell_visit_timestamp": bool(first_map.last_visit_time[second_cell] == 0.0),
        "second_received_first_cell_visit_timestamp": bool(second_map.last_visit_time[first_cell] == 0.0),
        "first_cellwise_coverage_age_after_exchange": first_cov_after,
        "first_cellwise_map_record_age_after_exchange": first_record_age_after,
        "second_cellwise_coverage_age_after_exchange": second_cov_after,
        "second_cellwise_map_record_age_after_exchange": second_record_age_after,
        "coverage_age_preserves_physical_visit_time": bool(first_cov_after == 20.0 and second_cov_after == 20.0),
        "map_record_age_resets_on_received_map_record": bool(first_record_age_after == 0.0 and second_record_age_after == 0.0),
    }


def information_exchange_checks() -> Dict[str, int | bool]:
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    same_cell = (mid.NY // 2) * mid.NX + mid.NX // 2
    far_cell = 0
    robots = [
        _robot_at_cell(0, w, same_cell, 25.0),
        _robot_at_cell(1, w, same_cell, 40.0),
        _robot_at_cell(2, w, far_cell, 55.0),
    ]
    robots[0].world_map.observe_cell(far_cell, 2.0)
    robots[1].world_map.observe_cell(w.K - 1, 3.0)
    informed = np.array([True, False, False], dtype=bool)
    encounters, communications, transmissions, map_updates = mid.perform_information_exchanges(
        robots,
        w,
        np.random.default_rng(123),
        t=7,
        informed=informed,
        beta=1.0,
        p_encounter_given_colocation=1.0,
    )
    return {
        "same_region_encounters": encounters,
        "successful_communications": communications,
        "new_transmissions": transmissions,
        "map_records_received": map_updates,
        "first_robot_reset": bool(robots[0].info_age == 0.0 and robots[0].last_meet == 7),
        "second_robot_reset": bool(robots[1].info_age == 0.0 and robots[1].last_meet == 7),
        "third_robot_unchanged": bool(robots[2].info_age == 55.0 and robots[2].last_meet == 0),
        "information_shared": bool(informed[0] and informed[1] and not informed[2]),
        "first_received_second_map_record": bool(robots[0].world_map.last_visit_time[w.K - 1] == 3.0),
        "second_received_first_map_record": bool(robots[1].world_map.last_visit_time[far_cell] == 2.0),
        "first_map_record_age_reset_for_received_record": bool(robots[0].world_map.map_record_age(7.0)[w.K - 1] == 0.0),
        "second_map_record_age_reset_for_received_record": bool(robots[1].world_map.map_record_age(7.0)[far_cell] == 0.0),
        "first_paper_aoi_reset_after_exchange": bool(robots[0].info_age == 0.0),
        "second_paper_aoi_reset_after_exchange": bool(robots[1].info_age == 0.0),
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
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    lambda_c = mid.build_lambda_C(w, mid.LAMBDA_C_VAL)
    pi_bar, _, _, _ = mid.solve_information_fixed_point(w, lambda_c, lambda_I_value, seed=seed)
    res = mid.run_simulation(
        T=T,
        lambda_I_value=lambda_I_value,
        seed=seed,
        fixed_initial_info_age=PURE_STALE_AGE,
        freeze_info_age=True,
        enable_meetings=False,
        information_field_mode="stationary_fixed_point",
        use_age_gate=False,
    )
    reps = representative_probabilities(res.pi_empirical)
    return SignSweepPoint(
        lambda_I_value=lambda_I_value,
        mean_dispersion=float(res.dispersion.mean()),
        mean_encounter_proxy=float(res.encounter_proxy.mean()),
        theoretical_p_enc=mid.encounter_probability_from_stationary(pi_bar),
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
    positive_points = [point for point in points if point.lambda_I_value > 0.0]

    return {
        "points": points,
        "zero_baseline": zero_point,
        "all_negative_reduce_dispersion_vs_zero": bool(
            all(point.mean_dispersion < zero_point.mean_dispersion for point in negative_points)
        ),
        "all_negative_increase_encounter_vs_zero": bool(
            all(point.mean_encounter_proxy > zero_point.mean_encounter_proxy for point in negative_points)
        ),
        "all_negative_increase_theoretical_p_enc_vs_zero": bool(
            all(point.theoretical_p_enc > zero_point.theoretical_p_enc for point in negative_points)
        ),
        "all_positive_reduce_theoretical_p_enc_vs_zero": bool(
            all(point.theoretical_p_enc < zero_point.theoretical_p_enc for point in positive_points)
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
        fixed_initial_info_age=None,
        freeze_info_age=False,
        enable_meetings=True,
        information_field_mode="stationary_fixed_point",
        use_age_gate=False,
    )
    negative_mode = mid.run_simulation(
        T=T,
        lambda_I_value=-400.0,
        seed=seed + 1,
        fixed_initial_info_age=None,
        freeze_info_age=False,
        enable_meetings=True,
        information_field_mode="stationary_fixed_point",
        use_age_gate=False,
    )
    positive_mode = mid.run_simulation(
        T=T,
        lambda_I_value=400.0,
        seed=seed + 2,
        fixed_initial_info_age=None,
        freeze_info_age=False,
        enable_meetings=True,
        information_field_mode="stationary_fixed_point",
        use_age_gate=False,
    )

    def pack(res: mid.SimResult) -> Dict[str, float]:
        return {
            "mean_dispersion": float(res.dispersion.mean()),
            "mean_encounter_proxy": float(res.encounter_proxy.mean()),
            "final_coverage_l1": float(res.coverage_l1[-1]),
            "mean_info_age": float(res.mean_info_age.mean()),
            "mean_coverage_age": float(res.mean_cov_age.mean()),
            "mean_local_coverage_age": float(res.mean_local_cov_age.mean()),
            "mean_map_record_age": float(res.mean_map_record_age.mean()),
            "mean_meetings_per_snap": float(res.meetings_per_snap.mean()),
        }

    negative_clusters_vs_zero = (
        float(negative_mode.encounter_proxy.mean()) > float(zero_baseline.encounter_proxy.mean())
        and float(negative_mode.meetings_per_snap.mean()) > float(zero_baseline.meetings_per_snap.mean())
        and float(negative_mode.dispersion.mean()) < float(zero_baseline.dispersion.mean())
    )
    positive_preserves_coverage_vs_zero = (
        float(positive_mode.encounter_proxy.mean()) <= 1.10 * float(zero_baseline.encounter_proxy.mean())
        and float(positive_mode.dispersion.mean()) >= 0.90 * float(zero_baseline.dispersion.mean())
    )
    negative_reduces_aoi_vs_positive = (
        float(negative_mode.mean_info_age.mean()) < float(positive_mode.mean_info_age.mean())
    )

    return {
        "zero_baseline": pack(zero_baseline),
        "negative_information_mode": pack(negative_mode),
        "positive_information_mode": pack(positive_mode),
        "negative_lambda_clusters_vs_zero": bool(negative_clusters_vs_zero),
        "positive_lambda_preserves_coverage_vs_zero": bool(positive_preserves_coverage_vs_zero),
        "negative_lambda_reduces_paper_aoi_vs_positive": bool(negative_reduces_aoi_vs_positive),
        "paper_aoi_is_time_since_robot_robot_exchange": True,
        "local_maps_are_freshness_records_not_the_paper_aoi": True,
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
    control_pi_bar: np.ndarray | None = None,
) -> tuple[InformationSpreadPoint, Dict[str, np.ndarray]]:
    """
    Track information diffusion directly as a robot-level contagion process.

    A small set of initially informed robots carries a binary information flag.
    When two robots occupy the same region, an encounter is registered. Given an
    encounter, communication succeeds with probability beta. Only successful
    communication can transmit the information flag and reset AoI.
    """
    rng = np.random.default_rng(seed)
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    lambda_c = mid.build_lambda_C(w, mid.LAMBDA_C_VAL)
    if control_pi_bar is None:
        pi_bar, _, _, _ = mid.solve_information_fixed_point(w, lambda_c, lambda_I_value, seed=seed)
    else:
        pi_bar = np.asarray(control_pi_bar, dtype=np.float64).copy()
        pi_bar = pi_bar / max(float(pi_bar.sum()), 1.0e-300)
    theoretical_p_enc = mid.encounter_probability_from_stationary(pi_bar)
    predicted_initial_dn_dt = mid.logistic_information_rate(
        informed_fraction=n_initial_informed / mid.N_ROBOTS,
        p_enc=theoretical_p_enc,
        beta=mid.BETA_TRANSMISSION,
        n_agents=mid.N_ROBOTS,
    )
    robots = [mid.make_robot(i, w, rng, fixed_initial_info_age=None) for i in range(mid.N_ROBOTS)]
    mid.initialise_targets(
        robots,
        w,
        lambda_c,
        lambda_I_value,
        rng,
        info_field=pi_bar,
        use_age_gate=False,
    )

    informed = np.zeros(mid.N_ROBOTS, dtype=bool)
    source_cell = int(np.argmax(pi_bar))
    source_xy = w.centers[source_cell]
    robot_xy = np.array([[robot.x, robot.y] for robot in robots], dtype=np.float64)
    source_order = np.argsort(np.linalg.norm(robot_xy - source_xy[None, :], axis=1))
    initial_indices = source_order[:n_initial_informed]
    informed[initial_indices] = True
    markov_visits = np.zeros(w.K, dtype=np.int64)
    coverage_age_field = np.full(w.K, -1.0, dtype=np.float64)
    for robot in robots:
        coverage_age_field[robot.from_k] = 0.0

    t_axis: List[int] = []
    informed_fraction: List[float] = []
    mean_coverage_age: List[float] = []
    mean_aoi: List[float] = []
    mean_local_coverage_age: List[float] = []
    mean_map_record_age: List[float] = []
    cumulative_meetings: List[int] = []
    cumulative_transmissions: List[int] = []
    meetings_per_interval: List[int] = []
    transmissions_per_interval: List[int] = []
    meeting_heatmap = np.zeros(w.K, dtype=np.float64)
    transmission_heatmap = np.zeros(w.K, dtype=np.float64)
    total_meetings = 0
    total_successful_communications = 0
    total_transmissions = 0
    total_map_records_received = 0
    interval_meetings = 0
    interval_transmissions = 0

    for t in range(1, T + 1):
        for r in robots:
            mid.step_robot(
                r=r,
                w=w,
                lambda_C=lambda_c,
                lambda_I_value=lambda_I_value,
                info_field=pi_bar,
                speed=mid.ROBOT_SPEED,
                markov_visits=markov_visits,
                coverage_age_field=coverage_age_field,
                t=t,
                rng=rng,
                freeze_info_age=False,
                use_age_gate=False,
            )

        for i, j, cell, mx, my in mid.same_region_pairs(robots, w):
            if rng.random() > mid.P_ENCOUNTER_GIVEN_COLOCATION:
                continue
            total_meetings += 1
            interval_meetings += 1
            meeting_heatmap[cell] += 1.0
            if rng.random() > mid.BETA_TRANSMISSION:
                continue
            total_successful_communications += 1
            robots[i].info_age = 0.0
            robots[j].info_age = 0.0
            robots[i].last_meet = t
            robots[j].last_meet = t
            total_map_records_received += mid.exchange_robot_world_maps(robots, w, i, j, t)
            before_i = bool(informed[i])
            before_j = bool(informed[j])
            if before_i or before_j:
                informed[i] = True
                informed[j] = True
            if before_i != before_j:
                total_transmissions += 1
                interval_transmissions += 1
                transmission_heatmap[cell] += 1.0

        if t % mid.RECORD_EVERY == 0 or t == T:
            t_axis.append(t)
            informed_fraction.append(float(informed.mean()))
            maps = [mid.ensure_robot_map(robot, w) for robot in robots]
            global_cov_age = np.where(
                coverage_age_field >= 0.0,
                float(t) - coverage_age_field,
                float(t) + 1.0,
            )
            mean_coverage_age.append(float(np.mean(global_cov_age)))
            mean_aoi.append(float(np.mean([robot.info_age for robot in robots])))
            mean_local_coverage_age.append(mid.mean_robot_coverage_age(maps, float(t)))
            mean_map_record_age.append(mid.mean_robot_map_record_age(maps, float(t)))
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
        theoretical_p_enc=float(theoretical_p_enc),
        beta_transmission=float(mid.BETA_TRANSMISSION),
        predicted_initial_dn_dt=float(predicted_initial_dn_dt),
        initial_source_cell=int(source_cell),
        final_informed_fraction=float(informed_fraction[-1]),
        time_to_50_percent=_threshold_time(t_axis, informed_fraction, 0.50),
        time_to_90_percent=_threshold_time(t_axis, informed_fraction, 0.90),
        time_to_all_informed=_threshold_time(t_axis, informed_fraction, 1.00),
        total_meetings=int(total_meetings),
        total_successful_communications=int(total_successful_communications),
        total_map_records_received=int(total_map_records_received),
        meeting_cells_visited=int(len(meeting_cells)),
        mean_meeting_distance_from_center=mean_meeting_distance,
        total_transmission_events=int(total_transmissions),
        transmission_cells_visited=int(len(transmission_cells)),
        mean_transmission_distance_from_center=mean_transmission_distance,
        transmission_efficiency=transmission_efficiency,
        final_mean_coverage_age=float(mean_coverage_age[-1]),
        final_mean_aoi=float(mean_aoi[-1]),
        final_mean_local_coverage_age=float(mean_local_coverage_age[-1]),
        final_mean_map_record_age=float(mean_map_record_age[-1]),
    )
    series = {
        "t_axis": np.array(t_axis, dtype=np.int64),
        "informed_fraction": np.array(informed_fraction, dtype=np.float64),
        "mean_coverage_age": np.array(mean_coverage_age, dtype=np.float64),
        "mean_aoi": np.array(mean_aoi, dtype=np.float64),
        "mean_local_coverage_age": np.array(mean_local_coverage_age, dtype=np.float64),
        "mean_map_record_age": np.array(mean_map_record_age, dtype=np.float64),
        "cumulative_meetings": np.array(cumulative_meetings, dtype=np.int64),
        "cumulative_transmissions": np.array(cumulative_transmissions, dtype=np.int64),
        "meetings_per_interval": np.array(meetings_per_interval, dtype=np.int64),
        "transmissions_per_interval": np.array(transmissions_per_interval, dtype=np.int64),
        "meeting_heatmap": meeting_heatmap,
        "transmission_heatmap": transmission_heatmap,
        "total_map_records_received": np.array([total_map_records_received], dtype=np.int64),
    }
    return point, series


def information_spread_validation(T: int, seed: int) -> Dict[str, object]:
    values = [-400.0, 0.0, 400.0]
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
            negative.transmission_cells_visited > 0
            and negative.transmission_cells_visited <= max(zero.transmission_cells_visited, positive.transmission_cells_visited, 1)
        ),
        "fastest_fraction_informed_lambda_I": float(
            min(points, key=lambda point: finite_or_large(point.time_to_90_percent)).lambda_I_value
        ),
    }


def inverse_information_rate_validation(
    T: int,
    seed: int,
    target_rate: float | None,
    target_p_enc: float,
    branch: str,
    continuation_step: float,
) -> Dict[str, object]:
    """
    Invert information-rate constraint to compute lambda_I.

    Coverage uses a target spatial distribution. Layer 1-I instead uses the
    mean-field spreading rate dn/dt = A beta p_enc f(1-f), where p_enc is
    induced by the stationary distribution of the information kernel.
    """
    w = mid.build_world(mid.NX, mid.NY, mid.CELL_SIZE)
    lambda_c = mid.build_lambda_C(w, mid.LAMBDA_C_VAL)
    informed_fraction = 1.0 / float(mid.N_ROBOTS)
    if target_rate is None:
        selected_target_p_enc = float(target_p_enc)
        selected_target_rate = mid.logistic_information_rate(
            informed_fraction=informed_fraction,
            p_enc=selected_target_p_enc,
            beta=mid.BETA_TRANSMISSION,
            n_agents=mid.N_ROBOTS,
        )
    else:
        selected_target_rate = float(target_rate)
        selected_target_p_enc = selected_target_rate / max(
            float(mid.N_ROBOTS) * mid.BETA_TRANSMISSION * informed_fraction * (1.0 - informed_fraction),
            1.0e-12,
        )

    inverse = mid.solve_lambda_I_for_information_rate(
        w=w,
        lambda_C=lambda_c,
        target_rate=selected_target_rate,
        informed_fraction=informed_fraction,
        beta=mid.BETA_TRANSMISSION,
        n_agents=mid.N_ROBOTS,
        seed=seed,
        branch=branch,
        continuation_step=continuation_step,
    )

    empirical_point, empirical_series = run_information_spread_case(
        T=T,
        lambda_I_value=float(inverse["lambda_I"]),
        seed=seed + 500,
        control_pi_bar=np.asarray(inverse["selected_pi_bar"], dtype=np.float64),
    )
    n_series = empirical_series["informed_fraction"] * float(mid.N_ROBOTS)
    t_axis = empirical_series["t_axis"]
    if len(t_axis) >= 1:
        empirical_initial_rate = float((n_series[0] - 1.0) / max(float(t_axis[0]), 1.0))
    else:
        empirical_initial_rate = float("nan")

    inverse["requested_target_p_enc"] = selected_target_p_enc
    inverse["empirical_spread_point"] = empirical_point
    inverse["empirical_initial_dn_dt_estimate"] = empirical_initial_rate
    inverse["empirical_series"] = empirical_series
    inverse["interpretation"] = (
        "This is the Layer 1-I analogue of inverse coverage, but the target is "
        "the information-spreading rate rather than a spatial visitation "
        "distribution. The solver follows a self-consistent fixed-point branch, "
        "computes p_enc = sum_k pi_bar_k^2 along that branch, and matches "
        "dn/dt = A beta p_enc f(1-f). When a branch brackets the target, a "
        "local bisection refines lambda_I; otherwise the validation reports the "
        "target as unreachable for the chosen scalar Layer 1-I kernel."
    )
    return inverse


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
    axes[1, 0].set_ylabel("final ||c(.,t) - π̄(.)||_1")

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

    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5))
    colors = {point.lambda_I_value: ("crimson" if point.lambda_I_value < 0 else "seagreen") for point in points}
    for point in points:
        if abs(point.lambda_I_value) < 1e-12:
            colors[point.lambda_I_value] = "grey"

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
            series[key]["mean_aoi"],
            lw=2.0,
            color=colors[point.lambda_I_value],
            label=f"lambda_I={point.lambda_I_value:+.0f}",
        )
    axes[0, 1].set_title("Mean Age of Information tau")
    axes[0, 1].set_xlabel("simulation step")
    axes[0, 1].set_ylabel("time since robot-robot exchange")
    axes[0, 1].legend(fontsize=8)

    for point in points:
        key = f"{point.lambda_I_value:+.1f}"
        axes[0, 2].plot(
            series[key]["t_axis"],
            np.maximum(series[key]["cumulative_meetings"], 1),
            lw=2.0,
            color=colors[point.lambda_I_value],
            label=f"lambda_I={point.lambda_I_value:+.0f}",
        )
    axes[0, 2].set_title("Cumulative physical meetings")
    axes[0, 2].set_xlabel("simulation step")
    axes[0, 2].set_ylabel("meetings, log scale")
    axes[0, 2].set_yscale("log")
    axes[0, 2].legend(fontsize=8)

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

    final_cov_age = [point.final_mean_coverage_age for point in points]
    final_aoi = [point.final_mean_aoi for point in points]
    axes[1, 2].bar(x - width / 2, final_cov_age, width=width, color="seagreen", label="coverage age")
    axes[1, 2].bar(x + width / 2, final_aoi, width=width, color="crimson", label="AoI tau")
    axes[1, 2].set_xticks(x, labels=labels)
    axes[1, 2].set_title("Final local age observables")
    axes[1, 2].set_xlabel("lambda_I")
    axes[1, 2].set_ylabel("mean age")
    axes[1, 2].legend(fontsize=8)

    fig.suptitle("Information diffusion validation: informed fraction and meeting locations")
    fig.tight_layout()
    fig.savefig(outdir / "maxcal_info_diffusion_validation_spread.png", dpi=140)
    plt.close(fig)


def make_inverse_rate_figure(outdir: Path, inverse: Dict[str, object]) -> None:
    evaluations = inverse["evaluations"]
    branch_evaluations = inverse.get("branch_evaluations", {})
    selected_lambda = float(inverse["lambda_I"])
    target_rate = float(inverse["target_information_spreading_rate"])
    selected_rate = float(inverse["predicted_information_spreading_rate"])
    series: Dict[str, np.ndarray] = inverse["empirical_series"]
    empirical_point: InformationSpreadPoint = inverse["empirical_spread_point"]

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))

    branch_styles = {
        "clustered": ("crimson", "clustered branch"),
        "coverage": ("seagreen", "coverage branch"),
        "fallback_scan": ("slategray", "fallback scan"),
    }
    plotted_any = False
    for branch_name, samples in branch_evaluations.items():
        if not samples:
            continue
        values = np.array([float(item["lambda_I"]) for item in samples], dtype=np.float64)
        rates = np.array([float(item["information_spreading_rate"]) for item in samples], dtype=np.float64)
        p_enc = np.array([float(item["p_enc"]) for item in samples], dtype=np.float64)
        color, label = branch_styles.get(branch_name, ("royalblue", branch_name))
        axes[0].plot(values, rates, marker="o", lw=1.8, color=color, label=label)
        axes[1].plot(values, p_enc, marker="o", lw=1.8, color=color, label=label)
        plotted_any = True
    if not plotted_any:
        values = np.array([float(item["lambda_I"]) for item in evaluations], dtype=np.float64)
        rates = np.array([float(item["information_spreading_rate"]) for item in evaluations], dtype=np.float64)
        p_enc = np.array([float(item["p_enc"]) for item in evaluations], dtype=np.float64)
        axes[0].plot(values, rates, marker="o", lw=1.8, color="crimson", label="evaluations")
        axes[1].plot(values, p_enc, marker="o", lw=1.8, color="darkorange", label="evaluations")
    axes[0].axhline(target_rate, color="black", ls=":", label="target dn/dt")
    axes[0].scatter([selected_lambda], [selected_rate], s=70, color="gold", edgecolor="black", zorder=5)
    axes[0].set_xlabel("lambda_I")
    axes[0].set_ylabel("predicted dn/dt")
    axes[0].set_title("Inverse information-rate solve")
    axes[0].legend(fontsize=8)

    axes[1].axhline(float(inverse["target_p_enc"]), color="black", ls=":", label="target p_enc")
    axes[1].scatter([selected_lambda], [float(inverse["predicted_p_enc"])], s=70, color="gold", edgecolor="black", zorder=5)
    axes[1].set_xlabel("lambda_I")
    axes[1].set_ylabel("sum_k π_k^2")
    axes[1].set_title("Encounter probability induced by π̄")
    axes[1].legend(fontsize=8)

    axes[2].plot(series["t_axis"], series["informed_fraction"], lw=2.0, color="navy")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].set_xlabel("simulation step")
    axes[2].set_ylabel("informed fraction")
    axes[2].set_title(
        f"Empirical spread at lambda_I={selected_lambda:+.0f}\n"
        f"t90={empirical_point.time_to_90_percent}"
    )

    fig.tight_layout()
    fig.savefig(outdir / "maxcal_info_diffusion_validation_inverse_rate.png", dpi=140)
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


def save_raw_data(
    outdir: Path,
    sweep: Dict[str, object],
    spread: Dict[str, object],
    inverse: Dict[str, object],
) -> None:
    points: List[SignSweepPoint] = sweep["points"]
    spread_points: List[InformationSpreadPoint] = spread["points"]
    spread_series: Dict[str, Dict[str, np.ndarray]] = spread["series"]
    inverse_evaluations = inverse["evaluations"]
    inverse_series: Dict[str, np.ndarray] = inverse["empirical_series"]
    spread_arrays: Dict[str, np.ndarray] = {}
    for point in spread_points:
        key = f"{point.lambda_I_value:+.1f}".replace("+", "pos_").replace("-", "neg_").replace(".", "_")
        spread_arrays[f"spread_{key}_t_axis"] = spread_series[f"{point.lambda_I_value:+.1f}"]["t_axis"]
        spread_arrays[f"spread_{key}_informed_fraction"] = spread_series[f"{point.lambda_I_value:+.1f}"]["informed_fraction"]
        spread_arrays[f"spread_{key}_mean_coverage_age"] = spread_series[f"{point.lambda_I_value:+.1f}"]["mean_coverage_age"]
        spread_arrays[f"spread_{key}_mean_aoi"] = spread_series[f"{point.lambda_I_value:+.1f}"]["mean_aoi"]
        spread_arrays[f"spread_{key}_mean_local_coverage_age"] = spread_series[f"{point.lambda_I_value:+.1f}"]["mean_local_coverage_age"]
        spread_arrays[f"spread_{key}_mean_map_record_age"] = spread_series[f"{point.lambda_I_value:+.1f}"]["mean_map_record_age"]
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
        theoretical_p_enc=np.array([point.theoretical_p_enc for point in points], dtype=np.float64),
        final_coverage_l1=np.array([point.final_coverage_l1 for point in points], dtype=np.float64),
        total_markov_arrivals=np.array([point.total_markov_arrivals for point in points], dtype=np.int64),
        corner_visit_probability=np.array([point.corner_visit_probability for point in points], dtype=np.float64),
        edge_visit_probability=np.array([point.edge_visit_probability for point in points], dtype=np.float64),
        interior_visit_probability=np.array([point.interior_visit_probability for point in points], dtype=np.float64),
        spread_lambda_I_values=np.array([point.lambda_I_value for point in spread_points], dtype=np.float64),
        spread_theoretical_p_enc=np.array([point.theoretical_p_enc for point in spread_points], dtype=np.float64),
        spread_beta_transmission=np.array([point.beta_transmission for point in spread_points], dtype=np.float64),
        spread_predicted_initial_dn_dt=np.array(
            [point.predicted_initial_dn_dt for point in spread_points],
            dtype=np.float64,
        ),
        spread_initial_source_cell=np.array([point.initial_source_cell for point in spread_points], dtype=np.int64),
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
        spread_total_successful_communications=np.array(
            [point.total_successful_communications for point in spread_points],
            dtype=np.int64,
        ),
        spread_total_map_records_received=np.array(
            [point.total_map_records_received for point in spread_points],
            dtype=np.int64,
        ),
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
        spread_final_mean_coverage_age=np.array(
            [point.final_mean_coverage_age for point in spread_points],
            dtype=np.float64,
        ),
        spread_final_mean_aoi=np.array(
            [point.final_mean_aoi for point in spread_points],
            dtype=np.float64,
        ),
        spread_final_mean_local_coverage_age=np.array(
            [point.final_mean_local_coverage_age for point in spread_points],
            dtype=np.float64,
        ),
        spread_final_mean_map_record_age=np.array(
            [point.final_mean_map_record_age for point in spread_points],
            dtype=np.float64,
        ),
        inverse_scan_lambda_I=np.array(
            [float(item["lambda_I"]) for item in inverse_evaluations],
            dtype=np.float64,
        ),
        inverse_scan_p_enc=np.array(
            [float(item["p_enc"]) for item in inverse_evaluations],
            dtype=np.float64,
        ),
        inverse_scan_information_spreading_rate=np.array(
            [float(item["information_spreading_rate"]) for item in inverse_evaluations],
            dtype=np.float64,
        ),
        inverse_selected_lambda_I=np.array([float(inverse["lambda_I"])], dtype=np.float64),
        inverse_target_information_spreading_rate=np.array(
            [float(inverse["target_information_spreading_rate"])],
            dtype=np.float64,
        ),
        inverse_predicted_information_spreading_rate=np.array(
            [float(inverse["predicted_information_spreading_rate"])],
            dtype=np.float64,
        ),
        inverse_empirical_t_axis=inverse_series["t_axis"],
        inverse_empirical_informed_fraction=inverse_series["informed_fraction"],
        inverse_empirical_cumulative_meetings=inverse_series["cumulative_meetings"],
        inverse_empirical_cumulative_transmissions=inverse_series["cumulative_transmissions"],
        **spread_arrays,
    )


def save_summary(
    outdir: Path,
    args: argparse.Namespace,
    kernel: Dict[str, float | bool],
    field: Dict[str, float | bool],
    local_maps: Dict[str, float | bool],
    exchange: Dict[str, int | bool],
    sweep: Dict[str, object],
    preview: Dict[str, object],
    spread: Dict[str, object],
    inverse: Dict[str, object],
) -> None:
    points: List[SignSweepPoint] = sweep["points"]
    zero_point: SignSweepPoint = sweep["zero_baseline"]
    spread_points: List[InformationSpreadPoint] = spread["points"]
    inverse_empirical_point: InformationSpreadPoint = inverse["empirical_spread_point"]
    inverse_summary = {
        key: value
        for key, value in inverse.items()
        if key not in {"empirical_series", "empirical_spread_point", "selected_pi_bar"}
    }
    inverse_summary["empirical_spread_point"] = asdict(inverse_empirical_point)

    summary = {
        "config": {
            "Nx": mid.NX,
            "Ny": mid.NY,
            "cell_size_m": mid.CELL_SIZE,
            "robots": mid.N_ROBOTS,
            "lambda_I_sign_convention": "lambda_I < 0 -> cluster; lambda_I >= 0 -> coverage-prioritizing",
            "lambda_I_sign_sweep": SIGN_SWEEP_VALUES,
            "information_observable": "pi_bar from fixed point pi_bar = stationary(P(pi_bar))",
            "theoretical_encounter_probability": "p_enc = sum_k pi_bar_k^2",
            "beta_transmission": mid.BETA_TRANSMISSION,
            "p_encounter_given_colocation": mid.P_ENCOUNTER_GIVEN_COLOCATION,
            "default_information_field_mode": mid.INFORMATION_FIELD_MODE,
            "age_observables": (
                "Paper coverage age is the mean over cells of time since last physical visit. "
                "Paper AoI tau is the time since a robot last exchanged information. "
                "Robot maps additionally store last_visit_time[k] and last_map_record_time[k] "
                "for local map freshness, but those cell-wise records are not tau."
            ),
            "sign_sweep_T": args.sign_sweep_T,
            "stage2_preview_T": args.stage2_preview_T,
            "spread_T": args.spread_T,
            "inverse_target_rate": args.inverse_target_rate,
            "inverse_target_p_enc": args.inverse_target_p_enc,
            "inverse_branch": args.inverse_branch,
            "inverse_continuation_step": args.inverse_continuation_step,
            "seed": args.seed,
            "pure_stale_age": PURE_STALE_AGE,
        },
        "kernel_sign_checks": kernel,
        "information_field_checks": field,
        "local_map_age_checks": local_maps,
        "information_exchange_checks": exchange,
        "sign_sweep": {
            "points": [asdict(point) for point in points],
            "zero_baseline": asdict(zero_point),
            "all_negative_reduce_dispersion_vs_zero": sweep["all_negative_reduce_dispersion_vs_zero"],
            "all_negative_increase_encounter_vs_zero": sweep["all_negative_increase_encounter_vs_zero"],
            "all_negative_increase_theoretical_p_enc_vs_zero": sweep[
                "all_negative_increase_theoretical_p_enc_vs_zero"
            ],
            "all_positive_reduce_theoretical_p_enc_vs_zero": sweep[
                "all_positive_reduce_theoretical_p_enc_vs_zero"
            ],
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
                "encounters occurred. Paper AoI is tracked as time since exchange and is updated only "
                "after successful communication; cell-wise map-record age is retained separately for "
                "local map freshness."
            ),
        },
        "inverse_information_rate": inverse_summary,
    }

    summary["layer1_i_ready_for_stage2"] = bool(
        kernel["zero_lambda_recovers_uniform_kernel"]
        and kernel["negative_lambda_prefers_high_information_neighbor"]
        and kernel["positive_lambda_penalizes_high_information_neighbor"]
        and kernel["negative_vs_positive_ordering_on_dense_neighbor"]
        and field["cluster_cell_attains_local_peak"]
        and field["global_peak_stays_near_robot_cluster"]
        and field["field_is_nonnegative"]
        and local_maps["coverage_age_preserves_physical_visit_time"]
        and local_maps["map_record_age_resets_on_received_map_record"]
        and exchange["same_region_encounters"] >= 1
        and exchange["successful_communications"] >= 1
        and exchange["new_transmissions"] >= 1
        and exchange["map_records_received"] >= 1
        and exchange["first_robot_reset"]
        and exchange["second_robot_reset"]
        and exchange["third_robot_unchanged"]
        and exchange["information_shared"]
        and exchange["first_received_second_map_record"]
        and exchange["second_received_first_map_record"]
        and exchange["first_map_record_age_reset_for_received_record"]
        and exchange["second_map_record_age_reset_for_received_record"]
        and exchange["first_paper_aoi_reset_after_exchange"]
        and exchange["second_paper_aoi_reset_after_exchange"]
        and sweep["all_negative_increase_theoretical_p_enc_vs_zero"]
        and sweep["all_positive_reduce_theoretical_p_enc_vs_zero"]
        and spread["negative_generates_transmissions"]
        and spread["transmission_locations_recorded"]
        and spread["negative_generates_more_meetings_than_zero_and_positive"]
        and inverse["target_within_scanned_range"]
        and inverse["converged"]
    )

    with open(outdir / "maxcal_info_diffusion_validation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    kernel = kernel_sign_checks()
    field = information_field_checks()
    local_maps = local_map_age_checks()
    exchange = information_exchange_checks()
    sweep = sign_sweep(T=args.sign_sweep_T, seed=args.seed)
    if args.stage2_preview_T > 0:
        preview = stage2_preview(T=args.stage2_preview_T, seed=args.seed)
    else:
        preview = {
            "skipped": True,
            "reason": (
                "The isolated Layer 1-I validation now uses the sign sweep, "
                "inverse information-rate solve, and stochastic sharing test. "
                "Set --stage2-preview-T to run this legacy diagnostic."
            ),
        }
    spread = information_spread_validation(T=args.spread_T, seed=args.seed)
    inverse = inverse_information_rate_validation(
        T=args.spread_T,
        seed=args.seed,
        target_rate=args.inverse_target_rate,
        target_p_enc=args.inverse_target_p_enc,
        branch=args.inverse_branch,
        continuation_step=args.inverse_continuation_step,
    )

    make_sign_sweep_figure(outdir, sweep)
    make_information_spread_figure(outdir, spread)
    make_inverse_rate_figure(outdir, inverse)
    make_information_meeting_maps_figure(outdir, spread)
    save_raw_data(outdir, sweep, spread, inverse)
    save_summary(outdir, args, kernel, field, local_maps, exchange, sweep, preview, spread, inverse)

    ready = (
        kernel["zero_lambda_recovers_uniform_kernel"]
        and kernel["negative_lambda_prefers_high_information_neighbor"]
        and kernel["positive_lambda_penalizes_high_information_neighbor"]
        and field["cluster_cell_attains_local_peak"]
        and local_maps["coverage_age_preserves_physical_visit_time"]
        and local_maps["map_record_age_resets_on_received_map_record"]
        and sweep["all_negative_increase_theoretical_p_enc_vs_zero"]
        and sweep["all_positive_reduce_theoretical_p_enc_vs_zero"]
        and spread["negative_generates_transmissions"]
        and spread["transmission_locations_recorded"]
        and spread["negative_generates_more_meetings_than_zero_and_positive"]
        and inverse["target_within_scanned_range"]
        and inverse["converged"]
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
        "  AoI preview           : "
        + (
            "skipped"
            if preview.get("skipped")
            else (
                f"negative->cluster={preview['negative_lambda_clusters_vs_zero']}, "
                f"positive->coverage={preview['positive_lambda_preserves_coverage_vs_zero']}, "
                f"negative lowers AoI={preview['negative_lambda_reduces_paper_aoi_vs_positive']}"
            )
        )
    )
    print(
        "  Informed spread       : "
        f"lambda_I={negative_spread.lambda_I_value:+.0f} reaches 90% at t={negative_spread.time_to_90_percent}, "
        f"meetings={negative_spread.total_meetings}, "
        f"transmission cells={negative_spread.transmission_cells_visited}, "
        f"fastest lambda_I={spread['fastest_fraction_informed_lambda_I']:+.0f}"
    )
    print(
        "  Inverse rate solve    : "
        f"target dn/dt={inverse['target_information_spreading_rate']:.5f}, "
        f"lambda_I={inverse['lambda_I']:+.0f}, "
        f"predicted dn/dt={inverse['predicted_information_spreading_rate']:.5f}, "
        f"converged={inverse['converged']}"
    )
    print("  Saved summary         : maxcal_info_diffusion_validation_summary.json")


if __name__ == "__main__":
    main()
