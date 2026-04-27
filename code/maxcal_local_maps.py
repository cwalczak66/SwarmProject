"""RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Per-robot world maps for local coverage-age observables and map freshness.

This module stores per-robot local map records used to measure local coverage
freshness.  That is deliberately separate from the paper-level AoI ``tau_i``,
which is updated by robot-robot information exchange events.

For robot ``i`` and cell ``k`` the local quantities are

    kappa_i^k(t) = t - T_i^k,  coverage age from the known visit time;
    eta_i^k(t)   = t - R_i^k,  age of the stored map record itself.

Pinciroli's AoI is instead ``tau_i(t)=t-last_meet_i`` and lives on the robot,
not on individual cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


UNKNOWN_TIME = -1.0


@dataclass
class RobotWorldMap:
    """
    Local cell-wise memory carried by one robot.

    last_visit_time[k]
        Timestamp of the most recent physical visit to cell k known by this
        robot.  This is the local estimate used to compute coverage age.

    last_map_record_time[k]
        Timestamp at which this robot most recently received or generated a map
        record about cell k.  This is a local map-freshness quantity, not the
        paper-level AoI tau_t, which is the time since robot-robot exchange.
    """

    last_visit_time: np.ndarray
    last_map_record_time: np.ndarray

    @classmethod
    def empty(cls, n_cells: int) -> "RobotWorldMap":
        """Create a completely unknown local map for a robot."""
        return cls(
            last_visit_time=np.full(n_cells, UNKNOWN_TIME, dtype=np.float64),
            last_map_record_time=np.full(n_cells, UNKNOWN_TIME, dtype=np.float64),
        )

    def observe_cell(self, cell: int, t: float) -> None:
        """Record a direct physical visit: ``T_i^k=R_i^k=t``."""
        self.last_visit_time[cell] = max(float(t), float(self.last_visit_time[cell]))
        self.last_map_record_time[cell] = float(t)

    def merge_from(self, other: "RobotWorldMap", t: float) -> int:
        """
        Receive the other robot's map at time t.

        The physical visit timestamp is copied only when the other robot knows a
        newer visit event.  The information timestamp is refreshed for every
        cell whose record is transmitted, including cells where both robots know
        the same latest visit event.  This separates coverage age
        ``t - last_visit_time`` from cell-wise map-record age
        ``t - last_map_record_time``.
        """
        other_known = other.last_visit_time >= 0.0
        self_known = self.last_visit_time >= 0.0

        newer_visit = other_known & (other.last_visit_time > self.last_visit_time)
        same_visit_known = other_known & self_known & np.isclose(
            other.last_visit_time,
            self.last_visit_time,
        )
        received_record = newer_visit | same_visit_known

        self.last_visit_time[newer_visit] = other.last_visit_time[newer_visit]
        self.last_map_record_time[received_record] = float(t)
        return int(np.count_nonzero(received_record))

    def coverage_age(self, t: float) -> np.ndarray:
        """Return ``kappa_i^k(t)=t-T_i^k``, with unknown cells maximally stale."""
        t_float = float(t)
        return np.where(
            self.last_visit_time >= 0.0,
            t_float - self.last_visit_time,
            t_float + 1.0,
        )

    def map_record_age(self, t: float) -> np.ndarray:
        """Return ``eta_i^k(t)=t-R_i^k`` for cell-wise map-record freshness."""
        t_float = float(t)
        return np.where(
            self.last_map_record_time >= 0.0,
            t_float - self.last_map_record_time,
            t_float + 1.0,
        )

    def mean_coverage_age(self, t: float) -> float:
        """Mean local coverage age over all cells in this robot's map."""
        return float(np.mean(self.coverage_age(t)))

    def mean_map_record_age(self, t: float) -> float:
        """Mean age of locally stored map records over all cells."""
        return float(np.mean(self.map_record_age(t)))


def exchange_maps(
    first: RobotWorldMap,
    second: RobotWorldMap,
    t: float,
) -> Tuple[int, int]:
    """
    Bidirectionally exchange local maps without order bias.

    Copies are used so both robots receive the pre-exchange state of the other
    robot, matching a simultaneous communication event.
    """
    first_visit_before = first.last_visit_time.copy()
    second_visit_before = second.last_visit_time.copy()

    first_known = first_visit_before >= 0.0
    second_known = second_visit_before >= 0.0

    first_receives_newer = second_known & (second_visit_before > first_visit_before)
    first_receives_same = second_known & first_known & np.isclose(second_visit_before, first_visit_before)
    first_receives_record = first_receives_newer | first_receives_same

    second_receives_newer = first_known & (first_visit_before > second_visit_before)
    second_receives_same = first_known & second_known & np.isclose(first_visit_before, second_visit_before)
    second_receives_record = second_receives_newer | second_receives_same

    first.last_visit_time[first_receives_newer] = second_visit_before[first_receives_newer]
    second.last_visit_time[second_receives_newer] = first_visit_before[second_receives_newer]
    first.last_map_record_time[first_receives_record] = float(t)
    second.last_map_record_time[second_receives_record] = float(t)

    first_updates = int(np.count_nonzero(first_receives_record))
    second_updates = int(np.count_nonzero(second_receives_record))
    return first_updates, second_updates


def mean_robot_coverage_age(maps: list[RobotWorldMap], t: float) -> float:
    """Swarm average of the robots' local coverage-age observable."""
    if not maps:
        return 0.0
    return float(np.mean([robot_map.mean_coverage_age(t) for robot_map in maps]))


def mean_robot_map_record_age(maps: list[RobotWorldMap], t: float) -> float:
    """Swarm average of the robots' local map-record freshness observable."""
    if not maps:
        return 0.0
    return float(np.mean([robot_map.mean_map_record_age(t) for robot_map in maps]))
