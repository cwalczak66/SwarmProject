"""
RBE 511 Swarm Intelligence Final Project - MaxCal-Derived Swarm Control: Emergent Oscillation Between Coverage and Information Diffusion

Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Pau Alcolea (palcolea@wpi.edu), Chris Walczak (cwalczak2@wpi.edu)
Course: RBE 511 - Swarm Intelligence (Prof. Carlo Pinciroli)  
Institution: Worcester Polytechnic Institute  

-------------------------------------------------------------------------------------------------------------------------------------------

Shared MaxCal utilities for all project scripts.

The MaxCal construction produces a log-linear transition kernel on the
region graph.  In this project the graph is always the 20 x 20, 8-connected grid. 
Both Layer 1 controllers reduce locally to

    P_ij = A_ij exp[-g_j] / sum_l A_il exp[-g_l],

where ``A`` is the adjacency matrix and ``g_j`` is the destination cost built
from the active Lagrange multipliers.  Because ``A`` is symmetric, this
destination-weighted kernel is reversible:

    pi_j = b_j (A b)_j / sum_m b_m (A b)_m,  b_j = exp[-g_j].

The coverage inverse solve, the information fixed point, and the hierarchical
controller all call these same helpers so the paper formulas are implemented in
one place instead of being rederived in each script.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class World:
    Nx: int
    Ny: int
    K: int
    cell_size: float
    adjacency: list[list[int]]
    centers: np.ndarray


def build_grid_world(Nx: int, Ny: int, cell_size: float) -> World:
    """Build an undirected 8-connected rectangular region graph."""
    K = Nx * Ny
    adjacency: list[list[int]] = [[] for _ in range(K)]
    centers = np.zeros((K, 2), dtype=np.float64)

    for k in range(K):
        row, col = divmod(k, Nx)
        centers[k] = ((col + 0.5) * cell_size, (row + 0.5) * cell_size)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < Ny and 0 <= nc < Nx:
                    adjacency[k].append(nr * Nx + nc)

    return World(Nx=Nx, Ny=Ny, K=K, cell_size=cell_size, adjacency=adjacency, centers=centers)


def region_center(world: World, cell: int) -> tuple[float, float]:
    """Return the continuous-space center of a cell."""
    return float(world.centers[cell, 0]), float(world.centers[cell, 1])


def position_to_cell(world: World, x: float, y: float) -> int:
    """Map a continuous position back to a grid cell index."""
    col = min(max(int(x / world.cell_size), 0), world.Nx - 1)
    row = min(max(int(y / world.cell_size), 0), world.Ny - 1)
    return row * world.Nx + col


def adjacency_matrix(world: World) -> np.ndarray:
    """Return the binary adjacency matrix of the region graph."""
    matrix = np.zeros((world.K, world.K), dtype=np.float64)
    for cell, neighbors in enumerate(world.adjacency):
        matrix[cell, neighbors] = 1.0
    return matrix


def normalize_probability(values: Sequence[float] | np.ndarray, floor: float = 0.0) -> np.ndarray:
    """Project a non-negative vector onto the probability simplex."""
    arr = np.asarray(values, dtype=np.float64)
    if floor > 0.0:
        arr = np.maximum(arr, floor)
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Probability vector must have a finite positive sum.")
    return arr / total


def destination_weight_transition(world: World, destination_cost: np.ndarray) -> np.ndarray:
    """Build ``P_ij`` from destination costs ``g_j``.

    Only neighbors of ``i`` receive probability mass.  The subtraction of the
    local minimum is a numerical stabilization and does not change the
    normalized probabilities.
    """
    cost = np.asarray(destination_cost, dtype=np.float64)
    if cost.shape != (world.K,):
        raise ValueError(f"destination_cost must have shape ({world.K},), got {cost.shape}.")

    transition = np.zeros((world.K, world.K), dtype=np.float64)
    for source, neighbors in enumerate(world.adjacency):
        local_cost = cost[neighbors]
        weights = np.exp(-(local_cost - float(local_cost.min())))
        transition[source, neighbors] = weights / float(weights.sum())
    return transition


def reversible_stationary_from_cost(world: World, destination_cost: np.ndarray) -> np.ndarray:
    """Return the reversible stationary law ``pi_j proportional to b_j(A b)_j``."""
    cost = np.asarray(destination_cost, dtype=np.float64)
    if cost.shape != (world.K,):
        raise ValueError(f"destination_cost must have shape ({world.K},), got {cost.shape}.")
    shifted = cost - float(cost.min())
    weights = np.exp(-shifted)
    neighbor_mass = np.array([weights[neighbors].sum() for neighbors in world.adjacency], dtype=np.float64)
    return normalize_probability(weights * neighbor_mass)


def sample_from_transition(
    world: World,
    transition: np.ndarray,
    source: int,
    rng: np.random.Generator,
) -> int:
    """Sample one next cell from a transition matrix row."""
    neighbors = world.adjacency[source]
    probs = np.asarray(transition[source, neighbors], dtype=np.float64)
    probs = probs / float(probs.sum())
    return int(rng.choice(neighbors, p=probs))


def power_stationary_distribution(
    transition: np.ndarray,
    tol: float = 1.0e-14,
    max_iters: int = 200_000,
) -> np.ndarray:
    """Generic stationary distribution solver for dense stochastic matrices."""
    pi = np.full(transition.shape[0], 1.0 / transition.shape[0], dtype=np.float64)
    for _ in range(max_iters):
        nxt = pi @ transition
        if float(np.max(np.abs(nxt - pi))) < tol:
            return normalize_probability(nxt)
        pi = nxt
    return normalize_probability(pi)


def coverage_age(last_visit_time: np.ndarray, t: float) -> np.ndarray:
    """Cell-wise coverage age ``kappa_k(t)=t-T_k``.

    ``T_k`` is the last physical visit time of cell ``k``.  Unknown cells use
    ``t + 1`` so they are treated as slightly staler than any known cell.
    """
    return np.where(last_visit_time >= 0.0, float(t) - last_visit_time, float(t) + 1.0)


def occupancy_distribution(world: World, positions: Sequence[tuple[float, float]]) -> np.ndarray:
    """Empirical cell occupancy distribution from continuous robot positions."""
    occ = np.zeros(world.K, dtype=np.float64)
    if not positions:
        return occ
    for x, y in positions:
        occ[position_to_cell(world, x, y)] += 1.0
    return occ / float(len(positions))


def move_toward(x: float, y: float, tx: float, ty: float, speed: float) -> tuple[float, float]:
    """Move toward a target by at most ``speed``, never overshooting."""
    dx = tx - x
    dy = ty - y
    dist = math.hypot(dx, dy)
    if dist <= 1.0e-10:
        return tx, ty
    step = min(float(speed), dist)
    return x + step * dx / dist, y + step * dy / dist


def l1_error(a: np.ndarray, b: np.ndarray) -> float:
    """Convenience wrapper for distribution L1 distance."""
    return float(np.sum(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
