"""
maxcal_info_diffusion.py

MaxCal Coverage + Information Diffusion — sign-consistent Stage II model.

This version aligns the information term with the sign convention the user
quoted from Pinciroli's draft:

    lambda_I < 0   -> attraction to high-information / high-density regions
                      -> clustering
    lambda_I = 0   -> pure coverage baseline
    lambda_I > 0   -> repulsion from high-density regions
                      -> coverage-prioritizing / anti-clustering

The transition kernel implemented here is

    p*(k2 | k1; i, t) ∝ w_{k2,k1}
                        exp( -lambda_C^{k2}
                             - g(A_I^i(t)) * lambda_I * I_t(k2) )

where
    A_I^i(t)  : information age of robot i
    g(.)      : monotone gate in [0, 1)
    I_t(k)    : dynamic local robot-density field (information field)

Because I_t(k) is positive, the sign of lambda_I directly controls the regime:
negative values reward high-density cells, while zero or positive values leave
coverage dominant or actively penalize clustering.

SETUP:
    pip install numpy matplotlib pillow

RUN:
    python maxcal_info_diffusion.py
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", tempfile.gettempdir())

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation


# ============================================================
# PARAMETERS
# ============================================================

NX = 20
NY = 20
CELL_SIZE = 1.0
N_ROBOTS = 50
ROBOT_SPEED = 0.15
T_SIM = 18_000
RECORD_EVERY = 30
SNAP_EVERY = 300
SEED = 42

# Coverage multiplier.
LAMBDA_C_VAL = 0.0

# Signed information multiplier.
# Pinciroli sign convention:
#   lambda_I < 0  -> cluster
#   lambda_I >= 0 -> coverage / anti-cluster
LAMBDA_I_VAL = 10.0        # Testing with negative and positive 10 give the opposite behaviors.

# Gate parameters.
A_HALF = 1.0  # Half-age for the information gate g(A_I) = A_I / (A_I + A_half). This controls how quickly
R_MEET = 1.5  # Meeting radius for information refreshment and encounter counting.

# Sweep used by the standalone script.
LAMBDA_I_SWEEP = (-10.0, -8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0, 10.0)
SWEEP_STALE_AGE = 1.0e6


# ============================================================
# 1. WORLD MODEL
# ============================================================

@dataclass
class World:
    Nx: int
    Ny: int
    K: int
    cell_size: float
    adjacency: List[List[int]]
    centers: np.ndarray


def build_world(Nx: int, Ny: int, cell_size: float) -> World:
    K = Nx * Ny
    adjacency: List[List[int]] = [[] for _ in range(K)]
    for k in range(K):
        row, col = divmod(k, Nx)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < Ny and 0 <= nc < Nx):
                    continue
                adjacency[k].append(nr * Nx + nc)

    centers = np.zeros((K, 2), dtype=np.float64)
    for k in range(K):
        row, col = divmod(k, Nx)
        centers[k, 0] = (col + 0.5) * cell_size
        centers[k, 1] = (row + 0.5) * cell_size

    return World(Nx, Ny, K, cell_size, adjacency, centers)


def region_center(w: World, k: int) -> Tuple[float, float]:
    return float(w.centers[k, 0]), float(w.centers[k, 1])


def position_to_cell(w: World, x: float, y: float) -> int:
    col = min(max(int(x / w.cell_size), 0), w.Nx - 1)
    row = min(max(int(y / w.cell_size), 0), w.Ny - 1)
    return row * w.Nx + col


def theoretical_stationary(w: World) -> np.ndarray:
    degrees = np.array([len(w.adjacency[k]) for k in range(w.K)], dtype=np.float64)
    return degrees / degrees.sum()


# ============================================================
# 2. STATIC COVERAGE TERM AND DYNAMIC INFORMATION FIELD
# ============================================================

def build_lambda_C(w: World, lambda_C_val: float) -> np.ndarray:
    return np.full(w.K, lambda_C_val, dtype=np.float64)


def occupancy_distribution(w: World, robots: Sequence["Robot"]) -> np.ndarray:
    occ = np.zeros(w.K, dtype=np.float64)
    if not robots:
        return occ
    for r in robots:
        occ[position_to_cell(w, r.x, r.y)] += 1.0
    return occ / float(len(robots))


def compute_information_field(w: World, robots: Sequence["Robot"]) -> np.ndarray:
    """
    Local information field I_t(k).

    We use the current robot occupancy, smoothed over each cell and its
    8-connected neighborhood. This yields a positive, local density-like field.
    With the kernel exp(- ... - lambda_I * I_t(k)), the sign of lambda_I
    directly determines attraction versus repulsion to crowded regions.
    """
    occ = occupancy_distribution(w, robots)
    field = np.zeros(w.K, dtype=np.float64)
    for k in range(w.K):
        local = [k] + w.adjacency[k]
        field[k] = float(occ[local].sum())
    max_val = float(field.max())
    if max_val > 0.0:
        field /= max_val
    return field


def encounter_proxy_from_occupancy(occ: np.ndarray) -> float:
    return float(np.sum(occ ** 2))


# ============================================================
# 3. MAXCAL TRANSITION KERNEL
# ============================================================

def info_gate(age: float, a_half: float = A_HALF) -> float:
    return age / (age + a_half)


def local_transition_probabilities(
    w: World,
    k1: int,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    info_age: float,
) -> Tuple[np.ndarray, np.ndarray]:
    neighbors = np.array(w.adjacency[k1], dtype=np.int64)
    gate = info_gate(info_age)
    effective_cost = lambda_C[neighbors] + gate * lambda_I_value * info_field[neighbors]
    shifted = effective_cost - effective_cost.min()
    scores = np.exp(-shifted)
    probs = scores / scores.sum()
    return neighbors, probs


def sample_next_region(
    w: World,
    k1: int,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    info_age: float,
    rng: np.random.Generator,
) -> int:
    neighbors, probs = local_transition_probabilities(
        w=w,
        k1=k1,
        lambda_C=lambda_C,
        lambda_I_value=lambda_I_value,
        info_field=info_field,
        info_age=info_age,
    )
    r = rng.random()
    s = 0.0
    for i, p in enumerate(probs):
        s += float(p)
        if s >= r:
            return int(neighbors[i])
    return int(neighbors[-1])


# ============================================================
# 4. ROBOT
# ============================================================

@dataclass
class Robot:
    id: int
    x: float
    y: float
    from_k: int
    to_k: int
    tx: float
    ty: float
    info_age: float = 0.0
    last_meet: int = 0


def make_robot(
    idx: int,
    w: World,
    rng: np.random.Generator,
    fixed_initial_info_age: float | None = None,
) -> Robot:
    k0 = int(rng.integers(0, w.K))
    cx, cy = region_center(w, k0)
    if fixed_initial_info_age is None:
        age0 = float(rng.uniform(0.0, A_HALF))
    else:
        age0 = float(fixed_initial_info_age)
    return Robot(
        id=idx,
        x=cx,
        y=cy,
        from_k=k0,
        to_k=k0,
        tx=cx,
        ty=cy,
        info_age=age0,
        last_meet=0,
    )


def initialise_targets(
    robots: Sequence[Robot],
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    rng: np.random.Generator,
) -> None:
    info_field = compute_information_field(w, robots)
    for r in robots:
        next_k = sample_next_region(
            w=w,
            k1=r.from_k,
            lambda_C=lambda_C,
            lambda_I_value=lambda_I_value,
            info_field=info_field,
            info_age=r.info_age,
            rng=rng,
        )
        r.to_k = next_k
        r.tx, r.ty = region_center(w, next_k)


def step_robot(
    r: Robot,
    w: World,
    lambda_C: np.ndarray,
    lambda_I_value: float,
    info_field: np.ndarray,
    speed: float,
    markov_visits: np.ndarray,
    coverage_age_field: np.ndarray,
    t: int,
    rng: np.random.Generator,
    freeze_info_age: bool = False,
) -> None:
    if not freeze_info_age:
        r.info_age += 1.0

    dx = r.tx - r.x
    dy = r.ty - r.y
    dist = math.sqrt(dx * dx + dy * dy)

    move = min(speed, dist)
    if dist > 1e-10:
        r.x += move * dx / dist
        r.y += move * dy / dist

    if abs(r.x - r.tx) < 1e-9 and abs(r.y - r.ty) < 1e-9:
        r.from_k = r.to_k
        markov_visits[r.from_k] += 1
        coverage_age_field[r.from_k] = t

        next_k = sample_next_region(
            w=w,
            k1=r.from_k,
            lambda_C=lambda_C,
            lambda_I_value=lambda_I_value,
            info_field=info_field,
            info_age=r.info_age,
            rng=rng,
        )
        r.to_k = next_k
        r.tx, r.ty = region_center(w, next_k)


def detect_meetings(robots: List[Robot], r_meet: float, t: int) -> int:
    n = len(robots)
    n_pairs = 0
    xs = np.array([r.x for r in robots])
    ys = np.array([r.y for r in robots])
    for i in range(n):
        dx = xs[i + 1:] - xs[i]
        dy = ys[i + 1:] - ys[i]
        hits = np.where(dx * dx + dy * dy <= r_meet * r_meet)[0]
        for j_off in hits:
            j = i + 1 + int(j_off)
            robots[i].info_age = 0.0
            robots[j].info_age = 0.0
            robots[i].last_meet = t
            robots[j].last_meet = t
            n_pairs += 1
    return n_pairs


# ============================================================
# 5. SIMULATION
# ============================================================

@dataclass
class SimResult:
    w: World
    lambda_I_value: float
    pi_empirical: np.ndarray
    mean_info_field: np.ndarray
    final_info_field: np.ndarray
    ck_history: List[np.ndarray]
    markov_step_history: List[int]
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]]
    t_axis: np.ndarray
    dispersion: np.ndarray
    mean_cov_age: np.ndarray
    mean_info_age: np.ndarray
    meetings_per_snap: np.ndarray
    encounter_proxy: np.ndarray
    coverage_l1: np.ndarray


def run_simulation(
    speed: float = ROBOT_SPEED,
    T: int = T_SIM,
    lambda_C_val: float = LAMBDA_C_VAL,
    lambda_I_value: float = LAMBDA_I_VAL,
    seed: int = SEED,
    n_robots: int = N_ROBOTS,
    fixed_initial_info_age: float | None = None,
    freeze_info_age: bool = False,
    enable_meetings: bool = True,
) -> SimResult:
    rng = np.random.default_rng(seed)
    w = build_world(NX, NY, CELL_SIZE)
    lambda_C = build_lambda_C(w, lambda_C_val)
    pi_theory = theoretical_stationary(w)

    robots = [
        make_robot(
            idx=i,
            w=w,
            rng=rng,
            fixed_initial_info_age=fixed_initial_info_age,
        )
        for i in range(n_robots)
    ]
    initialise_targets(
        robots=robots,
        w=w,
        lambda_C=lambda_C,
        lambda_I_value=lambda_I_value,
        rng=rng,
    )

    markov_visits = np.zeros(w.K, dtype=np.int64)
    coverage_age_field = np.zeros(w.K, dtype=np.float64)
    info_field_accumulator = np.zeros(w.K, dtype=np.float64)

    ck_history: List[np.ndarray] = []
    ms_history: List[int] = []
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]] = []

    t_log: List[int] = []
    disp_log: List[float] = []
    mean_cov_log: List[float] = []
    mean_info_log: List[float] = []
    meet_log: List[int] = []
    encounter_log: List[float] = []
    coverage_l1_log: List[float] = []
    meet_accumulator = 0

    final_info_field = compute_information_field(w, robots)

    for t in range(1, T + 1):
        info_field = compute_information_field(w, robots)
        info_field_accumulator += info_field
        final_info_field = info_field

        for r in robots:
            step_robot(
                r=r,
                w=w,
                lambda_C=lambda_C,
                lambda_I_value=lambda_I_value,
                info_field=info_field,
                speed=speed,
                markov_visits=markov_visits,
                coverage_age_field=coverage_age_field,
                t=t,
                rng=rng,
                freeze_info_age=freeze_info_age,
            )

        if enable_meetings:
            meet_accumulator += detect_meetings(robots, R_MEET, t)

        if t % RECORD_EVERY == 0:
            xs = np.array([r.x for r in robots])
            ys = np.array([r.y for r in robots])
            occ = occupancy_distribution(w, robots)
            total_arrivals = int(markov_visits.sum())
            ck = markov_visits / max(total_arrivals, 1)

            t_log.append(t)
            disp_log.append(float(np.var(xs) + np.var(ys)))
            mean_cov_log.append(float((t - coverage_age_field).mean()))
            mean_info_log.append(float(np.mean([r.info_age for r in robots])))
            meet_log.append(meet_accumulator)
            encounter_log.append(encounter_proxy_from_occupancy(occ))
            coverage_l1_log.append(float(np.sum(np.abs(ck - pi_theory))))
            meet_accumulator = 0

            if total_arrivals > 0:
                ck_history.append(ck.copy())
                ms_history.append(total_arrivals)

        if t % SNAP_EVERY == 0:
            xs = np.array([r.x for r in robots])
            ys = np.array([r.y for r in robots])
            pos_snapshots.append((xs, ys, t))

    pi_emp = markov_visits / max(int(markov_visits.sum()), 1)
    mean_info_field = info_field_accumulator / float(max(T, 1))

    return SimResult(
        w=w,
        lambda_I_value=lambda_I_value,
        pi_empirical=pi_emp,
        mean_info_field=mean_info_field,
        final_info_field=final_info_field,
        ck_history=ck_history,
        markov_step_history=ms_history,
        pos_snapshots=pos_snapshots,
        t_axis=np.array(t_log, dtype=np.int64),
        dispersion=np.array(disp_log, dtype=np.float64),
        mean_cov_age=np.array(mean_cov_log, dtype=np.float64),
        mean_info_age=np.array(mean_info_log, dtype=np.float64),
        meetings_per_snap=np.array(meet_log, dtype=np.float64),
        encounter_proxy=np.array(encounter_log, dtype=np.float64),
        coverage_l1=np.array(coverage_l1_log, dtype=np.float64),
    )


# ============================================================
# 6. DIAGNOSTICS
# ============================================================

def dominant_frequency(
    t_axis: np.ndarray,
    signal: np.ndarray,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    if len(signal) < 8:
        return 0.0, 0.0, np.array([]), np.array([])
    dt = float(t_axis[1] - t_axis[0])
    centered = signal - signal.mean()
    window = np.hanning(len(centered))
    spectrum = np.fft.rfft(centered * window)
    power = np.abs(spectrum) ** 2
    freqs = np.fft.rfftfreq(len(centered), d=dt)
    if len(power) <= 2:
        return 0.0, 0.0, freqs, power
    idx = int(np.argmax(power[1:])) + 1
    prominence_floor = float(np.median(power[1:]))
    prominence = float(power[idx] / prominence_floor) if prominence_floor > 0 else float("inf")
    return float(freqs[idx]), prominence, freqs, power


# ============================================================
# 7. VISUALISATION
# ============================================================

def make_main_figure(res: SimResult):
    w = res.w

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(res.mean_info_field.reshape(w.Ny, w.Nx), origin="lower", cmap="magma")
    ax1.set_title("(a) Mean information field I_bar(k)")
    ax1.set_xlabel("col")
    ax1.set_ylabel("row")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(res.pi_empirical.reshape(w.Ny, w.Nx), origin="lower", cmap="viridis")
    ax2.set_title("(b) Empirical visit distribution pi_hat(k)")
    ax2.set_xlabel("col")
    ax2.set_ylabel("row")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    ax3 = fig.add_subplot(gs[0, 2])
    xs, ys, t_end = res.pos_snapshots[-1]
    ax3.scatter(xs, ys, s=14, alpha=0.8, color="dodgerblue")
    ax3.set_xlim(0, w.Nx * w.cell_size)
    ax3.set_ylim(0, w.Ny * w.cell_size)
    ax3.set_aspect("equal")
    ax3.set_title(f"(c) Robot positions at t={t_end}")
    ax3.set_xlabel("x (m)")
    ax3.set_ylabel("y (m)")

    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.plot(res.t_axis, res.dispersion, color="navy", lw=1.4, label="S(t) dispersion")
    ax4.set_xlabel("simulation step")
    ax4.set_ylabel("S(t)", color="navy")
    ax4.tick_params(axis="y", labelcolor="navy")
    ax4b = ax4.twinx()
    ax4b.plot(res.t_axis, res.mean_info_age, color="crimson", lw=1.2, alpha=0.85, label="mean info age")
    ax4b.plot(res.t_axis, res.mean_cov_age, color="seagreen", lw=1.2, alpha=0.85, label="mean coverage age")
    ax4b.set_ylabel("age (steps)")
    ax4.set_title("(d) Dispersion and ages")
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(res.t_axis, res.encounter_proxy, color="darkorange", lw=1.3, label="encounter proxy")
    ax5.set_xlabel("simulation step")
    ax5.set_ylabel("sum_k occ_k^2", color="darkorange")
    ax5.tick_params(axis="y", labelcolor="darkorange")
    ax5b = ax5.twinx()
    ax5b.plot(res.t_axis, res.coverage_l1, color="slateblue", lw=1.2, alpha=0.85, label="coverage L1")
    ax5b.set_ylabel("||c(.,t)-pi_bar(.)||_1")
    ax5.set_title("(e) Clustering vs. coverage error")
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.suptitle(
        "MaxCal Coverage + Information Diffusion "
        f"(lambda_I = {res.lambda_I_value:+.2f})"
    )
    fig.tight_layout()
    return fig


def make_age_plane_figure(res: SimResult):
    fig, ax = plt.subplots(figsize=(6, 6))
    cov = res.mean_cov_age
    info = res.mean_info_age
    n0 = max(1, len(cov) // 10)
    sc = ax.scatter(cov[n0:], info[n0:], c=res.t_axis[n0:], cmap="viridis", s=8)
    ax.plot(cov[n0:], info[n0:], color="grey", lw=0.4, alpha=0.6)
    ax.set_xlabel("mean coverage age")
    ax.set_ylabel("mean information age")
    ax.set_title("Age-plane trajectory")
    fig.colorbar(sc, ax=ax, label="simulation step")
    fig.tight_layout()
    return fig


def make_constraint_sweep_figure(
    values: Sequence[float] = LAMBDA_I_SWEEP,
    T: int = 9_000,
):
    results = [
        run_simulation(
            lambda_I_value=value,
            T=T,
            fixed_initial_info_age=SWEEP_STALE_AGE,
            freeze_info_age=True,
            enable_meetings=False,
        )
        for value in values
    ]
    means_disp = np.array([float(res.dispersion.mean()) for res in results], dtype=np.float64)
    means_enc = np.array([float(res.encounter_proxy.mean()) for res in results], dtype=np.float64)
    final_cov_l1 = np.array([float(res.coverage_l1[-1]) for res in results], dtype=np.float64)
    values_arr = np.array(values, dtype=np.float64)

    zero_idx = int(np.where(np.isclose(values_arr, 0.0))[0][0])
    baseline_disp = means_disp[zero_idx]
    baseline_enc = means_enc[zero_idx]

    summary = []
    for value, res, mean_disp, mean_enc, cov_l1 in zip(values_arr, results, means_disp, means_enc, final_cov_l1):
        if mean_disp < baseline_disp and mean_enc > baseline_enc:
            regime = "cluster"
        elif mean_disp >= baseline_disp and mean_enc <= baseline_enc:
            regime = "coverage"
        else:
            regime = "mixed"
        f_star, prominence, _, _ = dominant_frequency(res.t_axis, res.dispersion)
        summary.append(
            (
                float(value),
                float(mean_disp),
                float(mean_enc),
                float(cov_l1),
                float(f_star),
                float(prominence),
                regime,
            )
        )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))

    axes[0, 0].plot(values_arr, means_disp, marker="o", lw=2.0, color="navy")
    axes[0, 0].axvline(0.0, color="grey", ls=":")
    axes[0, 0].set_title("Mean dispersion vs lambda_I")
    axes[0, 0].set_xlabel("lambda_I")
    axes[0, 0].set_ylabel("mean S(t)")

    axes[0, 1].plot(values_arr, means_enc, marker="o", lw=2.0, color="darkorange")
    axes[0, 1].axvline(0.0, color="grey", ls=":")
    axes[0, 1].set_title("Encounter proxy vs lambda_I")
    axes[0, 1].set_xlabel("lambda_I")
    axes[0, 1].set_ylabel("mean sum_k occ_k^2")

    axes[1, 0].plot(values_arr, final_cov_l1, marker="o", lw=2.0, color="slateblue")
    axes[1, 0].axvline(0.0, color="grey", ls=":")
    axes[1, 0].set_title("Final coverage error vs lambda_I")
    axes[1, 0].set_xlabel("lambda_I")
    axes[1, 0].set_ylabel("final ||c(.,t)-pi_bar(.)||_1")

    colors = ["crimson" if value < 0 else "seagreen" for value in values_arr]
    axes[1, 1].scatter(means_disp, means_enc, c=colors, s=55)
    for value, x, y in zip(values_arr, means_disp, means_enc):
        axes[1, 1].annotate(f"{value:+.0f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    axes[1, 1].set_title("Regime map")
    axes[1, 1].set_xlabel("mean S(t)")
    axes[1, 1].set_ylabel("mean sum_k occ_k^2")

    fig.suptitle("Signed lambda_I sweep (isolated stale-information test)")
    fig.tight_layout()
    return fig, summary


def make_animation(res: SimResult, fps: int = 12, filename: str = "maxcal_info_diffusion.gif"):
    w = res.w
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        res.mean_info_field.reshape(w.Ny, w.Nx),
        origin="lower",
        cmap="magma",
        alpha=0.45,
        extent=(0, w.Nx, 0, w.Ny),
    )
    ax.set_aspect("equal")
    ax.set_xlim(0, w.Nx)
    ax.set_ylim(0, w.Ny)
    ax.set_xlabel("col")
    ax.set_ylabel("row")
    scat = ax.scatter([], [], s=14, alpha=0.8, color="navy")
    title = ax.set_title("")

    def update(frame: int):
        xs, ys, t = res.pos_snapshots[frame]
        scat.set_offsets(np.column_stack([xs / w.cell_size, ys / w.cell_size]))
        title.set_text(f"t = {t}")
        return scat, title

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(res.pos_snapshots),
        interval=1000 // fps,
        blit=False,
    )
    try:
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        print(f"Saved {filename}")
    except Exception as exc:
        print(f"Could not save animation ({exc}); skipping.")
    plt.close(fig)


# ============================================================
# 8. ENTRY POINT
# ============================================================

def main():
    print("MaxCal Coverage + Information Diffusion (sign-consistent Stage II)")
    print(f"  World     : {NX}x{NY} grid, cell={CELL_SIZE} m")
    print(f"  Swarm     : {N_ROBOTS} robots, speed={ROBOT_SPEED} m/step")
    print(f"  Duration  : {T_SIM} steps")
    print(f"  lambda_C  : {LAMBDA_C_VAL}")
    print(f"  lambda_I  : {LAMBDA_I_VAL}")
    print(f"  A_half    : {A_HALF}")
    print(f"  R_meet    : {R_MEET}")
    print()
    print("Pinciroli sign interpretation implemented here:")
    print("  lambda_I < 0  -> attraction to high-information cells -> clustering")
    print("  lambda_I = 0  -> pure coverage baseline")
    print("  lambda_I > 0  -> repulsion from high-information cells -> coverage priority")
    print()

    print("Running central operating-point simulation...")
    result = run_simulation()
    f_star, prominence, _, _ = dominant_frequency(result.t_axis, result.dispersion)
    print(f"  Total Markov arrivals : {result.markov_step_history[-1]}")
    print(f"  Mean dispersion S     : {result.dispersion.mean():.3f}")
    print(f"  Mean encounter proxy  : {result.encounter_proxy.mean():.5f}")
    print(f"  Final coverage L1     : {result.coverage_l1[-1]:.5f}")
    if f_star > 0:
        print(f"  Dominant frequency f* : {f_star:.4e} (period ~ {1.0 / f_star:.0f} steps)")
    else:
        print("  Dominant frequency f* : none detected")
    print(f"  Spectral prominence   : {prominence:.2f}")
    print()

    print("Saving main figure -> maxcal_info_diffusion_main.png")
    fig_main = make_main_figure(result)
    fig_main.savefig("maxcal_info_diffusion_main.png", dpi=120)
    plt.close(fig_main)

    print("Saving age-plane figure -> maxcal_info_diffusion_age_plane.png")
    fig_age = make_age_plane_figure(result)
    fig_age.savefig("maxcal_info_diffusion_age_plane.png", dpi=120)
    plt.close(fig_age)

    print("Saving animation -> maxcal_info_diffusion.gif")
    make_animation(result)

    print("Running signed lambda_I sweep (isolated stale-information test)...")
    fig_sweep, summary = make_constraint_sweep_figure()
    fig_sweep.savefig("maxcal_info_diffusion_sweep.png", dpi=120)
    plt.close(fig_sweep)
    print("  Saved maxcal_info_diffusion_sweep.png")
    print("  Sweep summary (lambda_I, mean S, mean encounter, final L1, f*, prom, regime):")
    for row in summary:
        print(
            "    "
            f"{row[0]:+5.1f}  "
            f"S={row[1]:7.3f}  "
            f"E={row[2]:.5f}  "
            f"L1={row[3]:.5f}  "
            f"f*={row[4]:.4e}  "
            f"prom={row[5]:6.2f}  "
            f"-> {row[6]}"
        )


if __name__ == "__main__":
    main()
