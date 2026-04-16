"""
maxcal_coverage.py

SETUP:
    pip install numpy matplotlib pillow

RUN:
    python maxcal_coverage.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

import argparse
import os


# ============================================================
# PARAMETERS 
# ============================================================

NX = 20                  # grid columns
NY = 20                  # grid rows
CELL_SIZE = 1.0          # metres per cell → world is 20 × 20 m
N_ROBOTS = 50            # number of robots in the swarm   
ROBOT_SPEED  = 0.15      # m / simulation step
T_SIM = 12_000           # total simulation steps
RECORD_EVERY = 60        # steps between c_k(t) snapshots
SNAP_EVERY = 300         # steps between position snapshots
LAMBDA_C_VAL = 0.0       # symmetric multiplier (see §1 note)
SEED = 42

# NOTE on LAMBDA_C_VAL = 0:
#   Theory Eq.(1): p*(k2|k1) = w_{k2,k1}·exp(−λ_C^{k2}) / Z(k1)
#   With equal multipliers ∀k, exp(−λ_C) cancels in the ratio,
#   giving a uniform random walk: p*(k2|k1) = 1 / |N(k1)|.
#   The stationary distribution is then π_k ∝ deg(k) (see §3).
#   We keep the full formula so non-uniform λ_C^k slots in later.


# ============================================================
# 1.  WORLD MODEL
# ============================================================

@dataclass
class World:
    Nx: int
    Ny: int
    K: int
    cell_size: float
    adjacency: List[List[int]]   # 0-indexed; 8-connected


def build_world(Nx: int, Ny: int, cell_size: float) -> World:
    K = Nx * Ny     # grid size K, each cell is a state k in the the Markov Model
    adj: List[List[int]] = [[] for _ in range(K)]   # make a list that for each cell holds a list of integers (all the connected neighbors)

    for k in range(K):
        row, col = divmod(k, Nx)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if not (0 <= nr < Ny and 0 <= nc < Nx):
                    continue
                adj[k].append(nr * Nx + nc)

    return World(Nx, Ny, K, cell_size, adj)


def region_center(w: World, k: int) -> Tuple[float, float]:
    row, col = divmod(k, w.Nx)
    x = (col + 0.5) * w.cell_size
    y = (row + 0.5) * w.cell_size
    return x, y


# ============================================================
# 2. MAXCAL TRANSITION PROBABILITIES
# ============================================================
#
# Theory, Eq.(1):
#   p*(k2 | k1) = w_{k2,k1} · exp(−λ_C^{k2}) / Z(k1)
#   Z(k1)       = Σ_{k2 ∈ N(k1)} w_{k2,k1} · exp(−λ_C^{k2})
#
# Here w = 1 for all edges (isotropic cost, absorbed into adjacency).
# We pre-compute the full K×K matrix so the inner loop is a table look-up.
# Later: replace or augment lambda_C to add λ_E (energy) or λ_I (info).


def build_transition_matrix(w: World, lambda_C: np.ndarray) -> np.ndarray:
    P = np.zeros((w.K, w.K), dtype=np.float64)
    for k1 in range(w.K):
        neighbors = w.adjacency[k1]
        # scores = np.exp(-lambda_C[neighbors] -lambda_I * density[neighbors]) new equation for later
        scores = np.exp(-lambda_C[neighbors])
        Z = scores.sum()
        for i, k2 in enumerate(neighbors):
            P[k1, k2] = scores[i] / Z
    return P


def sample_next_region(w: World, k: int, P: np.ndarray, rng: np.random.Generator) -> int:
    nb = w.adjacency[k]
    probs = P[k, nb]
    r = rng.random()
    s = 0.0
    for i, p in enumerate(probs):
        s += p
        if s >= r:
            return nb[i]
    return nb[-1]   # fallback for floating-point edge case

#TODO: change this for diffusion, each timestep we want to increase prob of moving to cluster
def sample_next_region_diffusion(w,k,lambda_C,lambda_I,density,rng):
    #  neighbors = w.adjacency[k]
    
    # scores = np.exp(
    #     -lambda_C[neighbors]
    #     -lambda_I * density[neighbors]
    # )
    
    # probs = scores / scores.sum()
    # return rng.choice(neighbors, p=probs)
    pass

# ============================================================
# 3. THEORETICAL STATIONARY DISTRIBUTION
# ============================================================
#
# For a reversible MC with unit edge weights and equal λ_C^k,
# detailed balance π_k1 · P[k1,k2] = π_k2 · P[k2,k1] gives:
#
#   π_k1 / deg(k1) = π_k2 / deg(k2)  ⟹  π_k ∝ deg(k)
#
# This is the "ground truth" we compare the simulation against.
#
# Degree census for a 20×20 8-connected grid:
#   Corner cells (4 total):        degree 3
#   Non-corner edge cells (72):    degree 5
#   Interior cells (324):          degree 8

def theoretical_stationary(w: World) -> np.ndarray:
    degrees = np.array([len(w.adjacency[k]) for k in range(w.K)], dtype=np.float64)
    return degrees / degrees.sum()


# ============================================================
# 4. ROBOT
# ============================================================
# Fields
#   from_k  — the Markov-chain state: last region the robot arrived at.
#             This is the variable s_t in the theory.
#   to_k    — target region (already sampled from p*(· | from_k)).
#   (tx,ty) — continuous-space centre of to_k.
#
# DESIGN CHOICE — "sample on arrival":
#   The Markov transition p*(k2|k1) fires ONLY when the robot
#   physically reaches the centre of to_k. Between transitions
#   the Markov state is frozen. Continuous motion is not part
#   of the Markov chain; it is only its physical realisation.
#
# DESIGN CHOICE — capped movement:
#   At each step the robot moves min(speed, dist) toward the target.
#   This prevents the oscillation that arises in discrete-time
#   steering when speed > 2 · tolerance: if we move a fixed step
#   and overshoot, the robot bounces forever just outside the
#   tolerance radius. Capping guarantees arrival in ⌈dist/speed⌉
#   steps exactly, making transit time a deterministic function
#   of geometry — as the theory implicitly assumes.

@dataclass
class Robot:
    id: int
    x: float
    y: float
    from_k: int     # Markov state
    to_k: int       # next target region
    tx: float       # target centre x
    ty: float       # target centre y


def make_robot(idx: int, w: World, P: np.ndarray, rng: np.random.Generator) -> Robot:
    k0 = int(rng.integers(0, w.K))
    cx, cy = region_center(w, k0)
    #TODO: make sure the first step is initialized randomly, we dont have density yet so cannot compute 
    k1 = sample_next_region(w, k0, P, rng) 
    tx, ty = region_center(w, k1)
    return Robot(idx, cx, cy, k0, k1, tx, ty)


def step_robot(r: Robot, w: World, P: np.ndarray,
               speed: float, markov_visits: np.ndarray,
               rng: np.random.Generator) -> None:
    dx = r.tx - r.x
    dy = r.ty - r.y
    dist = math.sqrt(dx * dx + dy * dy)

    # ── IN TRANSIT: capped movement ──────────────────────────
    move = min(speed, dist)
    # Move toward target; never overshoot.
    if dist > 1e-10:
        r.x += move * dx / dist
        r.y += move * dy / dist

    # ── ARRIVAL TEST ─────────────────────────────────────────
    # Position matches target to floating-point precision.
    if abs(r.x - r.tx) < 1e-9 and abs(r.y - r.ty) < 1e-9:
        
        # Update Markov state
        r.from_k = r.to_k

        # Record arrival — this is the "visit" counted in c_k(t):
        #   c_k(t) = (# arrivals at k up to Markov-time t) / (total arrivals)
        markov_visits[r.from_k] += 1

        # ── MAXCAL TRANSITION: sample p*(k2 | from_k), Eq.(1) ───
        
        #TODO: change this to sample_next_region for diffusion
        next_k = sample_next_region(w, r.from_k, P, rng)
        
        r.to_k = next_k
        r.tx, r.ty = region_center(w, next_k)


# ============================================================
# 5.  MAIN SIMULATION
# ============================================================

@dataclass
class SimResult:
    w: World
    pi_empirical: np.ndarray
    ck_history: List[np.ndarray]                              # c_k(t) snapshots
    markov_step_history: List[int]                            # total arrivals at each snapshot
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]]   # (xs, ys, t)

# God-Like function to compute density at each grid cell
def compute_density(w, robots):
    density = np.zeros(w.K)
    for r in robots:
        density[r.from_k] += 1
    return density / len(robots)  # normalize

def run_simulation(speed: float = ROBOT_SPEED,
                   T: int = T_SIM,
                   lambda_C_val: float = LAMBDA_C_VAL,
                   seed: int = SEED) -> SimResult:
    rng = np.random.default_rng(seed)
    w = build_world(NX, NY, CELL_SIZE)
    lambda_C = np.full(w.K, lambda_C_val)
    lambda_I = 0.003 # new parameter to control diffusion strength?
   
    # TODO: do this once in the beginning
    P = build_transition_matrix(w, lambda_C, lambda_I, density) 

    # should accept 
    robots = [make_robot(i, w, P, rng) for i in range(N_ROBOTS)]
    markov_visits = np.zeros(w.K, dtype=np.int64)
    ck_history: List[np.ndarray] = []
    ms_history: List[int] = []
    pos_snapshots: List[Tuple[np.ndarray, np.ndarray, int]] = []
    
    density = compute_density(w, robots)


    for t in range(1, T + 1):
        for r in robots:
            step_robot(r, w, P, speed, markov_visits, rng) #TODO: update robot using density not P

        total = int(markov_visits.sum())
        if t % RECORD_EVERY == 0 and total > 0:
            ck_history.append(markov_visits / total)
            ms_history.append(total)

        if t % SNAP_EVERY == 0:
            xs = np.array([r.x for r in robots])
            ys = np.array([r.y for r in robots])
            pos_snapshots.append((xs, ys, t))

    pi_emp = markov_visits / markov_visits.sum()
    return SimResult(w, pi_emp, ck_history, ms_history, pos_snapshots)


# ============================================================
# 6. VISUALISATION
# ============================================================

def make_main_figure(res: SimResult):
    w = res.w
    pi_theory = theoretical_stationary(w)
    pi_emp = res.pi_empirical

    # Representative regions (one of each degree class)
    k_corner = 0                                  # (0,0) → degree 3
    k_edge = NX // 2                              # bottom-edge middle → degree 5
    k_interior = (NY // 2) * NX + NX // 2        # centre → degree 8

    vmin = float(pi_theory.min())
    vmax = float(pi_theory.max())

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(pi_theory.reshape(w.Ny, w.Nx), origin="lower",
                     cmap="viridis", vmin=vmin, vmax=vmax)
    ax1.set_title("(a) Theoretical π̄_k ∝ deg(k)")
    ax1.set_xlabel("col"); ax1.set_ylabel("row")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(pi_emp.reshape(w.Ny, w.Nx), origin="lower",
                     cmap="viridis", vmin=vmin, vmax=vmax)
    ax2.set_title("(b) Empirical π̂_k")
    ax2.set_xlabel("col"); ax2.set_ylabel("row")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    err = np.abs(pi_emp - pi_theory)
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(err.reshape(w.Ny, w.Nx), origin="lower", cmap="Reds")
    ax3.set_title("(c) |π̂_k − π̄_k|")
    ax3.set_xlabel("col"); ax3.set_ylabel("row")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    # Coverage convergence: theory's Eq.(3) predicts rate ∝ 1/(t+1)
    ax4 = fig.add_subplot(gs[1, 0:2])
    ms = res.markov_step_history
    for k, label, col in [
        (k_corner,   "Corner   (deg 3)", "red"),
        (k_edge,     "Edge     (deg 5)", "darkorange"),
        (k_interior, "Interior (deg 8)", "seagreen"),
    ]:
        ck = [snap[k] for snap in res.ck_history]
        ax4.plot(ms, ck, color=col, lw=2, label=label)
        ax4.axhline(pi_theory[k], color=col, ls="--", lw=1.2)
    ax4.set_title("(d) Coverage convergence (dashes = theory)")
    ax4.set_xlabel("Markov steps (arrivals)")
    ax4.set_ylabel("c_k(t)")
    ax4.legend(loc="right")

    ax5 = fig.add_subplot(gs[1, 2])
    xs, ys, t_end = res.pos_snapshots[-1]
    ax5.scatter(xs, ys, s=12, alpha=0.7, color="dodgerblue")
    ax5.set_xlim(0, w.Nx * w.cell_size)
    ax5.set_ylim(0, w.Ny * w.cell_size)
    ax5.set_aspect("equal")
    ax5.set_title(f"(e) Robot positions  t={t_end}")
    ax5.set_xlabel("x (m)"); ax5.set_ylabel("y (m)")

    fig.suptitle("MaxCal Coverage (Sec. 4.2.1) — symmetric λ_C, 8-connected grid")
    fig.tight_layout()
    return fig


def make_phase_figure(T: int = T_SIM):
    """
    Phase diagram: convergence speed vs robot speed.

    Left plot:  Markov time → curves should be speed-independent.
    Right plot: simulation time → curves spread out by speed.
    """
    # Phase diagram: convergence speed vs robot speed.
    #
    # KEY INSIGHT: the theory's convergence rate 1/(t+1) (Eq.3) is
    # in MARKOV time (number of arrivals).  In SIMULATION time the
    # rate scales with robot speed: faster robots = more Markov steps
    # per simulation step = faster apparent convergence.
    # Left plot: Markov time → curves should be speed-independent.
    # Right plot: simulation time → curves spread out by speed.


    w = build_world(NX, NY, CELL_SIZE)
    pi_theory = theoretical_stationary(w)
    k_int = (NY // 2) * NX + NX // 2
    speeds = [0.05, 0.10, 0.20, 0.50, 1.00]

    fig, (ax_m, ax_s) = plt.subplots(1, 2, figsize=(12, 5.2))

    for spd in speeds:
        res = run_simulation(speed=spd, T=T)
        sim_ts = np.arange(1, len(res.ck_history) + 1) * RECORD_EVERY
        errs = np.array([
            max(abs(snap[k_int] - pi_theory[k_int]), 1e-8)
            for snap in res.ck_history
        ])
        ax_m.plot(res.markov_step_history, errs, lw=2, label=f"speed={spd}")
        ax_s.plot(sim_ts, errs, lw=2, label=f"speed={spd}")

    ax_m.set_yscale("log")
    ax_m.set_title("Convergence in Markov time\n(speed-independent prediction)")
    ax_m.set_xlabel("Total Markov steps")
    ax_m.set_ylabel("|c_k(t) − π̄_k|")
    ax_m.legend(loc="upper right")

    ax_s.set_yscale("log")
    ax_s.set_title("Convergence in simulation time\n(faster robot → faster per step)")
    ax_s.set_xlabel("Simulation steps")
    ax_s.set_ylabel("|c_k(t) − π̄_k|")
    ax_s.legend(loc="upper right")

    fig.tight_layout()
    return fig


def make_animation(res: SimResult, fps: int = 12, filename: str = "maxcal_coverage.gif"):
    w = res.w
    pi_theory = theoretical_stationary(w)
    bg = pi_theory.reshape(w.Ny, w.Nx)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(bg, origin="lower", cmap="YlOrRd", alpha=0.5,
              extent=(0, w.Nx, 0, w.Ny))
    ax.set_aspect("equal")
    ax.set_xlim(0, w.Nx); ax.set_ylim(0, w.Ny)
    ax.set_xlabel("col"); ax.set_ylabel("row")
    scat = ax.scatter([], [], s=12, alpha=0.8, color="navy")
    title = ax.set_title("")

    def update(frame):
        xs, ys, t = res.pos_snapshots[frame]
        scat.set_offsets(np.column_stack([xs / w.cell_size, ys / w.cell_size]))
        title.set_text(f"t = {t}")
        return scat, title

    anim = animation.FuncAnimation(
        fig, update, frames=len(res.pos_snapshots), interval=1000 // fps, blit=False
    )

    try:
        anim.save(filename, writer=animation.PillowWriter(fps=fps))
        print(f"Saved {filename}")
    except Exception as e:
        print(f"Could not save animation ({e}); skipping.")
    plt.close(fig)


# ============================================================
# 7. PARSING ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="MaxCal Coverage Simulation")
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "Figures/Coverage"
        ),
        help="Directory to save output figures (default: Figures)"
    )
    return parser.parse_args()

# ============================================================
# 8. ENTRY POINT
# ============================================================

def main():
    args = parse_args()
    outdir = args.outdir

    # Create directory if it doesn't exist
    os.makedirs(outdir, exist_ok=True)

    print("MaxCal Coverage Simulation")
    print(f"  Output  : {outdir}/")
    print(f"  World   : {NX}x{NY} grid, cell={CELL_SIZE} m")
    print(f"  Swarm   : {N_ROBOTS} robots, speed={ROBOT_SPEED} m/step")
    print(f"  Duration: {T_SIM} steps")
    print()

    print("Running simulation...")
    result = run_simulation()
    print(f"  Done. Total Markov steps: {result.markov_step_history[-1]}")
    print(f"  Expected per robot: ~{round(result.markov_step_history[-1] / N_ROBOTS)}")
    print()

    # ---- Save outputs into directory ----
    main_path = os.path.join(outdir, "maxcal_coverage_main.png")
    gif_path = os.path.join(outdir, "maxcal_coverage.gif")
    phase_path = os.path.join(outdir, "maxcal_coverage_phase.png")

    print(f"Saving main figure → {main_path}")
    fig_main = make_main_figure(result)
    fig_main.savefig(main_path, dpi=120)
    plt.close(fig_main)

    print(f"Saving animation → {gif_path}")
    make_animation(result, filename=gif_path)

    print("Saving phase diagram (runs 5 additional simulations)...")
    fig_phase = make_phase_figure()
    fig_phase.savefig(phase_path, dpi=120)
    plt.close(fig_phase)
    print(f"  Saved {phase_path}")


if __name__ == "__main__":
    main()